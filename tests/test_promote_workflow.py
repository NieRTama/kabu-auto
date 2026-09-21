"""scripts/promote_workflow.py（候補モデルの学習→評価→昇格CLI）のテスト

このCLIは**人間が手で叩くときだけ動く**。自動昇格を実装しないという段階Eの
契約（src/strategy/promotion.py のdocstring）を、スケジューラへ登録されて
いないことのテストで固定する。
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from scripts import promote_workflow as pw
from src.core import config as cfg
from src.data import database as db


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """本番DBもリポジトリも汚さないための隔離。

    dataset.save_events() は base_dir 既定値（"data/datasets"、相対パス）で
    呼ばれる（train_v2() は base_dir をモデル保存にしか渡さない）。chdir
    せずに実行するとリポジトリ直下の data/datasets/ へ .csv.gz が生成され
    続けるため、cfg.load / db.init の後に tmp_path へ chdir して相対パス
    書き込みを閉じ込める（tests/test_v2_training.py と同じ対処）。
    """
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _ohlcv(n=300, start_price=1000.0, seed=0):
    """合成OHLCV（tests/test_v2_training.py の _ohlcv と同じ生成則）"""
    rng = np.random.default_rng(seed)
    start = date(2025, 1, 6)
    rows, p = [], start_price
    for i in range(n):
        p *= 1 + rng.normal(0, 0.01)
        rows.append({"date": start + timedelta(days=i), "open": p,
                     "high": p * 1.01, "low": p * 0.99, "close": p,
                     "volume": 1_000_000})
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


class TestSafety:
    def test_not_registered_in_scheduler_or_trading_services(self):
        """人手専用。自動実行される経路を1つも作らないこと"""
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for rel in ("src/core/scheduler.py", "src/services/trading.py", "main.py"):
            text = (root / rel).read_text(encoding="utf-8")
            assert "promote_workflow" not in text, rel


class TestBootstrap:
    def test_loads_config_in_the_same_order_as_main_py(self, monkeypatch):
        """cfg.load → watchlist.load → risk_profile.load → db.init の順

        risk_profile.load() は cfg.load() が読んだ trading/strategy 節を
        high_risk の値で上書きする。順序が崩れると config.yaml の素の値で
        ラベルが作られ、本番と別のタスクを学習してしまう
        （docs/kabu-auto-ml-real-data-comparison_20260921.md §1）。
        """
        from src.core import config as cfg_mod
        from src.core import risk_profile as rp_mod
        from src.core import watchlist as wl_mod
        from src.data import database as db_mod

        calls = []

        def fake_rp_load(path="risk_profile.json"):
            calls.append(f"risk_profile.load:{path}")
            return "high_risk"

        monkeypatch.setattr(cfg_mod, "load",
                            lambda path="config.yaml": calls.append(f"cfg.load:{path}"))
        monkeypatch.setattr(wl_mod, "load",
                            lambda path="watchlists.json", legacy_path="watchlist.json":
                            calls.append(f"watchlist.load:{path}"))
        monkeypatch.setattr(rp_mod, "load", fake_rp_load)
        monkeypatch.setattr(db_mod, "init", lambda: calls.append("db.init"))

        assert pw._bootstrap("config.yaml") == "high_risk"
        assert calls == [
            "cfg.load:config.yaml",
            "watchlist.load:watchlists.json",
            "risk_profile.load:risk_profile.json",
            "db.init",
        ]


class TestCollectOhlcv:
    def test_skips_symbols_with_fewer_than_200_bars(self, monkeypatch):
        """特徴量の助走が足りない銘柄は落とす（ml_retrain と同じ基準）"""
        from src.core import watchlist as wl_mod
        from src.data import market_data

        frames = {"7203": _ohlcv(n=300), "6758": _ohlcv(n=199),
                  "9984": _ohlcv(n=200)}
        monkeypatch.setattr(wl_mod, "get_all_codes",
                            lambda: ["7203", "6758", "9984"])
        monkeypatch.setattr(market_data, "load_ohlcv",
                            lambda symbol, limit=500: frames[symbol])

        got, skipped = pw._collect_ohlcv(500)
        assert sorted(got) == ["7203", "9984"]
        assert skipped == ["6758(199本)"]

    def test_passes_the_limit_through_and_keeps_going_on_failure(self, monkeypatch):
        """limit をそのまま渡す。1銘柄の読み込み失敗で全体を止めない"""
        from src.core import watchlist as wl_mod
        from src.data import market_data

        seen = []

        def fake_load(symbol, limit=500):
            seen.append((symbol, limit))
            if symbol == "6758":
                raise RuntimeError("DB読み込み失敗")
            return _ohlcv(n=300)

        monkeypatch.setattr(wl_mod, "get_all_codes", lambda: ["7203", "6758"])
        monkeypatch.setattr(market_data, "load_ohlcv", fake_load)

        got, skipped = pw._collect_ohlcv(None)
        assert seen == [("7203", None), ("6758", None)]
        assert list(got) == ["7203"]
        assert skipped == ["6758(読み込み失敗: DB読み込み失敗)"]


class TestTrain:
    def test_builds_policy_conf_after_the_risk_profile_is_applied(
            self, isolated_db, tmp_path, monkeypatch):
        """risk_profile 適用**後**の値で policy_conf / costs を作ること

        適用漏れだと sell_threshold が config.yaml の素の値のままになり、
        ラベル定義そのものが本番と別物になる（分析報告書 §1: イベント総数が
        5,484件→12,033件と2.2倍ずれた）。_bootstrap の中で trading/strategy
        節が書き換わる状況を再現し、その後に policy_conf が作られることを固定する。
        """
        from src.strategy import v2_training

        def fake_bootstrap(config_path="config.yaml"):
            cfg.get_section("trading")["stop_loss_pct"] = -0.10
            cfg.get_section("strategy")["sell_threshold"] = -0.08
            cfg.get_section("backtest")["retrain_window_sessions"] = None
            cfg.get_section("backtest")["slippage_pct"] = 0.001
            return "high_risk"

        monkeypatch.setattr(pw, "_bootstrap", fake_bootstrap)
        monkeypatch.setattr(pw, "_collect_ohlcv",
                            lambda limit: ({"7203": _ohlcv()}, []))

        captured = {}

        def fake_train_v2(ohlcv_by_symbol, **kwargs):
            captured["ohlcv"] = ohlcv_by_symbol
            captured.update(kwargs)
            return v2_training.V2TrainingResult(
                model_id="v2-20260921T120000-abcdef12", dataset_id="ds000001",
                n_events=300, n_resolved=250, positive_rate=0.47)

        monkeypatch.setattr(v2_training, "train_v2", fake_train_v2)

        base = str(tmp_path / "models")
        assert pw.main(["--base-dir", base, "train"]) == 0
        assert captured["policy_conf"].stop_loss_pct == pytest.approx(-0.10)
        assert captured["policy_conf"].sell_threshold == pytest.approx(-0.08)
        assert captured["costs"].slippage_pct == pytest.approx(0.001)
        assert captured["window_sessions"] is None
        assert captured["trigger"] == "manual_workflow"
        assert captured["base_dir"] == base
        assert list(captured["ohlcv"]) == ["7203"]

    def test_ohlcv_limit_zero_means_full_history(
            self, isolated_db, tmp_path, monkeypatch):
        """--ohlcv-limit 0 は limit=None（LIMIT句なし＝全期間）へ変換する"""
        from src.strategy import v2_training

        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        seen = {}

        def fake_collect(limit):
            seen["limit"] = limit
            return {"7203": _ohlcv()}, []

        monkeypatch.setattr(pw, "_collect_ohlcv", fake_collect)
        monkeypatch.setattr(
            v2_training, "train_v2",
            lambda ohlcv, **kw: v2_training.V2TrainingResult(
                model_id="v2-x", dataset_id="ds1", n_events=1, n_resolved=1,
                positive_rate=0.5))

        assert pw.main(["--base-dir", str(tmp_path / "models"),
                        "train", "--ohlcv-limit", "0"]) == 0
        assert seen["limit"] is None

    def test_returns_error_when_no_candidate_was_produced(
            self, isolated_db, tmp_path, monkeypatch):
        """決着イベント不足は例外にならず model_id=None で返る。終了コード1にする"""
        from src.strategy import v2_training

        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(pw, "_collect_ohlcv",
                            lambda limit: ({"7203": _ohlcv()}, []))
        monkeypatch.setattr(
            v2_training, "train_v2",
            lambda ohlcv, **kw: v2_training.V2TrainingResult(
                model_id=None, dataset_id="ds000001", n_events=300,
                n_resolved=51, positive_rate=None,
                skipped_reason="決着したイベントが不足しています: 51件 < 200件"))

        assert pw.main(["--base-dir", str(tmp_path / "models"), "train"]) == 1

    def test_returns_error_when_no_symbol_has_enough_history(
            self, isolated_db, tmp_path, monkeypatch):
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(pw, "_collect_ohlcv", lambda limit: ({}, ["7203(10本)"]))

        assert pw.main(["--base-dir", str(tmp_path / "models"), "train"]) == 1
