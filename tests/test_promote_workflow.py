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


class TestEvaluate:
    def _meta(self):
        from datetime import datetime

        from src.strategy import model_store as ms

        return ms.ModelMeta(
            model_id="v2-20260921T120000-abcdef12",
            trained_at=datetime(2026, 9, 21, 12, 0, 0),
            training_window_sessions=None,
            feature_cols=["rsi", "ma_dev"],
            dataset_id="ds000001",
            label_contract_id="lc0001",
        )

    def test_uses_the_candidate_model_id_as_the_factory_key(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        """model_factories の鍵は候補の model_id 自身であること

        check_promotable() は load_prediction_details(run_id, model_id=候補ID)
        で突き合わせる（src/strategy/promotion.py:106）。"current_lightgbm"
        のような汎用名で保存すると予測明細が見つからず永久に昇格できない。
        """
        from src.strategy import dataset as ds
        from src.strategy import evaluation
        from src.strategy import model_store as ms

        meta = self._meta()
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(ms, "read_meta",
                            lambda model_id, base_dir="models": meta)

        stored = tmp_path / "ds000001.csv.gz"
        stored.write_bytes(b"")
        events = pd.DataFrame({"label_contract_id": ["lc0001", "lc0001"]})
        monkeypatch.setattr(
            ds, "dataset_path",
            lambda dataset_id, base_dir="data/datasets": stored)
        monkeypatch.setattr(
            ds, "load_events",
            lambda dataset_id, base_dir="data/datasets": events)

        captured = {}

        def fake_run_evaluation(ev, **kwargs):
            captured["events"] = ev
            captured.update(kwargs)
            return {
                "evaluation_run_id": "20260921T130000-0badf00d",
                "fold_results": [],
                "summary": pd.DataFrame([
                    {"fold_index": 0, "model_id": meta.model_id, "n_val": 100,
                     "roc_auc": 0.5123, "brier": 0.2476,
                     "brier_vs_constant": -0.0004},
                ]),
                "degraded_reasons": [],
            }

        monkeypatch.setattr(evaluation, "run_evaluation", fake_run_evaluation)

        assert pw.main(["evaluate", meta.model_id]) == 0
        assert captured["model_factories"] == {
            meta.model_id: evaluation.CurrentLightGBM}
        assert captured["persist"] is True
        assert captured["n_splits"] == 5
        assert captured["window_sessions"] is None
        assert captured["feature_cols"] == ["rsi", "ma_dev"]
        assert "evaluation_run_id: 20260921T130000-0badf00d" in capsys.readouterr().out

    def test_reports_degraded_reasons_and_still_succeeds(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        """degraded は評価の失敗ではない。記録は残し、昇格できない旨を伝える"""
        from src.strategy import dataset as ds
        from src.strategy import evaluation
        from src.strategy import model_store as ms

        meta = self._meta()
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(ms, "read_meta",
                            lambda model_id, base_dir="models": meta)
        stored = tmp_path / "ds000001.csv.gz"
        stored.write_bytes(b"")
        monkeypatch.setattr(
            ds, "dataset_path", lambda dataset_id, base_dir="data/datasets": stored)
        monkeypatch.setattr(
            ds, "load_events", lambda dataset_id, base_dir="data/datasets":
            pd.DataFrame({"label_contract_id": ["lc0001"]}))
        monkeypatch.setattr(
            evaluation, "run_evaluation",
            lambda ev, **kw: {
                "evaluation_run_id": "run-x", "fold_results": [],
                "summary": pd.DataFrame([{"fold_index": 0, "roc_auc": None,
                                          "brier": 0.25,
                                          "brier_vs_constant": 0.0}]),
                "degraded_reasons": ["fold 0 model=v2-x: モデルが定数に縮退"]})

        assert pw.main(["evaluate", meta.model_id]) == 0
        out = capsys.readouterr().out
        assert "degraded: 1件" in out
        assert "モデルが定数に縮退" in out
        assert "AUC平均: None" in out

    def test_warns_when_the_label_contract_does_not_match(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        from src.strategy import dataset as ds
        from src.strategy import evaluation
        from src.strategy import model_store as ms

        meta = self._meta()
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(ms, "read_meta",
                            lambda model_id, base_dir="models": meta)
        stored = tmp_path / "ds000001.csv.gz"
        stored.write_bytes(b"")
        monkeypatch.setattr(
            ds, "dataset_path", lambda dataset_id, base_dir="data/datasets": stored)
        monkeypatch.setattr(
            ds, "load_events", lambda dataset_id, base_dir="data/datasets":
            pd.DataFrame({"label_contract_id": ["OTHER"]}))
        monkeypatch.setattr(
            evaluation, "run_evaluation",
            lambda ev, **kw: {
                "evaluation_run_id": "run-x", "fold_results": [],
                "summary": pd.DataFrame([{"fold_index": 0, "roc_auc": 0.5,
                                          "brier": 0.25,
                                          "brier_vs_constant": 0.0}]),
                "degraded_reasons": []})

        assert pw.main(["evaluate", meta.model_id]) == 0
        assert "ラベル契約が一致しません" in capsys.readouterr().out

    def test_errors_when_the_candidate_is_missing(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        from src.strategy import model_store as ms

        def raise_missing(model_id, base_dir="models"):
            raise FileNotFoundError(f"メタが見つかりません: {base_dir}/candidates/{model_id}/meta.json")

        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(ms, "read_meta", raise_missing)

        assert pw.main(["evaluate", "v2-nope"]) == 1
        assert "候補モデルが見つかりません" in capsys.readouterr().out

    def test_errors_when_the_saved_event_table_is_missing(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        """イベント表が無ければ評価しない（無言で作り直して別データを評価しない）"""
        from src.strategy import dataset as ds
        from src.strategy import model_store as ms

        meta = self._meta()
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(ms, "read_meta",
                            lambda model_id, base_dir="models": meta)
        monkeypatch.setattr(
            ds, "dataset_path",
            lambda dataset_id, base_dir="data/datasets":
            tmp_path / "missing" / "ds000001.csv.gz")

        assert pw.main(["evaluate", meta.model_id]) == 1
        out = capsys.readouterr().out
        assert "イベント表がありません" in out
        assert "ds000001" in out

    def test_errors_when_no_fold_produced_a_result(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        from src.strategy import dataset as ds
        from src.strategy import evaluation
        from src.strategy import model_store as ms

        meta = self._meta()
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(ms, "read_meta",
                            lambda model_id, base_dir="models": meta)
        stored = tmp_path / "ds000001.csv.gz"
        stored.write_bytes(b"")
        monkeypatch.setattr(
            ds, "dataset_path", lambda dataset_id, base_dir="data/datasets": stored)
        monkeypatch.setattr(
            ds, "load_events", lambda dataset_id, base_dir="data/datasets":
            pd.DataFrame({"label_contract_id": ["lc0001"]}))
        monkeypatch.setattr(
            evaluation, "run_evaluation",
            lambda ev, **kw: {"evaluation_run_id": "run-x", "fold_results": [],
                              "summary": pd.DataFrame(),
                              "degraded_reasons": []})

        assert pw.main(["evaluate", meta.model_id]) == 1
        assert "fold結果が0件" in capsys.readouterr().out


class TestPromote:
    _MODEL = "v2-20260921T120000-abcdef12"
    _RUN = "20260921T130000-0badf00d"

    def _args(self, *extra):
        return ["promote", self._MODEL, self._RUN,
                "--reason", "AUC・Brierを確認し現行より悪化がないため",
                "--decided-by", "garnet", *extra]

    def test_missing_reason_or_decided_by_exits_with_argparse_error(self):
        """--reason / --decided-by は必須引数。argparse が終了コード2で落とす"""
        with pytest.raises(SystemExit) as e:
            pw.main(["promote", self._MODEL, self._RUN, "--decided-by", "garnet"])
        assert e.value.code == 2

        with pytest.raises(SystemExit) as e:
            pw.main(["promote", self._MODEL, self._RUN, "--reason", "良さそう"])
        assert e.value.code == 2

    def test_blank_reason_is_rejected_before_anything_runs(
            self, monkeypatch, capsys):
        """空白のみの理由は promote() を呼ぶ前にCLIで拒否する

        promotion.promote() の `if not reason` は "   " を通してしまう。
        """
        from src.strategy import promotion

        called = []
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": called.append("bootstrap"))
        monkeypatch.setattr(promotion, "promote",
                            lambda *a, **kw: called.append("promote"))

        assert pw.main(["promote", self._MODEL, self._RUN,
                        "--reason", "   ", "--decided-by", "garnet"]) == 1
        assert called == []
        assert "--reason が空です" in capsys.readouterr().out

    def test_blank_decided_by_is_rejected_before_anything_runs(
            self, monkeypatch, capsys):
        from src.strategy import promotion

        called = []
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": called.append("bootstrap"))
        monkeypatch.setattr(promotion, "promote",
                            lambda *a, **kw: called.append("promote"))

        assert pw.main(["promote", self._MODEL, self._RUN,
                        "--reason", "良い", "--decided-by", "  "]) == 1
        assert called == []
        assert "--decided-by が空です" in capsys.readouterr().out

    def test_forwards_arguments_to_promotion_promote(
            self, tmp_path, monkeypatch, capsys):
        from src.strategy import model_store as ms
        from src.strategy import promotion
        from src.strategy.indicators import FEATURE_COLS

        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        refs = iter([
            None,
            ms.CurrentRef(model_id=self._MODEL, previous_model_id=None,
                          switched_at=None),
        ])
        monkeypatch.setattr(ms, "read_current",
                            lambda base_dir="models": next(refs))

        captured = {}

        def fake_promote(model_id, **kwargs):
            captured["model_id"] = model_id
            captured.update(kwargs)
            return 7

        monkeypatch.setattr(promotion, "promote", fake_promote)

        base = str(tmp_path / "models")
        assert pw.main(["--base-dir", base, *self._args()]) == 0
        assert captured["model_id"] == self._MODEL
        assert captured["evaluation_run_id"] == self._RUN
        assert captured["decided_by"] == "garnet"
        assert captured["reason"] == "AUC・Brierを確認し現行より悪化がないため"
        assert captured["degraded"] is False
        assert captured["base_dir"] == base
        assert captured["expected_feature_cols"] == list(FEATURE_COLS)
        assert "promotion_id=7" in capsys.readouterr().out

    def test_blockers_are_reported_and_current_is_untouched(
            self, monkeypatch, capsys):
        """check_promotable に落ちたら promote() が ValueError を投げる。
        CLIはその文面をそのまま出して終了コード1にする（再実装しない）。
        """
        from src.strategy import model_store as ms
        from src.strategy import promotion

        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(ms, "read_current", lambda base_dir="models": None)

        def raise_blocked(model_id, **kwargs):
            raise ValueError(
                "昇格できません: 実績が1件も確定していません（予測だけでは"
                "成績を測れません） / shadow記録だけでは昇格できません")

        monkeypatch.setattr(promotion, "promote", raise_blocked)

        assert pw.main(self._args()) == 1
        out = capsys.readouterr().out
        assert "実績が1件も確定していません" in out
        assert "shadow記録だけでは昇格できません" in out

    def test_dry_run_only_checks_and_never_promotes(self, monkeypatch, capsys):
        from src.strategy import promotion

        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        called = []
        monkeypatch.setattr(promotion, "promote",
                            lambda *a, **kw: called.append("promote"))
        monkeypatch.setattr(
            promotion, "check_promotable",
            lambda model_id, **kw: promotion.PromotionCheck(
                ok=False, blockers=["未評価です（evaluation_run_id がありません）"]))

        assert pw.main(self._args("--dry-run")) == 1
        assert called == []
        assert "未評価です" in capsys.readouterr().out

    def test_dry_run_reports_ok_without_changing_current(
            self, monkeypatch, capsys):
        from src.strategy import promotion

        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        called = []
        monkeypatch.setattr(promotion, "promote",
                            lambda *a, **kw: called.append("promote"))
        monkeypatch.setattr(
            promotion, "check_promotable",
            lambda model_id, **kw: promotion.PromotionCheck(ok=True, blockers=[]))

        assert pw.main(self._args("--dry-run")) == 0
        assert called == []
        assert "昇格可能です" in capsys.readouterr().out


# 合成OHLCV(n=300×2銘柄)の決着イベントは51件しかなく本番既定の200件に
# 届かない。届かせるには n>=1500 が要り train_v2() 1回で4〜5分かかる
# （tests/test_v2_training.py:103-110 の実測）。本番のしきい値は変えず、
# テストだけ下げる。
_TEST_MIN_RESOLVED_EVENTS = 10


def _bars():
    return {"7203": _ohlcv(seed=1), "9984": _ohlcv(seed=2, start_price=500.0)}


def _run_train(tmp_path, monkeypatch) -> str:
    """実データ経路で train を走らせ、できた候補の model_id を返す"""
    from src.strategy import v2_training

    monkeypatch.setattr(pw, "_bootstrap",
                        lambda config_path="config.yaml": "high_risk")
    monkeypatch.setattr(pw, "_collect_ohlcv", lambda limit: (_bars(), []))
    monkeypatch.setattr(v2_training, "MIN_RESOLVED_EVENTS",
                        _TEST_MIN_RESOLVED_EVENTS)

    base = str(tmp_path / "models")
    assert pw.main(["--base-dir", base, "train"]) == 0
    candidates = sorted((tmp_path / "models" / "candidates").iterdir())
    assert len(candidates) == 1
    return candidates[0].name


def _record_clean_evaluation(model_id, meta, run_id="run-integration-1"):
    """degraded でない評価記録を1件作る（既存の保存関数だけを使う）。

    51件の決着イベントで run_evaluation() を通すと内側foldの校正が identity へ
    縮退して degraded_reasons が付き、check_promotable() が必ず拒否する。
    昇格の経路そのものを固定したいので、ここは実イベント表から作った
    予測・実績・実行記録の3点セットを degraded 無しで保存する
    （tests/test_model_promotion.py の _recorded_evaluation と同じ考え方）。
    """
    from src.strategy import dataset as ds
    from src.strategy import evaluation
    from src.strategy.indicators import FEATURE_COLS

    events = ds.load_events(meta.dataset_id)
    resolved = events[events["status"] == ds.STATUS_RESOLVED].head(20)
    assert len(resolved) > 0

    preds = pd.DataFrame({
        "event_id": resolved["event_id"].astype(str).values,
        "label_contract_id": resolved["label_contract_id"].astype(str).values,
        "raw_probability": 0.6,
        "calibrated_probability": 0.55,
        "fold_index": 0,
    })
    evaluation.save_predictions(preds, run_id, model_id)
    evaluation.save_outcomes(events)
    run_config = evaluation.capture_run_config(
        events, n_splits=5, window_sessions=None,
        feature_cols=list(FEATURE_COLS))
    evaluation.save_evaluation_run(
        run_id, run_config, purpose=evaluation.PURPOSE_VALIDATION,
        model_id=model_id, n_folds=5, n_predictions=len(preds),
        degraded_reasons=[])
    return run_id


class TestIntegration:
    def test_train_then_evaluate_records_predictions_under_the_candidate_id(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        """train → evaluate で、候補ID自身の予測明細と実行記録が残ること"""
        from sqlalchemy import select

        from src.data.database import EvaluationRun, Prediction, get_session

        base = str(tmp_path / "models")
        model_id = _run_train(tmp_path, monkeypatch)
        capsys.readouterr()

        assert pw.main(["--base-dir", base, "evaluate", model_id]) == 0
        out = capsys.readouterr().out
        run_id = out.split("evaluation_run_id: ")[1].splitlines()[0].strip()

        with get_session() as session:
            run = session.scalar(select(EvaluationRun).where(
                EvaluationRun.evaluation_run_id == run_id))
            preds = list(session.scalars(select(Prediction).where(
                Prediction.evaluation_run_id == run_id)).all())

        assert run is not None
        # model_factories が1件なので EvaluationRun.model_id に候補IDが入る
        assert run.model_id == model_id
        assert preds
        assert {p.model_id for p in preds} == {model_id}

    def test_promote_switches_models_current_json(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        """昇格すると models/current.json が候補を指し、記録が committed になる"""
        from sqlalchemy import select

        from src.data.database import ModelPromotion, get_session
        from src.strategy import model_store as ms
        from src.strategy import promotion

        base = str(tmp_path / "models")
        model_id = _run_train(tmp_path, monkeypatch)
        capsys.readouterr()

        meta = ms.read_meta(model_id, base_dir=base)
        run_id = _record_clean_evaluation(model_id, meta)

        assert ms.read_current(base_dir=base) is None
        assert pw.main([
            "--base-dir", base, "promote", model_id, run_id,
            "--reason", "AUC・Brierを確認し現行より悪化がないため",
            "--decided-by", "garnet"]) == 0

        ref = ms.read_current(base_dir=base)
        assert ref is not None
        assert ref.model_id == model_id

        with get_session() as session:
            row = session.scalar(select(ModelPromotion).where(
                ModelPromotion.model_id == model_id))
        assert row.state == promotion.PROMOTION_COMMITTED
        assert row.evaluation_run_id == run_id
        assert row.decided_by == "garnet"
        assert row.reason == "AUC・Brierを確認し現行より悪化がないため"
        assert row.previous_model_id is None

    def test_promote_is_refused_when_the_evaluation_is_missing(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        """評価記録が無ければ現行は変わらない（安全機構が生きていること）"""
        from src.strategy import model_store as ms

        base = str(tmp_path / "models")
        model_id = _run_train(tmp_path, monkeypatch)
        capsys.readouterr()

        assert pw.main([
            "--base-dir", base, "promote", model_id, "run-does-not-exist",
            "--reason", "とりあえず上げたい",
            "--decided-by", "garnet"]) == 1
        assert ms.read_current(base_dir=base) is None
        assert "評価実行の記録がありません" in capsys.readouterr().out
