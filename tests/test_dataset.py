"""イベント表とラベル（src/strategy/dataset.py）のテスト

1行=1候補。候補はバージョン固定のルールだけで作り（MLの予測を候補生成に
使わない＝循環を断つ）、退出はpolicy、約定とコストはexecutionに委ねる。
未成熟・未約定・欠損は別ステータスにして学習対象から外す（spec §6）。
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest import execution
from src.core import config as cfg
from src.strategy import dataset
from src.strategy import indicators
from src.strategy import policy


@pytest.fixture(autouse=True)
def _load_config():
    cfg.load("config.yaml")


class TestSchema:
    def test_status_values_are_distinct(self):
        """4つの状態区分が互いに異なる（未成熟を損失0に潰さないため）"""
        values = {
            dataset.STATUS_RESOLVED,
            dataset.STATUS_IMMATURE,
            dataset.STATUS_UNFILLED,
            dataset.STATUS_INVALID_FEATURES,
        }
        assert len(values) == 4

    def test_event_columns_start_with_meta_then_features(self):
        """列順は固定。メタ列のあとに特徴量列が続く"""
        assert dataset.EVENT_COLUMNS == dataset.META_COLUMNS + list(indicators.FEATURE_COLS)

    def test_meta_columns_cover_spec_requirements(self):
        """spec §6 が要求する列が揃っている"""
        required = {
            "event_id", "label_contract_id", "symbol", "decision_at",
            "feature_as_of", "entry_at",
            "label_end_at", "status", "label", "net_return", "exit_reason",
            "feature_version", "strategy_version", "execution_model_version",
        }
        assert required <= set(dataset.META_COLUMNS)

    def test_sample_weight_is_not_a_column(self):
        """一意性重みは列に持たない（fold内で再計算する。本計画の差分2）"""
        assert "sample_weight" not in dataset.EVENT_COLUMNS


class TestMakeEventId:
    def test_is_deterministic(self):
        a = dataset.make_event_id("7203", date(2026, 9, 10))
        b = dataset.make_event_id("7203", date(2026, 9, 10))
        assert a == b

    def test_differs_by_symbol_and_session(self):
        base = dataset.make_event_id("7203", date(2026, 9, 10))
        assert dataset.make_event_id("9984", date(2026, 9, 10)) != base
        assert dataset.make_event_id("7203", date(2026, 9, 11)) != base


class TestLabelContractId:
    """ラベル契約ID。このクラスだけで完結するようヘルパを局所に置く
    （_policy_conf / _costs は後続タスクのテストで定義される）。"""

    @staticmethod
    def _p(stop=-0.07, breakeven=0.02, trailing=0.04,
           sell_thr=-0.25, max_holding=10):
        return policy.PolicyConfig(
            stop_loss_pct=stop, breakeven_trigger_pct=breakeven,
            trailing_stop_pct=trailing, sell_threshold=sell_thr,
            max_holding_sessions=max_holding)

    @staticmethod
    def _c(slip=0.0, comm=0.0):
        return execution.CostConfig(slippage_pct=slip, commission_pct=comm)

    def test_same_settings_give_the_same_id(self):
        a = dataset.make_label_contract_id(self._p(), self._c())
        b = dataset.make_label_contract_id(self._p(), self._c())
        assert a == b
        assert len(a) == 12

    def test_different_costs_give_a_different_id(self):
        """コストが違えば同じ銘柄・同じ日でもラベルは別物になる

        この2つが同じIDになると、別コストで評価をやり直したときに
        過去runの実績を上書きしてしまう（外部レビューR07）。
        """
        free = dataset.make_label_contract_id(self._p(), self._c())
        costly = dataset.make_label_contract_id(
            self._p(), self._c(slip=0.001, comm=0.001))
        assert free != costly

    def test_different_exit_policy_gives_a_different_id(self):
        base = dataset.make_label_contract_id(self._p(), self._c())
        assert dataset.make_label_contract_id(
            self._p(stop=-0.03), self._c()) != base
        assert dataset.make_label_contract_id(
            self._p(max_holding=5), self._c()) != base
        assert dataset.make_label_contract_id(
            self._p(sell_thr=-0.5), self._c()) != base

    def test_id_takes_no_data_argument(self):
        """契約IDはラベルの定義だけを表し、対象データには依存しない

        dataset_id（内容ハッシュ）を実績キーに使うと、銘柄を1つ足すだけで
        過去の実績と結び付かなくなる。銘柄・日付・件数を引数に取らないこと
        自体を契約として固定する。
        """
        import inspect
        params = set(inspect.signature(dataset.make_label_contract_id).parameters)
        assert params == {"policy_conf", "costs", "peak_basis"}


def _ohlcv(n: int, start_price: float = 1000.0) -> pd.DataFrame:
    """日付インデックス・昇順・重複なしの単一銘柄OHLCVを作る"""
    start = date(2025, 1, 6)
    rows = []
    price = start_price
    for i in range(n):
        price *= 1 + 0.002 * ((i % 7) - 3)
        rows.append({
            "date": start + timedelta(days=i),
            "open": price, "high": price * 1.01, "low": price * 0.99,
            "close": price, "volume": 100000,
        })
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


class TestRuleScores:
    def test_returns_one_score_per_session(self):
        feat = indicators.build_feature_frame(_ohlcv(120))
        scores = dataset.rule_scores(feat)
        assert len(scores) == len(feat)
        assert list(scores.index) == list(feat.index)

    def test_scores_are_within_rule_range(self):
        feat = indicators.build_feature_frame(_ohlcv(120))
        scores = dataset.rule_scores(feat).dropna()
        assert ((scores >= -1.0) & (scores <= 1.0)).all()

    def test_matches_signal_module_for_a_single_session(self):
        """既存の compute_rule_score と同じ値になる（ルールを二重に書いていない）"""
        from src.strategy import signal as signal_mod

        feat = indicators.build_feature_frame(_ohlcv(120))
        scores = dataset.rule_scores(feat)
        i = 100
        expected = signal_mod.compute_rule_score(feat.iloc[i - 1:i + 1])
        assert scores.iloc[i] == pytest.approx(expected)

    def test_first_session_has_no_score(self):
        """前日が無い先頭セッションはスコアを出さない（compute_rule_scoreが2行必要）"""
        feat = indicators.build_feature_frame(_ohlcv(120))
        scores = dataset.rule_scores(feat)
        assert pd.isna(scores.iloc[0])


class TestFindCandidates:
    def test_selects_sessions_at_or_above_threshold(self, monkeypatch):
        feat = indicators.build_feature_frame(_ohlcv(120))
        fake = pd.Series([np.nan] * len(feat), index=feat.index)
        fake.iloc[74] = 0.30
        fake.iloc[80] = 0.20
        fake.iloc[90] = 0.25
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        got = dataset.find_candidates(feat, buy_threshold=0.25)
        assert got == [74, 90]

    def test_excludes_sessions_with_invalid_features(self, monkeypatch):
        """特徴量が揃っていないセッションは候補にしない"""
        feat = indicators.build_feature_frame(_ohlcv(120))
        fake = pd.Series([0.99] * len(feat), index=feat.index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        got = dataset.find_candidates(feat, buy_threshold=0.25)
        valid_positions = [i for i, v in enumerate(feat["feature_valid"]) if bool(v)]
        assert got == valid_positions

    def test_returns_empty_when_nothing_reaches_threshold(self, monkeypatch):
        feat = indicators.build_feature_frame(_ohlcv(120))
        fake = pd.Series([0.01] * len(feat), index=feat.index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        assert dataset.find_candidates(feat, buy_threshold=0.25) == []
