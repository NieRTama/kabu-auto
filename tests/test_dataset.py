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


def _policy_conf(stop=-0.07, breakeven=0.02, trailing=0.04,
                 sell_thr=-0.25, max_holding=10):
    return policy.PolicyConfig(
        stop_loss_pct=stop, breakeven_trigger_pct=breakeven,
        trailing_stop_pct=trailing, sell_threshold=sell_thr,
        max_holding_sessions=max_holding,
    )


def _costs(slip=0.0, comm=0.0):
    return execution.CostConfig(slippage_pct=slip, commission_pct=comm)


def _frame(bars: list[dict]) -> pd.DataFrame:
    """simulate_event に渡す最小のフレーム（日付index・OHLC・feature_valid）"""
    df = pd.DataFrame(bars).set_index("date")
    df.index = pd.to_datetime(df.index)
    df["feature_valid"] = True
    return df


class TestSimulateEvent:
    def test_stop_loss_gives_label_zero(self):
        """損切りで終わった候補は label=0、純収益は負"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 2), "open": 1000, "high": 1005, "low": 900, "close": 910},
        ]
        out = dataset.simulate_event(_frame(bars), 0, _policy_conf(), _costs())
        assert out.status == dataset.STATUS_RESOLVED
        assert out.entry_at == date(2026, 9, 2)
        assert out.label_end_at == date(2026, 9, 2)
        assert out.exit_reason == policy.STOP_LINE
        assert out.entry_price == pytest.approx(1000.0)
        assert out.exit_price == pytest.approx(930.0)   # 基準線 1000*0.93
        assert out.net_return == pytest.approx(-0.07)
        assert out.label == 0

    def test_zero_net_return_is_label_zero(self):
        """純収益0は label=0 に含める（境界の明示）"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 2), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 3), "open": 1000, "high": 1005, "low": 995, "close": 1000},
        ]
        # max_holding=1 で満了 → 翌営業日(9/3)の寄り1000で退出。入りも1000なので0%
        out = dataset.simulate_event(
            _frame(bars), 0, _policy_conf(max_holding=1), _costs())
        assert out.status == dataset.STATUS_RESOLVED
        assert out.exit_reason == policy.TIME_LIMIT
        assert out.net_return == pytest.approx(0.0)
        assert out.label == 0

    def test_profitable_exit_gives_label_one(self):
        """トレーリングで利益が残れば label=1"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 2), "open": 1000, "high": 1200, "low": 1150, "close": 1190},
            {"date": date(2026, 9, 3), "open": 1190, "high": 1195, "low": 1100, "close": 1110},
        ]
        out = dataset.simulate_event(_frame(bars), 0, _policy_conf(), _costs())
        assert out.status == dataset.STATUS_RESOLVED
        assert out.exit_reason == policy.TRAILING
        assert out.label_end_at == date(2026, 9, 3)
        assert out.exit_price == pytest.approx(1152.0)  # ピーク1200 * 0.96
        assert out.net_return == pytest.approx(0.152)
        assert out.label == 1

    def test_immature_when_bars_run_out(self):
        """最大保有期間まで足が届かない候補は未成熟。ラベルを付けない"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 2), "open": 1000, "high": 1005, "low": 995, "close": 1000},
        ]
        out = dataset.simulate_event(
            _frame(bars), 0, _policy_conf(max_holding=10), _costs())
        assert out.status == dataset.STATUS_IMMATURE
        assert out.label is None
        assert out.net_return is None
        assert out.label_end_at is None

    def test_unfilled_when_no_entry_bar(self):
        """翌営業日の足が無ければエントリーできない（未約定）"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
        ]
        out = dataset.simulate_event(_frame(bars), 0, _policy_conf(), _costs())
        assert out.status == dataset.STATUS_UNFILLED
        assert out.label is None
        assert out.entry_at is None

    def test_unfilled_when_market_exit_has_no_next_bar(self):
        """満了の成行退出に必要な翌営業日の足が無ければ未約定"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 2), "open": 1000, "high": 1005, "low": 995, "close": 1000},
        ]
        # max_holding=1 → 9/2 に満了意図が出るが、翌足が無い
        out = dataset.simulate_event(
            _frame(bars), 0, _policy_conf(max_holding=1), _costs())
        assert out.status == dataset.STATUS_UNFILLED
        assert out.label is None

    def test_invalid_features_are_not_simulated(self):
        """特徴量が揃っていない判断セッションはシミュレートしない"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 2), "open": 1000, "high": 1005, "low": 900, "close": 910},
        ]
        feat = _frame(bars)
        feat.iloc[0, feat.columns.get_loc("feature_valid")] = False
        out = dataset.simulate_event(feat, 0, _policy_conf(), _costs())
        assert out.status == dataset.STATUS_INVALID_FEATURES
        assert out.label is None

    def test_entry_uses_next_open_not_decision_close(self):
        """判断した日の終値では約定しない（F04の回帰防止）"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1005},
            {"date": date(2026, 9, 2), "open": 980, "high": 1005, "low": 900, "close": 910},
        ]
        out = dataset.simulate_event(_frame(bars), 0, _policy_conf(), _costs())
        assert out.entry_price == pytest.approx(980.0)   # 翌日の寄り
        assert out.entry_price != pytest.approx(1005.0)  # 判断日の終値ではない

    def test_costs_are_reflected_in_net_return(self):
        """スリッページと手数料が純収益に反映される"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 2), "open": 1000, "high": 1005, "low": 900, "close": 910},
        ]
        free = dataset.simulate_event(_frame(bars), 0, _policy_conf(), _costs())
        charged = dataset.simulate_event(
            _frame(bars), 0, _policy_conf(), _costs(slip=0.001, comm=0.001))
        assert charged.net_return < free.net_return


class TestBuildEvents:
    def test_columns_and_order_are_fixed(self, monkeypatch):
        feat_len = 120
        ohlcv = _ohlcv(feat_len)
        fake = pd.Series([0.99] * feat_len, index=indicators.build_feature_frame(ohlcv).index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        events = dataset.build_events("7203", ohlcv, _policy_conf(), _costs())
        assert list(events.columns) == dataset.EVENT_COLUMNS

    def test_every_row_carries_identity_and_versions(self, monkeypatch):
        ohlcv = _ohlcv(120)
        fake = pd.Series([0.99] * 120, index=indicators.build_feature_frame(ohlcv).index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        events = dataset.build_events("7203", ohlcv, _policy_conf(), _costs())
        assert len(events) > 0
        assert (events["symbol"] == "7203").all()
        assert (events["feature_version"] == dataset.FEATURE_VERSION).all()
        assert (events["strategy_version"] == dataset.STRATEGY_VERSION).all()
        assert (events["execution_model_version"] == dataset.EXECUTION_MODEL_VERSION).all()
        assert events["event_id"].is_unique

    def test_decision_at_precedes_entry_at(self, monkeypatch):
        """判断は執行より前。同じセッションで判断して約定しない（F04）"""
        ohlcv = _ohlcv(120)
        fake = pd.Series([0.99] * 120, index=indicators.build_feature_frame(ohlcv).index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        events = dataset.build_events("7203", ohlcv, _policy_conf(), _costs())
        filled = events[events["entry_at"].notna()]
        assert len(filled) > 0
        assert (filled["decision_at"] < filled["entry_at"]).all()

    def test_label_end_at_is_not_before_entry_at(self, monkeypatch):
        ohlcv = _ohlcv(120)
        fake = pd.Series([0.99] * 120, index=indicators.build_feature_frame(ohlcv).index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        events = dataset.build_events("7203", ohlcv, _policy_conf(), _costs())
        resolved = events[events["status"] == dataset.STATUS_RESOLVED]
        assert len(resolved) > 0
        assert (resolved["label_end_at"] >= resolved["entry_at"]).all()

    def test_only_resolved_rows_have_labels(self, monkeypatch):
        """未成熟・未約定にラベルが付いていない（spec §14 段階B完了条件）"""
        ohlcv = _ohlcv(120)
        fake = pd.Series([0.99] * 120, index=indicators.build_feature_frame(ohlcv).index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        events = dataset.build_events("7203", ohlcv, _policy_conf(), _costs())
        unresolved = events[events["status"] != dataset.STATUS_RESOLVED]
        assert unresolved["label"].isna().all()
        resolved = events[events["status"] == dataset.STATUS_RESOLVED]
        assert resolved["label"].notna().all()
        assert set(resolved["label"].unique()) <= {0, 1}

    def test_tail_sessions_are_not_labelled(self, monkeypatch):
        """末尾の候補は決着に必要な足が無いのでラベルが付かない。

        足の残り本数により immature（決着しなかった）にも unfilled（約定
        できなかった）にもなり得るが、いずれもラベルは付けない。
        """
        ohlcv = _ohlcv(120)
        fake = pd.Series([0.99] * 120, index=indicators.build_feature_frame(ohlcv).index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        events = dataset.build_events(
            "7203", ohlcv, _policy_conf(max_holding=10), _costs())
        last = events.sort_values("decision_at").iloc[-1]
        assert last["status"] != dataset.STATUS_RESOLVED
        assert pd.isna(last["label"])

    def test_features_are_carried_on_each_row(self, monkeypatch):
        ohlcv = _ohlcv(120)
        fake = pd.Series([0.99] * 120, index=indicators.build_feature_frame(ohlcv).index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        events = dataset.build_events("7203", ohlcv, _policy_conf(), _costs())
        assert events[list(indicators.FEATURE_COLS)].notna().all().all()


class TestBuildEventsMulti:
    def test_concatenates_per_symbol(self, monkeypatch):
        ohlcv_a, ohlcv_b = _ohlcv(120), _ohlcv(120, start_price=500.0)
        monkeypatch.setattr(
            dataset, "rule_scores",
            lambda f: pd.Series([0.99] * len(f), index=f.index))

        events = dataset.build_events_multi(
            {"7203": ohlcv_a, "9984": ohlcv_b}, _policy_conf(), _costs())
        assert set(events["symbol"].unique()) == {"7203", "9984"}
        assert events["event_id"].is_unique

    def test_skips_symbols_with_insufficient_data(self, monkeypatch):
        """データが足りない銘柄は黙ってスキップし、他の銘柄を止めない"""
        monkeypatch.setattr(
            dataset, "rule_scores",
            lambda f: pd.Series([0.99] * len(f), index=f.index))

        events = dataset.build_events_multi(
            {"7203": _ohlcv(120), "0000": _ohlcv(5)}, _policy_conf(), _costs())
        assert set(events["symbol"].unique()) == {"7203"}

    def test_skips_a_symbol_that_actually_raises(self, monkeypatch):
        """本物の例外（必須列の欠落等）が起きた銘柄だけを、他を止めずにスキップする

        _ohlcv(5)のようなデータ不足は例外を投げず0行を返すだけなので、
        except Exceptionの分岐自体はこれまで検証されていなかった。
        """
        monkeypatch.setattr(
            dataset, "rule_scores",
            lambda f: pd.Series([0.99] * len(f), index=f.index))

        good = _ohlcv(120)
        broken = _ohlcv(120).drop(columns=["volume"])  # 必須列の欠落でKeyErrorを誘発

        events = dataset.build_events_multi(
            {"7203": good, "0000": broken}, _policy_conf(), _costs())
        assert set(events["symbol"].unique()) == {"7203"}
