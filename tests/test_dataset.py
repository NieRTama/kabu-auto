"""イベント表とラベル（src/strategy/dataset.py）のテスト

1行=1候補。候補はバージョン固定のルールだけで作り（MLの予測を候補生成に
使わない＝循環を断つ）、退出はpolicy、約定とコストはexecutionに委ねる。
未成熟・未約定・欠損は別ステータスにして学習対象から外す（spec §6）。
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.backtest import execution
from src.core import config as cfg
from src.data import database as db
from src.data.database import get_session
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


class TestSimulateEventObservationBudget:
    """simulate_event()が候補ごとに作る観測列の本数を検証する。

    run_session_series()はmax_holding_sessions個の観測で必ずTIME_LIMIT
    退出を返す（policy.step()参照）ため、それ以降の観測を作るのは無駄。
    候補数×残り日数のO(n^2)的な劣化を防ぐため、観測列の生成本数が
    系列長に依存せず一定であることを固定する（simulate-event-perf-brief.md）。
    """

    def test_observation_count_does_not_grow_with_series_length(self, monkeypatch):
        calls = {"n": 0}
        original = dataset._observation

        def counting(feat, i):
            calls["n"] += 1
            return original(feat, i)

        monkeypatch.setattr(dataset, "_observation", counting)

        # 価格を動かさない（stop/trailingが発火しない）ので、必ず
        # max_holding=5本目でTIME_LIMIT退出になる。この条件下では、
        # 観測の生成本数は系列の全長に関わらず一定になるはず。
        conf = _policy_conf(max_holding=5, stop=-0.5, trailing=0.0, breakeven=0.0)

        def flat_frame(n):
            bars = [
                {"date": date(2026, 1, 1) + timedelta(days=k),
                 "open": 1000, "high": 1005, "low": 995, "close": 1000}
                for k in range(n)
            ]
            return _frame(bars)

        calls["n"] = 0
        dataset.simulate_event(flat_frame(10), 0, conf, _costs())
        short_series_calls = calls["n"]

        calls["n"] = 0
        dataset.simulate_event(flat_frame(2000), 0, conf, _costs())
        long_series_calls = calls["n"]

        assert short_series_calls == long_series_calls
        # entry_bar(1) + observations(max_holding=5) + exit_bar(1) + next_bar(1)
        assert short_series_calls == 8


class TestSimulateEventMatchesPreOptimizationBehavior:
    """最適化前後で戻り値が1ビットも変わらないことを直接固定する。

    参照実装は最適化前のロジック（観測列を entry_idx から len(feat) まで
    毎回作る）をそのまま再現したもの。dataset.simulate_event() が
    max_holding_sessions で打ち切って作った観測列でも、
    run_session_series() が実際に消費する範囲は変わらないため、
    戻り値は完全に一致するはず（simulate-event-perf-brief.md）。
    """

    @staticmethod
    def _reference(feat, i, policy_conf, costs, *,
                   peak_basis=policy.PEAK_BASIS_PREVIOUS):
        """最適化前の実装を再現した参照実装（末尾まで観測列を作る）。"""
        if "feature_valid" in feat.columns and not bool(feat["feature_valid"].iloc[i]):
            return dataset._unresolved(dataset.STATUS_INVALID_FEATURES)

        entry_idx = i + 1
        if entry_idx >= len(feat):
            return dataset._unresolved(dataset.STATUS_UNFILLED)

        entry_bar = dataset._observation(feat, entry_idx)
        entry = execution.entry_fill(entry_bar, dataset.NOMINAL_QUANTITY, costs)

        state = policy.HoldingState(
            symbol=str(feat.attrs.get("symbol", "")),
            entry_at=entry.at,
            avg_cost=entry.price,
            quantity=entry.quantity,
            peak_price=entry.price,
            sessions_held=0,
        )
        observations = [dataset._observation(feat, k) for k in range(entry_idx, len(feat))]
        final, intent, _ = policy.run_session_series(
            state, observations, policy_conf, peak_basis=peak_basis)

        if intent is None:
            return dataset._unresolved(
                dataset.STATUS_IMMATURE, entry_at=entry.at,
                sessions_held=final.sessions_held)

        intent_idx = entry_idx + final.sessions_held - 1
        next_idx = intent_idx + 1
        exit_bar = dataset._observation(feat, intent_idx)
        next_bar = dataset._observation(feat, next_idx) if next_idx < len(feat) else None

        fill = execution.exit_fill(intent, exit_bar, next_bar, dataset.NOMINAL_QUANTITY, costs)
        if fill is None:
            return dataset._unresolved(
                dataset.STATUS_UNFILLED, entry_at=entry.at,
                sessions_held=final.sessions_held)

        ret = execution.net_return(entry, fill, costs)
        return dataset.EventOutcome(
            status=dataset.STATUS_RESOLVED,
            entry_at=entry.at,
            label_end_at=fill.at,
            label=1 if ret > 0 else 0,
            net_return=ret,
            exit_reason=intent.reason,
            entry_price=entry.price,
            exit_price=fill.price,
            sessions_held=final.sessions_held,
        )

    @staticmethod
    def _long_frame(n, seed, start_price=1000.0):
        """十分に長く、損切り・トレーリング・満了のいずれも起こりうる合成系列"""
        rng = np.random.default_rng(seed)
        start = date(2026, 1, 5)
        rows = []
        price = start_price
        for k in range(n):
            price *= 1 + rng.normal(0, 0.02)
            high = price * (1 + abs(rng.normal(0, 0.012)))
            low = price * (1 - abs(rng.normal(0, 0.012)))
            rows.append({
                "date": start + timedelta(days=k),
                "open": price, "high": high, "low": low, "close": price,
            })
        return _frame(rows)

    @pytest.mark.parametrize("max_holding", [3, 10, 20])
    @pytest.mark.parametrize("peak_basis", [
        policy.PEAK_BASIS_PREVIOUS, policy.PEAK_BASIS_SAME_SESSION])
    def test_matches_reference_across_all_candidate_positions(
            self, max_holding, peak_basis):
        """max_holdingの6倍長い系列の全候補位置で戻り値が完全一致する

        末尾付近の未成熟・未約定になる候補も含めて全位置を検査する。
        """
        n = max_holding * 6
        feat = self._long_frame(n, seed=max_holding * 100 + len(peak_basis))
        conf = _policy_conf(max_holding=max_holding)
        costs = _costs(slip=0.001, comm=0.0005)

        for i in range(n):
            got = dataset.simulate_event(feat, i, conf, costs, peak_basis=peak_basis)
            want = self._reference(feat, i, conf, costs, peak_basis=peak_basis)
            assert got == want, f"i={i} max_holding={max_holding} で不一致: {got} != {want}"


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


def _sample_events(monkeypatch, symbol="7203", n=120, start_price=1000.0):
    ohlcv = _ohlcv(n, start_price=start_price)
    monkeypatch.setattr(
        dataset, "rule_scores",
        lambda f: pd.Series([0.99] * len(f), index=f.index))
    return dataset.build_events(symbol, ohlcv, _policy_conf(), _costs())


class TestDatasetId:
    def test_same_content_gives_same_id(self, monkeypatch):
        """同一内容なら同じ dataset_id になる（spec §6）"""
        a = _sample_events(monkeypatch)
        b = _sample_events(monkeypatch)
        assert dataset.compute_dataset_id(a) == dataset.compute_dataset_id(b)

    def test_row_order_does_not_change_id(self, monkeypatch):
        """並び順は正規化で固定されるのでIDに影響しない"""
        events = _sample_events(monkeypatch)
        shuffled = events.sample(frac=1.0, random_state=7).reset_index(drop=True)
        assert dataset.compute_dataset_id(events) == dataset.compute_dataset_id(shuffled)

    def test_different_content_gives_different_id(self, monkeypatch):
        a = _sample_events(monkeypatch, symbol="7203")
        b = _sample_events(monkeypatch, symbol="9984", start_price=500.0)
        assert dataset.compute_dataset_id(a) != dataset.compute_dataset_id(b)

    def test_id_is_short_hex(self, monkeypatch):
        did = dataset.compute_dataset_id(_sample_events(monkeypatch))
        assert len(did) == 16
        assert all(c in "0123456789abcdef" for c in did)

    def test_id_is_deterministic_across_shuffles_with_mixed_label_contracts(self, monkeypatch):
        """異なるlabel_contract_idの行が混在しても、並び順を変えたら同じIDになる

        symbol+decision_atだけでは同値キーが生じ、並び替えが非決定的になる
        （実測: 40通りのシャッフルで40通りの別IDが出た）。
        """
        events = _sample_events(monkeypatch)
        # 同じ(symbol, decision_at)の組を持つ行を、別のlabel_contract_idで複製する
        duplicated = events.copy()
        duplicated["label_contract_id"] = "differentcontract"
        combined = pd.concat([events, duplicated], ignore_index=True)

        ids = set()
        for seed in range(10):
            shuffled = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            ids.add(dataset.compute_dataset_id(shuffled))
        assert len(ids) == 1

    def test_label_dtype_does_not_change_id(self, monkeypatch):
        """欠損の有無でlabelのdtypeが変わってもIDは変わらない。

        dtype推論に任せると、たまたま全件resolvedの回だけint64になって
        "1"と書かれ、欠損がある回の"1.0000000000"と別のハッシュになる。
        """
        events = _sample_events(monkeypatch)
        as_int = events.copy()
        as_int["label"] = as_int["label"].astype("object")
        as_float = events.copy()
        as_float["label"] = as_float["label"].astype("float64")
        assert dataset.compute_dataset_id(as_int) == dataset.compute_dataset_id(as_float)


class TestSaveLoad:
    def test_roundtrip_preserves_content_hash(self, monkeypatch, tmp_path):
        """保存して読み直しても dataset_id が変わらない"""
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        path = dataset.save_events(events, did, base_dir=str(tmp_path))
        assert path.exists()

        loaded = dataset.load_events(did, base_dir=str(tmp_path))
        assert dataset.compute_dataset_id(loaded) == did

    def test_roundtrip_preserves_columns_and_row_count(self, monkeypatch, tmp_path):
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        dataset.save_events(events, did, base_dir=str(tmp_path))
        loaded = dataset.load_events(did, base_dir=str(tmp_path))
        assert list(loaded.columns) == dataset.EVENT_COLUMNS
        assert len(loaded) == len(events)

    def test_roundtrip_preserves_status_and_label_semantics(self, monkeypatch, tmp_path):
        """読み直してもラベル無しのステータスにラベルが生えない"""
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        dataset.save_events(events, did, base_dir=str(tmp_path))
        loaded = dataset.load_events(did, base_dir=str(tmp_path))
        unresolved = loaded[loaded["status"] != dataset.STATUS_RESOLVED]
        assert unresolved["label"].isna().all()

    def test_written_bytes_are_deterministic(self, monkeypatch, tmp_path):
        """同一内容なら書き出したバイト列も同じ（gzipのmtimeを固定している）"""
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        p1 = dataset.save_events(events, did, base_dir=str(tmp_path / "a"))
        p2 = dataset.save_events(events, did, base_dir=str(tmp_path / "b"))
        assert dataset.file_sha256(p1) == dataset.file_sha256(p2)

    def test_file_name_is_the_dataset_id(self, monkeypatch, tmp_path):
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        path = dataset.save_events(events, did, base_dir=str(tmp_path))
        assert path.name == f"{did}.csv.gz"

    def test_roundtrip_preserves_numeric_looking_symbol(self, monkeypatch, tmp_path):
        """'0000'のような数字だけの銘柄コードが、保存・読込後もintに化けない"""
        events = _sample_events(monkeypatch, symbol="0000")
        did = dataset.compute_dataset_id(events)
        dataset.save_events(events, did, base_dir=str(tmp_path))
        loaded = dataset.load_events(did, base_dir=str(tmp_path))
        assert (loaded["symbol"] == "0000").all()
        assert dataset.compute_dataset_id(loaded) == did


class TestInputOhlcvHash:
    def test_same_input_gives_same_hash(self):
        a = {"7203": _ohlcv(50)}
        b = {"7203": _ohlcv(50)}
        assert dataset.input_ohlcv_hash(a) == dataset.input_ohlcv_hash(b)

    def test_different_input_gives_different_hash(self):
        a = {"7203": _ohlcv(50)}
        b = {"7203": _ohlcv(51)}
        assert dataset.input_ohlcv_hash(a) != dataset.input_ohlcv_hash(b)


class TestSaveDatasetMeta:
    def test_records_identity_and_provenance(self, monkeypatch, isolated_db, tmp_path):
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        path = dataset.save_events(events, did, base_dir=str(tmp_path / "ds"))
        input_hash = dataset.input_ohlcv_hash({"7203": _ohlcv(120)})

        dataset.save_dataset_meta(events, did, path, input_hash)

        with get_session() as session:
            row = session.scalar(select(db.Dataset))
        assert row.dataset_id == did
        assert row.file_sha256 == dataset.file_sha256(path)
        assert row.input_ohlcv_sha256 == input_hash
        assert row.n_events == len(events)
        assert row.feature_version == dataset.FEATURE_VERSION
        assert row.strategy_version == dataset.STRATEGY_VERSION
        assert row.execution_model_version == dataset.EXECUTION_MODEL_VERSION

    def test_collection_id_differs_between_runs(self, monkeypatch, isolated_db, tmp_path):
        """内容が同じでも採取履歴IDは実行ごとに変わる（dataset_idとは別物）"""
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        path = dataset.save_events(events, did, base_dir=str(tmp_path / "ds"))
        input_hash = dataset.input_ohlcv_hash({"7203": _ohlcv(120)})

        dataset.save_dataset_meta(events, did, path, input_hash)
        dataset.save_dataset_meta(events, did, path, input_hash)

        with get_session() as session:
            rows = list(session.scalars(select(db.Dataset)).all())
        assert len(rows) == 2
        assert rows[0].dataset_id == rows[1].dataset_id       # 内容は同じ
        assert rows[0].collection_id != rows[1].collection_id  # 採取は別

    def test_records_version_from_the_events_frame_not_the_module_constant(
            self, monkeypatch, isolated_db, tmp_path):
        """版はモジュール定数ではなくイベント表自身から記録する

        将来モジュール定数が上がった後に古い版のデータセットを再登録しても、
        DB行の版が現在のコード版で上書きされてはならない
        （コード変更による差とデータ改訂による差を分離するため）。
        """
        events = _sample_events(monkeypatch)
        events["feature_version"] = "f0-old"  # モジュール定数(FEATURE_VERSION)とは別の値
        did = dataset.compute_dataset_id(events)
        path = dataset.save_events(events, did, base_dir=str(tmp_path / "ds"))
        input_hash = dataset.input_ohlcv_hash({"7203": _ohlcv(120)})

        dataset.save_dataset_meta(events, did, path, input_hash)

        with get_session() as session:
            row = session.scalar(select(db.Dataset))
        assert row.feature_version == "f0-old"
        assert row.feature_version != dataset.FEATURE_VERSION

    def test_records_period_and_symbols(self, monkeypatch, isolated_db, tmp_path):
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        path = dataset.save_events(events, did, base_dir=str(tmp_path / "ds"))
        dataset.save_dataset_meta(events, did, path, "x")

        with get_session() as session:
            row = session.scalar(select(db.Dataset))
        assert row.period_start == events["decision_at"].min()
        assert row.period_end == events["decision_at"].max()
        assert "7203" in row.symbols_json

    def test_counts_resolved_events(self, monkeypatch, isolated_db, tmp_path):
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        path = dataset.save_events(events, did, base_dir=str(tmp_path / "ds"))
        dataset.save_dataset_meta(events, did, path, "x")

        with get_session() as session:
            row = session.scalar(select(db.Dataset))
        expected = int((events["status"] == dataset.STATUS_RESOLVED).sum())
        assert row.n_resolved == expected


def _events_with_spans(spans: list[tuple]) -> pd.DataFrame:
    """(entry_at, label_end_at) の並びから最小のイベント表を作る"""
    rows = []
    for k, (entry, end) in enumerate(spans):
        rows.append({
            "event_id": f"X:{k}",
            "symbol": "7203",
            "entry_at": entry,
            "label_end_at": end,
            "status": dataset.STATUS_RESOLVED if end is not None else dataset.STATUS_IMMATURE,
        })
    return pd.DataFrame(rows)


class TestUniquenessWeights:
    def test_isolated_events_get_weight_one(self):
        """重ならないイベントは重み1"""
        events = _events_with_spans([
            (date(2026, 9, 1), date(2026, 9, 2)),
            (date(2026, 9, 10), date(2026, 9, 11)),
        ])
        w = dataset.uniqueness_weights(events)
        assert w == pytest.approx([1.0, 1.0])

    def test_fully_overlapping_events_get_half(self):
        """完全に重なる2件はそれぞれ重み0.5"""
        events = _events_with_spans([
            (date(2026, 9, 1), date(2026, 9, 3)),
            (date(2026, 9, 1), date(2026, 9, 3)),
        ])
        w = dataset.uniqueness_weights(events)
        assert w == pytest.approx([0.5, 0.5])

    def test_partial_overlap_is_between(self):
        """一部だけ重なるイベントの重みは0.5と1.0の間"""
        events = _events_with_spans([
            (date(2026, 9, 1), date(2026, 9, 4)),
            (date(2026, 9, 3), date(2026, 9, 6)),
        ])
        w = dataset.uniqueness_weights(events)
        assert all(0.5 < x < 1.0 for x in w)

    def test_unresolved_events_get_zero(self):
        """決着していないイベントは重み0（学習に効かせない）"""
        events = _events_with_spans([
            (date(2026, 9, 1), date(2026, 9, 3)),
            (date(2026, 9, 1), None),
        ])
        w = dataset.uniqueness_weights(events)
        assert w[1] == pytest.approx(0.0)

    def test_length_matches_input(self):
        events = _events_with_spans([
            (date(2026, 9, 1), date(2026, 9, 3)),
            (date(2026, 9, 2), date(2026, 9, 5)),
            (date(2026, 9, 9), None),
        ])
        assert len(dataset.uniqueness_weights(events)) == 3

    def test_removing_an_event_changes_remaining_weights(self):
        """fold内で再計算する意味があること＝集合が変われば重みも変わる（spec §7）"""
        full = _events_with_spans([
            (date(2026, 9, 1), date(2026, 9, 3)),
            (date(2026, 9, 1), date(2026, 9, 3)),
        ])
        subset = _events_with_spans([
            (date(2026, 9, 1), date(2026, 9, 3)),
        ])
        assert dataset.uniqueness_weights(full)[0] != pytest.approx(
            dataset.uniqueness_weights(subset)[0])

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(columns=["entry_at", "label_end_at", "status"])
        assert len(dataset.uniqueness_weights(empty)) == 0


class TestLegacyLabelingUnchanged:
    """labeling.py は legacy 経路が依存しているため挙動を変えない（本計画の差分1）"""

    def test_build_training_set_still_returns_three_parts(self):
        from src.strategy import labeling

        X, y, w = labeling.build_training_set(_ohlcv(200))
        assert len(X) == len(y) == len(w)
        assert list(X.columns) == list(indicators.FEATURE_COLS)
        assert set(y.unique()) <= {0, 1}

    def test_triple_barrier_labels_still_available(self):
        from src.strategy import labeling

        feat = indicators.build_features(_ohlcv(200)).reset_index(drop=True)
        labels, t_ends = labeling.triple_barrier_labels(
            feat, pt_mult=2.0, sl_mult=2.0, max_holding=10)
        assert len(labels) == len(feat)
        assert len(t_ends) == len(feat)

    def test_docstring_marks_module_as_legacy(self):
        """v2経路はdataset.pyを使うことがモジュール自身に書かれている"""
        from src.strategy import labeling

        assert "dataset.py" in (labeling.__doc__ or "")
