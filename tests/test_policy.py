"""退出ポリシー（src/strategy/policy.py）のテスト

売買規則の唯一の実装。ラベル生成・バックテスト・実運用の3者が共有する。
退出条件は src/risk/manager.py:evaluate_exit() と同じ構造であること
（固定利確線は運用に存在しないため置かない）。
"""
from datetime import date

import pytest

from src.core import config as cfg
from src.strategy import policy


def _conf(stop=-0.07, breakeven=0.02, trailing=0.04,
          sell_thr=-0.25, max_holding=10) -> policy.PolicyConfig:
    return policy.PolicyConfig(
        stop_loss_pct=stop,
        breakeven_trigger_pct=breakeven,
        trailing_stop_pct=trailing,
        sell_threshold=sell_thr,
        max_holding_sessions=max_holding,
    )


def _state(avg_cost=1000.0, peak=1000.0, sessions_held=0) -> policy.HoldingState:
    return policy.HoldingState(
        symbol="7203",
        entry_at=date(2026, 9, 1),
        avg_cost=avg_cost,
        quantity=100,
        peak_price=peak,
        sessions_held=sessions_held,
    )


class TestIsArmed:
    def test_not_armed_before_trigger(self):
        """ピーク時の含み益がトリガー未満なら未発動"""
        assert policy.is_armed(_state(peak=1015.0), _conf()) is False

    def test_armed_at_trigger(self):
        """含み益+2%ちょうどで発動する"""
        assert policy.is_armed(_state(peak=1020.0), _conf()) is True

    def test_never_armed_when_trigger_disabled(self):
        """breakeven_trigger_pct=0 なら常に未発動（従来の損切りのみの挙動）"""
        assert policy.is_armed(_state(peak=2000.0), _conf(breakeven=0.0)) is False


class TestStopLine:
    def test_plain_stop_before_arming(self):
        """未発動なら基準線は取得単価×(1+stop_loss_pct)"""
        assert policy.stop_line(_state(peak=1010.0), _conf()) == pytest.approx(930.0)

    def test_raised_to_breakeven_after_arming(self):
        """発動後は取得単価まで引き上がる（元本割れリスクを取らない）"""
        # ピーク1020（+2%）でarmed。trailing線は1020*0.96=979.2で取得単価1000より下
        # なので、この時点の基準線は取得単価そのもの
        assert policy.stop_line(_state(peak=1020.0), _conf()) == pytest.approx(1000.0)

    def test_trailing_takes_over_when_higher(self):
        """ピークが伸びるとトレーリング線が取得単価を上回り、そちらが採用される"""
        # ピーク1100 → 1100*0.96=1056 > 取得単価1000
        assert policy.stop_line(_state(peak=1100.0), _conf()) == pytest.approx(1056.0)

    def test_trailing_disabled_keeps_breakeven(self):
        """trailing_stop_pct=0 なら発動後もブレークイーブン止まり"""
        line = policy.stop_line(_state(peak=1100.0), _conf(trailing=0.0))
        assert line == pytest.approx(1000.0)

    def test_matches_production_formula(self):
        """src/risk/manager.py:evaluate_exit と同じ式であること（数値で突き合わせる）"""
        avg_cost, peak = 1000.0, 1100.0
        stop_pct, trailing_pct, breakeven = -0.07, 0.04, 0.02

        # production の式をそのまま書き下したもの
        expected = avg_cost * (1 + stop_pct)
        peak_gain_pct = (peak - avg_cost) / avg_cost
        armed = breakeven > 0 and peak_gain_pct >= breakeven
        if armed:
            expected = max(expected, avg_cost)
            if trailing_pct > 0:
                expected = max(expected, peak * (1 - trailing_pct))

        actual = policy.stop_line(
            _state(avg_cost=avg_cost, peak=peak),
            _conf(stop=stop_pct, breakeven=breakeven, trailing=trailing_pct),
        )
        assert actual == pytest.approx(expected)


class TestConfigFromSettings:
    def test_reads_from_correct_sections(self):
        """trading節とstrategy節の双方から正しく読む"""
        cfg.load("config.yaml")
        conf = policy.config_from_settings()
        trading = cfg.get_section("trading")
        strategy = cfg.get_section("strategy")
        assert conf.stop_loss_pct == trading["stop_loss_pct"]
        assert conf.breakeven_trigger_pct == trading["breakeven_trigger_pct"]
        assert conf.trailing_stop_pct == trading["trailing_stop_pct"]
        assert conf.sell_threshold == strategy["sell_threshold"]
        assert conf.max_holding_sessions == strategy["tb_max_holding"]


def _obs(session=date(2026, 9, 2), o=1000.0, h=1010.0, l=990.0, c=1005.0,
         score=None) -> policy.Observation:
    return policy.Observation(session=session, open=o, high=h, low=l, close=c, score=score)


class TestStepStopTriggers:
    def test_no_exit_when_low_stays_above_line(self):
        """安値が基準線を割らなければ退出しない"""
        state = _state(avg_cost=1000.0, peak=1000.0)
        nxt, intent = policy.step(state, _obs(l=950.0), _conf())
        assert intent is None
        assert nxt.sessions_held == 1

    def test_stop_line_reason_before_arming(self):
        """未発動で基準線に到達したら STOP_LINE"""
        state = _state(avg_cost=1000.0, peak=1000.0)
        nxt, intent = policy.step(state, _obs(l=920.0), _conf())
        assert intent is not None
        assert intent.reason == policy.STOP_LINE
        assert intent.order_type == "STOP"
        assert intent.trigger_price == pytest.approx(930.0)

    def test_trailing_reason_after_arming(self):
        """発動後に基準線へ到達したら TRAILING"""
        state = _state(avg_cost=1000.0, peak=1100.0)  # armed、線は1056
        nxt, intent = policy.step(state, _obs(h=1100.0, l=1050.0), _conf())
        assert intent is not None
        assert intent.reason == policy.TRAILING
        assert intent.trigger_price == pytest.approx(1056.0)

    def test_signal_sell_is_market_order(self):
        """売りシグナルは翌営業日の寄りで成行（trigger_priceを持たない）"""
        state = _state(avg_cost=1000.0, peak=1000.0)
        nxt, intent = policy.step(state, _obs(l=990.0, score=-0.30), _conf())
        assert intent is not None
        assert intent.reason == policy.SIGNAL_SELL
        assert intent.order_type == "MARKET"
        assert intent.trigger_price is None

    def test_stop_takes_precedence_over_signal_sell(self):
        """同じ日に基準線到達と売りシグナルが揃ったら、基準線を優先する（不利側）"""
        state = _state(avg_cost=1000.0, peak=1000.0)
        nxt, intent = policy.step(state, _obs(l=920.0, score=-0.30), _conf())
        assert intent.reason == policy.STOP_LINE

    def test_time_limit_at_max_holding(self):
        """最大保有営業日数に達したら TIME_LIMIT（翌営業日の寄りで成行）"""
        state = _state(avg_cost=1000.0, peak=1000.0, sessions_held=9)
        nxt, intent = policy.step(state, _obs(l=990.0), _conf(max_holding=10))
        assert intent is not None
        assert intent.reason == policy.TIME_LIMIT
        assert intent.order_type == "MARKET"
        assert nxt.sessions_held == 10

    def test_no_time_limit_before_max_holding(self):
        state = _state(avg_cost=1000.0, peak=1000.0, sessions_held=8)
        nxt, intent = policy.step(state, _obs(l=990.0), _conf(max_holding=10))
        assert intent is None


class TestStepPeakOrdering:
    def test_same_session_high_does_not_raise_todays_line(self):
        """当日の高値でピークが更新されても、当日の基準線は前営業日ピークで固定する。

        未来（当日の高値）を遡ってストップへ使わないための規約。
        ピーク1000（未発動、線=930）の日に高値1100・安値1050が出た場合、
        同日にトレーリング線1056へ引き上げて安値1050で退出、とはしない。
        """
        state = _state(avg_cost=1000.0, peak=1000.0)
        nxt, intent = policy.step(state, _obs(h=1100.0, l=1050.0), _conf())
        assert intent is None                       # 当日は退出しない
        assert nxt.peak_price == pytest.approx(1100.0)  # ピークは当日終了後に反映

    def test_raised_line_applies_from_next_session(self):
        """引き上がった線は翌営業日から効く"""
        state = _state(avg_cost=1000.0, peak=1000.0)
        after_day1, _ = policy.step(state, _obs(h=1100.0, l=1050.0), _conf())
        # 翌日は peak=1100 に基づく線1056が有効
        _, intent = policy.step(after_day1, _obs(session=date(2026, 9, 3), h=1060.0, l=1050.0), _conf())
        assert intent is not None
        assert intent.reason == policy.TRAILING
        assert intent.trigger_price == pytest.approx(1056.0)

    def test_peak_never_decreases(self):
        """安値だけの日でもピークは下がらない"""
        state = _state(avg_cost=1000.0, peak=1100.0)
        nxt, _ = policy.step(state, _obs(h=1020.0, l=1010.0), _conf())
        assert nxt.peak_price == pytest.approx(1100.0)


class TestRunSessionSeries:
    def _series(self):
        """1日目に高値1100・安値1050、2日目に安値1050の観測列"""
        return [
            _obs(session=date(2026, 9, 2), o=1000.0, h=1100.0, l=1050.0, c=1090.0),
            _obs(session=date(2026, 9, 3), o=1090.0, h=1095.0, l=1050.0, c=1055.0),
        ]

    def test_pessimistic_exits_on_second_session(self):
        """既定（previous）: 1日目は線が引き上がらず退出せず、2日目に退出する"""
        final, intent, at = policy.run_session_series(
            _state(avg_cost=1000.0, peak=1000.0), self._series(), _conf())
        assert intent is not None
        assert intent.reason == policy.TRAILING
        assert at.session == date(2026, 9, 3)

    def test_optimistic_exits_on_first_session(self):
        """same_session: 1日目の高値で線が1056へ上がり、同じ日の安値1050で退出する"""
        final, intent, at = policy.run_session_series(
            _state(avg_cost=1000.0, peak=1000.0), self._series(), _conf(),
            peak_basis=policy.PEAK_BASIS_SAME_SESSION)
        assert intent is not None
        assert intent.reason == policy.TRAILING
        assert at.session == date(2026, 9, 2)

    def test_difference_between_bases_is_measurable(self):
        """2つの仮定の差（退出日）を測れる＝曖昧性を結果に記録できる"""
        pess = policy.run_session_series(
            _state(avg_cost=1000.0, peak=1000.0), self._series(), _conf())
        opt = policy.run_session_series(
            _state(avg_cost=1000.0, peak=1000.0), self._series(), _conf(),
            peak_basis=policy.PEAK_BASIS_SAME_SESSION)
        assert pess[2].session != opt[2].session

    def test_returns_none_intent_when_bars_run_out(self):
        """足が尽きても決着しない場合は意図なしで返す（未成熟として扱えるように）"""
        calm = [_obs(session=date(2026, 9, 2), h=1005.0, l=995.0)]
        final, intent, at = policy.run_session_series(
            _state(avg_cost=1000.0, peak=1000.0), calm, _conf(max_holding=10))
        assert intent is None
        assert at is None
        assert final.sessions_held == 1

    def test_stops_advancing_after_exit(self):
        """退出した時点で止まる（それ以降の足を消費しない）"""
        bars = [
            _obs(session=date(2026, 9, 2), l=920.0),   # ここで損切り
            _obs(session=date(2026, 9, 3), l=900.0),
        ]
        final, intent, at = policy.run_session_series(
            _state(avg_cost=1000.0, peak=1000.0), bars, _conf())
        assert intent.reason == policy.STOP_LINE
        assert at.session == date(2026, 9, 2)
        assert final.sessions_held == 1
