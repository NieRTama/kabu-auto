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
