"""過去検証の執行アダプタ（src/backtest/execution.py）のテスト

退出「意図」を約定へ変換し、スリッページと手数料を一元的に控除する。
Tの引けで判断し T+1 の寄りで約定する（同じ終値で判断・約定しない）。
"""
from datetime import date

import pytest

from src.core import config as cfg
from src.backtest import execution
from src.strategy import policy


def _costs(slip=0.001, comm=0.0) -> execution.CostConfig:
    return execution.CostConfig(slippage_pct=slip, commission_pct=comm)


def _bar(session=date(2026, 9, 2), o=1000.0, h=1010.0, l=990.0, c=1005.0):
    return policy.Observation(session=session, open=o, high=h, low=l, close=c)


class TestConfigFromSettings:
    def test_reads_backtest_section(self):
        cfg.load("config.yaml")
        costs = execution.config_from_settings()
        backtest = cfg.get_section("backtest")
        assert costs.slippage_pct == backtest["slippage_pct"]
        assert costs.commission_pct == backtest["commission_pct"]


class TestFillPrices:
    def test_buy_is_unfavourable(self):
        """買いはスリッページ分だけ高く約定する"""
        assert execution.buy_fill_price(1000.0, _costs(slip=0.001)) == pytest.approx(1001.0)

    def test_sell_is_unfavourable(self):
        """売りはスリッページ分だけ安く約定する"""
        assert execution.sell_fill_price(1000.0, _costs(slip=0.001)) == pytest.approx(999.0)

    def test_matches_existing_engine_convention(self):
        """既存 engine.py の _buy_fill_price/_sell_fill_price と同じ丸め（小数2桁）"""
        assert execution.buy_fill_price(1234.567, _costs(slip=0.001)) == pytest.approx(1235.80)
        assert execution.sell_fill_price(1234.567, _costs(slip=0.001)) == pytest.approx(1233.33)


class TestEntryFill:
    def test_fills_at_next_session_open(self):
        """Tの引けで決めた買いは T+1 の寄りで約定する"""
        next_bar = _bar(session=date(2026, 9, 3), o=1020.0, c=1050.0)
        fill = execution.entry_fill(next_bar, quantity=100, costs=_costs(slip=0.001))
        assert fill.at == date(2026, 9, 3)
        assert fill.price == pytest.approx(1021.02)  # 1020 * 1.001
        assert fill.quantity == 100
        assert fill.reason == "ENTRY"

    def test_does_not_use_the_decision_session_close(self):
        """判断した日の終値では約定しない（F04の回帰防止）"""
        decision_bar = _bar(session=date(2026, 9, 2), c=1005.0)
        next_bar = _bar(session=date(2026, 9, 3), o=1020.0)
        fill = execution.entry_fill(next_bar, quantity=100, costs=_costs(slip=0.0))
        assert fill.at != decision_bar.session
        assert fill.price != pytest.approx(decision_bar.close)
