"""ポートフォリオの状態機械（src/backtest/portfolio.py）のテスト

現金・保有・資金競合・セクター上限を実運用（src/risk/manager.py）と
同じ式で扱う。式が2つあると、バックテストで通った条件が運用で弾かれる。
"""
from datetime import date

import pytest

from src.backtest import portfolio as pf


def _holding(symbol="7203", qty=100, avg_cost=1000.0, sector="自動車",
             entry_at=date(2026, 9, 1), peak=1000.0, sessions_held=0):
    return pf.Holding(
        symbol=symbol, quantity=qty, avg_cost=avg_cost, sector=sector,
        entry_at=entry_at, peak_price=peak, sessions_held=sessions_held,
    )


class TestEmptyPortfolio:
    def test_starts_with_cash_and_nothing_held(self):
        p = pf.empty_portfolio(1_000_000.0)
        assert p.cash == pytest.approx(1_000_000.0)
        assert p.holdings == {}
        assert p.reserved == pytest.approx(0.0)

    def test_lot_size_matches_production(self):
        """単元は実運用（risk/manager.py:26）と同じ100株"""
        from src.risk.manager import LOT_SIZE as PROD_LOT_SIZE
        assert pf.LOT_SIZE == PROD_LOT_SIZE


class TestValuation:
    def test_holdings_value_uses_latest_price(self):
        p = pf.Portfolio(cash=0.0, holdings={"7203": _holding(qty=100, avg_cost=1000.0)},
                         reserved=0.0)
        assert pf.holdings_value(p, {"7203": 1200.0}) == pytest.approx(120_000.0)

    def test_falls_back_to_avg_cost_when_price_missing(self):
        """価格が取れない銘柄は取得単価で代用する（実運用と同じ規約）"""
        p = pf.Portfolio(cash=0.0, holdings={"7203": _holding(qty=100, avg_cost=1000.0)},
                         reserved=0.0)
        assert pf.holdings_value(p, {}) == pytest.approx(100_000.0)

    def test_nav_is_cash_plus_holdings(self):
        p = pf.Portfolio(cash=500_000.0,
                         holdings={"7203": _holding(qty=100, avg_cost=1000.0)},
                         reserved=0.0)
        assert pf.nav(p, {"7203": 1200.0}) == pytest.approx(620_000.0)

    def test_nav_ignores_reserved(self):
        """予約は現金の内訳であって総資産を減らさない"""
        a = pf.Portfolio(cash=500_000.0, holdings={}, reserved=0.0)
        b = pf.Portfolio(cash=500_000.0, holdings={}, reserved=100_000.0)
        assert pf.nav(a, {}) == pytest.approx(pf.nav(b, {}))

    def test_held_value_of_unheld_symbol_is_zero(self):
        p = pf.empty_portfolio(1_000_000.0)
        assert pf.held_value(p, "7203", 1000.0) == pytest.approx(0.0)

    def test_held_value_uses_current_price_not_book(self):
        """保有評価は簿価ではなく現在値（実運用 _held_value と同じ）"""
        p = pf.Portfolio(cash=0.0, holdings={"7203": _holding(qty=100, avg_cost=1000.0)},
                         reserved=0.0)
        assert pf.held_value(p, "7203", 1500.0) == pytest.approx(150_000.0)
