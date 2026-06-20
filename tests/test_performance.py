"""
パフォーマンス分析のテスト（Phase 5 / 7.8）
"""
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.analytics.performance import compute_performance


def _t(pnl, symbol="7203", sector="Tech", price=1100, qty=100,
       filled_price=None, filled_at=None, id=0, side="SELL"):
    return SimpleNamespace(
        pnl=pnl, symbol=symbol, sector=sector, price=price, quantity=qty,
        filled_price=filled_price, filled_quantity=qty,
        filled_at=filled_at or datetime(2026, 6, 20, 10, id), id=id, side=side,
    )


class TestComputePerformance:
    def test_empty(self):
        r = compute_performance([])
        assert r["total_trades"] == 0
        assert r["profit_factor"] is None

    def test_basic_metrics(self):
        trades = [_t(10000, id=1), _t(-5000, id=2), _t(20000, id=3), _t(-2000, id=4)]
        r = compute_performance(trades)
        assert r["total_trades"] == 4
        assert r["win_count"] == 2 and r["loss_count"] == 2
        assert r["win_rate"] == 0.5
        assert r["gross_profit"] == 30000
        assert r["gross_loss"] == 7000
        assert r["net_pnl"] == 23000
        assert r["profit_factor"] == round(30000 / 7000, 2)
        assert r["avg_win"] == 15000
        assert r["avg_loss"] == 3500
        assert r["expectancy"] == round(23000 / 4, 0)
        assert r["largest_win"] == 20000
        assert r["largest_loss"] == -5000

    def test_consecutive_streaks(self):
        # 時系列順: win, win, loss, loss, loss, win
        trades = [
            _t(100, id=1), _t(100, id=2), _t(-1, id=3),
            _t(-1, id=4), _t(-1, id=5), _t(100, id=6),
        ]
        r = compute_performance(trades)
        assert r["max_consecutive_wins"] == 2
        assert r["max_consecutive_losses"] == 3

    def test_profit_factor_none_when_no_loss(self):
        r = compute_performance([_t(100, id=1), _t(200, id=2)])
        assert r["profit_factor"] is None  # 無限大は返さない

    def test_return_pct_uses_cost_basis(self):
        # 売却総額110,000・損益+10,000 → 原価100,000 → +10%
        r = compute_performance([_t(10000, price=1100, qty=100, id=1)])
        assert r["avg_return_pct"] == 0.1

    def test_by_symbol_and_sector(self):
        trades = [
            _t(10000, symbol="7203", sector="Auto", id=1),
            _t(-5000, symbol="7203", sector="Auto", id=2),
            _t(3000, symbol="6758", sector="Tech", id=3),
        ]
        r = compute_performance(trades)
        assert r["by_symbol"]["7203"]["trades"] == 2
        assert r["by_symbol"]["7203"]["pnl"] == 5000
        assert r["by_symbol"]["6758"]["win_rate"] == 1.0
        assert r["by_sector"]["Auto"]["pnl"] == 5000

    def test_none_sector_grouped_as_unset(self):
        r = compute_performance([_t(100, sector=None, id=1)])
        assert "(未設定)" in r["by_sector"]


class TestPerformanceEndpoint:
    def test_endpoint(self, tmp_path):
        import src.core.config as cfg
        import src.data.database as db
        import src.dashboard.app as dash
        from src.data.database import Trade, get_session
        from fastapi.testclient import TestClient

        cfg.load("config.yaml")
        cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
        db.init()
        dash._auth_required = False
        try:
            with get_session() as s:
                s.add(Trade(order_id="s1", symbol="7203", side="SELL", quantity=100,
                            price=1100, pnl=10000, status="FILLED",
                            filled_at=datetime(2026, 6, 20, 10, 0)))
                s.commit()
            body = TestClient(dash.app).get("/api/performance").json()
            assert body["total_trades"] == 1
            assert body["net_pnl"] == 10000
        finally:
            db._engine = None
            db._Session = None
