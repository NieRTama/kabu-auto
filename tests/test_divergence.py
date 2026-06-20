"""
バックテスト実績乖離分析のテスト（Phase 5 / 7.7）
"""
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from src.analytics.performance import compute_divergence


def _bt(entry, exit_, pnl):
    return SimpleNamespace(entry_price=entry, exit_price=exit_, pnl=pnl)


def _act(pnl, price=1100, qty=100, filled_price=None):
    return SimpleNamespace(pnl=pnl, price=price, quantity=qty,
                           filled_price=filled_price, filled_quantity=qty)


class TestComputeDivergence:
    def test_basic(self):
        bt = [_bt(1000, 1100, 10000), _bt(1000, 950, -5000)]   # 1勝1敗, return +10%/-5%
        act = [_act(10000, price=1100, qty=100)]               # 実: 1勝, +10%
        r = compute_divergence(bt, act)
        assert r["backtest"]["trade_count"] == 2
        assert r["backtest"]["win_rate"] == 0.5
        assert r["actual"]["trade_count"] == 1
        assert r["actual"]["win_rate"] == 1.0
        assert r["divergence"]["win_rate_diff"] == round(1.0 - 0.5, 4)

    def test_avg_return_diff(self):
        bt = [_bt(1000, 1100, 10000)]   # +10%
        act = [_act(20000, price=1200, qty=100)]  # 売却24万 損益+2万 → 原価22万 → +9.09%
        r = compute_divergence(bt, act)
        assert r["backtest"]["avg_return_pct"] == 0.1
        assert r["divergence"]["avg_return_pct_diff"] is not None

    def test_few_actual_adds_note(self):
        r = compute_divergence([_bt(1000, 1100, 100)], [_act(100)])
        assert r["note"] != ""

    def test_empty_actual(self):
        r = compute_divergence([_bt(1000, 1100, 100)], [])
        assert r["actual"]["trade_count"] == 0
        assert r["divergence"]["win_rate_diff"] is None


class TestDivergenceEndpoint:
    def test_endpoint(self, tmp_path):
        import src.core.config as cfg
        import src.data.database as db
        import src.dashboard.app as dash
        from src.data.database import BacktestRun, BacktestTradeRecord, Trade, get_session
        from fastapi.testclient import TestClient

        cfg.load("config.yaml")
        cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
        db.init()
        dash._auth_required = False
        try:
            with get_session() as s:
                run = BacktestRun(symbol="7203", start_date=date(2025, 1, 1),
                                  end_date=date(2025, 6, 1), win_rate=0.5, trade_count=2)
                s.add(run)
                s.commit()
                rid = run.id
                s.add(BacktestTradeRecord(run_id=rid, symbol="7203", entry_price=1000,
                                          exit_price=1100, quantity=100, pnl=10000))
                s.add(Trade(order_id="s1", symbol="7203", side="SELL", quantity=100,
                            price=1100, pnl=10000, status="FILLED",
                            filled_at=datetime(2026, 6, 20, 10, 0)))
                s.commit()
            body = TestClient(dash.app).get(f"/api/backtest/{rid}/divergence").json()
            assert body["symbol"] == "7203"
            assert body["backtest"]["trade_count"] == 1
            assert body["actual"]["trade_count"] == 1
        finally:
            db._engine = None
            db._Session = None

    def test_missing_run_404(self, tmp_path):
        import src.core.config as cfg
        import src.data.database as db
        import src.dashboard.app as dash
        from fastapi.testclient import TestClient

        cfg.load("config.yaml")
        cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
        db.init()
        dash._auth_required = False
        try:
            assert TestClient(dash.app).get("/api/backtest/999/divergence").status_code == 404
        finally:
            db._engine = None
            db._Session = None
