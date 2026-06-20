"""
日次レポート・取引ジャーナルのテスト（Phase 5 / 7.4・7.9）
"""
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import src.core.config as cfg
import src.data.database as db
import src.dashboard.app as dash
from src.data.database import Trade, get_session


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    dash._auth_required = False
    try:
        yield tmp_path
    finally:
        db._engine = None
        db._Session = None


def _add_trade(order_id, side, qty, price, pnl=None, status="FILLED",
               filled_at=datetime(2026, 6, 20, 10, 0), filled_price=None, sector=None):
    with get_session() as session:
        session.add(Trade(
            order_id=order_id, symbol="7203", side=side, quantity=qty, price=price,
            filled_price=filled_price, filled_quantity=qty, pnl=pnl, status=status,
            filled_at=filled_at, sector=sector,
        ))
        session.commit()


class TestDailyReport:
    def test_aggregates_per_day(self, isolated_db):
        _add_trade("b1", "BUY", 100, 1000, filled_at=datetime(2026, 6, 20, 9, 0))
        _add_trade("s1", "SELL", 100, 1100, pnl=10000, filled_at=datetime(2026, 6, 20, 10, 0))
        _add_trade("s2", "SELL", 100, 900, pnl=-5000, filled_at=datetime(2026, 6, 20, 11, 0))
        client = TestClient(dash.app)
        body = client.get("/api/report/daily").json()
        day = next(r for r in body if r["date"] == "2026-06-20")
        assert day["trade_count"] == 3
        assert day["buy_count"] == 1 and day["sell_count"] == 2
        assert day["realized_pnl"] == 5000.0
        assert day["win_count"] == 1 and day["loss_count"] == 1
        assert day["win_rate"] == 0.5

    def test_excludes_dry_run(self, isolated_db):
        _add_trade("d1", "BUY", 100, 1000, status="DRY_RUN")
        client = TestClient(dash.app)
        body = client.get("/api/report/daily").json()
        assert body == []

    def test_newest_first(self, isolated_db):
        _add_trade("a", "SELL", 100, 1100, pnl=100, filled_at=datetime(2026, 6, 18, 10, 0))
        _add_trade("b", "SELL", 100, 1100, pnl=100, filled_at=datetime(2026, 6, 20, 10, 0))
        client = TestClient(dash.app)
        body = client.get("/api/report/daily").json()
        assert body[0]["date"] == "2026-06-20"


class TestTradeJournal:
    def test_only_realized_trades(self, isolated_db):
        _add_trade("b1", "BUY", 100, 1000)  # pnl None → 除外
        _add_trade("s1", "SELL", 100, 1100, pnl=10000)
        client = TestClient(dash.app)
        body = client.get("/api/journal").json()
        assert len(body) == 1
        assert body[0]["order_id"] == "s1"

    def test_return_pct_computed(self, isolated_db):
        # 売却総額 110,000、損益 +10,000 → 原価 100,000 → リターン 10%
        _add_trade("s1", "SELL", 100, 1100, pnl=10000)
        client = TestClient(dash.app)
        body = client.get("/api/journal").json()
        assert body[0]["return_pct"] == 0.1

    def test_return_pct_market_sell_uses_filled_price(self, isolated_db):
        # 成行: price=0, filled_price=1100 → 売却総額110,000, 損益+10,000 → 10%
        _add_trade("s1", "SELL", 100, 0, pnl=10000, filled_price=1100)
        client = TestClient(dash.app)
        body = client.get("/api/journal").json()
        assert body[0]["return_pct"] == 0.1

    def test_set_note(self, isolated_db):
        _add_trade("s1", "SELL", 100, 1100, pnl=10000)
        client = TestClient(dash.app)
        r = client.put("/api/journal/s1/note", json={"note": "ニュースで急騰、利確"})
        assert r.status_code == 200
        body = client.get("/api/journal").json()
        assert body[0]["note"] == "ニュースで急騰、利確"

    def test_note_missing_trade_404(self, isolated_db):
        client = TestClient(dash.app)
        r = client.put("/api/journal/nope/note", json={"note": "x"})
        assert r.status_code == 404
