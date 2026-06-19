"""
/api/positions, /api/pnl/enhanced_summary のテスト（N+1クエリ解消の回帰確認）

経緯: 保有銘柄ごとに最新OHLCVを個別クエリで取得していた(N+1)。
latest_closes() による一括取得に変更した後も、含み損益・リターン率の
計算結果が従来と同じ値になることを確認する。
"""
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.core.config as cfg
import src.data.database as db
import src.dashboard.app as dash
from src.data.database import Position, get_session
from src.data.market_data import upsert_ohlcv


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


def _add_position(symbol: str, quantity: int, avg_cost: float, sector: str = "") -> None:
    with get_session() as session:
        session.add(Position(symbol=symbol, quantity=quantity, avg_cost=avg_cost, sector=sector))
        session.commit()


def _set_close(symbol: str, close: float) -> None:
    df = pd.DataFrame(
        {"open": [close], "high": [close], "low": [close], "close": [close], "volume": [1000]},
        index=pd.to_datetime([date(2026, 1, 1)]),
    )
    df.index.name = "date"
    upsert_ohlcv(symbol, df)


class TestGetPositions:
    def test_returns_latest_price_and_unrealized_pnl(self, isolated_db):
        _add_position("7203", quantity=100, avg_cost=1000.0, sector="Consumer Cyclical")
        _set_close("7203", 1100.0)

        client = TestClient(dash.app)
        r = client.get("/api/positions")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 1
        assert body[0]["symbol"] == "7203"
        assert body[0]["latest_price"] == 1100.0
        assert body[0]["unrealized_pnl"] == 10000.0  # (1100-1000)*100
        assert body[0]["return_pct"] == 0.1

    def test_missing_ohlcv_gives_null_price_fields(self, isolated_db):
        _add_position("9999", quantity=100, avg_cost=1000.0)
        client = TestClient(dash.app)
        body = client.get("/api/positions").json()
        assert body[0]["latest_price"] is None
        assert body[0]["unrealized_pnl"] is None

    def test_multiple_positions_each_get_correct_price(self, isolated_db):
        """複数銘柄が、それぞれ正しい銘柄の最新終値と紐付くこと（一括取得のマッピング崩れ防止）"""
        _add_position("1111", quantity=100, avg_cost=1000.0)
        _set_close("1111", 1200.0)
        _add_position("2222", quantity=50, avg_cost=2000.0)
        _set_close("2222", 1800.0)

        client = TestClient(dash.app)
        body = client.get("/api/positions").json()
        by_symbol = {p["symbol"]: p for p in body}
        assert by_symbol["1111"]["latest_price"] == 1200.0
        assert by_symbol["2222"]["latest_price"] == 1800.0


class TestPnlEnhancedSummaryUnrealized:
    def test_total_unrealized_sums_across_positions(self, isolated_db):
        _add_position("1111", quantity=100, avg_cost=1000.0)
        _set_close("1111", 1100.0)  # +10,000
        _add_position("2222", quantity=10, avg_cost=5000.0)
        _set_close("2222", 4900.0)  # -1,000

        client = TestClient(dash.app)
        r = client.get("/api/pnl/enhanced_summary")
        assert r.status_code == 200
        assert r.json()["total_unrealized_pnl"] == 9000.0
