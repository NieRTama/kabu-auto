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


class TestPositionsUsesLivePriceWhenAvailable:
    """/api/positions・/api/pnl/enhanced_summary が、RiskManagerにprice_fnが
    注入されている場合はOHLCV終値ではなくリアルタイム価格を使うこと。

    2026-09-09: OHLCV終値だけで計算した結果、実際は+7,100円の含み益があるのに
    -440円の含み損とダッシュボードに表示される事故があった。
    """

    def test_positions_endpoint_prefers_live_price(self, isolated_db):
        from src.risk.manager import RiskManager
        from src.execution.order_manager import OrderManager
        from unittest.mock import MagicMock

        _add_position("7203", quantity=100, avg_cost=1000.0)
        _set_close("7203", 1050.0)  # 古い終値（低め）

        risk = RiskManager(price_fn=lambda syms: {"7203": 1200.0})  # リアルタイム現在値
        om = MagicMock(spec=OrderManager)
        om._risk = risk
        # set_order_manager() は init_auth() を呼び直し、isolated_dbフィクスチャが
        # 無効化した認証を再度有効化してしまう（このテストの関心事ではないため
        # 副作用を避け、_order_manager を直接差し替える）。
        dash._order_manager = om
        try:
            client = TestClient(dash.app)
            body = client.get("/api/positions").json()
            assert body[0]["latest_price"] == 1200.0
            assert body[0]["unrealized_pnl"] == 20000.0  # (1200-1000)*100
        finally:
            dash._order_manager = None

    def test_enhanced_summary_prefers_live_price(self, isolated_db):
        from src.risk.manager import RiskManager
        from src.execution.order_manager import OrderManager
        from unittest.mock import MagicMock

        _add_position("7203", quantity=100, avg_cost=1000.0)
        _set_close("7203", 900.0)  # 古い終値だと含み損に見える

        risk = RiskManager(price_fn=lambda syms: {"7203": 1200.0})  # 実際は含み益
        om = MagicMock(spec=OrderManager)
        om._risk = risk
        # set_order_manager() は init_auth() を呼び直し、isolated_dbフィクスチャが
        # 無効化した認証を再度有効化してしまう（このテストの関心事ではないため
        # 副作用を避け、_order_manager を直接差し替える）。
        dash._order_manager = om
        try:
            client = TestClient(dash.app)
            body = client.get("/api/pnl/enhanced_summary").json()
            assert body["total_unrealized_pnl"] == 20000.0
        finally:
            dash._order_manager = None

    def test_falls_back_to_ohlcv_when_no_order_manager(self, isolated_db):
        """_order_manager未設定（テスト環境等）では従来どおりOHLCV終値を使う（回帰防止）"""
        _add_position("7203", quantity=100, avg_cost=1000.0)
        _set_close("7203", 1100.0)
        assert dash._order_manager is None

        client = TestClient(dash.app)
        body = client.get("/api/positions").json()

        assert body[0]["latest_price"] == 1100.0
