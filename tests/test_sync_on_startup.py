"""
OrderManager.sync_on_startup() のテスト（再レビュー B-1 対応）

起動時、API側の実状態へ同期し、APIに見つからない未解決注文は
誤って CANCELLED にせず UNKNOWN にする（約定済みの取りこぼし誤判定を防ぐ）。
"""
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

import src.core.config as cfg
import src.data.database as db
import src.execution.order_manager as mod
import src.execution.order_status as st
from src.data.database import Trade, get_session
from src.execution.order_manager import OrderManager


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    try:
        yield tmp_path
    finally:
        db._engine = None
        db._Session = None


def _add_trade(order_id, status=st.PENDING, quantity=100):
    with get_session() as session:
        session.add(Trade(order_id=order_id, symbol="7203", side="BUY",
                          status=status, quantity=quantity, price=1000.0))
        session.commit()


def _status_of(order_id):
    with get_session() as session:
        t = session.scalar(select(Trade).where(Trade.order_id == order_id))
        return t.status


def _make_live_om(client):
    om = OrderManager(client, MagicMock())
    om._is_paper = False  # configはpaperだが、ライブ同期経路を試験する
    return om


class TestSyncOnStartup:
    def test_missing_order_becomes_unknown_not_cancelled(self, isolated_db):
        _add_trade("LIVE-1")
        client = MagicMock()
        client.get_orders.return_value = []  # APIに見つからない
        with patch.object(mod, "alert") as alert_mock:
            _make_live_om(client).sync_on_startup()
        assert _status_of("LIVE-1") == st.UNKNOWN
        alert_mock.assert_called_once()

    def test_filled_in_api_syncs_to_filled(self, isolated_db):
        _add_trade("LIVE-2", quantity=100)
        client = MagicMock()
        client.get_orders.return_value = [{"ID": "LIVE-2", "State": 5, "CumQty": 100}]
        _make_live_om(client).sync_on_startup()
        assert _status_of("LIVE-2") == st.FILLED

    def test_working_order_stays_pending(self, isolated_db):
        _add_trade("LIVE-3", quantity=100)
        client = MagicMock()
        client.get_orders.return_value = [{"ID": "LIVE-3", "State": 2, "CumQty": 0}]
        _make_live_om(client).sync_on_startup()
        assert _status_of("LIVE-3") == st.PENDING

    def test_ended_without_fill_is_cancelled(self, isolated_db):
        _add_trade("LIVE-4", quantity=100)
        client = MagicMock()
        client.get_orders.return_value = [{"ID": "LIVE-4", "State": 5, "CumQty": 0}]
        _make_live_om(client).sync_on_startup()
        assert _status_of("LIVE-4") == st.CANCELLED
