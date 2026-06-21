"""
注文ライフサイクルの統合テスト（レビュー P2-3）。

実DB（隔離tmp）+ モックbrokerで、以下の異常系・復旧経路をエンドツーエンドに検証する:
  - 部分約定 → 全約定（Fill積み上げ・Position再構成）
  - キャンセル失敗 → CANCEL_FAILED（未解決として新規発注を抑止）
  - 照会API障害 → 例外を飲み込みOPENのまま（次回再試行）
  - 陳腐化注文のタイムアウトキャンセル
  - 再起動時同期で行方不明注文を UNKNOWN 化
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

import src.core.config as cfg
import src.data.database as db
import src.execution.order_manager as mod
from src.data.database import Position, Trade, get_session
from src.execution import order_status as st


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


def _cfg(mode="live"):
    m = MagicMock()
    m.get_section.side_effect = lambda s: {
        "trading": {"mode": mode, "order_timeout_seconds": 300, "daily_order_limit": 100},
        "kabu_station": {"password": "pw"},
    }.get(s, {})
    return m


@contextmanager
def _make_om(mode="live"):
    client = MagicMock()
    client.send_order.return_value = {"Result": 0, "OrderId": "LIVE-1"}
    risk = MagicMock()
    risk.can_place_order.return_value = (True, "")
    with patch.object(mod, "cfg", _cfg(mode)):
        yield mod.OrderManager(client, risk), client


def _trade(order_id="LIVE-1"):
    with get_session() as s:
        return s.scalar(select(Trade).where(Trade.order_id == order_id))


class TestPartialThenFullFill:
    def test_partial_fill_then_complete(self, isolated_db):
        with _make_om("live") as (om, client):
            oid = om.buy("7203", 1000.0, 100, source="morning_execution")
            assert _trade(oid).status == st.PENDING

            # 1回目の照合: 50株部分約定（まだ生存）
            client.get_orders.return_value = [
                {"ID": oid, "State": 3, "CumQty": 50, "Price": 1000}
            ]
            om.reconcile_open_orders()
            t = _trade(oid)
            assert t.status == st.PARTIALLY_FILLED
            assert t.filled_quantity == 50

            # 2回目の照合: 全約定で確定終了
            client.get_orders.return_value = [
                {"ID": oid, "State": 5, "CumQty": 100, "Price": 1000}
            ]
            om.reconcile_open_orders()
            t = _trade(oid)
            assert t.status == st.FILLED
            assert t.filled_quantity == 100

        # Position が 100株で再構成されている
        with get_session() as s:
            pos = s.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.quantity == 100


class TestCancelFailure:
    def test_cancel_rejected_marks_cancel_failed(self, isolated_db):
        with _make_om("live") as (om, client):
            oid = om.buy("7203", 1000.0, 100)
            client.cancel_order.return_value = {"Result": 1, "Message": "no"}
            with patch.object(mod, "alert"):
                ok = om._cancel_order_now(oid)
        assert ok is False
        assert _trade(oid).status == st.CANCEL_FAILED
        # 未解決として残り、新規発注ゲートの抑止対象になる
        assert om._count_unresolved() == 1


class TestReconcileApiFailure:
    def test_get_orders_exception_leaves_open(self, isolated_db):
        with _make_om("live") as (om, client):
            oid = om.buy("7203", 1000.0, 100)
            client.get_orders.side_effect = RuntimeError("API down")
            om.reconcile_open_orders()  # 例外を飲み込んで継続
        assert _trade(oid).status == st.PENDING  # OPENのまま（次回再試行）


class TestTimeoutCancel:
    def test_stale_order_timeout_cancels(self, isolated_db):
        with _make_om("live") as (om, client):
            oid = om.buy("7203", 1000.0, 100)
            client.cancel_order.return_value = {"Result": 0}
            om._timeout_cancel(oid)
        assert _trade(oid).status == st.CANCELLED


class TestRestartReconciliation:
    def test_sync_on_startup_marks_missing_unknown(self, isolated_db):
        with _make_om("live") as (om, client):
            oid = om.buy("7203", 1000.0, 100)
            # 再起動後の照会で当該注文が見つからない（約定/失効/取り逃しの可能性）
            client.get_orders.return_value = []
            with patch.object(mod, "alert"):
                om.sync_on_startup()
        assert _trade(oid).status == st.UNKNOWN
