"""
OrderIntent（発注の意図）の生成・紐付けのテスト（Phase 5 / 4.2）

- buy/sell/sell_market/dry_run/semi_live のいずれでもOrderIntentが作られること
- semi_live承認(approve_order)では承認待ち時に作った同じIntentに紐付くこと
- 拒否(REJECTED)時もIntentが作られその状態がREJECTEDになること
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import src.core.config as cfg
import src.data.database as db
import src.execution.order_manager as mod
from src.data.database import OrderApproval, OrderIntent, Trade, get_session


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


def _cfg(mode: str) -> MagicMock:
    m = MagicMock()
    m.get_section.side_effect = lambda s: {
        "trading": {"mode": mode, "order_timeout_seconds": 300, "daily_order_limit": 100},
        "kabu_station": {"password": "pw"},
    }.get(s, {})
    return m


@contextmanager
def _make_om(mode: str, send_result=None):
    cfg_mock = _cfg(mode)
    client = MagicMock()
    client.send_order.return_value = send_result or {"Result": 0, "OrderId": "LIVE-1"}
    risk = MagicMock()
    risk.can_place_order.return_value = (True, "")
    with patch.object(mod, "cfg", cfg_mock):
        om = mod.OrderManager(client, risk)
        yield om, client


def _intent_for(trade_order_id: str):
    with get_session() as s:
        from sqlalchemy import select
        trade = s.scalar(select(Trade).where(Trade.order_id == trade_order_id))
        assert trade is not None
        assert trade.intent_id is not None
        intent = s.scalar(select(OrderIntent).where(OrderIntent.id == trade.intent_id))
        return trade, intent


class TestIntentCreationPaper:
    def test_buy_creates_intent(self, isolated_db):
        with _make_om("paper") as (om, client):
            order_id = om.buy("7203", 1000.0, 100, sector="Auto",
                              rationale="score=0.2", source="signal_scan")
        trade, intent = _intent_for(order_id)
        assert intent.symbol == "7203"
        assert intent.side == "BUY"
        assert intent.target_quantity == 100
        assert intent.source == "signal_scan"
        assert intent.rationale == "score=0.2"
        assert intent.mode == "paper"
        assert intent.status == "COMPLETED"  # paperは即時成立

    def test_sell_creates_intent(self, isolated_db):
        with _make_om("paper") as (om, client):
            order_id = om.sell("7203", 1100.0, 100, source="manual")
        _, intent = _intent_for(order_id)
        assert intent.side == "SELL"
        assert intent.order_type == "LIMIT"

    def test_sell_market_stop_loss_source(self, isolated_db):
        with patch.object(mod, "latest_closes", return_value={"7203": 1000.0}):
            with _make_om("paper") as (om, client):
                order_id = om.sell_market("7203", 100, reason="stop_loss")
        _, intent = _intent_for(order_id)
        assert intent.source == "stop_loss"
        assert intent.order_type == "MARKET"
        assert "損切り" in intent.rationale


class TestIntentCreationLive:
    def test_live_buy_pending_intent_status(self, isolated_db):
        with _make_om("live") as (om, client):
            order_id = om.buy("7203", 1000.0, 100, source="morning_execution")
        _, intent = _intent_for(order_id)
        assert intent.status == "SUBMITTED"
        assert intent.mode == "live"

    def test_rejected_order_intent_status(self, isolated_db):
        with _make_om("live", send_result={"Result": 1, "Message": "ng"}) as (om, client):
            result = om.buy("7203", 1000.0, 100)
        assert result is None
        with get_session() as s:
            from sqlalchemy import select
            trade = s.scalar(select(Trade).where(Trade.symbol == "7203", Trade.status == "REJECTED"))
            intent = s.scalar(select(OrderIntent).where(OrderIntent.id == trade.intent_id))
            assert intent.status == "REJECTED"


class TestSemiLiveApprovalSharesIntent:
    def test_enqueue_creates_pending_intent(self, isolated_db):
        with _make_om("semi_live") as (om, client):
            with patch.object(mod, "alert"):
                result = om.buy("7203", 1000.0, 100, sector="Auto", rationale="r1")
        assert result is None  # 未発注
        with get_session() as s:
            from sqlalchemy import select
            ap = s.scalar(select(OrderApproval).where(OrderApproval.symbol == "7203"))
            assert ap.intent_id is not None
            intent = s.scalar(select(OrderIntent).where(OrderIntent.id == ap.intent_id))
            assert intent.status == "PENDING"

    def test_approve_reuses_same_intent(self, isolated_db):
        with _make_om("semi_live") as (om, client):
            with patch.object(mod, "alert"):
                om.buy("7203", 1000.0, 100, sector="Auto", rationale="r1")
            with get_session() as s:
                from sqlalchemy import select
                ap = s.scalar(select(OrderApproval).where(OrderApproval.symbol == "7203"))
                approval_id = ap.id
                original_intent_id = ap.intent_id

            result = om.approve_order(approval_id)
        assert result["ok"] is True
        with get_session() as s:
            from sqlalchemy import select
            trade = s.scalar(select(Trade).where(Trade.order_id == result["order_id"]))
            assert trade.intent_id == original_intent_id  # 同じ意図に紐づく

    def test_concurrent_double_approve_only_sends_one_order(self, isolated_db):
        """再レビュー: 同じ承認IDへの同時承認リクエスト（ダッシュボードの二重クリック等）が
        来ても、実APIへの発注は1回だけになること（修正前は両方が成功し二重発注した）。
        """
        import threading
        import time

        def slow_send_order(order):
            time.sleep(0.05)
            return {"Result": 0, "OrderId": f"LIVE-{time.time()}"}

        with _make_om("semi_live") as (om, client):
            client.send_order.side_effect = slow_send_order
            with patch.object(mod, "alert"):
                om.buy("7203", 1000.0, 100, sector="Auto", rationale="r1")
            with get_session() as s:
                from sqlalchemy import select
                ap = s.scalar(select(OrderApproval).where(OrderApproval.symbol == "7203"))
                approval_id = ap.id

            results = []
            results_lock = threading.Lock()

            def do_approve():
                r = om.approve_order(approval_id)
                with results_lock:
                    results.append(r)

            threads = [threading.Thread(target=do_approve) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        ok_results = [r for r in results if r["ok"]]
        assert len(ok_results) == 1, f"2件以上が成功した（二重発注）: {results}"
        assert client.send_order.call_count == 1
