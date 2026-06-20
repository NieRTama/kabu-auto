"""
dry_run / semi_live モードのテスト（Phase 3 / 7.3）

- trading_mode ヘルパーの判定
- dry_run: 実APIへ send_order せず、DRY_RUN ステータスで記録すること
- semi_live: 通常注文を承認キューに積み（send_order しない）、退出（emergency/stop_loss）は
  即時実発注すること
- approve_order: 承認で実APIへ発注されること
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import src.execution.order_manager as mod
from src.core import trading_mode as tm
from src.execution import order_status as st


class TestTradingModeHelpers:
    def test_valid_modes(self):
        for m in ("paper", "live", "dry_run", "semi_live"):
            assert tm.is_valid(m)
        assert not tm.is_valid("bogus")

    def test_places_real_orders(self):
        assert tm.places_real_orders("live")
        assert tm.places_real_orders("semi_live")
        assert not tm.places_real_orders("paper")
        assert not tm.places_real_orders("dry_run")

    def test_uses_morning_execution(self):
        assert not tm.uses_morning_execution("paper")
        for m in ("live", "dry_run", "semi_live"):
            assert tm.uses_morning_execution(m)


def _cfg(mode: str) -> MagicMock:
    m = MagicMock()
    m.get_section.side_effect = lambda s: {
        "trading": {"mode": mode, "order_timeout_seconds": 300, "daily_order_limit": 100},
        "kabu_station": {"password": "pw"},
    }.get(s, {})
    return m


@contextmanager
def _make_om(mode: str):
    cfg_mock = _cfg(mode)
    client = MagicMock()
    client.send_order.return_value = {"Result": 0, "OrderId": "LIVE-1"}
    risk = MagicMock()
    risk.can_place_order.return_value = (True, "")
    session = MagicMock()
    session.scalar.return_value = None

    @contextmanager
    def session_ctx():
        yield session

    with patch.object(mod, "cfg", cfg_mock), \
         patch.object(mod, "get_session", session_ctx):
        om = mod.OrderManager(client, risk)
        yield om, client, risk, session


class TestDryRun:
    def test_buy_does_not_send_order(self):
        with _make_om("dry_run") as (om, client, risk, session):
            order_id = om.buy("7203", 1000.0, 100, sector="Tech")
        client.send_order.assert_not_called()
        assert order_id and order_id.startswith("DRYRUN-BUY")

    def test_buy_records_dry_run_status(self):
        from src.data.database import Trade
        with _make_om("dry_run") as (om, client, risk, session):
            om.buy("7203", 1000.0, 100)
        trade = next(c[0][0] for c in session.add.call_args_list if isinstance(c[0][0], Trade))
        assert trade.status == st.DRY_RUN

    def test_sell_market_does_not_send_even_on_exit(self):
        with _make_om("dry_run") as (om, client, risk, session):
            # latest_closes は使われない（dry_run は paper 分岐に入らない）
            order_id = om.sell_market("7203", 100, reason="emergency")
        client.send_order.assert_not_called()
        assert order_id and order_id.startswith("DRYRUN-SELL")


class TestSemiLive:
    def test_buy_enqueues_approval_without_sending(self):
        with _make_om("semi_live") as (om, client, risk, session):
            with patch.object(mod, "alert"):
                order_id = om.buy("7203", 1000.0, 100, sector="Tech")
        client.send_order.assert_not_called()
        assert order_id is None  # まだ発注していない
        approval = session.add.call_args_list[0][0][0]
        assert approval.symbol == "7203"
        assert approval.side == "BUY"
        assert approval.status == "PENDING"

    def test_normal_sell_market_enqueues(self):
        with _make_om("semi_live") as (om, client, risk, session):
            with patch.object(mod, "alert"):
                order_id = om.sell_market("7203", 100, reason="normal")
        client.send_order.assert_not_called()
        assert order_id is None
        approval = session.add.call_args_list[0][0][0]
        assert approval.order_type == "MARKET"

    def test_exit_sell_market_sends_immediately(self):
        """損切り・緊急決済（退出）は承認を介さず即時実発注する"""
        with _make_om("semi_live") as (om, client, risk, session):
            order_id = om.sell_market("7203", 100, reason="stop_loss")
        client.send_order.assert_called_once()
        assert order_id == "LIVE-1"

    def test_approve_order_sends_live(self):
        ap = MagicMock(id=1, status="PENDING", symbol="7203", side="BUY",
                       order_type="LIMIT", price=1000.0, quantity=100, sector="Tech")
        with _make_om("semi_live") as (om, client, risk, session):
            session.scalar.return_value = ap
            result = om.approve_order(1)
        client.send_order.assert_called_once()
        assert result["ok"] is True
        assert result["order_id"] == "LIVE-1"
        assert ap.status == "APPROVED"
        assert ap.resulting_order_id == "LIVE-1"

    def test_reject_order(self):
        ap = MagicMock(id=2, status="PENDING")
        with _make_om("semi_live") as (om, client, risk, session):
            session.scalar.return_value = ap
            result = om.reject_order(2)
        client.send_order.assert_not_called()
        assert result["ok"] is True
        assert ap.status == "REJECTED"

    def test_approve_already_decided_is_rejected(self):
        ap = MagicMock(id=3, status="APPROVED")
        with _make_om("semi_live") as (om, client, risk, session):
            session.scalar.return_value = ap
            result = om.approve_order(3)
        client.send_order.assert_not_called()
        assert result["ok"] is False
