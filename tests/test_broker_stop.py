"""
ブローカー側逆指値ストップのテスト（Phase 5 / 4.3）
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import src.execution.order_manager as mod


def _cfg(mode="live"):
    m = MagicMock()
    m.get_section.side_effect = lambda s: {
        "trading": {"mode": mode, "order_timeout_seconds": 300, "daily_order_limit": 100},
        "kabu_station": {"password": "pw"},
    }.get(s, {})
    return m


@contextmanager
def _make_om(mode="live"):
    cfg_mock = _cfg(mode)
    client = MagicMock()
    client.send_order.return_value = {"Result": 0, "OrderId": "STOP-1"}
    risk = MagicMock()
    session = MagicMock()
    session.scalar.return_value = None
    added = []
    session.add.side_effect = lambda o: added.append(o)

    @contextmanager
    def ctx():
        yield session

    with patch.object(mod, "cfg", cfg_mock), patch.object(mod, "get_session", ctx):
        om = mod.OrderManager(client, risk)
        yield om, client, added


class TestPlaceStopLoss:
    def test_live_sends_reverse_limit_order(self):
        with _make_om("live") as (om, client, added):
            oid = om.place_stop_loss("7203", 100, 950.0)
        assert oid == "STOP-1"
        sent = client.send_order.call_args[0][0]
        assert sent["FrontOrderType"] == 30           # 逆指値
        assert sent["Side"] == "1"                    # 売り
        assert sent["ReverseLimitOrder"]["TriggerPrice"] == 950.0
        assert sent["ReverseLimitOrder"]["UnderOver"] == 1     # 以下で発動
        assert sent["ReverseLimitOrder"]["AfterHitOrderType"] == 1  # 成行

    def test_records_trade_without_cancel_timer(self):
        with _make_om("live") as (om, client, added):
            om._set_cancel_timer = MagicMock()
            om.place_stop_loss("7203", 100, 950.0)
        # ストップは発動まで生かすのでタイムアウトキャンセルを設定しない
        om._set_cancel_timer.assert_not_called()
        assert added and "逆指値" in added[0].rationale

    def test_paper_is_noop(self):
        with _make_om("paper") as (om, client, added):
            oid = om.place_stop_loss("7203", 100, 950.0)
        assert oid is None
        client.send_order.assert_not_called()

    def test_dry_run_is_noop(self):
        with _make_om("dry_run") as (om, client, added):
            oid = om.place_stop_loss("7203", 100, 950.0)
        assert oid is None
        client.send_order.assert_not_called()

    def test_invalid_args_return_none(self):
        with _make_om("live") as (om, client, added):
            assert om.place_stop_loss("7203", 0, 950.0) is None
            assert om.place_stop_loss("7203", 100, 0) is None
        client.send_order.assert_not_called()

    def test_rejected_returns_none(self):
        with _make_om("live") as (om, client, added):
            client.send_order.return_value = {"Result": 1, "Message": "no"}
            assert om.place_stop_loss("7203", 100, 950.0) is None


class TestStopLossEndpoint:
    def test_endpoint_auto_computes_trigger(self, tmp_path):
        import src.core.config as cfg
        import src.data.database as db
        import src.dashboard.app as dash
        from src.data.database import Position, get_session
        from fastapi.testclient import TestClient

        cfg.load("config.yaml")
        cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
        cfg.get_section("trading")["stop_loss_pct"] = -0.05
        db.init()
        dash._auth_required = False
        dash._emergency_token = "tok"
        om = MagicMock()
        om.place_stop_loss.return_value = "STOP-9"
        dash._order_manager = om
        try:
            with get_session() as s:
                s.add(Position(symbol="7203", quantity=100, avg_cost=1000.0))
                s.commit()
            r = TestClient(dash.app).post("/api/stop_loss", json={"symbol": "7203"},
                                          headers={"X-Emergency-Token": "tok"})
            assert r.status_code == 200
            # trigger = 1000 * (1 - 0.05) = 950
            om.place_stop_loss.assert_called_once_with("7203", 100, 950.0)
        finally:
            db._engine = None
            db._Session = None
            dash._order_manager = None

    def test_endpoint_requires_token(self, tmp_path):
        import src.core.config as cfg
        import src.data.database as db
        import src.dashboard.app as dash
        from fastapi.testclient import TestClient

        cfg.load("config.yaml")
        cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
        db.init()
        dash._auth_required = False
        dash._emergency_token = "tok"
        dash._order_manager = MagicMock()
        try:
            r = TestClient(dash.app).post("/api/stop_loss", json={"symbol": "7203"})
            assert r.status_code in (403, 422)
        finally:
            db._engine = None
            db._Session = None
            dash._order_manager = None
