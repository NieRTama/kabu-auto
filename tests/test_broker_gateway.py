"""
BrokerGateway（ブローカーAPI境界層・レビュー C3分割）のテスト
"""
from unittest.mock import MagicMock

import pytest

import src.execution.broker_gateway as bg
from src.execution.broker_constants import FrontOrderType, Side


@pytest.fixture
def gw(monkeypatch):
    monkeypatch.setattr(bg.cfg, "get_api_password", lambda: "pw")
    client = MagicMock()
    client.send_order.return_value = {"Result": 0, "OrderId": "OID-1"}
    return bg.BrokerGateway(client), client


class TestPayloads:
    def test_buy_limit_payload(self, gw):
        gateway, client = gw
        gateway.send_buy_limit("7203", 1000.0, 100)
        sent = client.send_order.call_args[0][0]
        assert sent["Side"] == Side.BUY.value == "2"
        assert sent["FrontOrderType"] == FrontOrderType.LIMIT.value == 20
        assert sent["Price"] == 1000.0
        assert sent["Qty"] == 100
        assert sent["Password"] == "pw"

    def test_sell_limit_payload(self, gw):
        gateway, client = gw
        gateway.send_sell_limit("7203", 1100.0, 100)
        sent = client.send_order.call_args[0][0]
        assert sent["Side"] == "1"
        assert sent["FrontOrderType"] == 20

    def test_sell_market_payload(self, gw):
        gateway, client = gw
        gateway.send_sell_market("7203", 100)
        sent = client.send_order.call_args[0][0]
        assert sent["Side"] == "1"
        assert sent["FrontOrderType"] == 10
        assert sent["Price"] == 0

    def test_stop_loss_payload(self, gw):
        gateway, client = gw
        gateway.send_stop_loss_market("7203", 100, 950.0)
        sent = client.send_order.call_args[0][0]
        assert sent["FrontOrderType"] == 30
        assert sent["ReverseLimitOrder"]["TriggerPrice"] == 950.0
        assert sent["ReverseLimitOrder"]["AfterHitOrderType"] == 1


class TestResultHelpers:
    def test_is_accepted(self):
        assert bg.BrokerGateway.is_accepted({"Result": 0}) is True
        assert bg.BrokerGateway.is_accepted({"Result": 1}) is False
        assert bg.BrokerGateway.is_accepted(None) is False

    def test_order_id_of(self):
        assert bg.BrokerGateway.order_id_of({"OrderId": "X"}) == "X"
        assert bg.BrokerGateway.order_id_of({"Result": 0}) is None
