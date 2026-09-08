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


class TestFundType:
    """預り区分は売買で値が違う（2026-09-07 の発注拒否の回帰防止）。

    システムとして初めて出した現物買いが
    `{"Code":1010004,"Message":"預り区分が未設定です。"}` で拒否された。
    売買共通で "  "（空白2文字）を送っていたが、これは**現物売でのみ有効**な値。
    買いが一度も成立していなかったため2か月以上露見しなかった。
    """

    def test_buy_specifies_custody(self, gw):
        gateway, client = gw
        gateway.send_buy_limit("2157", 1006.0, 200)
        sent = client.send_order.call_args[0][0]
        assert sent["FundType"] == "02", "現物買いは預り区分（保護預り）を明示する"
        assert sent["FundType"].strip() != "", "空白は現物売用の値。買いでは拒否される"

    @pytest.mark.parametrize("call", [
        lambda g: g.send_sell_limit("7203", 1100.0, 100),
        lambda g: g.send_sell_market("7203", 100),
        lambda g: g.send_stop_loss_market("7203", 100, 900.0),
    ])
    def test_sell_keeps_blank_custody(self, gw, call):
        """売りは空白2文字のまま（買いに合わせて変えると今度は売りが壊れる）"""
        gateway, client = gw
        call(gateway)
        assert client.send_order.call_args[0][0]["FundType"] == "  "


class TestDelivType:
    """受渡区分も売買で値が違う（2026-09-08 の退出失敗の回帰防止）。

    トレーリングストップが発動して 9432 を成行売りしようとしたが
    `{"Code":100378,"Message":"指定された市場でのお取引はお受けできません。"}`
    で拒否された。売買共通で 2（お預り金）を送っていたが**現物売は 0（指定なし）**。
    エラー文言が「市場」を指すため、受渡区分が原因とは読み取れなかった。

    退出が通らないのは損失に直結する（建玉を切れない）ため、売り側を特に固定する。
    """

    def test_buy_uses_deposit(self, gw):
        gateway, client = gw
        gateway.send_buy_limit("7203", 1000.0, 100)
        assert client.send_order.call_args[0][0]["DelivType"] == 2

    @pytest.mark.parametrize("call", [
        lambda g: g.send_sell_limit("7203", 1100.0, 100),
        lambda g: g.send_sell_market("7203", 100),
        lambda g: g.send_stop_loss_market("7203", 100, 900.0),
    ])
    def test_sell_uses_unspecified(self, gw, call):
        """成行・指値・逆指値のどの退出経路でも 0 であること"""
        gateway, client = gw
        call(gateway)
        assert client.send_order.call_args[0][0]["DelivType"] == 0

    def test_constant_names_match_actual_values(self):
        """旧実装は `AUTO = 2  # 自動振替` で名前もコメントも誤っていた。
        2 は「お預り金」、自動振替は 1。名前で選ぶと間違うので値を固定する。
        """
        from src.execution.broker_constants import DelivType
        assert DelivType.UNSPECIFIED.value == 0
        assert DelivType.AUTO.value == 1
        assert DelivType.DEPOSIT.value == 2

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
