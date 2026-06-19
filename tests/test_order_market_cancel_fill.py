"""
OrderManager の成行発注・キャンセル失敗処理・実約定価格反映のテスト（再レビュー対応）

カバー範囲:
  A-1: sell_market が成行注文(FrontOrderType=10, Price=0)を送る
  A-2: _timeout_cancel がキャンセル失敗時に CANCELLED にせず CANCEL_FAILED にする
  A-3: _extract_fill / _resolve_fill が実約定単価(VWAP)・数量を取り出す
       on_order_event が部分約定を PARTIALLY_FILLED にする
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import src.execution.order_manager as mod
import src.execution.order_status as st


def _cfg(mode: str = "live") -> MagicMock:
    m = MagicMock()
    m.get_section.side_effect = lambda s: {
        "trading": {"mode": mode, "order_timeout_seconds": 300, "daily_order_limit": 100},
        "kabu_station": {"password": "pw"},
    }.get(s, {})
    return m


@contextmanager
def _make_om(mode: str = "live"):
    cfg_mock = _cfg(mode)
    client = MagicMock()
    client.send_order.return_value = {"Result": 0, "OrderId": "ORD-1"}
    risk = MagicMock()
    risk.can_place_order.return_value = (True, "")

    session = MagicMock()
    session.scalar.return_value = None

    @contextmanager
    def session_ctx():
        yield session

    with patch.object(mod, "cfg", cfg_mock), \
         patch.object(mod, "get_session", session_ctx):
        yield mod.OrderManager(client, risk), client, risk, session


# ─── A-1: 成行売り ────────────────────────────────────────────────────────


class TestSellMarket:
    def test_live_sends_market_order(self):
        """ライブ成行売りは FrontOrderType=10 / Price=0 / Side=1 で送る"""
        with _make_om(mode="live") as (om, client, _, _):
            oid = om.sell_market("7203", 100)
        assert oid == "ORD-1"
        sent = client.send_order.call_args[0][0]
        assert sent["FrontOrderType"] == 10, "成行は FrontOrderType=10"
        assert sent["Price"] == 0, "成行は Price=0"
        assert sent["Side"] == "1", "売りは Side=1"

    def test_paper_uses_latest_close(self):
        """ペーパー成行売りは最新終値で約定記録する"""
        with _make_om(mode="paper") as (om, _, _, _):
            om._record_trade = MagicMock()
            om._update_position = MagicMock()
            with patch.object(mod, "latest_closes", return_value={"7203": 1234.0}):
                oid = om.sell_market("7203", 100)
        assert oid is not None
        # _record_trade に終値1234が渡る
        args = om._record_trade.call_args
        assert args[0][4] == 1234.0

    def test_paper_skips_when_no_close(self):
        """終値が取得できなければ発注しない（0円約定の損益破損を防ぐ）"""
        with _make_om(mode="paper") as (om, _, _, _):
            with patch.object(mod, "latest_closes", return_value={}):
                oid = om.sell_market("9999", 100)
        assert oid is None


# ─── A-2: キャンセル失敗 ──────────────────────────────────────────────────


class TestTimeoutCancel:
    def test_exception_sets_cancel_failed_not_cancelled(self):
        """キャンセルAPIが例外 → CANCEL_FAILED（CANCELLEDにしない）＋アラート"""
        with _make_om(mode="live") as (om, client, _, _):
            client.cancel_order.side_effect = RuntimeError("network")
            om._update_trade_status = MagicMock()
            with patch.object(mod, "alert") as alert_mock:
                om._timeout_cancel("ORD-1")
        om._update_trade_status.assert_called_once_with("ORD-1", st.CANCEL_FAILED)
        alert_mock.assert_called_once()

    def test_result_nonzero_sets_cancel_failed(self):
        """キャンセルが拒否(Result!=0) → CANCEL_FAILED＋アラート"""
        with _make_om(mode="live") as (om, client, _, _):
            client.cancel_order.return_value = {"Result": 4}
            om._update_trade_status = MagicMock()
            with patch.object(mod, "alert") as alert_mock:
                om._timeout_cancel("ORD-1")
        om._update_trade_status.assert_called_once_with("ORD-1", st.CANCEL_FAILED)
        alert_mock.assert_called_once()

    def test_result_zero_sets_cancelled(self):
        """キャンセル成功(Result==0) → CANCELLED"""
        with _make_om(mode="live") as (om, client, _, _):
            client.cancel_order.return_value = {"Result": 0}
            om._update_trade_status = MagicMock()
            with patch.object(mod, "alert") as alert_mock:
                om._timeout_cancel("ORD-1")
        om._update_trade_status.assert_called_once_with("ORD-1", st.CANCELLED)
        alert_mock.assert_not_called()


# ─── A-3: 実約定価格の抽出 ────────────────────────────────────────────────


class TestExtractFill:
    def test_vwap_from_execution_details(self):
        """約定明細(RecType=8)から出来高加重平均単価と累計数量を算出"""
        order = {
            "CumQty": 300,
            "Details": [
                {"RecType": 8, "Price": 1000, "Qty": 100},
                {"RecType": 8, "Price": 1010, "Qty": 200},
                {"RecType": 1, "Price": 0, "Qty": 0},  # 受付（約定でない）→無視
            ],
        }
        price, qty = mod.OrderManager._extract_fill(order)
        assert qty == 300
        assert price == pytest.approx((1000 * 100 + 1010 * 200) / 300)

    def test_fallback_to_cumqty_price(self):
        """約定明細が無ければ CumQty / Price で代用"""
        price, qty = mod.OrderManager._extract_fill({"CumQty": 100, "Price": 1500})
        assert qty == 100
        assert price == 1500.0

    def test_empty_returns_none(self):
        assert mod.OrderManager._extract_fill({}) == (None, None)


class TestResolveFill:
    def test_uses_event_when_available(self):
        """イベントに約定明細があればAPI照会しない"""
        with _make_om(mode="live") as (om, client, _, _):
            event = {"CumQty": 100, "Details": [{"RecType": 8, "Price": 1200, "Qty": 100}]}
            price, qty = om._resolve_fill("ORD-1", event)
        assert (price, qty) == (1200.0, 100)
        client.get_orders.assert_not_called()

    def test_falls_back_to_get_orders(self):
        """イベントに明細が無ければ get_orders で当該注文を照会"""
        with _make_om(mode="live") as (om, client, _, _):
            client.get_orders.return_value = [
                {"ID": "OTHER", "CumQty": 50, "Price": 999},
                {"ID": "ORD-1", "CumQty": 100, "Price": 1300},
            ]
            price, qty = om._resolve_fill("ORD-1", {})
        assert (price, qty) == (1300.0, 100)


class TestOnOrderEventPartialFill:
    def test_partial_fill_marks_partially_filled(self):
        """累計約定 < 発注数量 なら PARTIALLY_FILLED とし、増分のみ反映"""
        trade = MagicMock()
        trade.status = "PENDING"
        trade.symbol = "7203"
        trade.side = "BUY"
        trade.quantity = 300
        trade.filled_quantity = None
        trade.price = 1000.0

        with _make_om(mode="live") as (om, _, _, session):
            session.scalar.return_value = trade
            om._resolve_fill = MagicMock(return_value=(1005.0, 100))  # 100/300のみ約定
            om._record_fill = MagicMock()
            om._update_position = MagicMock()
            om._cancel_timeout_timer = MagicMock()
            om.on_order_event({"OrderID": "ORD-1", "OrderState": 5})

        # 部分約定: ステータスは PARTIALLY_FILLED、タイマーは解除しない
        assert om._record_fill.call_args[0][1] == st.PARTIALLY_FILLED
        om._cancel_timeout_timer.assert_not_called()
        # 増分100のみ反映
        om._update_position.assert_called_once_with("7203", "BUY", 100, 1005.0, order_id=None)
