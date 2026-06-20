"""
OrderManager の成行発注・キャンセル失敗処理・実約定価格反映のテスト（再レビュー対応）

カバー範囲:
  A-1/P0-3,4,5: sell_market が成行注文(FrontOrderType=10, Price=0)を送る。
       reason="stop_loss"/"emergency" は新規発注ゲートをバイパスし退出を優先する
  A-2: _timeout_cancel/_cancel_order_now がキャンセル失敗時に CANCELLED にせず CANCEL_FAILED にする
  A-3/P0-2: _extract_fill が実約定単価(VWAP)・数量を取り出す（cum_qty=0を誤ってNone扱いしない）。
       _sync_trade_with_order が OrderState==5 を無条件でFILLED扱いせず、
       cum_qty に応じて FILLED/PARTIALLY_FILLED/CANCELLED を正しく区別する。
  P0-1: reconcile_open_orders / _reconcile_trade が /orders 照会でDBをブローカー状態へ収束させる
"""
from contextlib import contextmanager
from unittest.mock import ANY, MagicMock, patch

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
            om._apply_fill = MagicMock()
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


# ─── sell_market の reason によるリスクゲートバイパス（再レビュー P0-3/4/5）─────


class TestSellMarketExitBypass:
    def test_normal_reason_blocked_by_can_place_order(self):
        """reason='normal'（既定）は新規発注と同じゲートに従う"""
        with _make_om(mode="live") as (om, client, risk, _):
            risk.can_place_order.return_value = (False, "当日損失上限に達しました")
            oid = om.sell_market("7203", 100)
        assert oid is None
        client.send_order.assert_not_called()

    def test_stop_loss_bypasses_daily_loss_limit(self):
        """reason='stop_loss' は当日損失上限到達中でも発注を続行する"""
        with _make_om(mode="live") as (om, client, risk, _):
            risk.can_place_order.return_value = (False, "当日損失上限に達しました")
            om._cancel_open_orders_for_symbol = MagicMock()
            oid = om.sell_market("7203", 100, reason="stop_loss")
        assert oid == "ORD-1"
        client.send_order.assert_called_once()

    def test_emergency_bypasses_daily_loss_limit(self):
        """reason='emergency' も同様に当日損失上限をバイパスする"""
        with _make_om(mode="live") as (om, client, risk, _):
            risk.can_place_order.return_value = (False, "当日損失上限に達しました")
            om._cancel_open_orders_for_symbol = MagicMock()
            oid = om.sell_market("7203", 100, reason="emergency")
        assert oid == "ORD-1"

    def test_exit_reason_cancels_conflicting_pending_orders_first(self):
        """退出系は新規ゲートで弾かれず、同銘柄の未約定注文を先にキャンセルする"""
        with _make_om(mode="live") as (om, client, risk, _):
            om._cancel_open_orders_for_symbol = MagicMock()
            om.sell_market("7203", 100, reason="emergency")
        om._cancel_open_orders_for_symbol.assert_called_once_with("7203")

    def test_normal_reason_does_not_cancel_conflicting_orders(self):
        """通常売却(reason='normal')は競合キャンセルを行わない（_has_pending_orderでスキップのみ）"""
        with _make_om(mode="live") as (om, client, risk, _):
            om._cancel_open_orders_for_symbol = MagicMock()
            om.sell_market("7203", 100)
        om._cancel_open_orders_for_symbol.assert_not_called()

    def test_exit_reason_skips_pending_check_in_live(self):
        """退出系は _has_pending_order によるスキップを行わない（常に決済を試みる）"""
        with _make_om(mode="live") as (om, client, risk, _):
            om._has_pending_order = MagicMock(return_value=True)
            om._cancel_open_orders_for_symbol = MagicMock()
            oid = om.sell_market("7203", 100, reason="stop_loss")
        assert oid == "ORD-1"

    def test_paper_mode_exit_reason_still_fills(self):
        """ペーパーモードでも reason='emergency' は通常通り即時約定する"""
        with _make_om(mode="paper") as (om, _, risk, _):
            risk.can_place_order.return_value = (False, "ブロック理由")
            with patch.object(mod, "latest_closes", return_value={"7203": 1000.0}):
                oid = om.sell_market("7203", 100, reason="emergency")
        assert oid is not None


class TestCancelOpenOrdersForSymbol:
    def test_cancels_all_open_trades_for_symbol(self):
        """同銘柄の未約定注文をすべて _cancel_order_now に渡す"""
        with _make_om(mode="live") as (om, client, _, session):
            t1, t2 = MagicMock(order_id="O1"), MagicMock(order_id="O2")
            scal = MagicMock()
            scal.all.return_value = [t1, t2]
            session.scalars.return_value = scal
            om._cancel_order_now = MagicMock()
            om._cancel_open_orders_for_symbol("7203")
        assert om._cancel_order_now.call_args_list == [(("O1",),), (("O2",),)]

    def test_no_open_trades_does_nothing(self):
        with _make_om(mode="live") as (om, client, _, session):
            scal = MagicMock()
            scal.all.return_value = []
            session.scalars.return_value = scal
            om._cancel_order_now = MagicMock()
            om._cancel_open_orders_for_symbol("7203")
        om._cancel_order_now.assert_not_called()


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


class TestFindOrder:
    def test_finds_by_id_field(self):
        orders = [{"ID": "A"}, {"ID": "ORD-1", "CumQty": 1}]
        assert mod.OrderManager._find_order(orders, "ORD-1")["CumQty"] == 1

    def test_finds_by_orderid_field(self):
        orders = [{"OrderId": "ORD-1", "CumQty": 2}]
        assert mod.OrderManager._find_order(orders, "ORD-1")["CumQty"] == 2

    def test_not_found_returns_none(self):
        assert mod.OrderManager._find_order([{"ID": "OTHER"}], "ORD-1") is None


class TestSyncTradeWithOrderZeroFillBug:
    """再レビュー P0-2: cum_qty=0（取消/失効で0株約定終了）を誤ってFILLEDにしないことの検証

    _sync_trade_with_order はレースコンディション対策として、ロック取得後にDBから
    最新状態を読み直す（4.2再レビュー）。そのため各テストでは `session.scalar` が
    その「DB上の最新状態」を返すよう、渡す trade と同じ値を設定する。
    """

    def _trade(self, quantity=100, filled_quantity=None, status="PENDING", price=1000.0):
        return MagicMock(order_id="ORD-1", symbol="7203", side="BUY",
                         quantity=quantity, filled_quantity=filled_quantity,
                         status=status, price=price)

    def test_state5_zero_fill_becomes_cancelled_not_filled(self):
        """OrderState/State=5 だが CumQty=0 → CANCELLED（旧バグでは誤ってFILLEDになっていた）"""
        trade = self._trade()
        with _make_om(mode="live") as (om, _, _, session):
            session.scalar.return_value = trade
            om._record_fill = MagicMock()
            om._apply_fill = MagicMock()
            om._sync_trade_with_order(trade, {"State": 5, "CumQty": 0})
        assert om._record_fill.call_args[0][1] == st.CANCELLED
        om._apply_fill.assert_not_called()

    def test_state5_full_fill_becomes_filled(self):
        trade = self._trade(quantity=100)
        with _make_om(mode="live") as (om, _, _, session):
            session.scalar.return_value = trade
            om._record_fill = MagicMock()
            om._apply_fill = MagicMock()
            om._sync_trade_with_order(
                trade, {"State": 5, "CumQty": 100, "Price": 1010},
            )
        assert om._record_fill.call_args[0][1] == st.FILLED
        om._apply_fill.assert_called_once_with(
            "ORD-1", "7203", "BUY", 100, 1010.0, ANY, source="reconcile",
        )

    def test_partial_fill_marks_partially_filled_and_applies_delta_only(self):
        """累計約定 < 発注数量 なら PARTIALLY_FILLED とし、増分のみ反映"""
        trade = self._trade(quantity=300)
        with _make_om(mode="live") as (om, _, _, session):
            session.scalar.return_value = trade
            om._record_fill = MagicMock()
            om._apply_fill = MagicMock()
            om._cancel_timeout_timer = MagicMock()
            om._sync_trade_with_order(
                trade, {"State": 3, "CumQty": 100, "Price": 1005},
            )
        assert om._record_fill.call_args[0][1] == st.PARTIALLY_FILLED
        om._cancel_timeout_timer.assert_not_called()
        om._apply_fill.assert_called_once_with(
            "ORD-1", "7203", "BUY", 100, 1005.0, ANY, source="reconcile",
        )

    def test_second_partial_only_applies_new_delta(self):
        """既に100約定済みのTradeに累計200の通知が来たら増分100のみ反映する"""
        trade = self._trade(quantity=300, filled_quantity=100)
        with _make_om(mode="live") as (om, _, _, session):
            session.scalar.return_value = trade
            om._record_fill = MagicMock()
            om._apply_fill = MagicMock()
            om._sync_trade_with_order(
                trade, {"State": 3, "CumQty": 200, "Price": 1005},
            )
        om._apply_fill.assert_called_once_with(
            "ORD-1", "7203", "BUY", 100, 1005.0, ANY, source="reconcile",
        )

    def test_no_change_is_noop(self):
        """状態・約定数量に変化が無ければ何も更新しない"""
        trade = self._trade(quantity=100, filled_quantity=100, status=st.FILLED)
        with _make_om(mode="live") as (om, _, _, session):
            session.scalar.return_value = trade
            om._record_fill = MagicMock()
            om._sync_trade_with_order(
                trade, {"State": 5, "CumQty": 100, "Price": 1000},
            )
        om._record_fill.assert_not_called()

    def test_order_not_found_marks_unknown_and_alerts(self):
        trade = self._trade(status="PENDING")
        with _make_om(mode="live") as (om, _, _, session):
            session.scalar.return_value = trade
            om._record_fill = MagicMock()
            with patch.object(mod, "alert") as alert_mock:
                om._sync_trade_with_order(trade, None)
        om._record_fill.assert_called_once_with("ORD-1", st.UNKNOWN, None, None)
        alert_mock.assert_called_once()

    def test_already_unknown_order_not_found_does_not_realert(self):
        """既にUNKNOWNなら毎回アラートしない（同じ照合結果での重複通知を避ける）"""
        trade = self._trade(status=st.UNKNOWN)
        with _make_om(mode="live") as (om, _, _, session):
            session.scalar.return_value = trade
            om._record_fill = MagicMock()
            with patch.object(mod, "alert") as alert_mock:
                om._sync_trade_with_order(trade, None)
        om._record_fill.assert_not_called()
        alert_mock.assert_not_called()

    def test_rejected_state_cancels_timer(self):
        trade = self._trade(status="PENDING")
        with _make_om(mode="live") as (om, _, _, session):
            session.scalar.return_value = trade
            om._record_fill = MagicMock()
            om._cancel_timeout_timer = MagicMock()
            # REJECTEDは_status_from_api_orderの戻り値に無いため直接FILLED系以外を想定し、
            # CANCELLED系の終了パスでタイマー解除されることを確認する
            om._sync_trade_with_order(trade, {"State": 5, "CumQty": 0})
        om._cancel_timeout_timer.assert_called_once_with("ORD-1")


class TestTerminalPartialFill:
    """再レビュー P0-2: ブローカー側で確定終了(State=5)した部分約定は、未約定の
    PARTIALLY_FILLED とは別の終端状態(PARTIALLY_FILLED_DONE)になり、いつまでも
    未解決の未約定注文として扱われない（=同銘柄の新規発注を永久にブロックしない）ことの検証。
    """

    def _trade(self, quantity=100, filled_quantity=None, status="PENDING", price=1000.0):
        return MagicMock(order_id="ORD-1", symbol="7203", side="BUY",
                         quantity=quantity, filled_quantity=filled_quantity,
                         status=status, price=price)

    def test_state5_partial_fill_becomes_partially_filled_done(self):
        trade = self._trade(quantity=100)
        with _make_om(mode="live") as (om, _, _, session):
            session.scalar.return_value = trade
            om._record_fill = MagicMock()
            om._apply_fill = MagicMock()
            om._cancel_timeout_timer = MagicMock()
            om._sync_trade_with_order(
                trade, {"State": 5, "CumQty": 40, "Price": 1010},
            )
        assert om._record_fill.call_args[0][1] == st.PARTIALLY_FILLED_DONE
        om._apply_fill.assert_called_once_with(
            "ORD-1", "7203", "BUY", 40, 1010.0, ANY, source="reconcile",
        )
        # ブローカー側で確定終了済みなので、未約定のPARTIALLY_FILLEDと異なりタイマーは解除する
        om._cancel_timeout_timer.assert_called_once_with("ORD-1")

    def test_partially_filled_done_is_not_open_status(self):
        """PARTIALLY_FILLED_DONE は OPEN_STATUSES に含まれない
        （=同銘柄の新規発注ブロック・reconcile対象から外れ、永久にブロックされ続けない）"""
        assert st.PARTIALLY_FILLED_DONE not in st.OPEN_STATUSES
        assert st.PARTIALLY_FILLED in st.OPEN_STATUSES


class TestSellMarketExitCancelFailureBlocks:
    """再レビュー P0-4: 退出系(stop_loss/emergency)発注前の競合注文キャンセルが
    1件でも失敗した場合、その状態のまま成行売りを送信してはならない。"""

    def test_cancel_failure_blocks_emergency_sell(self):
        with _make_om(mode="live") as (om, client, _, _):
            om._cancel_open_orders_for_symbol = MagicMock(return_value=False)
            with patch.object(mod, "alert") as alert_mock:
                oid = om.sell_market("7203", 100, reason="emergency")
        assert oid is None
        client.send_order.assert_not_called()
        alert_mock.assert_called_once()

    def test_cancel_failure_blocks_stop_loss_sell(self):
        with _make_om(mode="live") as (om, client, _, _):
            om._cancel_open_orders_for_symbol = MagicMock(return_value=False)
            with patch.object(mod, "alert"):
                oid = om.sell_market("7203", 100, reason="stop_loss")
        assert oid is None
        client.send_order.assert_not_called()

    def test_cancel_success_allows_emergency_sell(self):
        with _make_om(mode="live") as (om, client, _, _):
            om._cancel_open_orders_for_symbol = MagicMock(return_value=True)
            oid = om.sell_market("7203", 100, reason="emergency")
        assert oid == "ORD-1"
        client.send_order.assert_called_once()


class TestCancelOpenOrdersForSymbolAggregateResult:
    def test_returns_true_when_all_cancellations_succeed(self):
        with _make_om(mode="live") as (om, client, _, session):
            t1, t2 = MagicMock(order_id="O1"), MagicMock(order_id="O2")
            scal = MagicMock()
            scal.all.return_value = [t1, t2]
            session.scalars.return_value = scal
            om._cancel_order_now = MagicMock(return_value=True)
            assert om._cancel_open_orders_for_symbol("7203") is True

    def test_returns_false_when_any_cancellation_fails(self):
        with _make_om(mode="live") as (om, client, _, session):
            t1, t2 = MagicMock(order_id="O1"), MagicMock(order_id="O2")
            scal = MagicMock()
            scal.all.return_value = [t1, t2]
            session.scalars.return_value = scal
            om._cancel_order_now = MagicMock(side_effect=[True, False])
            assert om._cancel_open_orders_for_symbol("7203") is False

    def test_returns_true_when_no_open_orders(self):
        with _make_om(mode="live") as (om, client, _, session):
            scal = MagicMock()
            scal.all.return_value = []
            session.scalars.return_value = scal
            om._cancel_order_now = MagicMock()
            assert om._cancel_open_orders_for_symbol("7203") is True


class TestCloseAllPositionsUsesBrokerSourceOfTruth:
    """再レビュー P0-1: ライブ/semi_liveの緊急全決済はローカルDBのPositionではなく
    ブローカー /positions を正本として使う。"""

    def test_uses_broker_leaves_qty_not_db(self):
        with _make_om(mode="live") as (om, client, _, _):
            client.get_positions.return_value = [
                {"Symbol": "7203", "LeavesQty": 150, "HoldQty": 0},
            ]
            om.sell_market = MagicMock()
            om.close_all_positions()
        om.sell_market.assert_called_once_with("7203", 150, reason="emergency")

    def test_skips_zero_quantity_broker_positions(self):
        with _make_om(mode="live") as (om, client, _, _):
            client.get_positions.return_value = [
                {"Symbol": "7203", "LeavesQty": 0, "HoldQty": 0},
            ]
            om.sell_market = MagicMock()
            om.close_all_positions()
        om.sell_market.assert_not_called()

    def test_positions_fetch_failure_blocks_and_alerts_instead_of_using_db(self):
        with _make_om(mode="live") as (om, client, _, _):
            client.get_positions.side_effect = RuntimeError("network")
            om.sell_market = MagicMock()
            with patch.object(mod, "alert") as alert_mock:
                om.close_all_positions()
        om.sell_market.assert_not_called()
        alert_mock.assert_called_once()

    def test_multiple_symbols_each_closed(self):
        with _make_om(mode="live") as (om, client, _, _):
            client.get_positions.return_value = [
                {"Symbol": "7203", "LeavesQty": 100, "HoldQty": 0},
                {"Symbol": "9984", "LeavesQty": 50, "HoldQty": 0},
            ]
            om.sell_market = MagicMock()
            om.close_all_positions()
        calls = om.sell_market.call_args_list
        assert calls[0].args == ("7203", 100)
        assert calls[0].kwargs == {"reason": "emergency"}
        assert calls[1].args == ("9984", 50)
        assert calls[1].kwargs == {"reason": "emergency"}


class TestReconcileTrade:
    def test_fetches_orders_and_syncs(self):
        trade = MagicMock(order_id="ORD-1")
        with _make_om(mode="live") as (om, client, _, _):
            client.get_orders.return_value = [{"ID": "ORD-1", "State": 5, "CumQty": 0}]
            om._sync_trade_with_order = MagicMock()
            om._reconcile_trade(trade)
        om._sync_trade_with_order.assert_called_once_with(
            trade, {"ID": "ORD-1", "State": 5, "CumQty": 0},
        )

    def test_api_failure_is_swallowed(self):
        trade = MagicMock(order_id="ORD-1")
        with _make_om(mode="live") as (om, client, _, _):
            client.get_orders.side_effect = RuntimeError("network")
            om._sync_trade_with_order = MagicMock()
            om._reconcile_trade(trade)  # raiseしない
        om._sync_trade_with_order.assert_not_called()


class TestReconcileOpenOrders:
    def test_paper_mode_does_nothing(self):
        with _make_om(mode="paper") as (om, client, _, _):
            om.reconcile_open_orders()
        client.get_orders.assert_not_called()

    def test_no_open_trades_skips_api_call(self):
        with _make_om(mode="live") as (om, client, _, session):
            scal = MagicMock()
            scal.all.return_value = []
            session.scalars.return_value = scal
            om.reconcile_open_orders()
        client.get_orders.assert_not_called()

    def test_syncs_each_open_trade_with_single_api_call(self):
        t1 = MagicMock(order_id="O1")
        t2 = MagicMock(order_id="O2")
        with _make_om(mode="live") as (om, client, _, session):
            scal = MagicMock()
            scal.all.return_value = [t1, t2]
            session.scalars.return_value = scal
            client.get_orders.return_value = [
                {"ID": "O1", "State": 5, "CumQty": 100},
                {"ID": "O2", "State": 5, "CumQty": 0},
            ]
            om._sync_trade_with_order = MagicMock()
            om.reconcile_open_orders()
        assert client.get_orders.call_count == 1
        assert om._sync_trade_with_order.call_count == 2

    def test_api_failure_does_not_raise(self):
        with _make_om(mode="live") as (om, client, _, session):
            scal = MagicMock()
            scal.all.return_value = [MagicMock(order_id="O1")]
            session.scalars.return_value = scal
            client.get_orders.side_effect = RuntimeError("down")
            om.reconcile_open_orders()  # raiseしない
