"""
約定(Fill)記録・Position/PnLロールアップのテスト（Phase 5 / 4.2）

- paper の即時約定が Fill を生成し Position・filled_quantity/filled_price を更新すること
- live の reconcile（_sync_trade_with_order 経由）が増分のみ Fill として記録されること
- 部分約定→残り約定の2段階で filled_quantity/filled_price がロールアップされること
- SELLの実現損益がFIFOで計算され Trade.pnl に反映されること
"""
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

import src.core.config as cfg
import src.data.database as db
import src.execution.order_manager as mod
from src.data.database import Fill, Position, Trade, get_session


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
def _make_om(mode: str):
    cfg_mock = _cfg(mode)
    client = MagicMock()
    client.send_order.return_value = {"Result": 0, "OrderId": "LIVE-1"}
    risk = MagicMock()
    risk.can_place_order.return_value = (True, "")
    with patch.object(mod, "cfg", cfg_mock):
        om = mod.OrderManager(client, risk)
        yield om, client, risk


class TestPaperFillCreatesLedger:
    def test_buy_creates_fill_and_position(self, isolated_db):
        with _make_om("paper") as (om, client, risk):
            order_id = om.buy("7203", 1000.0, 100, sector="Auto")
        with get_session() as s:
            fill = s.scalar(select(Fill).where(Fill.symbol == "7203", Fill.side == "BUY"))
            assert fill.fill_qty == 100
            assert fill.remaining_qty == 100
            pos = s.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.quantity == 100
            assert pos.avg_cost == 1000.0
            trade = s.scalar(select(Trade).where(Trade.order_id == order_id))
            assert trade.filled_quantity == 100
            assert trade.filled_price == 1000.0

    def test_sell_records_realized_pnl_on_trade(self, isolated_db):
        with _make_om("paper") as (om, client, risk):
            om.buy("7203", 1000.0, 100)
            sell_id = om.sell("7203", 1100.0, 100)
        with get_session() as s:
            trade = s.scalar(select(Trade).where(Trade.order_id == sell_id))
            assert trade.pnl == pytest.approx(10000.0)
        risk.record_loss.assert_called()  # SELLの度に呼ばれる（損失でなくても渡す）

    def test_sell_calls_risk_record_loss_with_realized(self, isolated_db):
        with _make_om("paper") as (om, client, risk):
            om.buy("7203", 1000.0, 100)
            om.sell("7203", 900.0, 100)  # 損失
        # 最後の record_loss 呼び出しが FIFO 実現損益(-10000)であること
        args = risk.record_loss.call_args_list[-1][0]
        assert args[0] == pytest.approx(-10000.0)

    def test_position_quantity_decreases_after_sell(self, isolated_db):
        with _make_om("paper") as (om, client, risk):
            om.buy("7203", 1000.0, 100)
            om.sell("7203", 1100.0, 60)
        with get_session() as s:
            pos = s.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.quantity == 40
            assert pos.avg_cost == 1000.0  # 残存ロットは同じ単価


class TestLiveReconcileFillRollup:
    def test_full_fill_creates_single_fill_record(self, isolated_db):
        with _make_om("live") as (om, client, risk):
            order_id = om.buy("7203", 1000.0, 100)
            trade = MagicMock(order_id=order_id, symbol="7203", side="BUY",
                              quantity=100, filled_quantity=None, status="PENDING", price=1000.0)
            om._sync_trade_with_order(trade, {"State": 5, "CumQty": 100, "Price": 1010})
        with get_session() as s:
            fills = s.scalars(select(Fill).where(Fill.symbol == "7203")).all()
            assert len(fills) == 1
            assert fills[0].fill_qty == 100
            assert fills[0].fill_price == 1010.0
            t = s.scalar(select(Trade).where(Trade.order_id == order_id))
            assert t.filled_quantity == 100
            assert t.filled_price == 1010.0

    def test_two_partial_fills_rollup_vwap(self, isolated_db):
        """2回の部分約定（別単価）が累積され、filled_quantity/filled_priceがVWAPになること"""
        with _make_om("live") as (om, client, risk):
            order_id = om.buy("7203", 1000.0, 300)
            trade_obj = MagicMock(order_id=order_id, symbol="7203", side="BUY",
                                  quantity=300, filled_quantity=None, status="PENDING", price=1000.0)
            # 1回目: 100株 @1000
            om._sync_trade_with_order(trade_obj, {"State": 3, "CumQty": 100, "Price": 1000})
            # 2回目: 累計200株（今回分100株） @1020
            trade_obj.filled_quantity = 100
            trade_obj.status = "PARTIALLY_FILLED"
            om._sync_trade_with_order(trade_obj, {"State": 3, "CumQty": 200, "Price": 1020})
        with get_session() as s:
            fills = s.scalars(
                select(Fill).where(Fill.symbol == "7203", Fill.side == "BUY")
            ).all()
            # 1件はbuy()自体の発注時点のpaper扱いではなくlive(PENDING)なので発注時Fillは無く、
            # 上記2回のreconcileでできた2件のみのはず
            assert len(fills) == 2
            total = sum(f.fill_qty for f in fills)
            vwap = sum(f.fill_qty * f.fill_price for f in fills) / total
            t = s.scalar(select(Trade).where(Trade.order_id == order_id))
            assert t.filled_quantity == total
            assert t.filled_price == pytest.approx(vwap)

    def test_two_partial_sell_fills_accumulate_pnl_and_vwap(self, isolated_db):
        """SELLが2回の部分約定（別単価）に分かれても、Trade.pnlがFIFO実現損益の
        合計になり、filled_price がVWAPになること（観点1: 部分約定の重複計上が無いこと）。
        """
        with _make_om("live") as (om, client, risk):
            buy_id = om.buy("7203", 1000.0, 300)
            buy_trade = MagicMock(order_id=buy_id, symbol="7203", side="BUY",
                                  quantity=300, filled_quantity=None,
                                  status="PENDING", price=1000.0)
            om._sync_trade_with_order(buy_trade, {"State": 5, "CumQty": 300, "Price": 1000})

            client.send_order.return_value = {"Result": 0, "OrderId": "SELL-1"}
            sell_id = om.sell("7203", 1100.0, 300)
            sell_trade = MagicMock(order_id=sell_id, symbol="7203", side="SELL",
                                   quantity=300, filled_quantity=None,
                                   status="PENDING", price=1100.0)
            # 1回目: 累計100株 @1100
            om._sync_trade_with_order(sell_trade, {"State": 3, "CumQty": 100, "Price": 1100})
            sell_trade.filled_quantity = 100
            sell_trade.status = "PARTIALLY_FILLED"
            # 2回目: 累計300株（残り200株） @1200
            om._sync_trade_with_order(sell_trade, {"State": 5, "CumQty": 300, "Price": 1200})

        with get_session() as s:
            t = s.scalar(select(Trade).where(Trade.order_id == sell_id))
            # FIFOで300株すべて元値1000円のロットから消費される:
            # (1100-1000)*100 + (1200-1000)*200 = 10000 + 40000 = 50000
            assert t.pnl == pytest.approx(50000.0)
            assert t.filled_quantity == 300
            # VWAP = (1100*100 + 1200*200) / 300
            assert t.filled_price == pytest.approx((1100 * 100 + 1200 * 200) / 300)
            pos = s.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.quantity == 0

    def test_concurrent_reconcile_and_ws_event_do_not_double_count_fill(self, isolated_db):
        """再レビュー: WebSocketコールバックと定期reconcile(15秒毎)が別スレッドから
        同時に同じ注文の約定完了を検知しても、Fillが二重生成されないこと。

        両者は別々のセッションでDBからTradeを読み（=互いに古いfilled_quantityの
        スナップショットを持つ）、その状態のまま _sync_trade_with_order を呼ぶ。
        ロックなしでは両方が同じdelta_qty(=fill_qty全量)を計算してFIFOロットへ
        二重計上してしまう（修正前は再現した）。注文IDごとのロック＋ロック内での
        DB再読み込みにより、片方だけが反映され、もう片方は変化なしとして無視される。
        """
        import threading

        with _make_om("live") as (om, client, risk):
            order_id = om.buy("7203", 1000.0, 100)

        # 2スレッドとも「まだ未約定」という古いスナップショットを持つTradeを渡す
        # （WSイベント・reconcileそれぞれが発注直後に読んだ状態を模している）
        trade_snapshot_1 = MagicMock(order_id=order_id, symbol="7203", side="BUY",
                                     quantity=100, filled_quantity=None,
                                     status="PENDING", price=1000.0)
        trade_snapshot_2 = MagicMock(order_id=order_id, symbol="7203", side="BUY",
                                     quantity=100, filled_quantity=None,
                                     status="PENDING", price=1000.0)
        order_data = {"State": 5, "CumQty": 100, "Price": 1010}

        t1 = threading.Thread(
            target=lambda: om._sync_trade_with_order(trade_snapshot_1, order_data)
        )
        t2 = threading.Thread(
            target=lambda: om._sync_trade_with_order(trade_snapshot_2, order_data)
        )
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        with get_session() as s:
            fills = s.scalars(
                select(Fill).where(Fill.symbol == "7203", Fill.side == "BUY")
            ).all()
            assert len(fills) == 1, (
                f"Fillが{len(fills)}件生成された。レースで二重計上された可能性がある"
            )
            assert fills[0].fill_qty == 100
            t = s.scalar(select(Trade).where(Trade.order_id == order_id))
            assert t.filled_quantity == 100
            assert t.status == "FILLED"
