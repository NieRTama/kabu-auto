"""
発注・ポジション管理モジュール
- 指値注文 + タイムアウトキャンセル
- 約定確認ループ（WebSocket OrderEvent）
- ペーパートレードモード対応
"""
import threading
import uuid
from datetime import datetime
from typing import Optional

from loguru import logger
from sqlalchemy import select

from src.api.kabu_client import KabuClient
from src.core import config as cfg
from src.data.database import Position, Trade, get_session
from src.risk.manager import RiskManager


class OrderManager:
    def __init__(self, client: KabuClient, risk: RiskManager):
        self._client = client
        self._risk = risk
        self._conf = cfg.get_section("trading")
        # パスワードは kabu_station セクションから取得する
        self._kabu_password = cfg.get_section("kabu_station").get("password", "")
        self._is_paper = self._conf.get("mode", "paper") == "paper"
        self._pending_orders: dict[str, threading.Timer] = {}
        self._orders_lock = threading.Lock()

    def on_order_event(self, event: dict) -> None:
        """WebSocket約定イベントのハンドラ"""
        order_id = event.get("OrderID")
        state = event.get("OrderState")
        if state == 5:  # 約定済み
            logger.info(f"約定確認: OrderID={order_id}")
            self._cancel_timeout_timer(order_id)
            self._update_trade_status(order_id, "FILLED", filled_at=datetime.now())
            # ライブモード: 約定後にポジションを更新（ペーパーは発注時点で更新済み）
            if not self._is_paper:
                self._update_position_from_fill(order_id)

    def buy(self, symbol: str, price: float, quantity: int,
            sector: Optional[str] = None) -> Optional[str]:
        """指値買い注文を発注する"""
        if quantity <= 0:
            return None
        ok, reason = self._risk.can_place_order()
        if not ok:
            logger.warning(f"発注スキップ: {symbol} - {reason}")
            return None
        self._risk.increment_order_count()

        if self._is_paper:
            order_id = f"PAPER-BUY-{symbol}-{uuid.uuid4().hex[:8]}"
            now = datetime.now()
            logger.info(f"[ペーパー] 買い: {symbol} {quantity}株 @{price:.0f}円")
            self._record_trade(order_id, symbol, "BUY", quantity, price,
                               status="FILLED", filled_at=now)
            self._update_position(symbol, "BUY", quantity, price, sector)
            return order_id

        order = {
            "Password": self._kabu_password,
            "Symbol": symbol,
            "Exchange": 1,
            "SecurityType": 1,  # 株式
            "Side": "2",  # 買い
            "CashMargin": 1,  # 現物
            "DelivType": 2,  # 自動振替
            "FundType": "  ",
            "AccountType": 4,  # 特定口座
            "Qty": quantity,
            "Price": price,
            "ExpireDay": 0,  # 当日中
            "FrontOrderType": 20,  # 指値
        }
        try:
            result = self._client.send_order(order)
            order_id = result.get("OrderId")
            self._record_trade(order_id, symbol, "BUY", quantity, price)
            self._set_cancel_timer(order_id)
            return order_id
        except Exception as e:
            logger.error(f"買い注文失敗: {symbol} {e}")
            return None

    def sell(self, symbol: str, price: float, quantity: int) -> Optional[str]:
        """指値売り注文を発注する"""
        if quantity <= 0:
            return None
        ok, reason = self._risk.can_place_order()
        if not ok:
            logger.warning(f"発注スキップ: {symbol} - {reason}")
            return None
        self._risk.increment_order_count()

        if self._is_paper:
            order_id = f"PAPER-SELL-{symbol}-{uuid.uuid4().hex[:8]}"
            now = datetime.now()
            logger.info(f"[ペーパー] 売り: {symbol} {quantity}株 @{price:.0f}円")
            self._record_trade(order_id, symbol, "SELL", quantity, price,
                               status="FILLED", filled_at=now)
            self._update_position(symbol, "SELL", quantity, price, order_id=order_id)
            return order_id

        order = {
            "Password": self._kabu_password,
            "Symbol": symbol,
            "Exchange": 1,
            "SecurityType": 1,
            "Side": "1",  # 売り
            "CashMargin": 1,
            "DelivType": 2,
            "FundType": "  ",
            "AccountType": 4,
            "Qty": quantity,
            "Price": price,
            "ExpireDay": 0,
            "FrontOrderType": 20,
        }
        try:
            result = self._client.send_order(order)
            order_id = result.get("OrderId")
            self._record_trade(order_id, symbol, "SELL", quantity, price)
            self._set_cancel_timer(order_id)
            return order_id
        except Exception as e:
            logger.error(f"売り注文失敗: {symbol} {e}")
            return None

    def close_all_positions(self) -> None:
        """全ポジションを強制決済する（緊急用）"""
        logger.warning("緊急全ポジション決済を実行します")
        with get_session() as session:
            positions = session.scalars(
                select(Position).where(Position.quantity > 0)
            ).all()
        for pos in positions:
            try:
                board = self._client.get_board(pos.symbol)
                price = board.get("CurrentPrice") or board.get("Sell1", {}).get("Price", 0)
                self.sell(pos.symbol, price, pos.quantity)
            except Exception as e:
                logger.error(f"緊急決済失敗: {pos.symbol} {e}")

    def _set_cancel_timer(self, order_id: str) -> None:
        timeout = self._conf.get("order_timeout_seconds", 300)
        timer = threading.Timer(timeout, self._timeout_cancel, args=[order_id])
        timer.daemon = True
        timer.start()
        with self._orders_lock:
            self._pending_orders[order_id] = timer

    def _cancel_timeout_timer(self, order_id: str) -> None:
        with self._orders_lock:
            timer = self._pending_orders.pop(order_id, None)
        if timer:
            timer.cancel()

    def _timeout_cancel(self, order_id: str) -> None:
        logger.info(f"注文タイムアウト → キャンセル: {order_id}")
        try:
            self._client.cancel_order(order_id)
        except Exception as e:
            logger.error(f"キャンセル失敗: {order_id} {e}")
        with self._orders_lock:
            self._pending_orders.pop(order_id, None)
        self._update_trade_status(order_id, "CANCELLED")

    def _record_trade(self, order_id: str, symbol: str, side: str,
                      quantity: int, price: float,
                      status: str = "PENDING",
                      filled_at: Optional[datetime] = None) -> None:
        with get_session() as session:
            session.add(Trade(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                status=status,
                filled_at=filled_at,
            ))
            session.commit()

    def _update_trade_status(self, order_id: str, status: str,
                             filled_at: Optional[datetime] = None) -> None:
        with get_session() as session:
            trade = session.scalar(select(Trade).where(Trade.order_id == order_id))
            if trade:
                trade.status = status
                if filled_at:
                    trade.filled_at = filled_at
                session.commit()

    def _update_position_from_fill(self, order_id: str) -> None:
        """約定後にポジションを更新する（ライブモード用）"""
        with get_session() as session:
            trade = session.scalar(select(Trade).where(Trade.order_id == order_id))
            if trade is None:
                logger.warning(f"約定トレード未発見: {order_id}")
                return
            symbol, side, quantity, price = (
                trade.symbol, trade.side, trade.quantity, trade.price
            )
        self._update_position(
            symbol, side, quantity, price,
            order_id=order_id if side == "SELL" else None,
        )

    def _update_position(self, symbol: str, side: str, quantity: int,
                         price: float, sector: Optional[str] = None,
                         order_id: Optional[str] = None) -> None:
        with get_session() as session:
            pos = session.scalar(select(Position).where(Position.symbol == symbol))
            if side == "BUY":
                if pos:
                    total_qty = pos.quantity + quantity
                    pos.avg_cost = (pos.avg_cost * pos.quantity + price * quantity) / total_qty
                    pos.quantity = total_qty
                    pos.updated_at = datetime.now()
                else:
                    session.add(Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_cost=price,
                        sector=sector,
                    ))
            elif side == "SELL" and pos:
                pnl = (price - pos.avg_cost) * quantity
                pos.quantity = max(0, pos.quantity - quantity)
                pos.updated_at = datetime.now()
                trade = session.scalar(
                    select(Trade).where(Trade.order_id == order_id)
                ) if order_id else None
                if trade:
                    trade.pnl = pnl
            session.commit()
