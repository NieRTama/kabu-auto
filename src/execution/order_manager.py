"""
発注・ポジション管理モジュール
- 指値注文 + タイムアウトキャンセル
- 二重発注防止（ライブモード）
- 約定確認ループ（WebSocket OrderEvent）+ 冪等性保証
- 約定レスポンス厳密検証
- 起動時API同期（ライブモード）
- ペーパートレードモード対応
"""
import threading
import uuid
from datetime import datetime
from typing import Optional

from loguru import logger
from sqlalchemy import func, select

from src.api.kabu_client import KabuClient
from src.core import clock
from src.core import config as cfg
from src.core.alerts import alert
from src.data.database import Position, Trade, get_session
from src.data.market_data import latest_closes
from src.execution import order_status as st
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

    def sync_on_startup(self) -> None:
        """起動時にAPIの注文状態と DB を同期する（ライブモードのみ）。

        API側で見つかった注文は実際の State / 約定数量へ同期する。
        APIに全く見つからない未解決注文は、約定済みを誤って取消扱いにしないよう
        CANCELLED ではなく UNKNOWN とし、人手確認を促す（UNKNOWN が残る間は発注抑止）。
        """
        if self._is_paper:
            return
        try:
            api_orders = self._client.get_orders()
            by_id = {}
            for o in api_orders:
                oid = o.get("ID") or o.get("OrderId")
                if oid:
                    by_id[oid] = o
            unknown_count = 0
            synced_count = 0
            with get_session() as session:
                open_trades = session.scalars(
                    select(Trade).where(Trade.status.in_(tuple(st.OPEN_STATUSES)))
                ).all()
                for t in open_trades:
                    o = by_id.get(t.order_id)
                    if o is None:
                        t.status = st.UNKNOWN
                        unknown_count += 1
                        continue
                    new_status = self._status_from_api_order(o, t.quantity)
                    if new_status and new_status != t.status:
                        t.status = new_status
                        synced_count += 1
                    cum = o.get("CumQty")
                    if cum:
                        t.filled_quantity = int(cum)
                session.commit()
            if synced_count:
                logger.info(f"起動同期: API状態へ {synced_count} 件を同期")
            if unknown_count:
                logger.warning(
                    f"起動同期: API未発見の未解決注文を {unknown_count} 件 UNKNOWN に設定（要確認）"
                )
                alert("起動同期: 状態不明の注文",
                      f"{unknown_count} 件。約定/残存の可能性があり要確認")
            logger.info("起動時注文同期完了")
        except Exception as e:
            logger.warning(f"起動時注文同期失敗（継続）: {e}")

    @staticmethod
    def _status_from_api_order(order: dict, ordered_qty: int) -> Optional[str]:
        """kabu注文照会の State / 約定数量(CumQty) から DBステータスを判定する。

        kabu State: 5=終了（約定/取消/失効で確定）、その他=処理中。
        """
        state = order.get("State")
        cum = order.get("CumQty") or 0
        if state == 5:  # 終了
            if ordered_qty > 0 and cum >= ordered_qty:
                return st.FILLED
            if cum > 0:
                return st.PARTIALLY_FILLED
            return st.CANCELLED
        # まだ生きている注文
        if cum > 0:
            return st.PARTIALLY_FILLED
        return st.PENDING

    def on_order_event(self, event: dict) -> None:
        """WebSocket約定イベントのハンドラ"""
        order_id = event.get("OrderID")
        state = event.get("OrderState")
        if state != 5:  # 5 = 完了（約定 or 取消で終了）
            return

        # 冪等性保証: 既に FILLED の注文は二重処理しない
        with get_session() as session:
            trade = session.scalar(select(Trade).where(Trade.order_id == order_id))
        if trade is None:
            logger.warning(f"不明な注文IDの約定通知を無視: {order_id}")
            return
        if trade.status == st.FILLED:
            logger.debug(f"重複約定通知を無視: {order_id}")
            return

        # 実約定単価・累計約定数量を取得（イベント優先、欠落時はAPI照会）。
        # 取得不能時のみ発注情報（指値price・発注数量）で代用する（従来挙動へのフォールバック）。
        fill_price, cum_qty = self._resolve_fill(order_id, event)
        eff_price = fill_price if fill_price else trade.price
        eff_cum_qty = cum_qty if cum_qty else trade.quantity
        already_applied = trade.filled_quantity or 0
        delta_qty = max(0, eff_cum_qty - already_applied)
        is_partial = eff_cum_qty < trade.quantity

        status = st.PARTIALLY_FILLED if is_partial else st.FILLED
        logger.info(
            f"約定確認: OrderID={order_id} 単価={eff_price} "
            f"累計約定={eff_cum_qty}/{trade.quantity} ({status})"
        )
        # 全量約定でタイマー解除（部分約定は残数の約定/タイムアウトを待つ）
        if not is_partial:
            self._cancel_timeout_timer(order_id)
        self._record_fill(
            order_id, status, eff_price, eff_cum_qty,
            filled_at=clock.now() if not is_partial else None,
        )
        # ライブモード: 増分のみポジションへ反映（ペーパーは発注時点で反映済み）
        if not self._is_paper and delta_qty > 0:
            self._update_position(
                trade.symbol, trade.side, delta_qty, eff_price,
                order_id=order_id if trade.side == "SELL" else None,
            )

    def buy(self, symbol: str, price: float, quantity: int,
            sector: Optional[str] = None) -> Optional[str]:
        """指値買い注文を発注する"""
        if quantity <= 0:
            return None
        ok, reason = self._risk.can_place_order()
        if not ok:
            logger.warning(f"発注スキップ: {symbol} - {reason}")
            return None

        # ライブモード: 二重発注防止
        if not self._is_paper and self._has_pending_order(symbol):
            logger.warning(f"未約定注文あり、重複発注スキップ: {symbol}")
            return None

        if self._is_paper:
            # ペーパーモードは常に即時成立するため、ここでカウントを確定する
            self._risk.increment_order_count()
            order_id = f"PAPER-BUY-{symbol}-{uuid.uuid4().hex[:8]}"
            now = clock.now()
            logger.info(f"[ペーパー] 買い: {symbol} {quantity}株 @{price:.0f}円")
            self._record_trade(order_id, symbol, "BUY", quantity, price,
                               status=st.FILLED, filled_at=now)
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
            if result.get("Result") != 0:
                logger.error(
                    f"買い注文拒否: {symbol} Result={result.get('Result')} "
                    f"Message={result.get('Message', '')}"
                )
                self._record_trade(
                    f"REJECTED-{symbol}-{uuid.uuid4().hex[:8]}", symbol, "BUY",
                    quantity, price, status=st.REJECTED,
                )
                return None
            order_id = result.get("OrderId")
            if not order_id:
                logger.error(f"買い注文: OrderId 未取得: {symbol} {result}")
                return None
            self._risk.increment_order_count()
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

        # ライブモード: 二重発注防止
        if not self._is_paper and self._has_pending_order(symbol):
            logger.warning(f"未約定注文あり、重複発注スキップ: {symbol}")
            return None

        if self._is_paper:
            # ペーパーモードは常に即時成立するため、ここでカウントを確定する
            self._risk.increment_order_count()
            order_id = f"PAPER-SELL-{symbol}-{uuid.uuid4().hex[:8]}"
            now = clock.now()
            logger.info(f"[ペーパー] 売り: {symbol} {quantity}株 @{price:.0f}円")
            self._record_trade(order_id, symbol, "SELL", quantity, price,
                               status=st.FILLED, filled_at=now)
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
            if result.get("Result") != 0:
                logger.error(
                    f"売り注文拒否: {symbol} Result={result.get('Result')} "
                    f"Message={result.get('Message', '')}"
                )
                self._record_trade(
                    f"REJECTED-{symbol}-{uuid.uuid4().hex[:8]}", symbol, "SELL",
                    quantity, price, status=st.REJECTED,
                )
                return None
            order_id = result.get("OrderId")
            if not order_id:
                logger.error(f"売り注文: OrderId 未取得: {symbol} {result}")
                return None
            self._risk.increment_order_count()
            self._record_trade(order_id, symbol, "SELL", quantity, price)
            self._set_cancel_timer(order_id)
            return order_id
        except Exception as e:
            logger.error(f"売り注文失敗: {symbol} {e}")
            return None

    def sell_market(self, symbol: str, quantity: int) -> Optional[str]:
        """成行売り注文を発注する（損切り・緊急決済など、確実な約定を優先する用途）。

        指値 sell() は急変時に約定しない恐れがあるため、損切り/全決済は本メソッドを使う。
        kabu現物の成行は FrontOrderType=10・Price=0。
        """
        if quantity <= 0:
            return None
        ok, reason = self._risk.can_place_order()
        if not ok:
            logger.warning(f"発注スキップ: {symbol} - {reason}")
            return None

        # ライブモード: 二重発注防止
        if not self._is_paper and self._has_pending_order(symbol):
            logger.warning(f"未約定注文あり、重複発注スキップ: {symbol}")
            return None

        if self._is_paper:
            # ペーパーは即時成立。約定価格は最新終値で近似する（成行のため板は無い）
            price = latest_closes([symbol]).get(symbol)
            if not price or price <= 0:
                logger.error(f"[ペーパー] 成行売りスキップ（終値取得失敗）: {symbol}")
                return None
            self._risk.increment_order_count()
            order_id = f"PAPER-SELLM-{symbol}-{uuid.uuid4().hex[:8]}"
            now = clock.now()
            logger.info(f"[ペーパー] 成行売り: {symbol} {quantity}株 @{price:.0f}円")
            self._record_trade(order_id, symbol, "SELL", quantity, price,
                               status=st.FILLED, filled_at=now)
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
            "Price": 0,  # 成行は0
            "ExpireDay": 0,
            "FrontOrderType": 10,  # 成行
        }
        try:
            result = self._client.send_order(order)
            if result.get("Result") != 0:
                logger.error(
                    f"成行売り注文拒否: {symbol} Result={result.get('Result')} "
                    f"Message={result.get('Message', '')}"
                )
                self._record_trade(
                    f"REJECTED-{symbol}-{uuid.uuid4().hex[:8]}", symbol, "SELL",
                    quantity, 0.0, status=st.REJECTED,
                )
                return None
            order_id = result.get("OrderId")
            if not order_id:
                logger.error(f"成行売り注文: OrderId 未取得: {symbol} {result}")
                return None
            self._risk.increment_order_count()
            # 成行は発注時に価格未確定。price=0 で記録し、約定時に filled_price で確定する
            self._record_trade(order_id, symbol, "SELL", quantity, 0.0)
            self._set_cancel_timer(order_id)
            return order_id
        except Exception as e:
            logger.error(f"成行売り注文失敗: {symbol} {e}")
            return None

    def close_all_positions(self) -> None:
        """全ポジションを成行で強制決済する（緊急用）"""
        logger.warning("緊急全ポジション決済を実行します")
        with get_session() as session:
            positions = session.scalars(
                select(Position).where(Position.quantity > 0)
            ).all()
        for pos in positions:
            try:
                self.sell_market(pos.symbol, pos.quantity)
            except Exception as e:
                logger.error(f"緊急決済失敗: {pos.symbol} {e}")

    def _has_pending_order(self, symbol: str) -> bool:
        """同銘柄に未約定注文があるか確認する（二重発注防止）"""
        with get_session() as session:
            count = session.scalar(
                select(func.count(Trade.id)).where(
                    Trade.symbol == symbol,
                    Trade.status.in_(tuple(st.OPEN_STATUSES)),
                )
            ) or 0
        return count > 0

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
        with self._orders_lock:
            self._pending_orders.pop(order_id, None)
        try:
            result = self._client.cancel_order(order_id)
        except Exception as e:
            # キャンセルAPI失敗 = 注文は証券会社側で生きている可能性がある。
            # DBを CANCELLED にすると実口座と乖離（二重発注・想定外約定）するため、
            # CANCEL_FAILED にして要人手確認とする。
            logger.error(f"キャンセル失敗: {order_id} {e}")
            self._update_trade_status(order_id, st.CANCEL_FAILED)
            alert("キャンセル失敗", f"OrderID={order_id} が未約定のまま残存の可能性。要確認")
            return
        if result.get("Result") == 0:
            self._update_trade_status(order_id, st.CANCELLED)
        else:
            logger.error(f"キャンセル拒否: {order_id} {result}")
            self._update_trade_status(order_id, st.CANCEL_FAILED)
            alert("キャンセル拒否",
                  f"OrderID={order_id} Result={result.get('Result')}。要確認")

    def _record_trade(self, order_id: str, symbol: str, side: str,
                      quantity: int, price: float,
                      status: str = st.PENDING,
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

    def _record_fill(self, order_id: str, status: str,
                     filled_price: Optional[float], filled_quantity: Optional[int],
                     filled_at: Optional[datetime] = None) -> None:
        """約定情報（実約定単価・累計約定数量・ステータス）をTradeへ記録する"""
        with get_session() as session:
            trade = session.scalar(select(Trade).where(Trade.order_id == order_id))
            if trade is None:
                return
            trade.status = status
            if filled_price is not None:
                trade.filled_price = filled_price
            if filled_quantity is not None:
                trade.filled_quantity = filled_quantity
            if filled_at:
                trade.filled_at = filled_at
            session.commit()

    def _resolve_fill(self, order_id: str,
                      event: dict) -> tuple[Optional[float], Optional[int]]:
        """実約定単価・累計約定数量を解決する。

        まず約定イベントから抽出し、明細が無ければ注文照会API(get_orders)で補完する。
        どちらからも取得できない場合は (None, None) を返し、呼び出し側で発注価格に
        フォールバックする。
        """
        price, qty = self._extract_fill(event)
        if qty is not None:
            return price, qty
        try:
            for o in self._client.get_orders():
                if order_id in (o.get("ID"), o.get("OrderId")):
                    return self._extract_fill(o)
        except Exception as e:
            logger.warning(f"約定明細のAPI照会失敗: {order_id} {e}")
        return None, None

    @staticmethod
    def _extract_fill(order: dict) -> tuple[Optional[float], Optional[int]]:
        """注文/約定イベントのdictから (実約定単価VWAP, 累計約定数量) を取り出す。

        kabuステーションAPIの約定明細 Details[].RecType==8（約定）を集計して
        出来高加重平均単価を算出する。明細が無い場合は CumQty / Price で代用する。
        スキーマ差異に強くするため全て .get で防御的に扱う。
        """
        if not isinstance(order, dict):
            return None, None
        details = order.get("Details") or []
        execs = [d for d in details
                 if d.get("RecType") == 8 and d.get("Qty") and d.get("Price")]
        if execs:
            tot_qty = sum(d["Qty"] for d in execs)
            vwap = (sum(d["Price"] * d["Qty"] for d in execs) / tot_qty
                    if tot_qty else None)
            cum = order.get("CumQty")
            return vwap, (int(cum) if cum is not None else int(tot_qty))
        cum = order.get("CumQty")
        price = order.get("Price")
        if cum:
            return (float(price) if price else None), int(cum)
        return None, None

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
                    pos.updated_at = clock.now()
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
                pos.updated_at = clock.now()
                trade = session.scalar(
                    select(Trade).where(Trade.order_id == order_id)
                ) if order_id else None
                if trade:
                    # 部分約定で複数回呼ばれても合算されるよう加算する
                    trade.pnl = (trade.pnl or 0) + pnl
                # 損失を RiskManager に記録（当日損失上限チェック用）
                self._risk.record_loss(pnl)
            elif side == "SELL" and not pos:
                logger.warning(
                    f"SELL約定だが保有ポジション無し: {symbol} {quantity}株 "
                    f"（PnL未集計・リスク管理に反映されません）"
                )
            session.commit()
