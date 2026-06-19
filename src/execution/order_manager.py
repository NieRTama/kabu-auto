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
        """WebSocket約定イベントのハンドラ（あくまで補助トリガ）。

        kabuステーションのWebSocket PUSHは本来板/価格情報が主体であり、注文状態の
        正本としては扱わない（イベント取り逃し・再起動・WS切断に弱いため）。
        ここでは「この注文を今すぐ照会すべき」というトリガとしてのみ使い、実際の
        状態確定は必ず /orders 照会の結果（_sync_trade_with_order）で行う。
        定期実行される reconcile_open_orders() と同じ収束処理を共有する。
        """
        order_id = event.get("OrderID")
        if order_id is None:
            return
        with get_session() as session:
            trade = session.scalar(select(Trade).where(Trade.order_id == order_id))
        if trade is None:
            logger.warning(f"不明な注文IDの約定通知を無視: {order_id}")
            return
        if trade.status not in st.OPEN_STATUSES:
            logger.debug(f"既に確定済みの注文通知を無視: {order_id} ({trade.status})")
            return
        self._reconcile_trade(trade)

    def reconcile_open_orders(self) -> None:
        """未約定（OPEN_STATUSES）の全Tradeをブローカーの /orders 照会結果へ収束させる。

        WebSocketイベントの取り逃し・接続断・プロセス再起動を跨いでも、定期的に
        この照合を実行することでDBとブローカーの状態ズレを検知・補正する
        （ペーパーモードでは実ブローカー注文が存在しないため何もしない）。
        """
        if self._is_paper:
            return
        with get_session() as session:
            open_trades = session.scalars(
                select(Trade).where(Trade.status.in_(tuple(st.OPEN_STATUSES)))
            ).all()
        if not open_trades:
            return
        try:
            orders = self._client.get_orders()
        except Exception as e:
            logger.warning(f"注文照合失敗（次回再試行）: {e}")
            return
        for trade in open_trades:
            order = self._find_order(orders, trade.order_id)
            self._sync_trade_with_order(trade, order)

    def _reconcile_trade(self, trade: Trade) -> None:
        """1件のTradeをAPI照会で取得し直して状態を収束させる（on_order_event用）。"""
        try:
            orders = self._client.get_orders()
        except Exception as e:
            logger.warning(f"約定通知トリガのAPI照会失敗: {trade.order_id} {e}")
            return
        order = self._find_order(orders, trade.order_id)
        self._sync_trade_with_order(trade, order)

    @staticmethod
    def _find_order(orders: list, order_id: str) -> Optional[dict]:
        for o in orders:
            if order_id in (o.get("ID"), o.get("OrderId")):
                return o
        return None

    def _sync_trade_with_order(self, trade: Trade, order: Optional[dict]) -> None:
        """1件のTradeをブローカー側注文（dict、Noneなら未発見）の実状態へ収束させる。

        `_status_from_api_order()`/`_extract_fill()` を使い、`OrderState==5`を
        無条件で約定とは扱わない（cum_qty=0なら取消、cum_qty<発注数量なら部分約定、
        と正しく区別する）。on_order_event と reconcile_open_orders の共通処理。
        """
        if order is None:
            if trade.status != st.UNKNOWN:
                logger.warning(f"API未発見の注文: OrderID={trade.order_id} {trade.symbol}")
                self._record_fill(trade.order_id, st.UNKNOWN, None, None)
                alert("注文が見つかりません",
                      f"OrderID={trade.order_id} {trade.symbol}。約定/残存の可能性があり要確認")
            return

        new_status = self._status_from_api_order(order, trade.quantity)
        fill_price, cum_qty = self._extract_fill(order)
        already_applied = trade.filled_quantity or 0
        eff_cum_qty = cum_qty if cum_qty is not None else already_applied
        delta_qty = max(0, eff_cum_qty - already_applied)
        eff_price = fill_price if fill_price is not None else trade.price

        if new_status == trade.status and delta_qty == 0:
            return  # 変化なし

        logger.info(
            f"注文状態同期: OrderID={trade.order_id} {trade.status}→{new_status} "
            f"約定={eff_cum_qty}/{trade.quantity}"
        )
        filled_at = clock.now() if new_status == st.FILLED else None
        self._record_fill(
            trade.order_id, new_status,
            eff_price if delta_qty > 0 else None, eff_cum_qty,
            filled_at=filled_at,
        )
        # ライブモード: 増分のみポジションへ反映（ペーパーは発注時点で反映済み）
        if (new_status in (st.FILLED, st.PARTIALLY_FILLED)
                and not self._is_paper and delta_qty > 0):
            self._update_position(
                trade.symbol, trade.side, delta_qty, eff_price,
                order_id=trade.order_id if trade.side == "SELL" else None,
            )
        # 終了状態ではタイムアウトキャンセルタイマーを解除（部分約定は残数を待つので解除しない）
        if new_status in (st.FILLED, st.CANCELLED, st.REJECTED):
            self._cancel_timeout_timer(trade.order_id)

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
                               status=st.FILLED, filled_at=now, sector=sector)
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
                    quantity, price, status=st.REJECTED, sector=sector,
                )
                return None
            order_id = result.get("OrderId")
            if not order_id:
                logger.error(f"買い注文: OrderId 未取得: {symbol} {result}")
                return None
            self._risk.increment_order_count()
            self._record_trade(order_id, symbol, "BUY", quantity, price, sector=sector)
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

    def sell_market(self, symbol: str, quantity: int, reason: str = "normal") -> Optional[str]:
        """成行売り注文を発注する（損切り・緊急決済など、確実な約定を優先する用途）。

        指値 sell() は急変時に約定しない恐れがあるため、損切り/全決済は本メソッドを使う。
        kabu現物の成行は FrontOrderType=10・Price=0。

        reason:
          - "normal":    通常の売却。新規発注と同じリスクゲート（日次上限・損失上限・
                         未解決注文ガード・同銘柄pending重複）を適用する。
          - "stop_loss" / "emergency": 既存リスクを減らす退出操作なので、新規発注用の
                         ゲートは適用しない（日次上限/損失上限到達中でも・未解決注文が
                         残っていても、退出だけは止めない）。ただし同銘柄の未約定注文は
                         先にキャンセルしてから成行売りを送る（競合発注を避けるため）。
        """
        if quantity <= 0:
            return None
        is_exit = reason in ("stop_loss", "emergency")

        if not is_exit:
            ok, block_reason = self._risk.can_place_order()
            if not ok:
                logger.warning(f"発注スキップ: {symbol} - {block_reason}")
                return None
            # ライブモード: 二重発注防止
            if not self._is_paper and self._has_pending_order(symbol):
                logger.warning(f"未約定注文あり、重複発注スキップ: {symbol}")
                return None
        elif not self._is_paper:
            # 退出系: ゲートでブロックせず、競合する未約定注文を先にキャンセルしてから進める
            self._cancel_open_orders_for_symbol(symbol)

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
                self.sell_market(pos.symbol, pos.quantity, reason="emergency")
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
        self._cancel_order_now(order_id)

    def _cancel_order_now(self, order_id: str) -> bool:
        """指定注文をキャンセルし、結果に応じてDBステータスを更新する。

        キャンセル成立で True、失敗/拒否（注文が生存の可能性）で False を返す。
        失敗時は CANCELLED にせず CANCEL_FAILED にして要人手確認とする
        （DBを CANCELLED にすると実口座と乖離し二重発注・想定外約定を招くため）。
        """
        self._cancel_timeout_timer(order_id)
        try:
            result = self._client.cancel_order(order_id)
        except Exception as e:
            logger.error(f"キャンセル失敗: {order_id} {e}")
            self._update_trade_status(order_id, st.CANCEL_FAILED)
            alert("キャンセル失敗", f"OrderID={order_id} が未約定のまま残存の可能性。要確認")
            return False
        if result.get("Result") == 0:
            self._update_trade_status(order_id, st.CANCELLED)
            return True
        logger.error(f"キャンセル拒否: {order_id} {result}")
        self._update_trade_status(order_id, st.CANCEL_FAILED)
        alert("キャンセル拒否",
              f"OrderID={order_id} Result={result.get('Result')}。要確認")
        return False

    def _cancel_open_orders_for_symbol(self, symbol: str) -> None:
        """同銘柄の未約定注文をすべてキャンセルする（退出系発注前の競合解消用）。"""
        with get_session() as session:
            open_ids = [
                t.order_id for t in session.scalars(
                    select(Trade).where(
                        Trade.symbol == symbol,
                        Trade.status.in_(tuple(st.OPEN_STATUSES)),
                    )
                ).all()
            ]
        for oid in open_ids:
            self._cancel_order_now(oid)

    def _record_trade(self, order_id: str, symbol: str, side: str,
                      quantity: int, price: float,
                      status: str = st.PENDING,
                      filled_at: Optional[datetime] = None,
                      sector: Optional[str] = None) -> None:
        with get_session() as session:
            session.add(Trade(
                order_id=order_id,
                symbol=symbol,
                side=side,
                sector=sector,
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
        if cum is not None:
            # cum=0（=確定済み0株約定）も真値として返す。0をfalsy判定すると
            # 「未約定」と「APIから値が取れない」を区別できなくなる
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
