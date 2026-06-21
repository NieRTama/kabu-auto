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
from src.core import halt
from src.core import trading_mode as tm
from src.core.alerts import alert
from src.data.database import Fill, OrderApproval, OrderIntent, Position, Trade, get_session
from src.data.market_data import latest_closes
from src.execution import lots
from src.execution import order_status as st
from src.execution.broker_gateway import BrokerGateway
from src.risk.manager import RiskManager


class OrderManager:
    def __init__(self, client: KabuClient, risk: RiskManager):
        # ブローカーAPIとの境界（発注ペイロード構築・送信・取消・照会）はここに集約する（C3）。
        # OrderManager 自身は client を直接保持しない（旧経路が誤って復活するのを防ぐ）。
        self._broker = BrokerGateway(client)
        self._risk = risk
        self._conf = cfg.get_section("trading")
        self._mode = self._conf.get("mode", "paper")
        self._is_paper = tm.is_paper(self._mode)
        self._pending_orders: dict[str, threading.Timer] = {}
        self._orders_lock = threading.Lock()
        # 注文IDごとの排他ロック（_sync_trade_with_order用）。
        # WebSocketコールバック(on_order_event)とAPSchedulerの定期reconcile(15秒毎)が
        # 別スレッドから同じ注文IDを同時に同期しようとすると、両方が同じ古いfilled_quantity
        # を基準に増分(delta_qty)を計算し、同一の約定をFillへ二重計上してしまう
        # （FIFOロット・Position・損益が壊れる）。注文IDごとに直列化することで防ぐ。
        self._sync_locks: dict[str, threading.Lock] = {}
        self._sync_locks_guard = threading.Lock()
        # 承認IDごとの排他ロック（approve_order用）。ダッシュボードから同じ承認IDに対する
        # 承認リクエストが（ダブルクリック・複数タブ等で）同時に来ると、両方が
        # status=="PENDING" の確認に通って実APIへ二重発注してしまう恐れがある。
        # 承認IDごとに直列化し、最初の1件だけが発注からAPPROVED確定までを完了する。
        self._approval_locks: dict[int, threading.Lock] = {}
        self._approval_locks_guard = threading.Lock()

    def sync_on_startup(self) -> None:
        """起動時にAPIの注文状態と DB を同期する（ライブモードのみ）。

        API側で見つかった注文は実際の State / 約定数量へ同期する。
        APIに全く見つからない未解決注文は、約定済みを誤って取消扱いにしないよう
        CANCELLED ではなく UNKNOWN とし、人手確認を促す（UNKNOWN が残る間は発注抑止）。
        """
        # semi_live承認中（APPROVING）にプロセスが落ちると、実発注が成立したか不明な
        # まま承認レコードが取り残される。再承認はブロックされ続けるが、運用者が気付かなければ
        # いつまでも放置されうるため、モードに関わらず起動時に検知してアラートする（レビュー再指摘 High）。
        self._alert_stuck_approvals()
        if self._is_paper:
            return
        try:
            api_orders = self._broker.get_orders()
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
                # 部分約定のままブローカー側で確定終了（残数は取消/失効で二度と
                # 約定しない）。未約定の PARTIALLY_FILLED とは区別する（P0-2）
                return st.PARTIALLY_FILLED_DONE
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
            orders = self._broker.get_orders()
        except Exception as e:
            logger.warning(f"注文照合失敗（次回再試行）: {e}")
            return
        for trade in open_trades:
            order = self._find_order(orders, trade.order_id)
            self._sync_trade_with_order(trade, order)

    def reconcile_positions_with_broker(self) -> dict:
        """DBの Position とブローカー実際の保有(/positions)を照合し、ズレを検出する（P0-3）。

        /orders 照会だけでは「Trade群からの積算が実口座の建玉と一致しているか」までは
        検証できない（バグ・未知の手動操作・取り逃したイベント等でズレうる）。ズレを
        検知したら、実口座の建玉を正しく把握できていない状態で発注ロジックを動かし続ける
        のは危険なため、既存の kill switch（取引停止スイッチ）を作動させて新規発注を
        止め、人手確認を要求する（fail-closed。損切り・緊急決済はhalt中でもバイパスされ
        止まらないため、退出操作で建玉を減らす方向には影響しない）。

        ウォレット（現金残高）照合は対象外（現金フロー台帳が無く「期待現金残高」を
        算出できないため。将来の拡張課題）。
        ペーパー/dry_runは実ブローカーが無いため何もしない。
        """
        if self._is_paper or self._mode == tm.DRY_RUN:
            return {"ok": True, "drift": []}
        try:
            broker_positions = self._broker.get_positions()
        except Exception as e:
            logger.warning(f"建玉照合失敗（次回再試行）: {e}")
            return {"ok": False, "drift": [], "error": str(e)}

        broker_qty: dict[str, int] = {}
        for p in broker_positions:
            symbol = p.get("Symbol")
            if not symbol:
                continue
            broker_qty[symbol] = broker_qty.get(symbol, 0) + int(p.get("LeavesQty") or 0)

        with get_session() as session:
            db_qty = {
                p.symbol: p.quantity
                for p in session.scalars(select(Position).where(Position.quantity > 0)).all()
            }

        drift = [
            {"symbol": sym, "db_qty": db_qty.get(sym, 0), "broker_qty": broker_qty.get(sym, 0)}
            for sym in (set(db_qty) | set(broker_qty))
            if db_qty.get(sym, 0) != broker_qty.get(sym, 0)
        ]
        if drift:
            detail = "; ".join(
                f"{d['symbol']}: DB={d['db_qty']}株 ブローカー={d['broker_qty']}株"
                for d in drift
            )
            logger.critical(f"建玉ドリフト検知: {detail}")
            alert(
                "建玉ドリフト検知（要確認）",
                f"DBとブローカー実建玉に差異があります。新規発注を停止しました: {detail}",
            )
            if not halt.is_halted():
                halt.engage(f"建玉ドリフト検知: {detail}")
            return {"ok": False, "drift": drift}
        return {"ok": True, "drift": []}

    def _reconcile_trade(self, trade: Trade) -> None:
        """1件のTradeをAPI照会で取得し直して状態を収束させる（on_order_event用）。"""
        try:
            orders = self._broker.get_orders()
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

    def _get_sync_lock(self, order_id: str) -> threading.Lock:
        with self._sync_locks_guard:
            lock = self._sync_locks.get(order_id)
            if lock is None:
                lock = threading.Lock()
                self._sync_locks[order_id] = lock
            return lock

    def _sync_trade_with_order(self, trade: Trade, order: Optional[dict]) -> None:
        """1件のTradeをブローカー側注文（dict、Noneなら未発見）の実状態へ収束させる。

        `_status_from_api_order()`/`_extract_fill()` を使い、`OrderState==5`を
        無条件で約定とは扱わない（cum_qty=0なら取消、cum_qty<発注数量なら部分約定、
        と正しく区別する）。on_order_event と reconcile_open_orders の共通処理。

        WebSocketコールバックと定期reconcileジョブが別スレッドから同じ注文IDを同時に
        同期すると、両方が同じ古い filled_quantity を基準に増分(delta_qty)を計算して
        同一約定を二重計上する恐れがある。注文IDごとのロックで直列化し、ロック取得後に
        DBから最新の filled_quantity/status を読み直してから判定する。
        """
        with self._get_sync_lock(trade.order_id):
            self._sync_trade_with_order_locked(trade, order)

    def _sync_trade_with_order_locked(self, trade: Trade, order: Optional[dict]) -> None:
        # 呼び出し元から渡された trade は別スレッドが読んだ時点のスナップショットの
        # 可能性があるため、ロック内でDBの最新状態へ読み直す（無ければ削除済み等のため何もしない）
        with get_session() as session:
            current = session.scalar(select(Trade).where(Trade.order_id == trade.order_id))
            if current is None:
                return
            cur_status = current.status
            cur_filled_quantity = current.filled_quantity
            cur_quantity = current.quantity
            cur_price = current.price
            cur_symbol = current.symbol
            cur_side = current.side

        if order is None:
            if cur_status != st.UNKNOWN:
                logger.warning(f"API未発見の注文: OrderID={trade.order_id} {cur_symbol}")
                self._record_fill(trade.order_id, st.UNKNOWN, None, None)
                alert("注文が見つかりません",
                      f"OrderID={trade.order_id} {cur_symbol}。約定/残存の可能性があり要確認")
            return

        new_status = self._status_from_api_order(order, cur_quantity)
        fill_price, cum_qty = self._extract_fill(order)
        already_applied = cur_filled_quantity or 0
        eff_cum_qty = cum_qty if cum_qty is not None else already_applied
        delta_qty = max(0, eff_cum_qty - already_applied)
        eff_price = fill_price if fill_price is not None else cur_price

        if new_status == cur_status and delta_qty == 0:
            return  # 変化なし

        logger.info(
            f"注文状態同期: OrderID={trade.order_id} {cur_status}→{new_status} "
            f"約定={eff_cum_qty}/{cur_quantity}"
        )
        filled_at = (clock.now() if new_status in (st.FILLED, st.PARTIALLY_FILLED_DONE)
                    else None)
        self._record_fill(
            trade.order_id, new_status,
            eff_price if delta_qty > 0 else None, eff_cum_qty,
            filled_at=filled_at,
        )
        # ライブモード: 増分のみポジションへ反映（ペーパーは発注時点で反映済み）
        if (new_status in (st.FILLED, st.PARTIALLY_FILLED, st.PARTIALLY_FILLED_DONE)
                and not self._is_paper and delta_qty > 0):
            self._apply_fill(
                trade.order_id, cur_symbol, cur_side, delta_qty, eff_price,
                clock.now(), source="reconcile",
            )
        # 終了状態ではタイムアウトキャンセルタイマーを解除
        # （未約定のPARTIALLY_FILLEDは残数を待つので解除しないが、PARTIALLY_FILLED_DONEは
        # ブローカー側で確定終了済みなので解除する。P0-2）
        if new_status in (st.FILLED, st.CANCELLED, st.REJECTED, st.PARTIALLY_FILLED_DONE):
            self._cancel_timeout_timer(trade.order_id)

    def buy(self, symbol: str, price: float, quantity: int,
            sector: Optional[str] = None, rationale: Optional[str] = None,
            source: str = "manual") -> Optional[str]:
        """指値買い注文を発注する。

        rationale=発注根拠（シグナルスコア等。7.6）。source=発注のきっかけ
        （signal_scan/morning_execution/manual等。OrderIntent.sourceに記録する。4.2）。
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
            # ペーパーモードは常に即時成立するため、ここでカウントを確定する
            self._risk.increment_order_count()
            order_id = f"PAPER-BUY-{symbol}-{uuid.uuid4().hex[:8]}"
            now = clock.now()
            logger.info(f"[ペーパー] 買い: {symbol} {quantity}株 @{price:.0f}円")
            self._record_trade(order_id, symbol, "BUY", quantity, price,
                               status=st.FILLED, filled_at=now, sector=sector,
                               rationale=rationale, source=source, order_type="LIMIT")
            self._apply_fill(order_id, symbol, "BUY", quantity, price, now,
                             source="paper", sector=sector)
            return order_id

        if self._mode == tm.DRY_RUN:
            return self._record_dry_run("BUY", symbol, quantity, price, sector, rationale,
                                        source=source, order_type="LIMIT")
        if self._mode == tm.SEMI_LIVE:
            return self._enqueue_approval("BUY", "LIMIT", symbol, price, quantity, sector,
                                          rationale, source=source)

        return self._live_buy(symbol, price, quantity, sector, rationale, source=source)

    def _live_buy(self, symbol: str, price: float, quantity: int,
                  sector: Optional[str] = None, rationale: Optional[str] = None, *,
                  intent_id: Optional[int] = None, source: str = "manual") -> Optional[str]:
        """実APIへ指値買いを送る（live / semi_live承認実行の共通実体）。

        intent_id を渡すと既存のOrderIntent（semi_live承認時等）に紐付ける。
        省略時は `_record_trade()` が新規にOrderIntentを作る。
        """
        try:
            result = self._broker.send_buy_limit(symbol, price, quantity)
            if not self._broker.is_accepted(result):
                logger.error(
                    f"買い注文拒否: {symbol} Result={result.get('Result')} "
                    f"Message={result.get('Message', '')}"
                )
                self._record_trade(
                    f"REJECTED-{symbol}-{uuid.uuid4().hex[:8]}", symbol, "BUY",
                    quantity, price, status=st.REJECTED, sector=sector,
                    intent_id=intent_id, source=source, order_type="LIMIT",
                )
                return None
            order_id = self._broker.order_id_of(result)
            if not order_id:
                return None
            self._risk.increment_order_count()
            self._record_trade(order_id, symbol, "BUY", quantity, price, sector=sector,
                               rationale=rationale, intent_id=intent_id, source=source,
                               order_type="LIMIT")
            self._set_cancel_timer(order_id)
            return order_id
        except Exception as e:
            logger.error(f"買い注文失敗: {symbol} {e}")
            return None

    def sell(self, symbol: str, price: float, quantity: int,
             rationale: Optional[str] = None, source: str = "manual") -> Optional[str]:
        """指値売り注文を発注する（rationale=発注根拠。7.6 / source=発注のきっかけ。4.2）"""
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
                               status=st.FILLED, filled_at=now, rationale=rationale,
                               source=source, order_type="LIMIT")
            self._apply_fill(order_id, symbol, "SELL", quantity, price, now, source="paper")
            return order_id

        if self._mode == tm.DRY_RUN:
            return self._record_dry_run("SELL", symbol, quantity, price, rationale=rationale,
                                        source=source, order_type="LIMIT")
        if self._mode == tm.SEMI_LIVE:
            return self._enqueue_approval("SELL", "LIMIT", symbol, price, quantity,
                                          rationale=rationale, source=source)

        return self._live_sell(symbol, price, quantity, rationale, source=source)

    def _live_sell(self, symbol: str, price: float, quantity: int,
                   rationale: Optional[str] = None, *,
                   intent_id: Optional[int] = None, source: str = "manual") -> Optional[str]:
        """実APIへ指値売りを送る（live / semi_live承認実行の共通実体）。"""
        try:
            result = self._broker.send_sell_limit(symbol, price, quantity)
            if not self._broker.is_accepted(result):
                logger.error(
                    f"売り注文拒否: {symbol} Result={result.get('Result')} "
                    f"Message={result.get('Message', '')}"
                )
                self._record_trade(
                    f"REJECTED-{symbol}-{uuid.uuid4().hex[:8]}", symbol, "SELL",
                    quantity, price, status=st.REJECTED,
                    intent_id=intent_id, source=source, order_type="LIMIT",
                )
                return None
            order_id = self._broker.order_id_of(result)
            if not order_id:
                return None
            self._risk.increment_order_count()
            self._record_trade(order_id, symbol, "SELL", quantity, price, rationale=rationale,
                               intent_id=intent_id, source=source, order_type="LIMIT")
            self._set_cancel_timer(order_id)
            return order_id
        except Exception as e:
            logger.error(f"売り注文失敗: {symbol} {e}")
            return None

    def sell_market(self, symbol: str, quantity: int, reason: str = "normal",
                    rationale: Optional[str] = None) -> Optional[str]:
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
        # OrderIntent.source へ記録する発注のきっかけ（4.2）。reasonを直接転記する。
        source = reason if is_exit else "manual"
        # 退出系は理由が明確な根拠なので、明示指定が無ければ reason を根拠として記録する（7.6）
        if rationale is None and is_exit:
            rationale = {"stop_loss": "損切り（stop_loss）", "emergency": "緊急決済（emergency）"}[reason]

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
            # 退出系: ゲートでブロックせず、競合する未約定注文を先にキャンセルしてから進める。
            # キャンセルに1件でも失敗（CANCEL_FAILED、実口座に注文が残っている可能性）が
            # あれば、その状態のまま成行売りを重ねて送るのは危険なため中断する（再レビュー P0-4）。
            if not self._cancel_open_orders_for_symbol(symbol):
                logger.critical(
                    f"退出注文ブロック: {symbol} の競合注文キャンセルに失敗。"
                    "実口座の状態が不確実なため成行売りを送信せず中断します"
                )
                alert(
                    "退出注文ブロック（要確認）",
                    f"{symbol}: 既存注文のキャンセル失敗のため緊急/損切り成行売りを"
                    "送信せず中断しました。証券会社サイトで直接ご確認ください",
                )
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
                               status=st.FILLED, filled_at=now, rationale=rationale,
                               source=source, order_type="MARKET")
            self._apply_fill(order_id, symbol, "SELL", quantity, price, now, source="paper")
            return order_id

        if self._mode == tm.DRY_RUN:
            # ドライランは退出系でも実発注しない（検証専用）。価格未確定なので0で記録する
            return self._record_dry_run("SELL", symbol, quantity, 0.0, rationale=rationale,
                                        source=source, order_type="MARKET")
        if self._mode == tm.SEMI_LIVE and not is_exit:
            # 通常売りは承認キューへ。損切り・緊急決済（退出）は承認を介さず即時発注する
            return self._enqueue_approval("SELL", "MARKET", symbol, 0.0, quantity,
                                          rationale=rationale, source=source)

        return self._live_sell_market(symbol, quantity, rationale, source=source)

    def _live_sell_market(self, symbol: str, quantity: int,
                          rationale: Optional[str] = None, *,
                          intent_id: Optional[int] = None,
                          source: str = "manual") -> Optional[str]:
        """実APIへ成行売りを送る（live / semi_live退出・承認実行の共通実体）。"""
        try:
            result = self._broker.send_sell_market(symbol, quantity)
            if not self._broker.is_accepted(result):
                logger.error(
                    f"成行売り注文拒否: {symbol} Result={result.get('Result')} "
                    f"Message={result.get('Message', '')}"
                )
                self._record_trade(
                    f"REJECTED-{symbol}-{uuid.uuid4().hex[:8]}", symbol, "SELL",
                    quantity, 0.0, status=st.REJECTED,
                    intent_id=intent_id, source=source, order_type="MARKET",
                )
                return None
            order_id = self._broker.order_id_of(result)
            if not order_id:
                return None
            self._risk.increment_order_count()
            # 成行は発注時に価格未確定。price=0 で記録し、約定時に filled_price で確定する
            self._record_trade(order_id, symbol, "SELL", quantity, 0.0, rationale=rationale,
                               intent_id=intent_id, source=source, order_type="MARKET")
            self._set_cancel_timer(order_id)
            return order_id
        except Exception as e:
            logger.error(f"成行売り注文失敗: {symbol} {e}")
            return None

    def place_stop_loss(self, symbol: str, quantity: int,
                        trigger_price: float) -> Optional[str]:
        """ブローカー側の逆指値ストップ（成行）を発注する（4.3・ライブのみ実発注）。

        PC/アプリが停止していても、株価がトリガー価格以下に下落したら証券会社側で
        自動的に成行決済される「保険」。アプリ側の損切り監視(stop_loss_check)が止まる
        ケースに備える。確実な約定を優先し、トリガー後は成行(AfterHitOrderType=1)とする。

        モード別:
          - live / semi_live: 実APIへ逆指値成行SELLを送る（保護目的なのでリスクゲートは
            適用しない。退出系と同じ思想）。約定/タイムアウトキャンセルの対象にしない
            （ストップは発動まで生かし続けるため、通常のタイムアウト解除はしない）。
          - paper / dry_run: 実ブローカーが無いため何もしない（ペーパーの損切りは
            stop_loss_check が日足終値で代替する）。
        """
        if quantity <= 0 or not trigger_price or trigger_price <= 0:
            return None
        if self._is_paper or self._mode == tm.DRY_RUN:
            logger.info(
                f"[{self._mode}] ブローカー側逆指値ストップはスキップ（実ブローカー無し）: "
                f"{symbol} {quantity}株 @{trigger_price:.0f}"
            )
            return None

        try:
            result = self._broker.send_stop_loss_market(symbol, quantity, trigger_price)
            if not self._broker.is_accepted(result):
                logger.error(
                    f"逆指値ストップ注文拒否: {symbol} Result={result.get('Result')} "
                    f"Message={result.get('Message', '')}"
                )
                return None
            order_id = self._broker.order_id_of(result)
            if not order_id:
                return None
            self._risk.increment_order_count()
            # ストップは発動まで生かすためタイムアウトキャンセルは設定しない
            self._record_trade(
                order_id, symbol, "SELL", quantity, 0.0,
                rationale=f"ブローカー側逆指値ストップ トリガー@{trigger_price:.0f}（成行）",
                source="broker_stop", order_type="STOP",
            )
            logger.warning(
                f"ブローカー側逆指値ストップ発注: {symbol} {quantity}株 トリガー@{trigger_price:.0f}"
            )
            return order_id
        except Exception as e:
            logger.error(f"逆指値ストップ注文失敗: {symbol} {e}")
            return None

    def close_all_positions(self) -> None:
        """全ポジションを成行で強制決済する（緊急用）。

        ライブ/semi_liveでは、ローカルDBの Position ではなくブローカー /positions を
        正本として使う（再レビュー P0-1）。DBはWS取り逃し・反映遅延・バグ等で実口座と
        ズレる可能性があり、緊急時にズレたDBを正本にすると「実際は200株あるのに100株しか
        売らず残存」「実際は0株なのに不要な売りを送る」といった事故になるため。
        /positions 取得自体に失敗した場合は実口座の状態を把握できないため、当てずっぽうで
        DBに基づいた自動決済はせず、critical alert を出して人手確認に委ねる（fail-closed）。
        ペーパー/dry_runは実ブローカーが無いためDBを正本のまま使う。
        """
        logger.warning("緊急全ポジション決済を実行します")
        if self._is_paper or self._mode == tm.DRY_RUN:
            with get_session() as session:
                positions = session.scalars(
                    select(Position).where(Position.quantity > 0)
                ).all()
            for pos in positions:
                try:
                    self.sell_market(pos.symbol, pos.quantity, reason="emergency")
                except Exception as e:
                    logger.error(f"緊急決済失敗: {pos.symbol} {e}")
            return

        try:
            broker_positions = self._broker.get_positions()
        except Exception as e:
            logger.critical(f"緊急決済ブロック: ブローカー /positions 取得失敗: {e}")
            alert(
                "緊急決済ブロック（要確認）",
                f"/positions 取得失敗のため自動決済を中断しました: {e}。"
                "証券会社サイトで建玉を直接確認し、必要であれば手動で決済してください",
            )
            return

        for pos in broker_positions:
            try:
                symbol = pos.get("Symbol")
                leaves_qty = int(pos.get("LeavesQty") or 0)
                if not symbol or leaves_qty <= 0:
                    continue
                # sell_market(reason="emergency") は送信前に同銘柄の未約定注文を
                # 先にキャンセルする（成立すれば HoldQty による引当は解放されるため、
                # ここでは LeavesQty=実保有数量をそのまま渡せばよい）
                self.sell_market(symbol, leaves_qty, reason="emergency")
            except Exception as e:
                logger.error(f"緊急決済失敗: {pos} {e}")

    def cancel_all_pending_buys(self) -> int:
        """未約定のBUY注文をすべてキャンセルする（kill switch 作動時の新規建玉防止）。

        SELL（決済・損切り）はリスクを減らす方向なので残す。BUYのみ取り消す。
        戻り値: キャンセルを試みた注文件数（ペーパーは対象外で常に0）。
        """
        if self._is_paper:
            return 0
        with get_session() as session:
            open_buy_ids = [
                t.order_id for t in session.scalars(
                    select(Trade).where(
                        Trade.side == "BUY",
                        Trade.status.in_(tuple(st.OPEN_STATUSES)),
                    )
                ).all()
            ]
        for oid in open_buy_ids:
            self._cancel_order_now(oid)
        return len(open_buy_ids)

    def halt_trading(self, reason: str = "", close_positions: bool = False) -> dict:
        """取引停止スイッチ（kill switch）を作動させる。

        手順: 停止フラグON → 未約定BUYをキャンセル →（任意で）全ポジション成行決済。
        以後 RiskManager.can_place_order() が新規発注を弾く。損切り・緊急決済は
        reason バイパスで引き続き実行可能。解除は resume_trading() で手動のみ。
        戻り値: {"state": <halt状態>, "cancelled_buys": int, "closed_positions": bool}
        """
        from src.core import halt
        state = halt.engage(reason)
        cancelled = self.cancel_all_pending_buys()
        if close_positions:
            self.close_all_positions()
        alert("取引停止スイッチ作動",
              f"理由: {state.get('reason')} / 未約定BUYキャンセル: {cancelled}件"
              + ("（全ポジション成行決済を実行）" if close_positions else ""))
        return {"state": state, "cancelled_buys": cancelled, "closed_positions": close_positions}

    def resume_trading(self) -> dict:
        """取引停止を解除する。未解決注文（UNKNOWN/CANCEL_FAILED）が残る間は解除しない。

        実口座と乖離した状態（状態不明・キャンセル失敗）のまま再開すると
        二重発注・想定外建玉を招くため、未解決ゼロを解除の条件とする。
        戻り値: {"ok": bool, "reason": str, "state": <halt状態>}
        """
        from src.core import halt
        unresolved = self._count_unresolved()
        if unresolved:
            return {
                "ok": False,
                "reason": f"未解決の注文が{unresolved}件あります。"
                          "/orders・/positions を確認し解消してから再開してください",
                "state": halt.get_state(),
            }
        state = halt.release()
        return {"ok": True, "reason": "", "state": state}

    def _alert_stuck_approvals(self) -> None:
        """前回終了時にクラッシュ等で APPROVING のまま残った承認があれば critical alert する。

        APPROVING はクラッシュ安全のための一時状態であり、正常終了時は APPROVED/PENDING
        いずれかに必ず遷移している（_approve_order_locked 参照）。これが残っている場合、
        実発注が成立したか不明なまま再承認もブロックされ続けるため、人手で実口座と
        突合する必要がある（レビュー再指摘 High）。
        """
        with get_session() as session:
            stuck = session.scalars(
                select(OrderApproval).where(OrderApproval.status == "APPROVING")
            ).all()
        if not stuck:
            return
        detail = ", ".join(f"id={a.id} {a.side} {a.symbol} {a.quantity}株" for a in stuck)
        logger.critical(
            f"前回終了時に承認処理が中断された注文が{len(stuck)}件残っています: {detail}。"
            "実口座で発注が成立しているか確認してください"
        )
        alert("承認処理が中断されたまま残っています（要確認）",
              f"{len(stuck)}件: {detail}。実口座と突合し、必要なら手動で記録を是正してください")

    def _count_unresolved(self) -> int:
        with get_session() as session:
            return session.scalar(
                select(func.count(Trade.id)).where(
                    Trade.status.in_(tuple(st.UNRESOLVED_STATUSES))
                )
            ) or 0

    def _count_pending_approvals(self) -> int:
        with get_session() as session:
            return session.scalar(
                select(func.count(OrderApproval.id)).where(
                    OrderApproval.status == "PENDING"
                )
            ) or 0

    def status_snapshot(self) -> dict:
        """ダッシュボード・起動ログ用の運用状態スナップショット（Phase 3 / 11.1）。

        モード・発注可否・停止スイッチ・未解決注文・承認待ち件数をまとめて返す。
        発注可否(can_place_order)は kill switch・日次上限・損失上限・未解決ガードを
        すべて織り込んだ実効的な判定。
        """
        from src.core import halt
        can_order, reason = self._risk.can_place_order()
        return {
            "mode": self._mode,
            "can_place_order": can_order,
            "block_reason": reason,
            "halt": halt.get_state(),
            "unresolved_orders": self._count_unresolved(),
            "pending_approvals": self._count_pending_approvals(),
        }

    # ─── dry_run / semi_live ────────────────────────────────────────────

    def _record_dry_run(self, side: str, symbol: str, quantity: int,
                        price: float, sector: Optional[str] = None,
                        rationale: Optional[str] = None, *,
                        source: str = "manual", order_type: str = "LIMIT") -> str:
        """dry_run モードの「発注しようとした」記録を残す（実発注はしない）。

        DRY_RUN ステータスは OPEN/UNRESOLVED いずれにも属さないため、余力引当・建玉・
        未解決ガードに影響しない。日次注文数だけはセッション内のゲート挙動を
        ライブと揃えるため計上する。Fillは生成しない（実約定が無いため）。
        """
        self._risk.increment_order_count()
        order_id = f"DRYRUN-{side}-{symbol}-{uuid.uuid4().hex[:8]}"
        logger.warning(
            f"[DRY-RUN] {side} {symbol} {quantity}株 @{price:.0f}円 — 実発注はスキップしました"
        )
        self._record_trade(order_id, symbol, side, quantity, price,
                           status=st.DRY_RUN, filled_at=clock.now(), sector=sector,
                           rationale=rationale, source=source, order_type=order_type)
        return order_id

    def _enqueue_approval(self, side: str, order_type: str, symbol: str,
                          price: float, quantity: int,
                          sector: Optional[str] = None,
                          rationale: Optional[str] = None, *,
                          source: str = "manual") -> Optional[str]:
        """semi_live モードの計画注文を承認キューに積む（実発注はまだしない）。

        この時点でOrderIntent（status=PENDING）を作り、OrderApprovalへ紐付ける（4.2）。
        承認時（approve_order）に同じ意図へBrokerOrderを紐付けるため、
        「承認待ち→承認→実発注」の全行程が1つの意図で追跡できる。
        """
        with get_session() as session:
            intent = OrderIntent(
                symbol=symbol, side=side, target_quantity=quantity, order_type=order_type,
                limit_price=price if order_type == "LIMIT" else None, sector=sector,
                rationale=rationale, source=source, mode=self._mode, status="PENDING",
            )
            session.add(intent)
            session.flush()
            ap = OrderApproval(
                symbol=symbol, side=side, order_type=order_type,
                price=price, quantity=quantity, sector=sector, status="PENDING",
                rationale=rationale, intent_id=intent.id,
            )
            session.add(ap)
            session.commit()
            ap_id = ap.id
        logger.warning(
            f"[semi-live] 承認待ちに登録: {side} {symbol} {quantity}株 "
            f"({order_type}, approval_id={ap_id})"
        )
        alert("発注承認待ち",
              f"{side} {symbol} {quantity}株 をダッシュボードで承認してください（id={ap_id}）")
        return None  # 承認されるまで実発注しない

    def list_pending_approvals(self) -> list[dict]:
        """承認待ち（PENDING）の計画注文を新しい順に返す。"""
        with get_session() as session:
            rows = session.scalars(
                select(OrderApproval)
                .where(OrderApproval.status == "PENDING")
                .order_by(OrderApproval.id.desc())
            ).all()
            return [
                {
                    "id": r.id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "symbol": r.symbol,
                    "side": r.side,
                    "order_type": r.order_type,
                    "price": r.price,
                    "quantity": r.quantity,
                    "sector": r.sector,
                    "rationale": r.rationale,
                }
                for r in rows
            ]

    def _get_approval_lock(self, approval_id: int) -> threading.Lock:
        with self._approval_locks_guard:
            lock = self._approval_locks.get(approval_id)
            if lock is None:
                lock = threading.Lock()
                self._approval_locks[approval_id] = lock
            return lock

    def approve_order(self, approval_id: int) -> dict:
        """承認待ちの計画注文を承認し、実APIへ発注する（semi_live）。

        同じ承認IDへの同時承認リクエスト（ダッシュボードの二重クリック等）が
        いずれも status=="PENDING" の確認を通過して二重発注するのを防ぐため、
        承認IDごとのロックでチェック→発注→ステータス更新を直列化する。

        戻り値: {"ok": bool, "order_id": str|None, "reason": str}
        """
        with self._get_approval_lock(approval_id):
            return self._approve_order_locked(approval_id)

    def _approve_order_locked(self, approval_id: int) -> dict:
        # ── 1. 「承認中(APPROVING)」へ原子的にクレームする ─────────────────
        # 実発注より「前」に状態を PENDING→APPROVING で確定コミットしておく。これにより
        # 「実発注は成功したが APPROVED への更新前にプロセスが落ちた」場合でも、再起動後に
        # PENDING のまま残って再承認＝二重発注、という事故を防ぐ（クラッシュ安全。レビュー P1-3）。
        # APPROVING が残っていたら自動再試行はせず、人手での実口座照合に委ねる。
        with get_session() as session:
            ap = session.scalar(select(OrderApproval).where(OrderApproval.id == approval_id))
            if ap is None:
                return {"ok": False, "order_id": None, "reason": "承認対象が見つかりません"}
            if ap.status != "PENDING":
                return {"ok": False, "order_id": None,
                        "reason": f"既に処理済みです（{ap.status}）"}
            symbol, side = ap.symbol, ap.side
            order_type, price = ap.order_type, ap.price or 0.0
            quantity, sector = ap.quantity, ap.sector
            rationale = ap.rationale
            intent_id = ap.intent_id
            ap.status = "APPROVING"
            session.commit()

        # ── 2. 実APIへ発注 ───────────────────────────────────────────────
        if side == "BUY":
            order_id = self._live_buy(symbol, price, quantity, sector, rationale,
                                      intent_id=intent_id, source="approval")
        elif order_type == "MARKET":
            order_id = self._live_sell_market(symbol, quantity, rationale,
                                              intent_id=intent_id, source="approval")
        else:
            order_id = self._live_sell(symbol, price, quantity, rationale,
                                       intent_id=intent_id, source="approval")

        # ── 3a. 発注失敗（拒否=Result!=0 など実発注に至らなかったケース）─────
        # _live_* は拒否時に REJECTED の Trade を記録して None を返す。発注は成立して
        # いないため、承認を PENDING へ戻して再試行可能にする（従来挙動を維持）。
        if not order_id:
            with get_session() as session:
                ap = session.scalar(select(OrderApproval).where(OrderApproval.id == approval_id))
                if ap is not None and ap.status == "APPROVING":
                    ap.status = "PENDING"
                    session.commit()
            return {"ok": False, "order_id": None,
                    "reason": "発注に失敗しました（拒否/接続エラー等）。承認は保留のままです"}

        # ── 3b. 発注成功 → APPROVED 確定 ─────────────────────────────────
        try:
            with get_session() as session:
                ap = session.scalar(select(OrderApproval).where(OrderApproval.id == approval_id))
                ap.status = "APPROVED"
                ap.decided_at = clock.now()
                ap.resulting_order_id = order_id
                session.commit()
        except Exception as e:
            # 発注は成立済みだが承認レコードの確定に失敗。APPROVING のまま残るため
            # 再承認はされない（PENDINGに戻らない）。人手で order_id を照合する。
            logger.critical(
                f"[semi-live] 発注成功(order_id={order_id})後の承認確定に失敗: id={approval_id} {e}。"
                "承認は APPROVING のまま残ります（再発注はされません）。実口座を確認してください"
            )
            alert("承認確定エラー（要確認）",
                  f"id={approval_id} の発注({order_id})は成立しましたが記録更新に失敗しました。"
                  "実口座と突合してください")
            return {"ok": True, "order_id": order_id,
                    "reason": "発注成立。ただし承認記録の確定に失敗（要確認）"}
        logger.warning(f"[semi-live] 承認・発注: id={approval_id} {side} {symbol} → {order_id}")
        return {"ok": True, "order_id": order_id, "reason": ""}

    def reject_order(self, approval_id: int) -> dict:
        """承認待ちの計画注文を却下する（発注しない）。

        approve_order と同じ承認IDロックを使う。これが無いと「承認」と「却下」が
        同時に来た場合、両方が status=="PENDING" の確認を通過してしまい、
        却下したはずの注文が実発注される競合が起きうる。
        """
        with self._get_approval_lock(approval_id):
            return self._reject_order_locked(approval_id)

    def _reject_order_locked(self, approval_id: int) -> dict:
        with get_session() as session:
            ap = session.scalar(select(OrderApproval).where(OrderApproval.id == approval_id))
            if ap is None:
                return {"ok": False, "reason": "承認対象が見つかりません"}
            if ap.status != "PENDING":
                return {"ok": False, "reason": f"既に処理済みです（{ap.status}）"}
            ap.status = "REJECTED"
            ap.decided_at = clock.now()
            session.commit()
        logger.info(f"[semi-live] 却下: id={approval_id}")
        return {"ok": True, "reason": ""}

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
            result = self._broker.cancel(order_id)
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

    def _cancel_open_orders_for_symbol(self, symbol: str) -> bool:
        """同銘柄の未約定注文をすべてキャンセルする（退出系発注前の競合解消用）。

        戻り値: 全件キャンセル確定で True。1件でも失敗（CANCEL_FAILED）があれば False
        （呼び出し元はこの場合、成行売りの送信を中断する。再レビュー P0-4）。
        """
        with get_session() as session:
            open_ids = [
                t.order_id for t in session.scalars(
                    select(Trade).where(
                        Trade.symbol == symbol,
                        Trade.status.in_(tuple(st.OPEN_STATUSES)),
                    )
                ).all()
            ]
        all_ok = True
        for oid in open_ids:
            if not self._cancel_order_now(oid):
                all_ok = False
        return all_ok

    def _record_trade(self, order_id: str, symbol: str, side: str,
                      quantity: int, price: float,
                      status: str = st.PENDING,
                      filled_at: Optional[datetime] = None,
                      sector: Optional[str] = None,
                      rationale: Optional[str] = None, *,
                      intent_id: Optional[int] = None,
                      source: str = "manual",
                      order_type: str = "LIMIT") -> int:
        """BrokerOrder（Trade）を記録する唯一のchoke point（Phase 5 / 4.2）。

        `intent_id` を渡さなければ、この発注のための OrderIntent を新規作成して紐付ける
        （semi_live承認やREJECTED再記録のように既存の意図を継続する場合のみ明示的に渡す）。
        戻り値: 作成したTrade.id。
        """
        with get_session() as session:
            if intent_id is None:
                intent = OrderIntent(
                    symbol=symbol, side=side, target_quantity=quantity,
                    order_type=order_type,
                    limit_price=price if order_type == "LIMIT" else None,
                    sector=sector, rationale=rationale, source=source, mode=self._mode,
                    status=self._initial_intent_status(status),
                )
                session.add(intent)
                session.flush()
                intent_id = intent.id
            trade = Trade(
                order_id=order_id,
                intent_id=intent_id,
                symbol=symbol,
                side=side,
                sector=sector,
                quantity=quantity,
                price=price,
                status=status,
                filled_at=filled_at,
                rationale=rationale,
            )
            session.add(trade)
            session.commit()
            trade_id = trade.id
        # 注文ライフサイクルの相関ログ（P2-5）。order_id / intent_id を相関IDとして
        # 全モード共通のこの1点に必ず出すことで、発注→同期→約定→取消の追跡を
        # `order_id=...` のgrepで辿れるようにする。
        logger.info(
            f"注文記録: {side} {symbol} {quantity}株 @{price:.0f} status={status} "
            f"mode={self._mode} source={source} order_id={order_id} intent_id={intent_id}"
        )
        return trade_id

    @staticmethod
    def _initial_intent_status(trade_status: str) -> str:
        """Trade作成時のステータスから、紐づくOrderIntentの初期状態を決める。"""
        if trade_status == st.REJECTED:
            return "REJECTED"
        if trade_status in (st.FILLED, st.DRY_RUN):
            return "COMPLETED"
        return "SUBMITTED"

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

    def _apply_fill(self, order_id: str, symbol: str, side: str, fill_qty: int,
                    fill_price: float, filled_at: datetime, source: str = "paper",
                    sector: Optional[str] = None) -> None:
        """1回の約定をFIFOロット台帳（Fill）へ記録し、Position・Trade派生列・
        RiskManagerの損益へロールアップする（Phase 5 / 4.2。`_update_position()` の後継）。

        BUYはFIFOロットとして`Fill`に積む。SELLは`src/execution/lots.py`が古いロットから
        順に消費して確定損益を返す（平均単価会計からFIFOロット会計へ移行）。Positionは
        常に「残っているロットの集計」として再構成するため、二重計上や平均単価のズレが
        起きない。Trade.filled_quantity/filled_price/pnl は、この注文に紐づく全Fillから
        導出する派生列として更新する。
        """
        with get_session() as session:
            trade = session.scalar(select(Trade).where(Trade.order_id == order_id))
            broker_order_id = trade.id if trade else None
            if side == "BUY":
                lots.record_buy_fill(session, broker_order_id, symbol, fill_qty, fill_price,
                                     filled_at, source)
            else:
                _, realized, consumed = lots.consume_fifo(
                    session, broker_order_id, symbol, fill_qty, fill_price, filled_at, source,
                )
                if consumed < fill_qty:
                    logger.warning(
                        f"SELL約定だが保有ロット不足: {symbol} 要求{fill_qty}株 "
                        f"消費{consumed}株（不足分の損益は未計上・リスク管理に反映されません）"
                    )
                if trade:
                    # 部分約定で複数回呼ばれても合算されるよう加算する
                    trade.pnl = (trade.pnl or 0) + realized
                # 損失を RiskManager に記録（当日損失上限チェック用）
                self._risk.record_loss(realized)
            lots.rebuild_position(session, symbol, sector=sector)
            if trade:
                self._rollup_trade_fills(session, trade)
            session.commit()

    @staticmethod
    def _rollup_trade_fills(session, trade: Trade) -> None:
        """この注文(broker_order_id=trade.id)に紐づく全Fillから filled_quantity/filled_price
        （出来高加重平均=VWAP）を再計算し、Tradeの派生列へ反映する。"""
        fills = session.scalars(select(Fill).where(Fill.broker_order_id == trade.id)).all()
        if not fills:
            return
        total_qty = sum(f.fill_qty for f in fills)
        vwap = (sum(f.fill_qty * f.fill_price for f in fills) / total_qty
                if total_qty else None)
        trade.filled_quantity = total_qty
        trade.filled_price = vwap
