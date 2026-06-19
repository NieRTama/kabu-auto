"""
リスク管理モジュール
- 1銘柄あたりの最大投資額
- 損切りライン
- 最大保有銘柄数
- セクター集中制限
- ギャップリスク考慮
"""
from typing import Optional

from loguru import logger
from sqlalchemy import func, select

from src.core import clock
from src.core import config as cfg
from src.data.database import Position, Trade, get_session
from src.data.market_data import latest_closes
from src.execution import order_status as st


class RiskManager:
    def __init__(self):
        self._daily_order_count = 0
        self._daily_loss_yen = 0.0
        self._conf = cfg.get_section("trading")

    def reset_daily_counters(self) -> None:
        self._daily_order_count = 0
        self._daily_loss_yen = 0.0

    def restore_daily_state(self) -> None:
        """起動時に当日の注文数・実現損失をDBから復元する。

        日次カウンタはメモリ上のみのため、当日損失上限に達して停止した後に
        プロセスを再起動するとセーフティが消えてしまう。これを防ぐため、
        当日（JST）約定済みTradeから注文数と損失額を再構築する。
        """
        today = clock.today()
        with get_session() as session:
            trades = session.scalars(
                select(Trade).where(Trade.filled_at.isnot(None))
            ).all()
            order_count = 0
            loss_yen = 0.0
            for t in trades:
                if t.filled_at and t.filled_at.date() == today:
                    order_count += 1
                    if t.pnl is not None and t.pnl < 0:
                        loss_yen += abs(t.pnl)
        self._daily_order_count = order_count
        self._daily_loss_yen = loss_yen
        if order_count or loss_yen:
            logger.info(
                f"日次リスク状態を復元: 当日注文数={order_count} 当日実現損失={loss_yen:,.0f}円"
            )

    def record_loss(self, pnl: float) -> None:
        """損益を記録する（損失のみ累積）"""
        if pnl < 0:
            self._daily_loss_yen += abs(pnl)

    def is_daily_loss_limit_reached(self) -> tuple[bool, str]:
        """当日損失上限チェック。(over_limit, reason) を返す"""
        limit = self._conf.get("max_daily_loss", 0)
        if limit <= 0:
            return False, ""
        if self._daily_loss_yen >= limit:
            return True, f"当日損失上限({limit:,.0f}円)に達しました"
        return False, ""

    def can_place_order(self) -> tuple[bool, str]:
        """注文可能かチェック。(ok, reason) を返す"""
        limit = self._conf.get("daily_order_limit", 100)
        if self._daily_order_count >= limit:
            return False, f"1日の注文上限({limit})に達しました"
        over, reason = self.is_daily_loss_limit_reached()
        if over:
            return False, reason
        # 状態不明・キャンセル失敗の注文が残っている間は新規発注を抑止する
        # （実口座と乖離したまま発注すると二重発注・想定外建玉になるため、要人手確認）
        unresolved = self._count_unresolved_orders()
        if unresolved:
            return False, f"未解決の注文が{unresolved}件あります（要確認）"
        return True, ""

    def _count_unresolved_orders(self) -> int:
        """要人手確認の異常注文（UNKNOWN / CANCEL_FAILED）の件数を返す"""
        with get_session() as session:
            return session.scalar(
                select(func.count(Trade.id)).where(
                    Trade.status.in_(tuple(st.UNRESOLVED_STATUSES))
                )
            ) or 0

    def _reserved_buy_by_sector(self) -> tuple[float, dict[str, float]]:
        """未約定BUY注文の引当金額 (総額, セクター別内訳) を返す。

        price × 残数（発注数量 - 約定済数量）で算定する。セクターは発注時に記録した
        `Trade.sector` を優先する（新規銘柄はPositionが未作成のため、Position経由の
        解決では漏れる）。`Trade.sector` が無い古いレコードのみPositionへフォールバック。
        """
        with get_session() as session:
            trades = session.scalars(
                select(Trade).where(
                    Trade.side == "BUY",
                    Trade.status.in_(tuple(st.OPEN_STATUSES)),
                )
            ).all()
            pos_sector = {
                p.symbol: p.sector
                for p in session.scalars(select(Position)).all()
            }
        total = 0.0
        by_sector: dict[str, float] = {}
        for t in trades:
            remaining = (t.quantity or 0) - (t.filled_quantity or 0)
            if remaining <= 0 or not t.price:
                continue
            notional = t.price * remaining
            total += notional
            sec = t.sector or pos_sector.get(t.symbol)
            if sec:
                by_sector[sec] = by_sector.get(sec, 0.0) + notional
        return total, by_sector

    def increment_order_count(self) -> None:
        self._daily_order_count += 1

    def calc_position_size(self, symbol: str, price: float,
                           cash_balance: float) -> int:
        """購入株数を計算する（最大投資額を超えない範囲）"""
        max_ratio = self._conf.get("max_position_ratio", 0.20)
        # 未約定BUYの引当を差し引いた実効余力で上限を計算する
        # （未約定中の多重発注で余力を二重に使う事故を防ぐ。main側の逐次減算と二重で守る）
        reserved, _ = self._reserved_buy_by_sector()
        available = max(0.0, cash_balance - reserved)
        max_amount = available * max_ratio
        if price <= 0:
            return 0
        lot = 100  # 東証は通常100株単位
        units = int(max_amount / (price * lot))
        quantity = units * lot
        logger.debug(f"{symbol}: 購入可能数={quantity}株 (価格={price:.0f}, 上限={max_amount:.0f}円)")
        return quantity

    def check_max_positions(self, candidate_symbol: Optional[str] = None) -> tuple[bool, str]:
        """最大保有銘柄数チェック。

        保有Positionの数だけでなく、未約定BUYの銘柄（まだPosition化されていない
        新規銘柄も含む）も合算した銘柄集合で判定する。保有3/上限4のときに未約定BUYが
        2件あると、どちらも個別には通って合計5銘柄になってしまう抜けを防ぐ
        （再レビュー P1-1）。`candidate_symbol` を渡すと、その銘柄を追加した場合を
        含めて判定する（既に保有/未約定の銘柄であれば集合は増えないため通る）。
        """
        max_pos = self._conf.get("max_positions", 5)
        with get_session() as session:
            active_symbols = {
                p.symbol for p in session.scalars(
                    select(Position).where(Position.quantity > 0)
                ).all()
            }
            reserved_symbols = {
                t.symbol for t in session.scalars(
                    select(Trade).where(
                        Trade.side == "BUY",
                        Trade.status.in_(tuple(st.OPEN_STATUSES)),
                    )
                ).all()
            }
        after_symbols = active_symbols | reserved_symbols
        if candidate_symbol:
            after_symbols = after_symbols | {candidate_symbol}
        if len(after_symbols) > max_pos:
            return False, f"最大保有銘柄数({max_pos})に達しています"
        return True, ""

    def check_sector_concentration(self, sector: str,
                                   candidate_notional: float = 0.0) -> tuple[bool, str]:
        """同一セクターの集中投資チェック（建玉の時価評価額ベース）。

        銘柄数の比率ではなく、quantity × 最新終値 のエクスポージャー金額で判定する
        （小口1銘柄と大型1銘柄が同じ「1銘柄」として扱われる粗さを避けるため）。
        最新終値が取得できない銘柄は avg_cost（取得平均単価）で代用する。
        `candidate_notional` には**これから出す注文自体の金額**を渡す（再レビュー P1-2）。
        既存保有・未約定BUYの引当だけでは「この注文を加えたら超過する」ケースを
        発注前に弾けないため、判定対象の候補金額も合算してから比較する。
        """
        max_ratio = self._conf.get("max_sector_ratio", 0.40)
        with get_session() as session:
            all_pos = session.scalars(select(Position).where(Position.quantity > 0)).all()
        # 未約定BUYの引当も「買った後の集中度」として加味する（発注前に上限超過を防ぐ）
        reserved_total, reserved_by_sector = self._reserved_buy_by_sector()
        if not all_pos and reserved_total <= 0 and candidate_notional <= 0:
            return True, ""
        closes = latest_closes([p.symbol for p in all_pos])
        total_value = reserved_total + candidate_notional
        same_sector_value = reserved_by_sector.get(sector, 0.0) + candidate_notional
        for p in all_pos:
            price = closes.get(p.symbol) or p.avg_cost
            value = p.quantity * price
            total_value += value
            if p.sector == sector:
                same_sector_value += value
        if total_value <= 0:
            return True, ""
        ratio = same_sector_value / total_value
        if ratio >= max_ratio:
            return False, f"セクター集中率が上限({max_ratio:.0%})超: {sector}"
        return True, ""

    def should_stop_loss(self, symbol: str, current_price: float) -> bool:
        """損切りラインを超えているか判定"""
        stop_pct = self._conf.get("stop_loss_pct", -0.05)
        with get_session() as session:
            pos = session.scalar(select(Position).where(Position.symbol == symbol))
        if pos is None or pos.avg_cost <= 0:
            return False
        pnl_pct = (current_price - pos.avg_cost) / pos.avg_cost
        if pnl_pct <= stop_pct:
            logger.warning(f"損切りライン到達: {symbol} 損益率={pnl_pct:.2%}")
            return True
        return False

    def validate_buy(self, symbol: str, price: float,
                     cash_balance: float, sector: Optional[str] = None) -> tuple[bool, str]:
        """買い注文の総合バリデーション"""
        ok, reason = self.can_place_order()
        if not ok:
            return False, reason
        ok, reason = self.check_max_positions(candidate_symbol=symbol)
        if not ok:
            return False, reason
        quantity = self.calc_position_size(symbol, price, cash_balance)
        if quantity <= 0:
            return False, f"余力不足: {symbol}"
        if sector:
            # この注文自体の金額もセクター集中度に加味する（発注前に超過を弾く。再レビュー P1-2）
            ok, reason = self.check_sector_concentration(sector, candidate_notional=price * quantity)
            if not ok:
                return False, reason
        return True, ""
