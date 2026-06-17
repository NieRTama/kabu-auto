"""
リスク管理モジュール
- 1銘柄あたりの最大投資額
- 損切りライン
- 最大保有銘柄数
- セクター集中制限
- ギャップリスク考慮
"""
from datetime import datetime
from typing import Optional

from loguru import logger
from sqlalchemy import func, select

from src.core import config as cfg
from src.data.database import Position, Trade, get_session


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
        today = datetime.now().date()
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
        return True, ""

    def increment_order_count(self) -> None:
        self._daily_order_count += 1

    def calc_position_size(self, symbol: str, price: float,
                           cash_balance: float) -> int:
        """購入株数を計算する（最大投資額を超えない範囲）"""
        max_ratio = self._conf.get("max_position_ratio", 0.20)
        max_amount = cash_balance * max_ratio
        if price <= 0:
            return 0
        lot = 100  # 東証は通常100株単位
        units = int(max_amount / (price * lot))
        quantity = units * lot
        logger.debug(f"{symbol}: 購入可能数={quantity}株 (価格={price:.0f}, 上限={max_amount:.0f}円)")
        return quantity

    def check_max_positions(self) -> tuple[bool, str]:
        """最大保有銘柄数チェック"""
        max_pos = self._conf.get("max_positions", 5)
        with get_session() as session:
            active = session.scalar(
                select(func.count(Position.id)).where(Position.quantity > 0)
            ) or 0
        if active >= max_pos:
            return False, f"最大保有銘柄数({max_pos})に達しています"
        return True, ""

    def check_sector_concentration(self, sector: str) -> tuple[bool, str]:
        """同一セクターの集中投資チェック"""
        max_ratio = self._conf.get("max_sector_ratio", 0.40)
        with get_session() as session:
            all_pos = session.scalars(select(Position).where(Position.quantity > 0)).all()
        if not all_pos:
            return True, ""
        same_sector = [p for p in all_pos if p.sector == sector]
        ratio = len(same_sector) / len(all_pos)
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
        ok, reason = self.check_max_positions()
        if not ok:
            return False, reason
        if sector:
            ok, reason = self.check_sector_concentration(sector)
            if not ok:
                return False, reason
        quantity = self.calc_position_size(symbol, price, cash_balance)
        if quantity <= 0:
            return False, f"余力不足: {symbol}"
        return True, ""
