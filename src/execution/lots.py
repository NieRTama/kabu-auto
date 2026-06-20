"""
FIFOロット台帳（Phase 5 / 4.2）。

BUYの約定(Fill)をロット（先入れ）として記録し、SELLの約定が古いロットから順に消費して
実現損益を確定する。Position（建玉数量・平均取得単価）は常に「残っているロットの集計」
として再構成するため、二重計上や平均単価のズレが生まれない。

このモジュールは `Session` を受け取る純粋な操作関数の集合であり、commit はしない
（呼び出し元のセッション境界に委ねる）。
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from src.core import clock
from src.data.database import Fill, Position


def record_buy_fill(session, broker_order_id: int, symbol: str, qty: int, price: float,
                    filled_at: datetime, source: str = "paper") -> Fill:
    """BUY約定を新しいFIFOロットとして記録する（remaining_qty=qty で開始）。"""
    fill = Fill(
        broker_order_id=broker_order_id, symbol=symbol, side="BUY",
        fill_qty=qty, fill_price=price, filled_at=filled_at,
        source=source, remaining_qty=qty, realized_pnl=None,
    )
    session.add(fill)
    session.flush()
    return fill


def consume_fifo(session, broker_order_id: int, symbol: str, qty: int, sell_price: float,
                 filled_at: datetime, source: str = "paper") -> tuple[Fill, float, int]:
    """SELL約定をFIFOで古いBUYロットから消費し、確定損益を返す。

    保有ロットが不足する場合（在庫無しSELL等）は、ある分だけ消費し残りは無コストとして
    扱わず警告対象とする（呼び出し元が `consumed < qty` を見て警告を出す）。
    戻り値: (Fill[SELL], realized_pnl, consumed_qty)
    """
    remaining_to_consume = qty
    realized_pnl = 0.0
    consumed_qty = 0

    lots = session.scalars(
        select(Fill).where(
            Fill.symbol == symbol, Fill.side == "BUY", Fill.remaining_qty > 0,
        ).order_by(Fill.filled_at.asc(), Fill.id.asc())
    ).all()

    for lot in lots:
        if remaining_to_consume <= 0:
            break
        take = min(lot.remaining_qty, remaining_to_consume)
        lot.remaining_qty -= take
        realized_pnl += (sell_price - lot.fill_price) * take
        remaining_to_consume -= take
        consumed_qty += take

    sell_fill = Fill(
        broker_order_id=broker_order_id, symbol=symbol, side="SELL",
        fill_qty=qty, fill_price=sell_price, filled_at=filled_at,
        source=source, remaining_qty=None, realized_pnl=round(realized_pnl, 2),
    )
    session.add(sell_fill)
    session.flush()
    return sell_fill, round(realized_pnl, 2), consumed_qty


def rebuild_position(session, symbol: str, sector: Optional[str] = None) -> Optional[Position]:
    """残存ロット（remaining_qty>0のBUY Fill）からPositionを再構成する。

    残ロットが無ければ Position は quantity=0 に更新する（行自体は消さない。
    既存コードが `Position` の存在をsymbol単位で前提にしている箇所があるため）。
    `sector` は新規作成時、または既存行にセクターが未設定の場合のみ補完する。
    """
    lots = session.scalars(
        select(Fill).where(
            Fill.symbol == symbol, Fill.side == "BUY", Fill.remaining_qty > 0,
        )
    ).all()
    total_qty = sum(lot.remaining_qty for lot in lots)
    total_cost = sum(lot.remaining_qty * lot.fill_price for lot in lots)
    avg_cost = (total_cost / total_qty) if total_qty > 0 else 0.0

    pos = session.scalar(select(Position).where(Position.symbol == symbol))
    if pos is None:
        if total_qty <= 0:
            return None
        pos = Position(symbol=symbol, quantity=total_qty, avg_cost=avg_cost, sector=sector)
        session.add(pos)
    else:
        pos.quantity = total_qty
        pos.avg_cost = avg_cost
        pos.updated_at = clock.now()
        if sector and not pos.sector:
            pos.sector = sector
    return pos
