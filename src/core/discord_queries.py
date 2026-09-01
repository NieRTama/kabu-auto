"""Discordリモコンの照会コマンド用の整形処理。

「今どうなっているか」を外出先から確認するための読み取り専用の問い合わせ。
集計そのものは pnl_report / DB の既存関数を使い、ここは**表示の整形だけ**を担う
（同じ数字が日次レポートとDiscordで食い違わないようにするため）。

main.py に直接書くと composition root が肥大化するので分離した。
発注や状態変更は一切行わない（照会のみ）。
"""
from typing import Optional

from sqlalchemy import select

from src.data.database import Position, Trade, get_session
from src.execution import order_status as st


def format_positions(reference_capital: float = 0.0) -> str:
    """保有建玉を現在値・含み損益つきで整形する。

    取得単価だけでは「今どうなっているか」が分からないため、
    日次レポートと同じ build_holdings() を使って評価額・含み損益も出す。
    """
    from src.core.pnl_report import build_holdings
    from src.data.market_data import latest_closes

    with get_session() as session:
        positions = list(session.scalars(
            select(Position).where(Position.quantity > 0)
        ).all())
    if not positions:
        return "保有建玉: なし"

    closes = latest_closes([p.symbol for p in positions])
    lines = ["保有建玉:"]
    for p in positions:
        close = closes.get(p.symbol)
        if close and p.avg_cost:
            pnl = (close - p.avg_cost) * p.quantity
            pct = (close - p.avg_cost) / p.avg_cost
            sign = "+" if pnl >= 0 else ""
            lines.append(
                f"  {p.symbol} {p.quantity}株 取得{p.avg_cost:,.0f} → 現在{close:,.0f}"
                f"  {sign}{pnl:,.0f}円 ({sign}{pct:.1%})"
            )
        else:
            # 価格が取れない銘柄を「損益0」に見せない（実態を隠さない）
            lines.append(f"  {p.symbol} {p.quantity}株 取得{p.avg_cost:,.0f} → 現在値取得不可")

    h = build_holdings(reference_capital)
    sign = "+" if h["unrealized"] >= 0 else ""
    total = f"合計: 評価額 {h['market_value']:,.0f}円 / 含み損益 {sign}{h['unrealized']:,.0f}円"
    if h.get("pct") is not None:
        total += f" ({sign}{h['pct']:.1%})"
    lines.append(total)
    return "\n".join(lines)


def format_today_trades(today: Optional[str] = None) -> str:
    """本日の約定履歴を整形する（何をいくらで売買したか）。"""
    from src.core import clock
    day = today or clock.today().isoformat()

    with get_session() as session:
        trades = list(session.scalars(
            select(Trade)
            .where(Trade.filled_at.isnot(None))
            .order_by(Trade.filled_at)
        ).all())
    rows = [t for t in trades if t.filled_at and t.filled_at.date().isoformat() == day]
    if not rows:
        return f"本日({day})の約定: なし"

    lines = [f"本日({day})の約定: {len(rows)}件"]
    realized = 0.0
    for t in rows:
        side = "買" if t.side == "BUY" else "売"
        price = t.filled_price or 0
        qty = t.filled_quantity or t.quantity
        line = f"  {t.filled_at.strftime('%H:%M')} {side} {t.symbol} {qty}株 @{price:,.0f}"
        if t.pnl is not None:
            sign = "+" if t.pnl >= 0 else ""
            line += f"  損益{sign}{t.pnl:,.0f}円"
            realized += t.pnl
        lines.append(line)
    if realized:
        sign = "+" if realized >= 0 else ""
        lines.append(f"本日の確定損益: {sign}{realized:,.0f}円")
    return "\n".join(lines)


def format_pnl(reference_capital: float = 0.0) -> str:
    """当日/週次/月次/総合の損益サマリを整形する（日次レポートと同じ集計）。"""
    from src.core.pnl_report import _format_period, build_report
    report = build_report(reference_capital)
    lines = ["損益サマリ:"]
    for key in ("daily", "weekly", "monthly", "overall"):
        lines.append(f"  {_format_period(report[key])}")
    return "\n".join(lines)


def format_open_orders() -> str:
    """未約定注文の一覧を整形する（今どんな注文が出ているか）。"""
    with get_session() as session:
        orders = list(session.scalars(
            select(Trade)
            .where(Trade.status.in_(tuple(st.OPEN_STATUSES)))
            .order_by(Trade.id)
        ).all())
    if not orders:
        return "未約定注文: なし"

    lines = [f"未約定注文: {len(orders)}件"]
    for t in orders:
        side = "買" if t.side == "BUY" else "売"
        price = f"@{t.price:,.0f}" if t.price else "成行"
        filled = f" (約定{t.filled_quantity}株)" if t.filled_quantity else ""
        lines.append(f"  {side} {t.symbol} {t.quantity}株 {price} [{t.status}]{filled}")
    return "\n".join(lines)
