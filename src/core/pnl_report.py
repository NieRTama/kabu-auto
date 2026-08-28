"""
日次・週次・月次・総合の損益サマリ集計（X/Discordの日次レポート向け）。

`/api/report/daily`（ダッシュボード）と同じ Trade.pnl ベースの集計方式を、
複数期間（当日/週次/月次/総合）に対して横断的に算出する。DRY_RUN は実取引でないため
（他の取引活動レポートと同様）対象外とする。

`format_report_text()` はプラットフォーム非依存のテンプレート整形。文字数上限は
投稿先（X=280字・Discord=2000字）ごとに異なるため、切り詰めは呼び出し側
（src/core/x_poster.py・src/core/discord_report.py）の責務とする。
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import select

from src.core import trading_mode as tm
from src.data.database import Trade, get_session


@dataclass
class PeriodPnL:
    label: str          # "当日" / "週次" / "月次" / "総合"
    realized_pnl: float  # 円（正=利益・負=損失）
    pct: Optional[float]  # 基準資金に対する比率（基準資金未設定ならNone）
    win_count: int
    loss_count: int

    @property
    def win_rate(self) -> Optional[float]:
        decided = self.win_count + self.loss_count
        return round(self.win_count / decided, 3) if decided else None


def build_holdings(reference_capital: float = 0.0) -> dict:
    """現在の保有建玉の評価額・簿価・含み損益を返す。

    実現損益（build_report）は「確定した成績」だが、それだけでは決済前の
    含み損益が見えない。日次レポートで現在のポジション状況も併せて把握できるようにする。

    最新終値が取得できない銘柄は評価額・含み損益の計算から除外し、`unpriced` に
    銘柄コードを載せる（avg_costで代用すると含み損益が常に0になり「データが無い」
    ことを隠してしまうため。RiskManager._unrealized_pnl_with_gaps と同じ方針）。

    戻り値:
      count        : 保有銘柄数
      cost         : 取得原価の合計（円）
      market_value : 時価評価額の合計（円。価格不明銘柄は含まない）
      unrealized   : 含み損益（円。正=含み益）
      pct          : 基準資金に対する含み損益の比率（基準資金が0ならNone）
      unpriced     : 価格を取得できず計算から除外した銘柄コード
    """
    from src.data.database import Position
    from src.data.market_data import latest_closes

    with get_session() as session:
        positions = list(session.scalars(
            select(Position).where(Position.quantity > 0)
        ).all())
    closes = latest_closes([p.symbol for p in positions]) if positions else {}

    cost = market_value = 0.0
    unpriced: list[str] = []
    for p in positions:
        close = closes.get(p.symbol)
        if close and p.avg_cost:
            cost += p.avg_cost * p.quantity
            market_value += close * p.quantity
        elif p.avg_cost:
            unpriced.append(p.symbol)

    unrealized = market_value - cost
    pct = (unrealized / reference_capital) if reference_capital else None
    return {
        "count": len(positions),
        "cost": cost,
        "market_value": market_value,
        "unrealized": unrealized,
        "pct": pct,
        "unpriced": unpriced,
    }


def _week_start(d: date) -> date:
    """その週の月曜日を返す（ISO週開始）。"""
    return d - timedelta(days=d.weekday())


def _aggregate(trades: list, start: Optional[date], end: date, label: str,
              reference_capital: float) -> PeriodPnL:
    realized = 0.0
    win = loss = 0
    for t in trades:
        d = t.filled_at.date()
        if start is not None and d < start:
            continue
        if d > end:
            continue
        if t.pnl is None:
            continue
        realized += t.pnl
        if t.pnl > 0:
            win += 1
        elif t.pnl < 0:
            loss += 1
    pct = round(realized / reference_capital, 4) if reference_capital > 0 else None
    return PeriodPnL(label=label, realized_pnl=round(realized, 0), pct=pct,
                     win_count=win, loss_count=loss)


def build_report(reference_capital: float, today: Optional[date] = None) -> dict:
    """当日・週次・月次・総合の損益サマリを返す。

    reference_capital が0（未設定）の場合、各期間の pct は None になる
    （呼び出し側は%を省略して円額のみ表示すること）。
    """
    from src.core import clock
    today = today or clock.today()
    week_start = _week_start(today)
    month_start = today.replace(day=1)

    with get_session() as session:
        trades = session.scalars(
            select(Trade).where(
                Trade.filled_at.isnot(None),
                Trade.status.in_(("FILLED", "PARTIALLY_FILLED", "PARTIALLY_FILLED_DONE")),
            )
        ).all()

    return {
        "daily": _aggregate(trades, today, today, "当日", reference_capital),
        "weekly": _aggregate(trades, week_start, today, "週次", reference_capital),
        "monthly": _aggregate(trades, month_start, today, "月次", reference_capital),
        "overall": _aggregate(trades, None, today, "総合", reference_capital),
    }


def _format_period(p: PeriodPnL) -> str:
    sign = "+" if p.realized_pnl >= 0 else ""
    yen = f"{sign}{p.realized_pnl:,.0f}円"
    if p.pct is not None:
        pct_sign = "+" if p.pct >= 0 else ""
        yen += f" ({pct_sign}{p.pct:.1%})"
    wr = f" 勝率{p.win_rate:.0%}" if p.win_rate is not None else ""
    return f"{p.label}: {yen}{wr}"


def _format_holdings(h: dict) -> list[str]:
    """保有建玉（評価額・含み損益）の表示行を組み立てる。"""
    if not h or not h.get("count"):
        return ["保有: なし"]
    unrealized = h["unrealized"]
    sign = "+" if unrealized >= 0 else ""
    line = f"含み損益: {sign}{unrealized:,.0f}円"
    if h.get("pct") is not None:
        pct_sign = "+" if h["pct"] >= 0 else ""
        line += f" ({pct_sign}{h['pct']:.1%})"
    lines = [
        f"保有: {h['count']}銘柄",
        f"評価額: {h['market_value']:,.0f}円（取得 {h['cost']:,.0f}円）",
        line,
    ]
    if h.get("unpriced"):
        # 価格を取れなかった銘柄は評価額に含まれていない。黙って過小表示しない
        lines.append(f"※価格取得不可のため未算入: {', '.join(h['unpriced'])}")
    return lines


def format_report_text(mode: str, report: dict, holdings: Optional[dict] = None) -> str:
    """日次レポートの投稿文を組み立てる（モード・当日/週次/月次/総合・勝率）。

    holdings を渡すと、確定した実現損益に加えて「現在のポジション状況」
    （保有銘柄数・評価額・含み損益）も併記する。省略時は従来どおり実現損益のみ。

    プラットフォーム非依存（文字数上限の切り詰めは行わない）。X/Discordそれぞれの
    投稿関数が、各プラットフォームの上限に合わせて切り詰めを行う。
    """
    lines = [
        "【kabu-auto 日次レポート】",
        f"モード: {tm.description(mode)}",
        "",
        _format_period(report["daily"]),
        _format_period(report["weekly"]),
        _format_period(report["monthly"]),
        _format_period(report["overall"]),
    ]
    if holdings is not None:
        lines.append("")
        lines.extend(_format_holdings(holdings))
    return "\n".join(lines)
