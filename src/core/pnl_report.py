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


def format_report_text(mode: str, report: dict) -> str:
    """日次レポートの投稿文を組み立てる（モード・当日/週次/月次/総合・勝率）。

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
    return "\n".join(lines)
