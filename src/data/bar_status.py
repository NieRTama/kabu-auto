"""日足が確定しているかの判定。

`market_calendar.is_business_day()` は営業日か否かを返すだけで、その足が
確定済みかは分からない。場中の未確定足・引け後の配信待ち・休場を区別しないと、
「日付は今日だがまだ確定していない足」を新規候補の根拠に使ってしまう。

本モジュールは純粋関数だけで構成し、DB・ネットワークに触らない。
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from src.core import market_calendar

# 東証の大引け（後場終了）。この時刻＋猶予を過ぎたらその日の足を確定とみなす。
_CLOSE_HOUR = 15
_CLOSE_MINUTE = 0


@dataclass(frozen=True)
class BarStatus:
    """ある銘柄の最終足が、基準セッションに対してどういう状態かを表す。"""
    symbol: str
    last_bar_session: date | None
    observed_at: datetime
    is_final: bool
    state: str  # "fresh" | "stale" | "missing" | "provisional"


def _previous_business_day(d: date) -> date:
    """d より前の直近営業日を返す。"""
    cur = d - timedelta(days=1)
    while not market_calendar.is_business_day(cur):
        cur -= timedelta(days=1)
    return cur


def as_of_session(now: datetime, close_grace_minutes: int = 20) -> date:
    """now の時点で確定しているべき直近セッションを返す。

    当日が営業日で、かつ大引け＋猶予を過ぎていれば当日。
    それ以外（場中・引け直後の配信待ち・休場）は直近の過去営業日。
    """
    today = now.date()
    if market_calendar.is_business_day(today):
        cutoff = now.replace(
            hour=_CLOSE_HOUR, minute=_CLOSE_MINUTE, second=0, microsecond=0
        ) + timedelta(minutes=close_grace_minutes)
        if now >= cutoff:
            return today
    return _previous_business_day(today)


def classify(symbol: str, last_bar_session: date | None, now: datetime,
             close_grace_minutes: int = 20) -> BarStatus:
    """最終足の営業日を基準セッションと突き合わせて鮮度を分類する。"""
    expected = as_of_session(now, close_grace_minutes)
    if last_bar_session is None:
        state, is_final = "missing", False
    elif last_bar_session == expected:
        state, is_final = "fresh", True
    elif last_bar_session < expected:
        state, is_final = "stale", True
    else:
        # 基準より新しい＝まだ確定していない場中の足
        state, is_final = "provisional", False
    return BarStatus(
        symbol=symbol,
        last_bar_session=last_bar_session,
        observed_at=now,
        is_final=is_final,
        state=state,
    )
