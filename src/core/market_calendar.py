"""東証の営業日判定（土日・祝日・年末年始）。

## 背景

`is_market_open()` は当初「土日」しか除外しておらず、**祝日を「場中」と誤判定**していた。
祝日には次の問題が起きる:

  - stop_loss_check が5分毎に板取得を試みて全て失敗 → エラー率監視が発報
  - liveness が「場中なのにデータ取得なし」→ サイレント故障として誤検知
  - morning_execution が無駄に発注を試みる

2026-08 に昼休み・夜間で同種の誤検知を2度経験しており、祝日でも同じことが起きる。

## 東証の休場日

  - 土曜・日曜
  - 国民の祝日（振替休日・国民の休日を含む）→ jpholiday が判定する
  - 年末年始: 12/31 〜 1/3（祝日ではないため別途指定が必要）

大発会・大納会の日程は年により変わりうるが、12/31-1/3 は一貫して休場。
1/2・1/3 は祝日ではないので jpholiday では判定できない点に注意。
"""
from datetime import date
from typing import Optional

import jpholiday

# 年末年始の休場（(月, 日) の範囲。祝日ではないため個別に持つ）
_YEAR_END_START = (12, 31)
_NEW_YEAR_END = (1, 3)


def is_holiday(d: date) -> bool:
    """東証の休場日か（土日・祝日・年末年始）。"""
    if d.weekday() >= 5:  # 土日
        return True
    if jpholiday.is_holiday(d):
        return True
    return _is_year_end_or_new_year(d)


def _is_year_end_or_new_year(d: date) -> bool:
    """12/31 〜 1/3 の年末年始休場か。"""
    if (d.month, d.day) >= _YEAR_END_START:
        return True
    return (d.month, d.day) <= _NEW_YEAR_END


def is_business_day(d: date) -> bool:
    """東証の営業日か（is_holiday の逆）。"""
    return not is_holiday(d)


def holiday_name(d: date) -> Optional[str]:
    """休場理由を返す（ログ・通知用）。営業日なら None。"""
    if d.weekday() >= 5:
        return "土日"
    name = jpholiday.is_holiday_name(d)
    if name:
        return name
    if _is_year_end_or_new_year(d):
        return "年末年始休場"
    return None
