"""アプリ全体の現在時刻を一元化するモジュール。

本プロジェクトは日時を **JST naive**（タイムゾーン情報を持たない、日本時間としての
datetime）で統一している。DB保存・スケジューラ・シグナルのcutoff比較がすべてこの前提で
組まれており、aware/UTC を混在させると morning_execution の前日シグナル判定が
9時間ずれて取りこぼす（src/data/database.py のコメント参照）。

そのため now() は「タイムゾーンを Asia/Tokyo に明示したうえで naive に落とした」現在時刻を
返す。これによりホスト機がJST以外でも一貫してJSTで動作しつつ、既存のnaive前提を壊さない。
zoneinfo が利用できない環境（tzdata未導入のWindows等）では従来どおりローカル時刻の
naive datetime にフォールバックする。
"""
from datetime import date, datetime

try:
    from zoneinfo import ZoneInfo

    _JST = ZoneInfo("Asia/Tokyo")
except Exception:  # tzdata未導入など
    _JST = None


def now() -> datetime:
    """JSTの現在時刻を naive datetime（tzinfoなし）で返す。"""
    if _JST is not None:
        return datetime.now(_JST).replace(tzinfo=None)
    return datetime.now()


def today() -> date:
    """JSTの本日の日付を返す。"""
    return now().date()
