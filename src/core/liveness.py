"""「動いているべき時に動いた形跡があるか」を記録・判定する（サイレント故障の検知）。

エラー率監視（error_rate）は「壊れたらエラーが増える」故障を捉えるが、
**静かに何もしなくなる**故障は捉えられない。例:

  - スケジューラのジョブが例外で死に、以後実行されなくなる
  - 市場データの取得が例外を出さずに空を返し続ける
  - WebSocketが繋がったまま無反応になる

これらは「エラーが出ない」ため error_rate をすり抜け、
「注文も損益も動かない」ため既存の異常検知にも掛からない。
そこで正常系の動作（板取得の成功）に印をつけ、場中に一定時間途絶えたら異常とみなす。

判定は純粋関数で、時刻を注入できるためテストで実時間に依存しない。
"""
import threading
from typing import Optional

_lock = threading.Lock()
_last_success: Optional[float] = None

# 場中にこの秒数以上、板取得の成功が無ければ異常とみなす（既定15分）。
# stop_loss_check は5分毎に走るため、3回連続で成果が無い状態に相当する。
DEFAULT_SILENCE_SECONDS = 900


def mark_alive(*, now: float) -> None:
    """正常な動作（板取得の成功など）を記録する。"""
    global _last_success
    with _lock:
        _last_success = now


def seconds_since_alive(*, now: float) -> Optional[float]:
    """最後の成功からの経過秒。1度も成功していなければ None。"""
    with _lock:
        if _last_success is None:
            return None
        return now - _last_success


def mark_closed(*, now: float) -> None:
    """市場が閉じている間、基準時刻を進める（休止を「途絶」と数えないため）。

    2026-08-31 に誤検知が発生: 前場最終取得(11:25) → 昼休み(11:30-12:30) →
    後場開始(12:30)の時点で「65分間データ取得なし」と判定された。
    65分 = 昼休み60分 + ジョブ間隔5分 で、実際にはシステムは正常だった。

    is_silent() は「場が閉じていれば False」を返すので検知自体はしないが、
    経過時間は素通しで積み上がるため、再開直後に閾値超過が残ってしまう。
    閉じている間は基準を進めておくことで、昼休み・夜間・休日を跨いでも
    「再開後に実際に途絶えた時間」だけを見られるようにする。
    """
    global _last_success
    with _lock:
        if _last_success is not None:
            _last_success = now


def is_silent(*, now: float, market_open: bool,
              threshold_seconds: float = DEFAULT_SILENCE_SECONDS) -> bool:
    """場中に動作が途絶えているか。

    場が閉じているときは常に False（場外に動きが無いのは正常）。
    1度も成功していない場合も False とする（起動直後の誤検知を避けるため。
    起動時の接続失敗は preflight と broker_auth が別途カバーしている）。
    """
    if not market_open or threshold_seconds <= 0:
        return False
    elapsed = seconds_since_alive(now=now)
    if elapsed is None:
        return False
    return elapsed >= threshold_seconds


def reset() -> None:
    """テスト用に状態を初期化する。"""
    global _last_success
    with _lock:
        _last_success = None
