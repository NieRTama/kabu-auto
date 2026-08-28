"""直近のエラー発生数を数え、異常の「予兆」ではなく「実害」を検知する。

背景: 既存の health.py は未解決注文・損失上限といった「取引の結果」を見ているが、
「システムが正常に機能しているか」は見ていなかった。そのため 2026-08 に起きた
2件の障害をどちらも検知できなかった:

  - ログイン認証切れ（8/26-27）: `損切りチェックエラー: 9432 401` が178回出続けたが
    2営業日気づけず、保有建玉の監視が止まっていた
  - トレーリングストップの KeyError: 利確条件を満たすたびに例外で決済が中断していた

個々のバグは予測できないが、「壊れたらエラーが増える」のは普遍的に成り立つ。
そこでログのERROR以上を時間窓で数え、閾値を超えたら通知する（未知の障害に対する防御）。

loguru のシンクとして登録し、記録と同時にカウントする。判定は純粋関数に分けてあり、
時刻を注入できるためテストで実時間に依存しない。
"""
import threading
from collections import deque
from typing import Callable, Deque, Optional, Tuple

_lock = threading.Lock()
_events: Deque[Tuple[float, str]] = deque()

# 窓の長さと、この件数以上で異常とみなす閾値（既定値。config で上書きできる）
DEFAULT_WINDOW_SECONDS = 900   # 15分
DEFAULT_THRESHOLD = 10


def record(message: str, *, now: float) -> None:
    """エラー1件を記録する（シンクから呼ばれる）。"""
    with _lock:
        _events.append((now, message))


def _prune(now: float, window_seconds: float) -> None:
    cutoff = now - window_seconds
    while _events and _events[0][0] < cutoff:
        _events.popleft()


def snapshot(*, now: float, window_seconds: float = DEFAULT_WINDOW_SECONDS) -> dict:
    """時間窓内のエラー件数と代表メッセージを返す（副作用は古い要素の破棄のみ）。"""
    with _lock:
        _prune(now, window_seconds)
        count = len(_events)
        # 代表として直近のメッセージを1つ添える（何が起きているかの手がかり）
        latest = _events[-1][1] if _events else ""
    return {"count": count, "latest": latest, "window_seconds": window_seconds}


def reset() -> None:
    """テスト・再起動時にカウンタを空にする。"""
    with _lock:
        _events.clear()


def make_sink(clock: Optional[Callable[[], float]] = None) -> Callable:
    """loguru へ登録するシンクを作る。ERROR以上のみを数える。

    シンクは loguru のロック内で呼ばれるため、ここでログを出してはいけない
    （再入して停止する）。数えるだけに徹する。
    """
    import time as _time
    now_fn = clock or _time.monotonic

    def sink(message) -> None:
        record(message.record["message"], now=now_fn())

    return sink
