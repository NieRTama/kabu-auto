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
# (時刻, レベル名, メッセージ)
_events: Deque[Tuple[float, str, str]] = deque()

# 窓の長さと、この件数以上で異常とみなす閾値（既定値。config で上書きできる）
DEFAULT_WINDOW_SECONDS = 900   # 15分
DEFAULT_THRESHOLD = 10

# WARNING の閾値は別に持つ。リトライ前提の失敗（「次回再試行」等）は WARNING で
# 記録されるため件数が桁違いに多く、ERROR と同じ閾値では誤検知になる。
# 2026-09-02 の事故では401が499件出たが、その大半は15秒毎の建玉照合の WARNING で、
# ERROR は28件（15分あたり約3.5件）にとどまり閾値10に届かなかった。
# 「497回失敗しているのにエラーは少ない」と判定されていた。
DEFAULT_WARNING_THRESHOLD = 50  # 15分で50件 = 平均3.3件/分の失敗が続いている状態


def record(message: str, *, now: float, level: str = "ERROR") -> None:
    """ログ1件を記録する（シンクから呼ばれる）。"""
    with _lock:
        _events.append((now, level, message))


def _prune(now: float, window_seconds: float) -> None:
    cutoff = now - window_seconds
    while _events and _events[0][0] < cutoff:
        _events.popleft()


def snapshot(*, now: float, window_seconds: float = DEFAULT_WINDOW_SECONDS) -> dict:
    """時間窓内の件数と代表メッセージを返す（副作用は古い要素の破棄のみ）。

    `count` は ERROR 以上、`warning_count` は WARNING のみ。閾値が別なので分けて数える。
    """
    with _lock:
        _prune(now, window_seconds)
        errors = [e for e in _events if e[1] != "WARNING"]
        warnings = [e for e in _events if e[1] == "WARNING"]
        # 代表として直近のメッセージを1つ添える（何が起きているかの手がかり）
        latest = errors[-1][2] if errors else ""
        latest_warning = warnings[-1][2] if warnings else ""
    return {
        "count": len(errors),
        "latest": latest,
        "warning_count": len(warnings),
        "latest_warning": latest_warning,
        "window_seconds": window_seconds,
    }


def reset() -> None:
    """テスト・再起動時にカウンタを空にする。"""
    with _lock:
        _events.clear()


def make_sink(clock: Optional[Callable[[], float]] = None) -> Callable:
    """loguru へ登録するシンクを作る。WARNING以上をレベル付きで数える。

    シンクは loguru のロック内で呼ばれるため、ここでログを出してはいけない
    （再入して停止する）。数えるだけに徹する。
    """
    import time as _time
    now_fn = clock or _time.monotonic

    def sink(message) -> None:
        rec = message.record
        record(rec["message"], now=now_fn(), level=rec["level"].name)

    return sink
