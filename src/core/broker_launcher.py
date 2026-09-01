"""kabuステーション（KabuS.exe）の起動制御。

## 背景

kabuステーションが落ちると kabu-auto は何もできなくなるが、復旧には
「アプリを起動する」という物理的な操作が必要だった（2026-08-31 に
アプリがクラッシュし、翌朝までkabu-autoが待機状態のままだった）。

kabu-auto はデスクトップセッション内で動いているため、そこから起動すれば
同じセッションに画面が出る。認証（2段階認証）は**自動化しない**
——専用認証アプリでの承認は人が行う。ここが自動化してよい範囲の線引き。

## 責務

このモジュールは「プロセスを起動する」ことだけを担う。
起動後に接続できたかの確認は broker_wait / auth_recovery が行う（責務分離）。

起動ループを避けるため、1日あたりの試行回数に上限を設ける。
証券会社側の障害時は起動しても認証できないため、繰り返しても無意味なため。
"""
import os
import subprocess
import threading
from datetime import date
from typing import Optional

from loguru import logger

DEFAULT_EXE_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", ""), "kabuStation", "KabuS.exe"
)
DEFAULT_MAX_ATTEMPTS_PER_DAY = 3

_lock = threading.Lock()
_attempts: int = 0
_attempts_date: Optional[date] = None


def _today() -> date:
    from src.core import clock
    return clock.today()


def is_running(process_name: str = "KabuS") -> bool:
    """kabuステーションのプロセスが起動しているか。

    psutil を持たないため tasklist で確認する（Windows前提のアプリなので可）。
    判定に失敗した場合は「起動している」とみなす（多重起動を避ける安全側）。
    """
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {process_name}.exe", "/NH"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception as e:
        logger.warning(f"プロセス確認に失敗しました（起動中とみなします）: {e}")
        return True
    return f"{process_name}.exe" in out


def attempts_today() -> int:
    """本日の起動試行回数（日付が変わればリセットされる）。"""
    with _lock:
        _roll_over_if_new_day()
        return _attempts


def _roll_over_if_new_day() -> None:
    """呼び出し側で _lock を保持している前提。"""
    global _attempts, _attempts_date
    today = _today()
    if _attempts_date != today:
        _attempts_date = today
        _attempts = 0


def reset() -> None:
    """テスト用に試行回数を初期化する。"""
    global _attempts, _attempts_date
    with _lock:
        _attempts = 0
        _attempts_date = None


def launch(exe_path: str = "", *,
           max_attempts_per_day: int = DEFAULT_MAX_ATTEMPTS_PER_DAY) -> tuple[bool, str]:
    """kabuステーションを起動する。(起動したか, 説明) を返す。

    既に起動している場合は起動せず (False, 理由) を返す（多重起動防止）。
    1日の試行上限に達している場合も起動しない（証券会社側の障害時に
    無意味な起動を繰り返さないため）。
    """
    path = exe_path or DEFAULT_EXE_PATH
    if is_running():
        return False, "kabuステーションは既に起動しています"
    if not os.path.isfile(path):
        return False, f"実行ファイルが見つかりません: {path}"

    global _attempts
    with _lock:
        _roll_over_if_new_day()
        if max_attempts_per_day > 0 and _attempts >= max_attempts_per_day:
            return False, (
                f"本日の起動試行が上限({max_attempts_per_day}回)に達しています。"
                "証券会社側の障害の可能性があるため、手動で確認してください"
            )
        _attempts += 1
        attempt_no = _attempts

    try:
        # 同じデスクトップセッションで起動する（GUIを人が操作できるように）。
        # 親プロセス終了に巻き込まれないよう切り離す。
        subprocess.Popen([path], close_fds=True)
    except Exception as e:
        logger.error(f"kabuステーションの起動に失敗しました: {e}")
        return False, f"起動に失敗しました: {e}"

    logger.warning(f"kabuステーションを起動しました（本日{attempt_no}回目）: {path}")
    return True, f"kabuステーションを起動しました（本日{attempt_no}回目）"
