"""
プロセス多重起動防止のファイルロック（再レビュー P1-1）。

同一PCで kabu-auto を誤って2つ起動すると、スケジューラジョブの重複実行・二重発注・
WebSocket接続の競合・DB状態の破壊といった重大な事故につながる。ダッシュボードの
ポート競合チェック（main.py）だけでは検知が間に合わない可能性があるため、PIDを書いた
ロックファイルでプロセスレベルの多重起動を直接検知する。

設計:
- ロックファイル: 既定 data/kabu_auto.lock に自プロセスのPIDを書く。
- 既存ロックファイルがあれば、そのPIDのプロセスが実際に生きているか確認する
  （前回が異常終了（強制終了・電源断等）してファイルだけ残った「stale lock」を
  誤って多重起動と判定しないため）。生きていれば多重起動と判定し、
  死んでいれば古いロックを上書きして起動を許可する。
- release() はプロセス正常終了時にロックファイルを削除する（自分が書いたものの時のみ）。
"""
import os
from pathlib import Path
from typing import Optional

from loguru import logger

_path: Optional[Path] = None


def _is_process_running(pid: int) -> bool:
    """指定PIDのプロセスが現在も実行中かを確認する（Windows/POSIX両対応）。"""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid,
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire(path: str = "data/kabu_auto.lock") -> tuple[bool, str]:
    """多重起動防止ロックを取得する。

    戻り値: (ok, detail)。ok=False の場合、detail に既に稼働中のPID等の理由を入れる。
    """
    global _path
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)
    if _path.exists():
        try:
            existing_pid = int(_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            existing_pid = -1
        if existing_pid > 0 and existing_pid != os.getpid() and _is_process_running(existing_pid):
            return False, f"別プロセス（PID={existing_pid}）が既に稼働中です"
        logger.warning(
            f"古いロックファイルを検知しました（前回プロセス PID={existing_pid} は終了済み）。"
            "上書きして起動を続けます"
        )
    _path.write_text(str(os.getpid()), encoding="utf-8")
    return True, ""


def release() -> None:
    """ロックファイルを削除する（現プロセスが書いたものの場合のみ。安全な多重解放）。"""
    if _path is None:
        return
    try:
        if _path.exists() and _path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            _path.unlink()
    except OSError as e:
        logger.warning(f"ロックファイル削除失敗（無視して終了します）: {e}")
