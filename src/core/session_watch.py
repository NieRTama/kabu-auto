"""Windowsの画面セッション（コンソール）切断の検知・復帰。

## 背景

kabuステーションへの自動ログインはSendKeys等のGUI自動化に依存しており、
アタッチされたデスクトップセッション（コンソール）が無いとウィンドウを
見つけられず静かに失敗する。リモートデスクトップ(RDP)で接続後に切断すると、
ローカルのコンソールセッションが "Disc"（切断）状態のまま残り、この状態を
引き起こす（2026-09-27 早朝、実際にこれで完全自動ログインが失敗し、
`tscon <セッションID> /dest:console` の手動実行でのみ復旧した）。

本モジュールは `query session` で現在ユーザーのセッション状態を調べ、
切断されていれば `tscon` でコンソールへ再接続する。プロセス
（\\kabu\\kabu-auto タスク）は Principal RunLevel=Highest で既に管理者権限を
持っているため、`tscon` に必要な権限は満たしている。
"""
import getpass
import os
import subprocess
from typing import Optional

from loguru import logger

# tasklist 等と同じ規約（src/core/broker_launcher.py 参照）: PATH問題を避けるため
# System32配下のフルパスを使う。%SystemRoot%はWindowsでは常に定義される。
_QUERY_EXE = os.path.join(
    os.environ.get("SystemRoot", r"C:\Windows"), "System32", "query.exe"
)
_TSCON_EXE = os.path.join(
    os.environ.get("SystemRoot", r"C:\Windows"), "System32", "tscon.exe"
)


def _current_username() -> str:
    try:
        return os.environ["USERNAME"]
    except KeyError:
        return getpass.getuser()


def is_console_disconnected() -> Optional[str]:
    """現在ユーザーのセッションが切断(Disc)状態ならそのセッションIDを返す。

    判定できない場合（非Windows・query.exe不在・タイムアウト等）はNoneを返し、
    スケジューラを止めない（サイレントにno-op、tasklist系と同じ安全側の方針）。
    """
    try:
        result = subprocess.run(
            [_QUERY_EXE, "session"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        logger.warning(f"セッション状態の確認に失敗しました: {e}")
        return None

    if result.returncode != 0:
        logger.warning(
            f"query session が異常終了しました（判定不能扱い）: rc={result.returncode}"
        )
        return None

    username = _current_username()
    for line in result.stdout.splitlines():
        # 例: ">                  garnet                    1  Disc"
        #     " console                                     2  Conn"
        # 先頭の">"はアタッチ中セッションの印。列はスペース区切りで、
        # セッション0はユーザー名列が空になる（services等）。
        tokens = line.replace(">", " ").split()
        if username not in tokens:
            continue
        idx = tokens.index(username)
        # ユーザー名の次がセッションID、その次がSTATE
        if idx + 2 >= len(tokens):
            continue
        session_id, state = tokens[idx + 1], tokens[idx + 2]
        if state.upper().startswith("DISC"):
            return session_id
        return None
    return None


def reconnect_to_console(session_id: str) -> tuple[bool, str]:
    """指定セッションをコンソールへ再接続する。(成功したか, 説明) を返す。"""
    try:
        result = subprocess.run(
            [_TSCON_EXE, session_id, "/dest:console"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception as e:
        return False, f"tscon実行に失敗しました: {e}"

    if result.returncode == 0:
        return True, f"セッション{session_id}をコンソールへ再接続しました"
    detail = (result.stderr or result.stdout).strip()
    return False, f"tsconが失敗しました(rc={result.returncode}): {detail}"
