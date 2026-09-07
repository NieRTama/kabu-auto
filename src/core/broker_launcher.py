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
import time
from datetime import date
from typing import Optional

from loguru import logger

DEFAULT_EXE_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", ""), "kabuStation", "KabuS.exe"
)
DEFAULT_MAX_ATTEMPTS_PER_DAY = 3

# 起動直後に再度起動しないための待ち時間。
# Popen してから KabuS.exe が tasklist に現れるまで数秒かかるため、その間は
# is_running() が False を返し続ける。近接した2回目の呼び出し（定期チェックと
# Discord の launch コマンド等）がそこを踏むと二重起動になる。
RELAUNCH_COOLDOWN_SECONDS = 60

_lock = threading.Lock()
_attempts: int = 0
_attempts_date: Optional[date] = None
_last_launch_at: float = 0.0


def _today() -> date:
    from src.core import clock
    return clock.today()


def _now() -> float:
    """クールダウン判定用の時刻（単調増加。テストで差し替える）。"""
    return time.monotonic()


def probe_running(process_name: str = "KabuS") -> Optional[bool]:
    """プロセスの有無を調べる。**判定できなければ None**。

    psutil を持たないため tasklist で確認する（Windows前提のアプリなので可）。

    利用者への案内文にはこちらを使う。「判定できなかった」を「起動中」と
    混同すると、本当に落ちているときに「新しく起動せず窓でログインして」と
    誤誘導してしまう。
    """
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {process_name}.exe", "/NH"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        logger.warning(f"プロセス確認に失敗しました: {e}")
        return None
    # 該当なしのときも tasklist は 0 で終了し、日本語の案内文を stdout に出す。
    # 非ゼロ終了（RPC unavailable 等）は stdout が空になるため、これを「未起動」と
    # 読むと誤った通知が飛び、自動起動が有効なら2つ目を起動してしまう。
    if result.returncode != 0:
        logger.warning(
            f"プロセス確認コマンドが異常終了しました（判定不能扱い）: "
            f"rc={result.returncode} {result.stderr.strip()[:120]}"
        )
        return None
    return f"{process_name}.exe" in result.stdout


def is_running(process_name: str = "KabuS") -> bool:
    """起動してよいかの判断用。

    判定に失敗した場合は「起動している」とみなす（多重起動を避ける安全側）。
    """
    return probe_running(process_name) is not False


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
    """テスト用に試行回数とクールダウンを初期化する。"""
    global _attempts, _attempts_date, _last_launch_at
    with _lock:
        _attempts = 0
        _attempts_date = None
        _last_launch_at = 0.0


def launch(exe_path: str = "", *,
           max_attempts_per_day: int = DEFAULT_MAX_ATTEMPTS_PER_DAY,
           manual: bool = False) -> tuple[bool, str]:
    """kabuステーションを起動する。(起動したか, 説明) を返す。

    既に起動している場合は起動せず (False, 理由) を返す（多重起動防止）。

    manual=True は**人が明示的に頼んだ起動**（Discord の launch コマンド）。
    自動起動と違い、日次上限で止めず・その残枠も消費せず・起動の有無を
    確認できなかった場合も通す。自動起動を無効にした構成では、これが唯一の
    起動経路になるため、機械的な安全弁で人の操作まで塞がないようにする。
    """
    path = exe_path or DEFAULT_EXE_PATH

    global _attempts, _last_launch_at
    # 二重起動を実際に止めているのは下の**クールダウン**（起動直後は tasklist に
    # 現れず probe_running() が False を返し続けるため、これが無いと近接した2回目が
    # 必ず通る）。生存確認をロック内に入れているのは多重防御で、
    # 「確認してから起動するまで」の隙（tasklist の実行に数百ミリ秒かかる）を詰める。
    with _lock:
        # 生存確認を実行ファイルの有無より**先**に行う。逆にすると、既定以外の
        # 場所にインストールされた環境で、正常稼働中でも「実行ファイルが
        # 見つかりません」と返り、利用者を手動起動＝二重起動へ誘導してしまう。
        alive = probe_running()
        if alive is True:
            return False, "kabuステーションは既に起動しています"
        if alive is None and not manual:
            # 自動起動は安全側（多重起動を避ける）。人が頼んだときは通す——
            # 自動起動を無効にした構成では手動が唯一の経路であり、tasklist が
            # 失敗し続ける間ずっと起動できないと終日発注できなくなる。
            # 人は画面を見て起動済みかどうか判断できる。
            return False, "kabuステーションが起動しているか確認できませんでした"
        if not os.path.isfile(path):
            return False, f"実行ファイルが見つかりません: {path}"

        elapsed = _now() - _last_launch_at
        if _last_launch_at and elapsed < RELAUNCH_COOLDOWN_SECONDS:
            return False, (
                f"起動直後です（{int(elapsed)}秒前に起動）。プロセス一覧に現れるまで"
                f"{RELAUNCH_COOLDOWN_SECONDS}秒は再起動しません"
            )

        # 日次上限は「自動起動の暴走」を止めるためのもの。人が明示的に頼んだ
        # 起動は上限で止めず、自動起動の残枠も消費しない（消費すると、自動起動が
        # 一度も走っていないのに手動3回でその日の自動起動が止まる）。
        _roll_over_if_new_day()
        if manual:
            attempt_no = None
        else:
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
        _last_launch_at = _now()

    how = "手動" if attempt_no is None else f"本日{attempt_no}回目"
    logger.warning(f"kabuステーションを起動しました（{how}）: {path}")
    return True, f"kabuステーションを起動しました（{how}）"
