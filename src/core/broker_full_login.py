"""kabuステーション（KabuS.exe）の完全自動ログイン制御。

## 背景

kabu-auto は従来、Kabuステーションの起動のみを自動化し、2段階認証
（ワンタイムパスワード入力）は意図的に自動化していなかった
（`broker_launcher.py` 参照。「認証は自動化しない、専用認証アプリでの承認は
人が行う」という方針）。

2026-09-11、Gmail API経由でワンタイムパスワードを自動取得・自動入力する
外部リポジトリ（kabusapi-auto-login-template、WSL2/Docker側に導入済み）を
統合し、起動〜ログイン〜2段階認証入力までの完全自動化に方針転換した。

## 責務

このモジュールは、WSL側の完全自動ログインスクリプト
（kabusapi-auto-login-template/scripts/kabustation/run_login_only.sh）を
呼び出すことだけを担う。ログイン自体のロジック（パスワード入力、Gmail API
連携、SendKeys等）はすべてWSL側スクリプトに任せる（本モジュールは
パスワード等の秘密情報を一切扱わない）。

起動後にAPI接続できたかの確認は、既存の broker_wait / auth_recovery が
行う（責務分離。broker_launcher.py と同じ設計思想）。

同時実行・連続実行を避けるため、broker_launcher.py と同様のクールダウン・
日次試行回数上限を持つ。
"""
import subprocess
import time
from datetime import date
from typing import Optional

from loguru import logger

from src.core import broker_process_lock

DEFAULT_WSL_DISTRO = "Ubuntu"
DEFAULT_PROJECT_DIR = "~/projects/kabusapi-auto-login-template"
DEFAULT_SCRIPT_PATH = "scripts/kabustation/run_login_only.sh"
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_MAX_ATTEMPTS_PER_DAY = 3

# 直前の実行からこの秒数以内の再実行を抑止する（broker_launcher.py と同じ考え方。
# WSL側スクリプトはKabuS.exeを一旦killしてから再起動するため、近接した2回目が
# 走ると起動直後のプロセスをまた落とすことになる）。
RERUN_COOLDOWN_SECONDS = 60

# KabuS.exe の起動・再起動を行う全モジュールで共有するロック（broker_process_lock.py 参照）。
# 別々のロックを持つと、broker_launcher.py の生存監視自動起動と同時に有効化した際
# 互いを知らずに二重起動しうる（2026-09-13 に発見）。
#
# クールダウン起点（直近の操作完了時刻）と実行中フラグも broker_process_lock.py の
# ものを使う。ロックだけ共有してもこれらがモジュールごとに別々のままだと、
# broker_full_login.run() が subprocess.run() 実行中にロックを手放す間隙で
# broker_launcher.launch() が「自分は実行中でない」と誤判定して二重起動できて
# しまうことが再現テストで判明したため（2026-09-13）。
_lock = broker_process_lock.lock
_attempts: int = 0
_attempts_date: Optional[date] = None


def _today() -> date:
    from src.core import clock
    return clock.today()


def _now() -> float:
    """クールダウン判定用の時刻（単調増加。テストで差し替える）。"""
    return time.monotonic()


def reset() -> None:
    """テスト用に試行回数とクールダウンを初期化する。"""
    global _attempts, _attempts_date
    with _lock:
        _attempts = 0
        _attempts_date = None
    broker_process_lock.reset()


def attempts_today() -> int:
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


def run(*, manual: bool = False,
        max_attempts_per_day: int = DEFAULT_MAX_ATTEMPTS_PER_DAY,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        wsl_distro: str = DEFAULT_WSL_DISTRO,
        project_dir: str = DEFAULT_PROJECT_DIR,
        script_path: str = DEFAULT_SCRIPT_PATH) -> tuple[bool, str]:
    """Kabuステーションの起動〜ログイン〜2段階認証入力までを完全自動化する。

    (実行できたか, 説明) を返す。WSL側スクリプトの完了まで**同期的にブロックする**
    （起動確認は tasklist で済む broker_launcher.launch() と異なり、ログイン完了
    までを1回の呼び出しで待つ設計。呼び出し元は数十秒〜数分のブロックを想定する）。

    manual=True は人が明示的に頼んだ実行（Discordコマンド）。broker_launcher.launch()
    と同じく、日次上限では止めない。

    `_lock` は「クールダウン・日次上限・実行中フラグのチェックと予約」
    「完了時刻の記録」だけを保持し、`subprocess.run()`（最大 timeout_seconds 秒、
    既定180秒）の実行中は手放す。Discordリモコンのポーリング等、他の呼び出しが
    この長い外部呼び出しの間ずっとブロックされるのを避けるため（緊急停止(halt)
    経路が数分止まるのは取引システムとして安全上望ましくない）。

    二重実行の防止は「経過時間ベースのクールダウン（60秒）」と
    「実行中フラグ（`broker_process_lock.in_progress`）」の2段構え。クールダウン
    だけでは、タイムアウト上限（既定180秒）がクールダウン秒数を超えるため、
    実行中の60〜180秒の間に到着した呼び出しが素通りしてしまう（`elapsed` が
    `RERUN_COOLDOWN_SECONDS` を超えるため）。実行中フラグは経過時間に関係なく
    「今まさに1本実行中かどうか」だけを見るので、この隙間を塞ぐ。

    クールダウン起点・実行中フラグは broker_launcher.py と共有する
    （broker_process_lock.py 参照）。モジュールごとに別々の状態を持つと、
    このモジュールが subprocess.run() 実行中にロックを手放す間隙で
    broker_launcher.launch() が「自分は実行中でない」と誤判定し、同じ
    KabuS.exe を二重に起動/再起動できてしまう（2026-09-13、再現テストで確認）。

    注意: WSLコマンドは wsl_distro/project_dir/script_path を f-string で
    シェルエスケープ無しに埋め込んでいる。現状 main.py は既定値以外を渡さない
    ため安全だが、将来これらを設定ファイル等の外部入力で差し替え可能にする場合は
    埋め込み前に shlex.quote() を通すこと。
    """
    global _attempts

    with _lock:
        now = _now()
        elapsed = now - broker_process_lock.last_operation_at
        if elapsed < RERUN_COOLDOWN_SECONDS:
            return False, (
                f"直前に実行しています（{int(elapsed)}秒前）。"
                f"{RERUN_COOLDOWN_SECONDS}秒は再実行しません"
            )
        if broker_process_lock.in_progress:
            # クールダウン（経過時間）だけでは、タイムアウト上限（既定180秒）が
            # クールダウン秒数（60秒）を超えるため、実行中の60〜180秒の間に
            # 到着した呼び出しがこの上のチェックを素通りしてしまう。経過時間に
            # 関係なく「実行中は無条件で拒否する」ことでその隙間を塞ぐ。
            return False, "既に起動/再起動の操作が進行中です。完了までお待ちください"

        _roll_over_if_new_day()
        if manual:
            attempt_no = None
        else:
            if max_attempts_per_day > 0 and _attempts >= max_attempts_per_day:
                return False, (
                    f"本日の実行回数が上限({max_attempts_per_day}回)に達しています。"
                    "証券会社側の障害の可能性があるため、手動で確認してください"
                )
            _attempts += 1
            attempt_no = _attempts

        # 「これから実行する」印として現在時刻を仮置きする。これは同時実行の
        # ガードを兼ねる（ロック解放直後に別呼び出しが来ても、この時刻で
        # クールダウンに掛かり subprocess を二重に走らせない）。完了後に
        # 実際の完了時刻で上書きする（cooldownは完了時刻basisにするため）。
        broker_process_lock.last_operation_at = now
        broker_process_lock.in_progress = True

    command = [
        "wsl", "-d", wsl_distro, "--", "bash", "-lc",
        f"cd {project_dir} && ./{script_path}",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        with _lock:
            broker_process_lock.last_operation_at = _now()
            broker_process_lock.in_progress = False
        logger.error(f"完全自動ログインがタイムアウトしました（{timeout_seconds}秒）")
        return False, (
            f"完全自動ログインがタイムアウトしました（{timeout_seconds}秒）。"
            "WSL側の処理が残っている可能性があるため、手動でログインする前に"
            "既存のログイン試行が完了していないか確認してください"
        )
    except Exception as e:
        with _lock:
            broker_process_lock.last_operation_at = _now()
            broker_process_lock.in_progress = False
        logger.error(f"完全自動ログインの実行に失敗しました: {e}")
        return False, f"実行に失敗しました: {e}"

    with _lock:
        broker_process_lock.last_operation_at = _now()
        broker_process_lock.in_progress = False

    if result.returncode != 0:
        # WSL側スクリプトは set -e のため、docker compose up --build 等の
        # 診断出力の多くが stdout に出る。stderr が空だと失敗理由が丸ごと
        # 消えるので、空のときは stdout にフォールバックする。
        detail_source = result.stderr.strip() or result.stdout.strip()
        if "auth_gmailapi" in detail_source:
            # OTP取得側がGmail認証失効（テスト中アプリの7日失効）を報告している。
            # 失敗理由は長い出力の末尾にあり [:300] では切れるため、ここで判定する。
            logger.error(f"完全自動ログイン失敗: Gmail認証が失効しています: {detail_source[-300:]}")
            return False, (
                "Gmail認証（ワンタイムパスワード取得用）が失効しています。"
                "Discordで `gmail_auth` を送る（スマホ可）か、"
                "PCで scripts\\gmail_reauth.bat をダブルクリックし、"
                "開いたブラウザで「許可」を押してください"
            )
        logger.error(
            f"完全自動ログインが失敗しました（rc={result.returncode}）: "
            f"{detail_source[:500]}"
        )
        return False, (
            f"完全自動ログインに失敗しました（rc={result.returncode}）: "
            f"{detail_source[:300]}"
        )

    how = "手動" if attempt_no is None else f"本日{attempt_no}回目"
    logger.warning(f"完全自動ログインを実行しました（{how}）")
    return True, f"完全自動ログインを実行しました（{how}）"
