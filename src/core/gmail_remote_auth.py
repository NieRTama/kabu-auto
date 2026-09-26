"""Gmail再認証をDiscord経由（スマホ）で行う。

## 背景

Gmail再認証は従来 `scripts/gmail_reauth.bat` のみで、PC上のブラウザから
実行する前提だった（`InstalledAppFlow.run_local_server` が
http://localhost:8090/ でコードを受け取るため）。外出先で失効に気づいても
PCの前に戻るまで再認証できない。

本モジュールは、外部リポジトリ側の2段階版スクリプト
（kabusapi-auto-login-template/scripts/gmail_api/auth_gmailapi_remote.py）
を呼び出し、スマホから完結できるようにする。

1. `gmail_auth`（引数なし）→ `start()` が認証URLを発行して返す。
2. 人がスマホでURLを開き、Googleの同意画面で許可すると
   http://localhost:8090/?state=...&code=...&scope=... へリダイレクトされる
   （このページ自体はスマホでは開けないが、アドレスバーのURLはコピーできる）。
3. `gmail_auth <コピーしたURL>` → `finish()` がコードをトークンに交換し、
   token.pickle を更新して同意時刻を記録する。

秘密情報（認可コード・state）はこのモジュール内で正規表現検証と
shlex.quote() を経てからのみシェルコマンドに渡す（Discord経由の文字列を
そのままシェルへ渡さない）。
"""
import re
import shlex
import subprocess
import threading
from urllib.parse import urlparse, parse_qs

from src.core import clock, gmail_token

PROJECT = "~/projects/kabusapi-auto-login-template"
SCRIPT = "./scripts/gmail_api/auth_gmailapi_remote.py"
TIMEOUT_SECONDS = 300

# 認可コード・stateはURLの一部としてのみ現れる値で、シェルコマンドへ渡す前に
# この形式を満たすことを確認する（それ以外は拒否し、subprocessを呼ばない）。
_TOKEN_RE = re.compile(r"^[A-Za-z0-9/_\-.~]+$")

_lock = threading.Lock()


def _run(subcmd: str) -> subprocess.CompletedProcess:
    command = (
        f"cd {PROJECT} && docker compose -f docker/docker-compose.yml "
        f"run --rm -T onetime_password python -u {SCRIPT} {subcmd}"
    )
    return subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=TIMEOUT_SECONDS,
    )


def _tail(text: str, n: int = 300) -> str:
    return text.strip()[-n:]


def start() -> str:
    """認証URLを発行する（gmail_auth コマンド、引数なし）。"""
    if not _lock.acquire(blocking=False):
        return "Gmail認証の処理が実行中です。完了までお待ちください"
    try:
        try:
            result = _run("start")
        except subprocess.TimeoutExpired:
            return "認証URLの発行がタイムアウトしました。しばらくしてもう一度お試しください"
        except Exception as e:
            return f"認証URLの発行に失敗しました: {e}"

        m = re.search(r"AUTH_URL=(\S+)", result.stdout)
        if result.returncode != 0 or not m:
            detail = _tail(result.stderr) or _tail(result.stdout)
            return f"認証URLの発行に失敗しました: {detail}"

        url = m.group(1)
        return (
            "以下のURLをスマホのブラウザで開き、アカウントを選んで「続行」→「許可」を"
            "押してください。\n"
            f"{url}\n\n"
            "許可すると「ページを開けません」等のエラー画面になりますが、想定どおりです。"
            "そのページのアドレスバーに表示されているURLをコピーし、10分以内に\n"
            "`gmail_auth <コピーしたURL>`\n"
            "の形で送ってください。"
        )
    finally:
        _lock.release()


def finish(pasted: str) -> str:
    """コピーされたリダイレクト先URLでトークンを交換する（gmail_auth <URL>）。"""
    if not _lock.acquire(blocking=False):
        return "Gmail認証の処理が実行中です。完了までお待ちください"
    try:
        text = pasted.strip()
        # Discordはリンクを <...> で囲むことがあるため取り除く
        if text.startswith("<") and text.endswith(">"):
            text = text[1:-1]

        if text.startswith(("http://", "https://")):
            query = urlparse(text).query
        elif "?" in text:
            query = text.split("?", 1)[1]
        else:
            query = text
        params = parse_qs(query)

        if "error" in params:
            return f"認証がキャンセルされました（{params['error'][0]}）。もう一度 `gmail_auth` からやり直してください"

        code_list = params.get("code")
        state_list = params.get("state")
        if not code_list or not state_list:
            return "URLに code または state が含まれていません。ブラウザのアドレスバーに表示されたURL全体を貼り付けてください"

        code, state = code_list[0], state_list[0]
        if not _TOKEN_RE.match(code) or not _TOKEN_RE.match(state):
            return "URLの形式が正しくありません。もう一度 `gmail_auth` を送ってやり直してください"

        subcmd = f"finish {shlex.quote(code)} {shlex.quote(state)}"
        try:
            result = _run(subcmd)
        except subprocess.TimeoutExpired:
            return "トークン交換がタイムアウトしました。もう一度 `gmail_auth` からやり直してください"
        except Exception as e:
            return f"トークン交換に失敗しました: {e}"

        if result.returncode == 0 and "TOKEN_SAVED" in result.stdout:
            now = clock.now()
            gmail_token.record_issued(now)
            expires = now + gmail_token.LIFETIME
            return f"Gmail認証が完了しました。次の期限: {expires:%m/%d %H:%M}"

        detail = _tail(result.stderr) or _tail(result.stdout)
        return f"Gmail認証に失敗しました: {detail}"
    finally:
        _lock.release()
