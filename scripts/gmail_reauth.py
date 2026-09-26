"""Gmail API（ワンタイムパスワード取得用）の再認証。ブラウザで「許可」を押す以外は自動。

失効理由と運用方針は src/core/gmail_token.py を参照。gmail_reauth.bat から起動する。
"""
import re
import subprocess
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import clock, gmail_token  # noqa: E402

PROJECT = "~/projects/kabusapi-auto-login-template"
AUTH = (
    "docker compose -f docker/docker-compose.yml run --rm -T -e PYTHONUNBUFFERED=1 "
    "onetime_password python -u ./scripts/gmail_api/auth_gmailapi.py"
)


def wsl(cmd: str, **kwargs):
    return subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "bash", "-lc", f"cd {PROJECT} && {{ {cmd}; }}"], **kwargs
    )


def main() -> int:
    # 中断した前回のコンテナが8090を掴んだままだと起動できない（2026-09-25実例）
    wsl("docker ps -aq --filter name=onetime_password-run | xargs -r docker rm -f >/dev/null")
    # 失効トークンが残っていると認証スクリプトは更新を試みて失敗終了するため退避する
    wsl("[ ! -f secrets/token.pickle ] || mv -f secrets/token.pickle secrets/token.pickle.bak")

    proc = subprocess.Popen(
        ["wsl", "-d", "Ubuntu", "--", "bash", "-lc", f"cd {PROJECT} && {AUTH}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    for line in proc.stdout:
        print(line, end="")
        m = re.search(r"https://accounts\.google\.com/\S+", line)
        if m:
            webbrowser.open(m.group(0))
            print(">>> ブラウザを開きました。アカウントを選び「続行」→「許可」を押してください（5分以内）")

    if proc.wait() == 0 and wsl("test -s secrets/token.pickle").returncode == 0:
        now = clock.now()
        gmail_token.record_issued(now)
        print(f"\n完了しました。次の期限: {now + gmail_token.LIFETIME:%m/%d %H:%M}")
        return 0

    wsl("[ -f secrets/token.pickle ] || [ ! -f secrets/token.pickle.bak ] || mv secrets/token.pickle.bak secrets/token.pickle")
    print("\n再認証に失敗しました（元のトークンに戻しました）。もう一度実行してください。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
