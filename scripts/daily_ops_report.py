"""毎日の運用状況概要を log/ops_daily/ に書き出す（Windowsタスク \kabu\kabu-ops-daily から18:30に実行）。

当日の [ALERT] 行（約定・再ログイン・異常検知など）と損益を1枚にまとめる。
ログには ops_daily_YYYY-MM-DD.md を新規作成し、同じ内容を標準出力にも出す。
ponytail: 通知はDiscordのみ（未設定ならスキップ）。期間（1ヶ月）の打ち切りはタスク側の終了日で行う。
"""
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")  # Webhook URL 等は .env にある（本体の main.py と同じ）

from src.core.ops_summary import build  # noqa: E402


if __name__ == "__main__":
    today = date.today()
    text = build(today)
    d = REPO / "log" / "ops_daily"
    d.mkdir(exist_ok=True)
    (d / f"ops_daily_{today}.md").write_text(text + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    try:
        from src.core.discord_report import post_text
        post_text(build(today, full=False))  # Discord には短い版（詳細はファイル）
    except Exception as e:  # 通知失敗でもファイルは残す
        print(f"(Discord投稿失敗: {e})")
