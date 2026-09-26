"""Gmail API（ワンタイムパスワード自動取得用）のリフレッシュトークン失効の予告。

OAuth同意画面が「テスト中」のため、リフレッシュトークンは同意から7日で失効する。
gmail.readonly は制限付きスコープで、本番公開にはGoogleの審査（CASA監査を含みうる）が
要るため、個人利用ではテスト中のまま定期的に再認証する運用とした（2026-09-26）。
token.pickle はOTP取得のたびに書き換わり更新日時から同意時刻を逆算できないので、
再認証スクリプト（scripts/gmail_reauth.py）が同意時刻をここへ記録する。
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

LIFETIME = timedelta(days=7)
WARN_BEFORE = timedelta(days=2)
ISSUED_AT_PATH = Path(__file__).resolve().parents[2] / "data" / "gmail_token_issued_at.txt"
_HOW_TO = "PCで scripts\\gmail_reauth.bat をダブルクリックし、開いたブラウザで「許可」を押してください。"


def record_issued(at: datetime, path: Path = ISSUED_AT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(at.isoformat(timespec="seconds"), encoding="utf-8")


def reminder(now: datetime, path: Path = ISSUED_AT_PATH) -> Optional[tuple[str, str]]:
    """通知が要るなら (level, message)、不要なら None を返す。level は alerts の LEVEL_*。"""
    try:
        issued = datetime.fromisoformat(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return "warning", f"Gmail認証の実施日時が記録されていません。{_HOW_TO}"
    expires = issued + LIFETIME
    if now >= expires:
        return "critical", (
            f"Gmail認証が失効しています（{expires:%m/%d %H:%M}）。"
            f"朝8:30の自動ログインが失敗します。{_HOW_TO}"
        )
    if expires - now <= WARN_BEFORE:
        return "warning", f"Gmail認証が {expires:%m/%d %H:%M} 頃に失効します。{_HOW_TO}"
    return None
