"""
ダッシュボードのログイン認証（認証情報の保存とセッション管理）

認証情報（ユーザーID・パスワード）は Authファイル（既定: data/auth.json）に保存する。
パスワードは復号可能な「暗号化」ではなく、ソルト付きの PBKDF2-HMAC-SHA256 ハッシュとして
保存する（一方向ハッシュ。Authファイルが漏えいしても元パスワードを復元できず、鍵管理も不要）。

セッションはメモリ上で管理する（プロセス再起動で無効化され、再ログインが必要）。
いずれも Python 標準ライブラリ（hashlib / hmac / secrets）のみで実装し、追加依存はない。
"""
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

_ALGO = "pbkdf2_sha256"
_ITERATIONS = 200_000
_SALT_BYTES = 16
_SESSION_TTL_HOURS = 168  # セッション既定有効期間（7日）

_path: Path = Path("data/auth.json")
_data: Optional[dict] = None
_sessions: dict[str, datetime] = {}  # token -> expiry


def load(path: str = "data/auth.json", session_ttl_hours: Optional[int] = None) -> None:
    """Authファイルのパスを設定し、存在すれば読み込む。"""
    global _path, _data, _SESSION_TTL_HOURS
    _path = Path(path)
    if session_ttl_hours:
        _SESSION_TTL_HOURS = int(session_ttl_hours)
    if _path.exists():
        try:
            with open(_path, encoding="utf-8") as f:
                _data = json.load(f)
            logger.info(f"認証情報を読み込み: ユーザー='{_data.get('username')}' ({_path})")
        except Exception as e:
            logger.error(f"認証ファイルの読み込みに失敗: {e}")
            _data = None
    else:
        _data = None
        logger.info(f"認証ファイル未作成（初回はGUIで初期設定）: {_path}")


def is_configured() -> bool:
    """認証情報（ユーザーID・パスワードハッシュ）が登録済みか。"""
    return bool(_data and _data.get("username") and _data.get("hash"))


def _hash(password: str, salt: bytes, iterations: int) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return dk.hex()


def create_user(username: str, password: str) -> None:
    """初期設定：認証情報を作成しAuthファイルへPBKDF2ハッシュで保存する。

    既に登録済みの場合は ValueError（初期設定は1度だけ）。
    """
    global _data
    if is_configured():
        raise ValueError("認証情報は既に設定されています")
    username = (username or "").strip()
    if not username:
        raise ValueError("ユーザーIDを入力してください")
    if not password or len(password) < 8:
        raise ValueError("パスワードは8文字以上にしてください")
    salt = secrets.token_bytes(_SALT_BYTES)
    _data = {
        "username": username,
        "salt": salt.hex(),
        "hash": _hash(password, salt, _ITERATIONS),
        "iterations": _ITERATIONS,
        "algo": _ALGO,
        "created_at": datetime.now().isoformat(),
    }
    _path.parent.mkdir(parents=True, exist_ok=True)
    with open(_path, "w", encoding="utf-8") as f:
        json.dump(_data, f, ensure_ascii=False, indent=2)
    logger.warning(f"認証情報を作成しました: ユーザー='{username}' ({_path})")


def verify(username: str, password: str) -> bool:
    """ユーザーID・パスワードを照合する（定数時間比較）。"""
    if not is_configured():
        return False
    salt = bytes.fromhex(_data["salt"])
    iterations = int(_data.get("iterations", _ITERATIONS))
    expected = _data["hash"]
    candidate = _hash(password or "", salt, iterations)
    user_ok = hmac.compare_digest((username or "").strip(), _data["username"])
    pw_ok = hmac.compare_digest(candidate, expected)
    return user_ok and pw_ok


# ─── セッション管理（メモリ内） ────────────────────────────────────────────


def create_session() -> str:
    """ログイン成功時にセッショントークンを発行する。"""
    token = secrets.token_urlsafe(32)
    _sessions[token] = datetime.now() + timedelta(hours=_SESSION_TTL_HOURS)
    return token


def validate_session(token: Optional[str]) -> bool:
    """セッショントークンが有効か。期限切れは破棄する。"""
    if not token:
        return False
    expiry = _sessions.get(token)
    if expiry is None:
        return False
    if datetime.now() >= expiry:
        _sessions.pop(token, None)
        return False
    return True


def destroy_session(token: Optional[str]) -> None:
    if token:
        _sessions.pop(token, None)
