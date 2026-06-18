"""
ログイン認証モジュール（src/core/auth.py）のテスト

- 認証情報の作成・照合（PBKDF2ハッシュ）
- 初期設定済み判定
- セッションの発行・検証・期限切れ・破棄
"""
from datetime import datetime, timedelta

import pytest

import src.core.auth as auth


@pytest.fixture(autouse=True)
def isolated_auth(tmp_path):
    """各テストで Authファイルとセッションを隔離・初期化する。"""
    auth.load(str(tmp_path / "auth.json"))
    auth._sessions.clear()
    yield
    auth._sessions.clear()


class TestCredentials:
    def test_initially_not_configured(self):
        assert auth.is_configured() is False

    def test_create_and_verify(self):
        auth.create_user("trader", "s3cretpw!")
        assert auth.is_configured() is True
        assert auth.verify("trader", "s3cretpw!") is True

    def test_verify_wrong_password(self):
        auth.create_user("trader", "s3cretpw!")
        assert auth.verify("trader", "wrong") is False

    def test_verify_wrong_username(self):
        auth.create_user("trader", "s3cretpw!")
        assert auth.verify("someone", "s3cretpw!") is False

    def test_password_not_stored_in_plaintext(self, tmp_path):
        auth.create_user("trader", "s3cretpw!")
        content = (tmp_path / "auth.json").read_text(encoding="utf-8")
        assert "s3cretpw!" not in content
        assert "hash" in content and "salt" in content

    def test_short_password_rejected(self):
        with pytest.raises(ValueError):
            auth.create_user("trader", "short")

    def test_empty_username_rejected(self):
        with pytest.raises(ValueError):
            auth.create_user("", "longenough123")

    def test_create_twice_rejected(self):
        auth.create_user("trader", "s3cretpw!")
        with pytest.raises(ValueError):
            auth.create_user("other", "anotherpw123")

    def test_salt_makes_hash_unique(self, tmp_path):
        """同じパスワードでもソルトが異なればハッシュは一致しない"""
        auth.create_user("a", "samepassword1")
        h1 = auth._data["hash"]
        auth.load(str(tmp_path / "auth2.json"))
        auth.create_user("b", "samepassword1")
        h2 = auth._data["hash"]
        assert h1 != h2

    def test_reload_persists(self, tmp_path):
        auth.create_user("trader", "s3cretpw!")
        auth.load(str(tmp_path / "auth.json"))  # 同じファイルを再読込
        assert auth.is_configured() is True
        assert auth.verify("trader", "s3cretpw!") is True


class TestSessions:
    def test_create_and_validate(self):
        token = auth.create_session()
        assert auth.validate_session(token) is True

    def test_invalid_token(self):
        assert auth.validate_session("nope") is False
        assert auth.validate_session(None) is False

    def test_expired_session(self):
        token = auth.create_session()
        auth._sessions[token] = datetime.now() - timedelta(seconds=1)  # 強制的に期限切れ
        assert auth.validate_session(token) is False
        assert token not in auth._sessions  # 期限切れは破棄される

    def test_destroy_session(self):
        token = auth.create_session()
        auth.destroy_session(token)
        assert auth.validate_session(token) is False
