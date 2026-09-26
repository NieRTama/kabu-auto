"""Gmail認証をDiscord経由（スマホ）でやり直す gmail_remote_auth のテスト。

WSL/dockerへの実接続はしない。subprocess.run をモックして、URL抽出・
入力検証（シェルへ渡す前のバリデーション）・排他制御・失敗時の応答内容を
検証する。
"""
import subprocess
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.core import gmail_remote_auth as gra
from src.core import gmail_token


def _cp(rc=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["wsl"], returncode=rc, stdout=stdout, stderr=stderr)


@pytest.fixture(autouse=True)
def _release_lock():
    # 前のテストが例外等でロックを持ったまま終わらないようにする
    if gra._lock.locked():
        gra._lock.release()
    yield
    if gra._lock.locked():
        gra._lock.release()


class TestStart:
    def test_extracts_auth_url(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: _cp(0, stdout="AUTH_URL=https://accounts.google.com/o/oauth2/auth?x=1\n"),
        )
        reply = gra.start()
        assert "https://accounts.google.com/o/oauth2/auth?x=1" in reply
        assert "gmail_auth" in reply

    def test_failure_shows_tail(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: _cp(1, stdout="", stderr="x" * 400 + "認証情報ファイルが見つかりません"),
        )
        reply = gra.start()
        assert "認証情報ファイルが見つかりません" in reply

    def test_timeout(self, monkeypatch):
        def _raise(*a, **k):
            raise subprocess.TimeoutExpired(cmd="wsl", timeout=300)
        monkeypatch.setattr(subprocess, "run", _raise)
        reply = gra.start()
        assert "タイムアウト" in reply

    def test_busy_lock(self, monkeypatch):
        called = MagicMock()
        monkeypatch.setattr(subprocess, "run", called)
        gra._lock.acquire()
        try:
            reply = gra.start()
        finally:
            gra._lock.release()
        assert "実行中" in reply
        called.assert_not_called()


class TestFinish:
    def test_success_records_issued_time_and_quotes_args(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return _cp(0, stdout="TOKEN_SAVED\n")

        monkeypatch.setattr(subprocess, "run", fake_run)

        recorded = {}
        monkeypatch.setattr(
            gmail_token, "record_issued",
            lambda at, path=None: recorded.setdefault("at", at),
        )
        fixed_now = datetime(2026, 9, 27, 8, 0)
        monkeypatch.setattr(gra.clock, "now", lambda: fixed_now)

        url = "http://localhost:8090/?state=abcSTATE&code=abcCODE-123&scope=xyz"
        reply = gra.finish(url)

        assert "完了" in reply
        assert f"{(fixed_now + gmail_token.LIFETIME):%m/%d %H:%M}" in reply
        assert recorded["at"] == fixed_now

        # シェルコマンド文字列にcode/stateがshlex.quote済みで含まれること
        shell_cmd = captured["cmd"][-1]
        assert "abcSTATE" in shell_cmd
        assert "abcCODE-123" in shell_cmd

    def test_strips_angle_brackets(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return _cp(0, stdout="TOKEN_SAVED\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(gmail_token, "record_issued", lambda at, path=None: None)
        monkeypatch.setattr(gra.clock, "now", lambda: datetime(2026, 9, 27, 8, 0))

        url = "<http://localhost:8090/?state=abcSTATE&code=abcCODE&scope=xyz>"
        reply = gra.finish(url)
        assert "完了" in reply
        assert "cmd" in captured

    def test_bare_query_string_accepted(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _cp(0, stdout="TOKEN_SAVED\n"))
        monkeypatch.setattr(gmail_token, "record_issued", lambda at, path=None: None)
        monkeypatch.setattr(gra.clock, "now", lambda: datetime(2026, 9, 27, 8, 0))

        reply = gra.finish("state=abcSTATE&code=abcCODE&scope=xyz")
        assert "完了" in reply

    def test_missing_code_no_subprocess(self, monkeypatch):
        called = MagicMock()
        monkeypatch.setattr(subprocess, "run", called)
        reply = gra.finish("http://localhost:8090/?state=abcSTATE&scope=xyz")
        assert "code" in reply or "URL" in reply
        called.assert_not_called()

    def test_error_param_is_cancelled(self, monkeypatch):
        called = MagicMock()
        monkeypatch.setattr(subprocess, "run", called)
        reply = gra.finish("http://localhost:8090/?error=access_denied&state=abcSTATE")
        assert "キャンセル" in reply
        called.assert_not_called()

    @pytest.mark.parametrize("bad_code", [
        "abc;rm -rf /",
        "abc$(whoami)",
        "abc`whoami`",
        "abc|cat",
        "abc code",
    ])
    def test_rejects_injection_like_code_without_subprocess(self, monkeypatch, bad_code):
        called = MagicMock()
        monkeypatch.setattr(subprocess, "run", called)
        from urllib.parse import quote
        url = f"http://localhost:8090/?state=abcSTATE&code={quote(bad_code)}"
        reply = gra.finish(url)
        assert reply
        called.assert_not_called()

    def test_non_zero_rc_is_failure(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: _cp(1, stdout="", stderr="x" * 400 + "stateが一致しません"),
        )
        reply = gra.finish("http://localhost:8090/?state=abcSTATE&code=abcCODE")
        assert "失敗" in reply
        assert "stateが一致しません" in reply

    def test_busy_lock(self, monkeypatch):
        called = MagicMock()
        monkeypatch.setattr(subprocess, "run", called)
        gra._lock.acquire()
        try:
            reply = gra.finish("http://localhost:8090/?state=abcSTATE&code=abcCODE")
        finally:
            gra._lock.release()
        assert "実行中" in reply
        called.assert_not_called()


def test_gmail_auth_mentioned_in_reminder():
    assert "gmail_auth" in gmail_token._HOW_TO
