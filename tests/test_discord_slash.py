"""
Discord スラッシュコマンド（discord_slash）のテスト

メンション方式は「補完がBotのロールを選ぶ」問題で使いにくく、応答も最大30秒
かかっていた（2026-09-01）。Gateway 経由なら即時応答でき、`/` で候補も出る。

Gateway への実接続はテストしない（外部依存）。認可・コマンド実行・
無効化条件といった**自分たちのロジック**を検証する。
"""
from unittest.mock import MagicMock, patch

import pytest

from src.core import discord_slash as ds

CHANNEL = "1000"
OWNER = "111"
STRANGER = "222"


def _server(handlers=None):
    return ds.SlashCommandServer(
        token="t", channel_id=CHANNEL, allowed_user_ids={OWNER},
        handlers=handlers or {"status": (lambda a: "OK", "稼働状況")},
    )


class TestAuthorization:
    """認可の基準はメンション方式と同じ（緩めない）"""

    def test_allows_owner_in_correct_channel(self):
        ok, _ = _server()._is_authorized(OWNER, CHANNEL)
        assert ok is True

    def test_rejects_other_user(self):
        ok, reason = _server()._is_authorized(STRANGER, CHANNEL)
        assert ok is False
        assert "権限" in reason

    def test_rejects_other_channel(self):
        ok, reason = _server()._is_authorized(OWNER, "9999")
        assert ok is False
        assert "チャンネル" in reason

    def test_accepts_int_ids(self):
        """DiscordのIDは整数で渡るため、文字列比較で落とさないこと"""
        ok, _ = _server()._is_authorized(int(OWNER), int(CHANNEL))
        assert ok is True


class TestHandlerExecution:
    def test_runs_handler(self):
        assert _server()._run_handler("status", "") == "OK"

    def test_passes_arguments(self):
        s = _server({"halt": (lambda a: f"reason={a}", "停止")})
        assert s._run_handler("halt", "様子見") == "reason=様子見"

    def test_unknown_command(self):
        assert "不明なコマンド" in _server()._run_handler("bogus", "")

    def test_exception_is_reported_not_raised(self):
        """ハンドラが落ちてもBotごと死なせない"""
        def boom(a):
            raise RuntimeError("db down")
        s = _server({"status": (boom, "稼働状況")})
        assert "失敗" in s._run_handler("status", "")

    def test_empty_result_is_replaced(self):
        """空文字はDiscordが受け付けないため置き換える"""
        s = _server({"status": (lambda a: "", "稼働状況")})
        assert s._run_handler("status", "") == "(応答なし)"


class TestDisabled:
    def test_returns_none_without_token(self):
        assert ds.build("", CHANNEL, {OWNER}, {}) is None

    def test_returns_none_without_channel(self):
        assert ds.build("t", "", {OWNER}, {}) is None

    def test_returns_none_when_library_missing(self):
        """discord.py 未導入でも起動を妨げない（機能ごと無効化）"""
        with patch.object(ds, "available", return_value=False):
            assert ds.build("t", CHANNEL, {OWNER}, {}) is None


class TestSlowCommands:
    def test_external_api_commands_are_deferred(self):
        """3秒以内に返せない可能性があるものは defer 対象にする。

        kabuステーション起動や再接続は外部APIを叩くため、
        defer しないと Discord 側でタイムアウトする。
        """
        assert "launch" in ds._SLOW_COMMANDS
        assert "reconnect" in ds._SLOW_COMMANDS

    def test_read_only_commands_are_not_deferred(self):
        """DB照会は速いので即応答（defer すると2通に分かれて見づらい）"""
        for name in ("status", "positions", "pnl", "orders", "today"):
            assert name not in ds._SLOW_COMMANDS


class TestWiring:
    def _main_src(self):
        with open("main.py", encoding="utf-8") as f:
            return f.read()

    def test_shares_handlers_with_mention_mode(self):
        """メンション方式とスラッシュ方式が同じ定義を共有すること（二重実装を防ぐ）"""
        src = self._main_src()
        assert "_discord_handlers" in src
        assert "discord_bot.build(" in src and "discord_slash.build(" in src
        # 両方に同じ変数が渡されている
        assert src.count("handlers=_discord_handlers") == 2

    def test_mention_mode_is_kept(self):
        """Gateway が切れていても操作できる経路として残す"""
        assert "discord_bot.build(" in self._main_src()

    def test_no_secret_handling(self):
        """秘密情報をDiscord経由で扱わない方針が守られていること"""
        import inspect
        src = inspect.getsource(ds)
        for forbidden in ("KABU_API_PASSWORD", "os.environ[", ".env"):
            assert forbidden not in src, f"秘密情報を扱う実装が入っている: {forbidden}"
