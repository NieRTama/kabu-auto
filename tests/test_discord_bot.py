"""
Discordリモコン（discord_bot）のテスト

外出先から状態確認・緊急停止を行うためのポーリング方式の受信口。
Discord API はモックし、コマンド解釈・権限チェック・既読管理を検証する。

セキュリティ上の要件（実装で担保すべきこと）:
  - 許可されていないユーザーのコマンドは実行しない
  - Bot宛メンション以外は反応しない（雑談に反応しない・特権intent不要）
  - Bot自身/他Botの発言に反応しない（無限ループ防止）
  - 起動前の古いメッセージを実行しない（再起動でコマンドが暴発しない）
  - 未設定なら機能ごと無効
"""
from unittest.mock import MagicMock, patch

import pytest

from src.core import discord_bot as mod

BOT_ID = "999"
OWNER = "111"
STRANGER = "222"


def _msg(msg_id: str, content: str, author_id: str = OWNER,
         mention_bot: bool = True, is_bot: bool = False) -> dict:
    return {
        "id": msg_id,
        "content": content,
        "author": {"id": author_id, "bot": is_bot},
        "mentions": [{"id": BOT_ID}] if mention_bot else [],
    }


def _rc(messages, allowed={OWNER}, handlers=None):
    client = MagicMock()
    client.fetch_messages.return_value = messages
    executed = []
    handlers = handlers or {
        "status": lambda a: "STATUS_OK",
        "halt": lambda a: f"HALTED:{a}",
    }
    wrapped = {k: (lambda a, f=v, n=k: (executed.append(n), f(a))[1])
               for k, v in handlers.items()}
    rc = mod.RemoteControl(client, mod.CommandHandler(wrapped),
                           bot_id=BOT_ID, allowed_user_ids=allowed)
    return rc, client, executed


class TestMentionParsing:
    def test_strips_plain_mention(self):
        assert mod.strip_mention(f"<@{BOT_ID}> status", BOT_ID) == "status"

    def test_strips_nickname_mention(self):
        assert mod.strip_mention(f"<@!{BOT_ID}> halt 理由", BOT_ID) == "halt 理由"

    def test_returns_empty_without_mention(self):
        assert mod.strip_mention("status", BOT_ID) == ""

    def test_is_mentioned_detects_bot(self):
        assert mod.is_mentioned(_msg("1", "x"), BOT_ID) is True
        assert mod.is_mentioned(_msg("1", "x", mention_bot=False), BOT_ID) is False


class TestCommandHandler:
    def test_executes_known_command(self):
        h = mod.CommandHandler({"status": lambda a: "OK"})
        assert h.execute("status") == "OK"

    def test_passes_arguments(self):
        h = mod.CommandHandler({"halt": lambda a: f"reason={a}"})
        assert h.execute("halt 急変のため") == "reason=急変のため"

    def test_unknown_command_returns_help(self):
        h = mod.CommandHandler({"status": lambda a: "OK"})
        reply = h.execute("destroy")
        assert "不明なコマンド" in reply and "status" in reply

    def test_help_lists_commands(self):
        h = mod.CommandHandler({"status": lambda a: "", "halt": lambda a: ""})
        assert "status" in h.execute("help")

    def test_empty_returns_none(self):
        assert mod.CommandHandler({}).execute("   ") is None

    def test_handler_exception_is_reported_not_raised(self):
        def boom(a):
            raise RuntimeError("db down")
        h = mod.CommandHandler({"status": boom})
        assert "失敗" in h.execute("status")

    def test_command_is_case_insensitive(self):
        h = mod.CommandHandler({"status": lambda a: "OK"})
        assert h.execute("STATUS") == "OK"


class TestAuthorization:
    def test_rejects_unauthorized_user(self):
        rc, client, executed = _rc([_msg("1", f"<@{BOT_ID}> halt", author_id=STRANGER)])
        rc.poll_once()
        assert executed == [], "許可外ユーザーのコマンドは実行しない"
        client.send.assert_not_called()

    def test_accepts_allowed_user(self):
        rc, client, executed = _rc([_msg("1", f"<@{BOT_ID}> status")])
        rc.poll_once()
        assert executed == ["status"]
        client.send.assert_called_once()

    def test_ignores_message_without_mention(self):
        rc, _, executed = _rc([_msg("1", "status", mention_bot=False)])
        rc.poll_once()
        assert executed == [], "メンション無しには反応しない"

    def test_ignores_bot_authors(self):
        rc, _, executed = _rc([_msg("1", f"<@{BOT_ID}> status", is_bot=True)])
        rc.poll_once()
        assert executed == [], "Botの発言には反応しない（無限ループ防止）"


class TestReadState:
    def test_advances_last_id(self):
        rc, client, _ = _rc([_msg("10", f"<@{BOT_ID}> status")])
        rc.poll_once()
        rc.poll_once()
        assert client.fetch_messages.call_args.kwargs["after"] == "10"

    def test_prime_marks_existing_as_read(self):
        client = MagicMock()
        client.fetch_messages.return_value = [_msg("55", "古い発言")]
        rc = mod.RemoteControl(client, mod.CommandHandler({}),
                               bot_id=BOT_ID, allowed_user_ids={OWNER})
        rc.prime()
        assert rc._last_id == "55", "起動前のメッセージは既読扱いにする"

    def test_fetch_failure_is_swallowed(self):
        client = MagicMock()
        client.fetch_messages.side_effect = RuntimeError("network down")
        rc = mod.RemoteControl(client, mod.CommandHandler({}),
                               bot_id=BOT_ID, allowed_user_ids={OWNER})
        assert rc.poll_once() == 0, "取得失敗で例外を投げない（次回再試行）"


class TestBuildDisabled:
    def test_returns_none_without_token(self):
        assert mod.build("", "chan", {OWNER}, {}) is None

    def test_returns_none_without_channel(self):
        assert mod.build("token", "", {OWNER}, {}) is None

    def test_returns_none_when_api_unreachable(self):
        with patch.object(mod.DiscordBotClient, "get_me", side_effect=RuntimeError("401")):
            assert mod.build("token", "chan", {OWNER}, {}) is None


class TestNoSecretCommands:
    def test_env_write_command_is_not_implemented(self):
        """秘密情報をDiscord経由で運ばせない方針が守られていること"""
        import inspect
        src = inspect.getsource(mod)
        for forbidden in ("KABU_API_PASSWORD", "os.environ[", "open('.env'", '.env"'):
            assert forbidden not in src, (
                f"Discord経由で秘密情報や.envを扱う実装が入っている: {forbidden}"
            )
