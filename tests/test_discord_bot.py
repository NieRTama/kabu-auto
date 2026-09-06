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
    """本番と同じ結線でリモコンを作る。

    本番は build() が prime() を呼んでから poll_once を回す。ここでは
    「起動時点ではメッセージが無く、そのあとに新着が届く」状況を模す
    （起動前のメッセージは実行しない仕様のため、prime を省くと実態とズレる）。
    """
    client = MagicMock()
    client.fetch_messages.return_value = []
    executed = []
    handlers = handlers or {
        "status": lambda a: "STATUS_OK",
        "halt": lambda a: f"HALTED:{a}",
    }
    wrapped = {k: (lambda a, f=v, n=k: (executed.append(n), f(a))[1])
               for k, v in handlers.items()}
    rc = mod.RemoteControl(client, mod.CommandHandler(wrapped),
                           bot_id=BOT_ID, allowed_user_ids=allowed)
    rc.prime()                                     # 起動時の既読化（本番と同じ）
    client.fetch_messages.return_value = messages  # 以降に新着が届く
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


class TestHelpWithDescriptions:
    """help が「何ができるか」まで案内すること。

    名前の羅列だけでは外出先で思い出せない（コマンドが9個に増えた）。
    """

    def _handler(self):
        return mod.CommandHandler({
            "status": (lambda a: "", "稼働状況を表示"),
            "halt": (lambda a: "", "取引を停止する"),
        })

    def test_lists_names_and_descriptions(self):
        out = self._handler().help_text()
        assert "status" in out and "稼働状況を表示" in out
        assert "halt" in out and "取引を停止する" in out

    def test_shows_how_to_invoke(self):
        """メンションを付けて実行することを案内する（打ち方が分からないと使えない）"""
        assert "@kabu-auto" in self._handler().help_text()

    def test_help_command_returns_the_list(self):
        assert "稼働状況を表示" in self._handler().execute("help")

    def test_question_mark_is_alias(self):
        assert "稼働状況を表示" in self._handler().execute("?")

    def test_unknown_command_shows_help(self):
        out = self._handler().execute("bogus")
        assert "不明なコマンド" in out and "稼働状況を表示" in out

    def test_tuple_and_plain_handlers_both_work(self):
        """説明なし（関数のみ）の登録とも混在できる（後方互換）"""
        h = mod.CommandHandler({
            "a": (lambda x: "A", "説明あり"),
            "b": lambda x: "B",
        })
        assert h.execute("a") == "A"
        assert h.execute("b") == "B"
        assert "説明あり" in h.help_text()

    def test_falls_back_to_names_when_no_descriptions(self):
        h = mod.CommandHandler({"a": lambda x: "", "b": lambda x: ""})
        assert h.help_text() == "利用できるコマンド: a  b"


class TestRoleMention:
    """ロールメンションにも反応すること（2026-09-01 に実際に踏んだ）。

    Discordの入力補完は「Botユーザー」ではなく「Botに紐づくロール」を選ぶことがあり、
    その場合 <@&ロールID> になる。ユーザーメンションだけを見ていたため、
    Intentを有効にしても一切反応しなかった。利用者に選び分けを強いるのは
    非現実的なので、実装側で両方を受け付ける。
    """

    ROLE = "555"

    def _msg_role(self, text):
        return {
            "id": "1", "content": f"<@&{self.ROLE}> {text}",
            "author": {"id": OWNER, "bot": False},
            "mentions": [], "mention_roles": [self.ROLE],
        }

    def test_detects_role_mention(self):
        assert mod.is_mentioned(self._msg_role("help"), BOT_ID, {self.ROLE}) is True

    def test_ignores_role_mention_without_role_ids(self):
        """ロールIDを知らなければ従来どおり無視（他ロールへの言及に反応しない）"""
        assert mod.is_mentioned(self._msg_role("help"), BOT_ID, set()) is False

    def test_strips_role_mention(self):
        assert mod.strip_mention(f"<@&{self.ROLE}> status", BOT_ID, {self.ROLE}) == "status"

    def test_strips_user_mention_still_works(self):
        assert mod.strip_mention(f"<@{BOT_ID}> status", BOT_ID, {self.ROLE}) == "status"
        assert mod.strip_mention(f"<@!{BOT_ID}> status", BOT_ID, {self.ROLE}) == "status"

    def test_rejects_other_role(self):
        """別のロールへのメンションは自分宛ではない"""
        assert mod.strip_mention("<@&999> status", BOT_ID, {self.ROLE}) == ""

    def test_rejects_other_user(self):
        assert mod.strip_mention("<@777> status", BOT_ID, {self.ROLE}) == ""

    def test_executes_command_via_role_mention(self):
        """ロールメンション経由でも実際にコマンドが動くこと（結線の検証）"""
        client = MagicMock()
        client.fetch_messages.return_value = []
        rc = mod.RemoteControl(
            client, mod.CommandHandler({"status": lambda a: "OK"}),
            bot_id=BOT_ID, allowed_user_ids={OWNER}, role_ids={self.ROLE},
        )
        rc.prime()                                                   # 起動時の既読化
        client.fetch_messages.return_value = [self._msg_role("status")]
        assert rc.poll_once() == 1
        client.send.assert_called_once_with("OK")

    def test_authorization_still_applies_to_role_mention(self):
        """ロール経由でも送信者IDの認可は効く（誰でも操作できてはいけない）"""
        msg = self._msg_role("status")
        msg["author"]["id"] = STRANGER
        client = MagicMock()
        client.fetch_messages.return_value = []
        rc = mod.RemoteControl(
            client, mod.CommandHandler({"status": lambda a: "OK"}),
            bot_id=BOT_ID, allowed_user_ids={OWNER}, role_ids={self.ROLE},
        )
        rc.prime()                               # 既読化を通す（通さないと認可を検証せず素通りする）
        client.fetch_messages.return_value = [msg]
        assert rc.poll_once() == 0
        client.send.assert_not_called()


class TestRequestTimeoutIsBounded:
    """接続待ちを読み取りより短く取り、到達できないIPでの足止めを縮める。

    2026-09-06: ネットワーク断で fetch_messages が約53秒かかり、30秒間隔の
    ポーリングジョブが APScheduler にスキップされた。requests の timeout は
    「接続1回あたり」の上限で、接続は候補IPごとに順に試される。

    なお、これは超過の確率を下げるだけで防止の保証ではない（DNS解決は上限の対象外、
    候補IP数は環境依存、poll_once は返信ごとに send も呼ぶ）。超過に気づけるように
    する側が本命の対策なので、ここでは「上限が分かれていること」だけを固定する。
    """

    def test_fetch_messages_separates_connect_and_read_timeout(self):
        client = mod.DiscordBotClient("tok", "chan")
        resp = MagicMock()
        resp.json.return_value = []
        resp.raise_for_status.return_value = None
        with patch.object(mod.requests, "get", return_value=resp) as req:
            client.fetch_messages()
        timeout = req.call_args.kwargs["timeout"]
        assert isinstance(timeout, tuple), \
            "接続と読み取りで別々の上限を持つこと（接続待ちだけを短くしたい）"
        assert len(timeout) == 2

    def test_send_uses_the_same_bounded_timeout(self):
        client = mod.DiscordBotClient("tok", "chan")
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        with patch.object(mod.requests, "post", return_value=resp) as req:
            client.send("hello")
        assert isinstance(req.call_args.kwargs["timeout"], tuple)

    def test_connect_timeout_is_shorter_than_read_timeout(self):
        """接続待ちの上限が読み取りより短いこと（足止めされるのは接続側のため）。"""
        connect, read = mod.DEFAULT_TIMEOUT
        assert 0 < connect < read


class TestStartupHandshakeTimeout:
    """起動時のハンドシェイクは、定期ポーリングより待ち時間を長く取る。

    get_me() / fetch_bot_role_ids() は build() からしか呼ばれず、ここで失敗すると
    build() が None を返す。すると main.py は discord_poll を登録しないため、
    **プロセスの生涯にわたってリモコン（緊急停止）が使えない**。
    ポーリングと違って「次回の再試行」が無いので、短く切ってはいけない。
    """

    def _resp(self, payload):
        resp = MagicMock()
        resp.json.return_value = payload
        resp.raise_for_status.return_value = None
        return resp

    def test_get_me_waits_longer_than_the_polling_timeout(self):
        client = mod.DiscordBotClient("tok", "chan")
        with patch.object(mod.requests, "get", return_value=self._resp({"id": "1"})) as req:
            client.get_me()
        connect, read = req.call_args.kwargs["timeout"]
        assert connect > mod.DEFAULT_TIMEOUT[0]
        assert read >= mod.DEFAULT_TIMEOUT[1]

    def test_fetch_bot_role_ids_waits_longer_than_the_polling_timeout(self):
        client = mod.DiscordBotClient("tok", "chan")
        with patch.object(mod.requests, "get",
                          return_value=self._resp({"guild_id": "g1", "roles": ["r1"]})) as req:
            client.fetch_bot_role_ids("999")
        connect, _read = req.call_args.kwargs["timeout"]
        assert connect > mod.DEFAULT_TIMEOUT[0]

    def test_polling_calls_still_use_the_shorter_timeout(self):
        """ポーリング側は短いまま（起動用を流用して長くしない）。"""
        client = mod.DiscordBotClient("tok", "chan")
        with patch.object(mod.requests, "get", return_value=self._resp([])) as req:
            client.fetch_messages()
        assert req.call_args.kwargs["timeout"] == mod.DEFAULT_TIMEOUT


class TestPrimeFailureDoesNotReplayHistory:
    """起動時の既読化に失敗しても、起動前のコマンドを実行しないこと。

    prime() が失敗すると _last_id が None のままになり、次の poll_once が
    after=None で直近20件を取得して**起動前のコマンドを実行**してしまう。
    モジュール冒頭が「起動時点より前のメッセージは実行しない（再起動で過去の
    コマンドが暴発しない）」と保証している性質なので、失敗経路でも守る。
    """

    def _old_command(self):
        return {
            "id": "1", "content": f"<@{BOT_ID}> halt 昨日の理由",
            "author": {"id": OWNER, "bot": False}, "mentions": [{"id": BOT_ID}],
        }

    def _rc_with(self, client):
        self.executed = []
        return mod.RemoteControl(
            client,
            mod.CommandHandler({"halt": lambda a: self.executed.append(a) or "停止"}),
            bot_id=BOT_ID, allowed_user_ids={OWNER},
        )

    def test_history_is_not_executed_after_prime_failure(self):
        client = MagicMock()
        client.fetch_messages.side_effect = [RuntimeError("network down"),
                                             [self._old_command()]]
        rc = self._rc_with(client)
        rc.prime()          # 既読化に失敗
        rc.poll_once()      # 直後のポーリング

        assert self.executed == [], "起動前のコマンドを実行してはいけない"

    def test_history_is_marked_read_so_new_commands_still_work(self):
        """既読化のやり直しは行い、以降の新着はちゃんと動くこと（機能を殺さない）。"""
        new_command = {
            "id": "2", "content": f"<@{BOT_ID}> halt 今の理由",
            "author": {"id": OWNER, "bot": False}, "mentions": [{"id": BOT_ID}],
        }
        client = MagicMock()
        client.fetch_messages.side_effect = [RuntimeError("network down"),
                                             [self._old_command()],
                                             [new_command]]
        rc = self._rc_with(client)
        rc.prime()
        rc.poll_once()      # ここで過去分を既読化するだけ
        rc.poll_once()      # 以降は通常動作

        assert self.executed == ["今の理由"]
        assert client.fetch_messages.call_args.kwargs["after"] == "1"

    def test_prime_uses_the_startup_timeout(self):
        """prime も再試行の無い起動時1回きりの呼び出しなので、短く切らない。"""
        client = mod.DiscordBotClient("tok", "chan")
        resp = MagicMock()
        resp.json.return_value = []
        resp.raise_for_status.return_value = None
        with patch.object(mod.requests, "get", return_value=resp) as req:
            client.fetch_messages(limit=1, startup=True)
        assert req.call_args.kwargs["timeout"] == mod.STARTUP_TIMEOUT


def _snowflake(offset_ms: int = 0) -> str:
    """Discord のスノーフレークID（上位42bitが 2015-01-01 起点のミリ秒）を作る。"""
    import time as _t
    ms = int(_t.time() * 1000) + offset_ms
    return str((ms - mod.DISCORD_EPOCH_MS) << 22)


class TestPrimeFailureKeepsPostStartupCommands:
    """prime 失敗の受け皿が、起動**後**に届いたコマンドまで捨てないこと。

    捨てたいのは起動前の分だけ。再起動直後の瞬断で prime が失敗し、その直後に
    運用者が緊急停止を送ると、無言で破棄され「halt が効いた」と誤認する。
    """

    def _rc(self, client):
        self.executed = []
        return mod.RemoteControl(
            client,
            mod.CommandHandler({"halt": lambda a: self.executed.append(a) or "停止"}),
            bot_id=BOT_ID, allowed_user_ids={OWNER},
        )

    def _cmd(self, msg_id, arg):
        return {"id": msg_id, "content": f"<@{BOT_ID}> halt {arg}",
                "author": {"id": OWNER, "bot": False}, "mentions": [{"id": BOT_ID}]}

    def test_command_sent_after_startup_is_executed(self):
        client = MagicMock()
        rc = self._rc(client)                       # ここで起動時刻が決まる
        fresh = self._cmd(_snowflake(offset_ms=1000), "今すぐ")
        client.fetch_messages.side_effect = [RuntimeError("network down"), [fresh]]

        rc.prime()
        rc.poll_once()

        assert self.executed == ["今すぐ"], "起動後に届いたコマンドは実行する"

    def test_command_sent_before_startup_is_discarded(self):
        client = MagicMock()
        rc = self._rc(client)
        old = self._cmd(_snowflake(offset_ms=-3600_000), "1時間前")
        client.fetch_messages.side_effect = [RuntimeError("network down"), [old]]

        rc.prime()
        rc.poll_once()

        assert self.executed == [], "起動前のコマンドは実行しない"

    def test_mixed_batch_runs_only_the_new_one(self):
        client = MagicMock()
        rc = self._rc(client)
        old = self._cmd(_snowflake(offset_ms=-3600_000), "1時間前")
        fresh = self._cmd(_snowflake(offset_ms=1000), "今すぐ")
        client.fetch_messages.side_effect = [RuntimeError("network down"), [old, fresh]]

        rc.prime()
        rc.poll_once()

        assert self.executed == ["今すぐ"]


class TestReadPositionNeverRewinds:
    """既読位置を巻き戻さないこと。

    巻き戻すと、破棄したはずのメッセージを次回のポーリングで取り直し、
    今度は _primed=True なので**実行してしまう**。
    """

    def _rc(self, client):
        self.executed = []
        return mod.RemoteControl(
            client,
            mod.CommandHandler({"halt": lambda a: self.executed.append(a) or "停止"}),
            bot_id=BOT_ID, allowed_user_ids={OWNER},
        )

    def _cmd(self, msg_id, arg):
        return {"id": msg_id, "content": f"<@{BOT_ID}> halt {arg}",
                "author": {"id": OWNER, "bot": False}, "mentions": [{"id": BOT_ID}]}

    def test_read_position_advances_past_discarded_messages(self):
        """除外した分を含め、既読位置はバッチ末尾まで進むこと。

        途中で止まると、除外したメッセージを次回また取得し、そのときは
        _primed=True なので**実行してしまう**。
        """
        client = MagicMock()
        rc = self._rc(client)
        old1 = self._cmd(_snowflake(offset_ms=-7200_000), "2時間前")
        old2 = self._cmd(_snowflake(offset_ms=-3600_000), "1時間前")
        client.fetch_messages.side_effect = [
            RuntimeError("network down"), [old1, old2], [],
        ]
        rc.prime()
        rc.poll_once()
        rc.poll_once()

        assert self.executed == [], "起動前のコマンドは実行しない"
        assert client.fetch_messages.call_args.kwargs["after"] == old2["id"],             "既読位置がバッチ末尾まで進んでいない（次回また取得してしまう）"

    def test_missing_id_does_not_crash(self):
        client = MagicMock()
        rc = self._rc(client)
        no_id = {"content": f"<@{BOT_ID}> halt x",
                 "author": {"id": OWNER, "bot": False}, "mentions": [{"id": BOT_ID}]}
        client.fetch_messages.side_effect = [RuntimeError("down"), [no_id]]
        rc.prime()
        rc.poll_once()   # KeyError で落ちないこと


class TestStartupCutoffTolerance:
    def test_has_a_margin_for_clock_skew(self):
        """ホスト時計とDiscordサーバ時刻のズレを吸収する余裕があること。

        厳密比較だと、端末の時計が進んでいる場合に起動**後**の緊急停止を
        取りこぼす（＝halt が効いたと誤認させる）。
        """
        assert mod.STARTUP_CUTOFF_MARGIN_MS > 0


class TestStartupTimeIsProcessStart:
    """起動時刻の基準は「プロセス起動」であって「RemoteControl 生成」ではない。

    build() は get_me / fetch_bot_role_ids を叩いてから RemoteControl を作る。
    ネットワークが劣化していると（各10秒×3リクエスト）生成は数十秒後になり、
    同じ障害で prime() も失敗する。生成時刻を基準にすると、その間に送られた
    緊急停止が「起動前」に分類されて無言で捨てられる。
    """

    def _rc(self, client, **kw):
        self.executed = []
        return mod.RemoteControl(
            client,
            mod.CommandHandler({"halt": lambda a: self.executed.append(a) or "停止"}),
            bot_id=BOT_ID, allowed_user_ids={OWNER}, **kw,
        )

    def _cmd(self, msg_id, arg):
        return {"id": msg_id, "content": f"<@{BOT_ID}> halt {arg}",
                "author": {"id": OWNER, "bot": False}, "mentions": [{"id": BOT_ID}]}

    def test_default_start_time_is_process_start_not_construction(self):
        rc = self._rc(MagicMock())
        assert rc._started_at_ms == mod.PROCESS_START_MS

    def test_command_sent_during_a_slow_handshake_is_executed(self):
        """ハンドシェイク中（起動後・生成前）に届いたコマンドは実行すること。"""
        import time as _t
        started = int(_t.time() * 1000) - 120_000          # 2分前にプロセス起動
        client = MagicMock()
        rc = self._rc(client, started_at_ms=started)
        during = self._cmd(_snowflake(offset_ms=-60_000), "起動直後")  # 1分前＝起動後
        client.fetch_messages.side_effect = [RuntimeError("down"), [during]]

        rc.prime()
        rc.poll_once()

        assert self.executed == ["起動直後"]

    def test_command_sent_before_process_start_is_still_discarded(self):
        import time as _t
        started = int(_t.time() * 1000) - 120_000
        client = MagicMock()
        rc = self._rc(client, started_at_ms=started)
        before = self._cmd(_snowflake(offset_ms=-300_000), "5分前")   # 起動より前
        client.fetch_messages.side_effect = [RuntimeError("down"), [before]]

        rc.prime()
        rc.poll_once()

        assert self.executed == []


class TestUnreadablePositionKeepsGuard:
    """既読位置を確定できなければ、既読化済みにしないこと。

    確定しないまま _primed=True にすると、次回 after=None で同じバッチを取り直し、
    今度は破棄ロジックを通らずに起動前のコマンドを実行してしまう。
    """

    def test_batch_without_ids_does_not_disable_the_guard(self):
        executed = []
        no_id = {"content": f"<@{BOT_ID}> halt 過去分",
                 "author": {"id": OWNER, "bot": False}, "mentions": [{"id": BOT_ID}]}
        client = MagicMock()
        client.fetch_messages.side_effect = [
            RuntimeError("down"), [no_id], [no_id],
        ]
        rc = mod.RemoteControl(
            client,
            mod.CommandHandler({"halt": lambda a: executed.append(a) or "停止"}),
            bot_id=BOT_ID, allowed_user_ids={OWNER},
        )
        rc.prime()
        rc.poll_once()
        rc.poll_once()

        assert executed == [], "既読位置が確定できない間は起動前扱いを続ける"


class TestPrimeWithoutReadPositionKeepsGuard:
    """prime() が既読位置を確定できなければ、既読化済みにしないこと。

    poll_once 側は同じ条件を防いでいるのに prime 側が素通りだと、
    次の poll_once が after=None で直近20件を取得し、破棄ロジックを
    通らずに起動前の halt 等を実行してしまう。
    """

    def test_message_without_id_does_not_disable_the_guard(self):
        executed = []
        no_id = {"content": f"<@{BOT_ID}> halt 前日分",
                 "author": {"id": OWNER, "bot": False}, "mentions": [{"id": BOT_ID}]}
        old = {"id": _snowflake(offset_ms=-86_400_000),
               "content": f"<@{BOT_ID}> halt 前日分",
               "author": {"id": OWNER, "bot": False}, "mentions": [{"id": BOT_ID}]}
        client = MagicMock()
        # prime は成功するがIDが無い -> 既読位置が確定しない
        client.fetch_messages.side_effect = [[no_id], [old]]
        rc = mod.RemoteControl(
            client,
            mod.CommandHandler({"halt": lambda a: executed.append(a) or "停止"}),
            bot_id=BOT_ID, allowed_user_ids={OWNER},
        )
        rc.prime()
        rc.poll_once()

        assert executed == [], "既読位置が確定していない間は起動前扱いを続ける"

    def test_empty_channel_still_counts_as_primed(self):
        """空チャンネルは既読化済みとして扱う（以降の新着は普通に動く）。"""
        executed = []
        fresh = {"id": _snowflake(offset_ms=1000),
                 "content": f"<@{BOT_ID}> halt 今すぐ",
                 "author": {"id": OWNER, "bot": False}, "mentions": [{"id": BOT_ID}]}
        client = MagicMock()
        client.fetch_messages.side_effect = [[], [fresh]]
        rc = mod.RemoteControl(
            client,
            mod.CommandHandler({"halt": lambda a: executed.append(a) or "停止"}),
            bot_id=BOT_ID, allowed_user_ids={OWNER},
        )
        rc.prime()
        rc.poll_once()

        assert executed == ["今すぐ"]


class TestUnconfiguredAllowListFailsClosed:
    """許可ユーザーが未設定なら、誰のコマンドも実行しないこと（fail-closed）。

    `DISCORD_ALLOWED_USER_ID` 未設定だと main.py は空セットを渡す。
    `if self._allowed and ...` という書き方だと**判定ごとスキップ**され、
    同じチャンネルにいる誰でも halt / resume を実行できてしまう。
    モジュール冒頭が謳う「送信者IDが許可リストと一致するメッセージだけを実行する」
    「未設定なら機能ごと無効」の両方に反する。
    """

    def test_poll_once_executes_nothing_when_allow_list_is_empty(self):
        executed = []
        client = MagicMock()
        client.fetch_messages.return_value = []
        rc = mod.RemoteControl(
            client,
            mod.CommandHandler({"halt": lambda a: executed.append(a) or "停止"}),
            bot_id=BOT_ID, allowed_user_ids=set(),
        )
        rc.prime()
        client.fetch_messages.return_value = [
            _msg("1", f"<@{BOT_ID}> halt 誰でも実行できてはいけない", author_id=STRANGER)
        ]

        assert rc.poll_once() == 0
        assert executed == []
        client.send.assert_not_called()

    def test_build_disables_the_feature_when_allow_list_is_empty(self):
        """未設定なら機能ごと無効にする（設定漏れで開放状態にしない）。"""
        with patch.object(mod.DiscordBotClient, "get_me", return_value={"id": BOT_ID}):
            assert mod.build("token", "chan", set(), {"status": lambda a: "OK"}) is None

    def test_build_ignores_blank_user_ids(self):
        """空文字だけの指定も未設定と同じ扱いにする（main.py は {""} を渡す）。"""
        with patch.object(mod.DiscordBotClient, "get_me", return_value={"id": BOT_ID}):
            assert mod.build("token", "chan", {""}, {"status": lambda a: "OK"}) is None
