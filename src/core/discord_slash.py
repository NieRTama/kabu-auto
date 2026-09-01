"""Discord スラッシュコマンド（Gateway 常時接続）。

## 位置づけ

既存の `discord_bot.py`（30秒ポーリング＋メンション）と**併存**する。
どちらも同じコマンド関数を呼ぶだけで、実装は共有される。

| | メンション方式 | スラッシュ方式（本モジュール） |
|---|---|---|
| 受信 | REST を30秒ポーリング | Gateway 常時接続 |
| 応答 | 最大30秒 | 即時（3秒以内） |
| 入力 | 補完でロールを選ぶ誤りが起きた | `/` で候補が出る |
| 依存 | なし | discord.py（任意） |

メンション方式を残すのは、Gateway 切断中でも操作できる経路を確保するため
（新しい常時接続は新しい障害点でもある）。

## ポート開放は不要

スラッシュコマンドの受信方法は「HTTP Interactions Endpoint」と「Gateway」の
排他2択で、**Gateway を選べば公開HTTPSは不要**（公式ドキュメント:
"The INTERACTION_CREATE Gateway Event may be handled by connected clients"）。
本実装は Gateway 方式を使う。

## 注意点

- Discord の制約により **3秒以内に応答**しなければならない。外部APIを叩く
  コマンド（launch / reconnect）は `defer()` してから結果を送る
- コマンドはギルド単位で登録する（グローバル登録は反映に最大1時間かかる）
- discord.py は asyncio ベースのため専用スレッドで独自ループを回し、
  既存の同期コード（APScheduler / uvicorn）に影響を与えない
"""
import asyncio
import threading
from typing import Callable, Optional

from loguru import logger

# 応答に時間がかかる可能性があるコマンド（先に defer して3秒制約を回避する）
_SLOW_COMMANDS = {"launch", "reconnect"}

MAX_REPLY_LENGTH = 1900  # Discordの2000字制限に対する余裕


def available() -> bool:
    """discord.py が導入されているか（未導入ならスラッシュ機能を無効化する）。"""
    try:
        import discord  # noqa: F401
        return True
    except ImportError:
        return False


class SlashCommandServer:
    """Gateway に接続してスラッシュコマンドを処理する。

    start() で専用スレッドを立ち上げ、以後はそのスレッド内で動く。
    停止は stop()（プロセス終了時は daemon スレッドなので放置でもよい）。
    """

    def __init__(self, token: str, channel_id: str, allowed_user_ids: set,
                 handlers: dict):
        self._token = token
        self._channel_id = str(channel_id)
        self._allowed = {str(u) for u in allowed_user_ids if str(u).strip()}
        # handlers は {名前: 関数} または {名前: (関数, 説明)}
        self._handlers: dict = {}
        self._descriptions: dict = {}
        for name, spec in handlers.items():
            fn, desc = spec if isinstance(spec, tuple) else (spec, "")
            self._handlers[name] = fn
            self._descriptions[name] = desc or name
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client = None

    # ─── 認可 ────────────────────────────────────────────
    def _is_authorized(self, user_id: str, channel_id: str) -> tuple[bool, str]:
        """実行してよい相手・場所か。メンション方式と同じ基準を使う。"""
        if str(channel_id) != self._channel_id:
            return False, "このチャンネルでは実行できません"
        if self._allowed and str(user_id) not in self._allowed:
            logger.warning(
                f"許可されていないユーザーのスラッシュコマンドを拒否しました: user_id={user_id}"
            )
            return False, "このコマンドを実行する権限がありません"
        return True, ""

    def _run_handler(self, name: str, args: str) -> str:
        """コマンド本体を実行する（例外は文字列にして返し、Botを落とさない）。"""
        fn = self._handlers.get(name)
        if fn is None:
            return f"不明なコマンド: {name}"
        try:
            return fn(args) or "(応答なし)"
        except Exception as e:
            logger.error(f"スラッシュコマンド実行エラー({name}): {e}")
            return f"コマンド実行に失敗しました: {e}"

    # ─── 起動・停止 ──────────────────────────────────────
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="discord-slash",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._loop and self._client:
            asyncio.run_coroutine_threadsafe(self._client.close(), self._loop)

    def _run(self) -> None:
        """専用スレッドで独自イベントループを回す。"""
        import discord
        from discord import app_commands

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        intents = discord.Intents.default()
        client = discord.Client(intents=intents)
        tree = app_commands.CommandTree(client)
        self._client = client
        server = self

        def _make_callback(cmd_name: str):
            async def callback(interaction: "discord.Interaction",
                               args: Optional[str] = None):
                ok, reason = server._is_authorized(
                    interaction.user.id, interaction.channel_id)
                if not ok:
                    await interaction.response.send_message(reason, ephemeral=True)
                    return
                args = args or ""
                # 3秒以内に応答できない可能性があるものは先に defer する
                if cmd_name in _SLOW_COMMANDS:
                    await interaction.response.defer(thinking=True)
                    text = await asyncio.to_thread(server._run_handler, cmd_name, args)
                    await interaction.followup.send(text[:MAX_REPLY_LENGTH])
                else:
                    text = await asyncio.to_thread(server._run_handler, cmd_name, args)
                    await interaction.response.send_message(text[:MAX_REPLY_LENGTH])
            return callback

        for name, desc in self._descriptions.items():
            tree.command(name=name, description=desc[:100])(_make_callback(name))

        @client.event
        async def on_ready():
            try:
                channel = client.get_channel(int(self._channel_id)) \
                    or await client.fetch_channel(int(self._channel_id))
                guild = discord.Object(id=channel.guild.id)
                # ギルド単位で登録すると即時反映される（グローバルは最大1時間）
                tree.copy_global_to(guild=guild)
                await tree.sync(guild=guild)
                logger.info(
                    f"Discordスラッシュコマンドを登録しました"
                    f"（{len(self._descriptions)}件・bot={client.user}）"
                )
            except Exception as e:
                logger.error(f"スラッシュコマンドの登録に失敗しました: {e}")

        try:
            loop.run_until_complete(client.start(self._token))
        except Exception as e:
            logger.warning(f"Discordスラッシュ接続が終了しました: {e}")
        finally:
            loop.close()


def build(token: str, channel_id: str, allowed_user_ids: set,
          handlers: dict) -> Optional[SlashCommandServer]:
    """設定からスラッシュコマンドサーバを構築する。

    未設定・discord.py 未導入なら None（機能ごと無効。既存挙動と完全一致）。
    """
    if not token or not channel_id:
        return None
    if not available():
        logger.info(
            "discord.py が未導入のためスラッシュコマンドは無効です"
            "（メンション方式は引き続き利用できます）"
        )
        return None
    server = SlashCommandServer(token, channel_id, allowed_user_ids, handlers)
    server.start()
    return server
