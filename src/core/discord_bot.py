"""Discord からの遠隔操作（ポーリング方式のリモコン）。

外出先から状態確認・緊急停止を行うための最小限の受信口。ポート開放が不要
（アウトバウンドのみ）で、Discordアプリがそのままリモコンになる。

## 設計方針

- **Botへのメンションのみを読む**。Discord の Message Content は特権intentだが、
  「Botへのメンション」「BotへのDM」は intent 無しでも本文を取得できる
  （公式ドキュメントの例外規定）。これにより Developer Portal での特権申請が不要。
- **秘密情報を運ばせない**。`.env` の書き込みやパスワード・トークンの送受信は
  実装しない。Discordのメッセージ履歴に認証情報が残る構成は、たとえ自分専用の
  チャンネルでも作らない（漏洩時の影響が口座操作に直結するため）。
- **緊急全決済（emergency_close）は対象外**。誤爆時の損害が大きすぎるため、
  ダッシュボード（要トークン）に限定する。停止(halt)は安全側なので許可する。
- **未設定なら機能ごと無効**。トークン未設定時は何もしない（既存挙動と完全一致）。

## セキュリティ

- 送信者IDが許可リストと一致するメッセージだけを実行する（同じチャンネルに
  他人がいても操作できない）
- 指定チャンネル以外は見ない
- 起動時点より前のメッセージは実行しない（再起動で過去のコマンドが暴発しない）

REST API のみを使い、新規依存は無い（既存の requests を使う）。
"""
from typing import Callable, Optional

import requests
from loguru import logger

API_BASE = "https://discord.com/api/v10"
MAX_REPLY_LENGTH = 1900  # Discordの2000字制限に対する余裕


class DiscordBotClient:
    """Discord REST API の薄いラッパー（取得と返信のみ）。"""

    def __init__(self, token: str, channel_id: str, timeout: int = 10):
        self._token = token
        self._channel_id = channel_id
        self._timeout = timeout

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bot {self._token}"}

    def get_me(self) -> dict:
        resp = requests.get(f"{API_BASE}/users/@me", headers=self._headers,
                            timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def fetch_messages(self, after: Optional[str] = None, limit: int = 20) -> list:
        params = {"limit": limit}
        if after:
            params["after"] = after
        resp = requests.get(
            f"{API_BASE}/channels/{self._channel_id}/messages",
            headers=self._headers, params=params, timeout=self._timeout,
        )
        resp.raise_for_status()
        # Discord は新しい順で返すため、処理しやすいよう古い順に直す
        return list(reversed(resp.json()))

    def send(self, content: str) -> None:
        if len(content) > MAX_REPLY_LENGTH:
            content = content[:MAX_REPLY_LENGTH] + "…(略)"
        resp = requests.post(
            f"{API_BASE}/channels/{self._channel_id}/messages",
            headers=self._headers, json={"content": content}, timeout=self._timeout,
        )
        resp.raise_for_status()


def strip_mention(content: str, bot_id: str) -> str:
    """メッセージ先頭のBotメンションを取り除いてコマンド部分を返す。

    Discord のメンションは `<@123>` または `<@!123>`（ニックネーム時）の形式。
    """
    for prefix in (f"<@{bot_id}>", f"<@!{bot_id}>"):
        if content.startswith(prefix):
            return content[len(prefix):].strip()
    return ""


def is_mentioned(message: dict, bot_id: str) -> bool:
    """このメッセージがBot宛のメンションか。"""
    return any(u.get("id") == bot_id for u in message.get("mentions", []))


class CommandHandler:
    """コマンド文字列を解釈して実行する。Discord API を知らない（テスト容易性）。

    handlers は {コマンド名: 関数(args:str)->str} 。関数は返信文字列を返す。
    """

    def __init__(self, handlers: dict):
        self._handlers = handlers

    def help_text(self) -> str:
        names = "  ".join(sorted(self._handlers))
        return f"利用できるコマンド: {names}"

    def execute(self, command_line: str) -> Optional[str]:
        """コマンドを実行して返信文を返す。空・未知のコマンドは案内を返す。"""
        text = (command_line or "").strip()
        if not text:
            return None
        parts = text.split(maxsplit=1)
        name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        if name in ("help", "?"):
            return self.help_text()
        fn = self._handlers.get(name)
        if fn is None:
            return f"不明なコマンド: {name}\n{self.help_text()}"
        try:
            return fn(args)
        except Exception as e:
            logger.error(f"Discordコマンド実行エラー({name}): {e}")
            return f"コマンド実行に失敗しました: {e}"


class RemoteControl:
    """ポーリングして許可された相手のコマンドだけを実行する。

    poll_once() を定期ジョブから呼ぶ。状態（最後に見たメッセージID）は
    インスタンスが保持し、永続化はしない（再起動時は「起動後の分だけ」見る）。
    """

    def __init__(self, client: DiscordBotClient, handler: CommandHandler,
                 *, bot_id: str, allowed_user_ids: set):
        self._client = client
        self._handler = handler
        self._bot_id = bot_id
        self._allowed = {str(u) for u in allowed_user_ids if str(u).strip()}
        self._last_id: Optional[str] = None

    def prime(self) -> None:
        """起動時に既存メッセージを既読扱いにする（過去コマンドの暴発防止）。"""
        try:
            messages = self._client.fetch_messages(limit=1)
            if messages:
                self._last_id = messages[-1]["id"]
        except Exception as e:
            logger.warning(f"Discordリモコンの初期化に失敗（次回再試行）: {e}")

    def poll_once(self) -> int:
        """新着を1回分処理する。処理したコマンド数を返す。"""
        try:
            messages = self._client.fetch_messages(after=self._last_id)
        except Exception as e:
            logger.warning(f"Discordメッセージ取得に失敗（次回再試行）: {e}")
            return 0

        executed = 0
        for msg in messages:
            self._last_id = msg["id"]
            author = msg.get("author", {}) or {}
            if author.get("bot"):
                continue  # 自分やほかのBotの発言は無視（無限ループ防止）
            if not is_mentioned(msg, self._bot_id):
                continue
            if self._allowed and str(author.get("id")) not in self._allowed:
                logger.warning(
                    f"許可されていないユーザーからのDiscordコマンドを拒否しました: "
                    f"user_id={author.get('id')}"
                )
                continue
            command = strip_mention(msg.get("content", ""), self._bot_id)
            reply = self._handler.execute(command)
            if reply is None:
                continue
            executed += 1
            try:
                self._client.send(reply)
            except Exception as e:
                logger.error(f"Discordへの返信に失敗しました: {e}")
        return executed


def build(token: str, channel_id: str, allowed_user_ids: set,
          handlers: dict) -> Optional[RemoteControl]:
    """設定からリモコンを構築する。未設定・接続失敗なら None（機能無効）。"""
    if not token or not channel_id:
        return None
    client = DiscordBotClient(token, channel_id)
    try:
        me = client.get_me()
    except Exception as e:
        logger.warning(f"Discordリモコンを初期化できませんでした（無効化して継続）: {e}")
        return None
    bot_id = str(me.get("id", ""))
    if not bot_id:
        return None
    rc = RemoteControl(client, CommandHandler(handlers),
                       bot_id=bot_id, allowed_user_ids=allowed_user_ids)
    rc.prime()
    logger.info(f"Discordリモコンを有効化しました（bot={me.get('username')}）")
    return rc
