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
import re
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

    def fetch_bot_role_ids(self, bot_id: str) -> set:
        """このBotに紐づくロールIDを取得する。

        Discordの入力補完は「Botユーザー」ではなく「Botに紐づくロール」を選ぶことが
        あり、その場合メッセージは <@&ロールID> になる。宛先として認識するために
        自分のロールIDを控えておく（取得できなければ空集合＝従来どおり動く）。
        """
        try:
            ch = requests.get(f"{API_BASE}/channels/{self._channel_id}",
                              headers=self._headers, timeout=self._timeout)
            ch.raise_for_status()
            guild_id = ch.json().get("guild_id")
            if not guild_id:
                return set()
            me = requests.get(f"{API_BASE}/guilds/{guild_id}/members/{bot_id}",
                              headers=self._headers, timeout=self._timeout)
            me.raise_for_status()
            return {str(r) for r in me.json().get("roles", [])}
        except Exception as e:
            logger.warning(f"Botのロール取得に失敗しました（ロールメンションは無効）: {e}")
            return set()

    def send(self, content: str) -> None:
        if len(content) > MAX_REPLY_LENGTH:
            content = content[:MAX_REPLY_LENGTH] + "…(略)"
        resp = requests.post(
            f"{API_BASE}/channels/{self._channel_id}/messages",
            headers=self._headers, json={"content": content}, timeout=self._timeout,
        )
        resp.raise_for_status()


# メンション記法。Discordの入力補完は「Botユーザー」ではなく「Botに紐づくロール」を
# 選ぶことがあり、その場合 <@&ロールID> になる（2026-09-01 に実際に発生し、
# ユーザーメンションだけを見ていたため一切反応しなかった）。
# 利用者に選び分けを強いるのは非現実的なので、両方を受け付ける。
_MENTION_RE = re.compile(r"^<@(?P<role>&)?!?(?P<id>\d+)>")


def strip_mention(content: str, bot_id: str, role_ids: Optional[set] = None) -> str:
    """メッセージ先頭のメンションを取り除いてコマンド部分を返す。

    受け付ける形式:
      <@123>   ユーザー（Bot本体）
      <@!123>  ニックネーム付きユーザー
      <@&456>  ロール（Botに紐づくロールを補完で選んだ場合）

    role_ids を渡すとロールメンションも許可する。宛先が一致しなければ空文字。
    """
    m = _MENTION_RE.match(content or "")
    if not m:
        return ""
    target = m.group("id")
    # ロール記法(<@&ID>)とユーザー記法(<@ID>)は別物として突き合わせる。
    # 混同すると「たまたま同じ数値の別ロール」に反応してしまう。
    if m.group("role"):
        allowed = target in {str(r) for r in (role_ids or set())}
    else:
        allowed = target == str(bot_id)
    return content[m.end():].strip() if allowed else ""


def is_mentioned(message: dict, bot_id: str, role_ids: Optional[set] = None) -> bool:
    """このメッセージがBot宛（ユーザー or Botのロール）のメンションか。"""
    if any(u.get("id") == str(bot_id) for u in message.get("mentions", [])):
        return True
    if not role_ids:
        return False
    mentioned_roles = {str(r) for r in message.get("mention_roles", [])}
    return bool(mentioned_roles & {str(r) for r in role_ids})


class CommandHandler:
    """コマンド文字列を解釈して実行する。Discord API を知らない（テスト容易性）。

    handlers は {コマンド名: 関数} または {コマンド名: (関数, 説明)}。
    説明を添えると help が「何ができるか」まで案内する（名前の羅列だけでは
    外出先で思い出せないため）。関数は返信文字列を返す。
    """

    def __init__(self, handlers: dict):
        self._handlers: dict = {}
        self._descriptions: dict = {}
        for name, spec in handlers.items():
            if isinstance(spec, tuple):
                fn, desc = spec
            else:
                fn, desc = spec, ""
            self._handlers[name] = fn
            self._descriptions[name] = desc

    def help_text(self) -> str:
        """コマンド一覧を説明つきで返す。

        説明が1つも無い場合は名前だけを並べる（説明を持たない使い方との互換）。
        """
        names = sorted(self._handlers)
        if not any(self._descriptions.get(n) for n in names):
            return "利用できるコマンド: " + "  ".join(names)
        width = max(len(n) for n in names)
        lines = ["kabu-auto コマンド一覧（先頭に @kabu-auto を付けて実行）", ""]
        for n in names:
            desc = self._descriptions.get(n, "")
            lines.append(f"  {n.ljust(width)}  {desc}" if desc else f"  {n}")
        return "\n".join(lines)

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
                 *, bot_id: str, allowed_user_ids: set,
                 role_ids: Optional[set] = None):
        self._client = client
        self._handler = handler
        self._bot_id = bot_id
        # Discordの補完で「Botのロール」を選ばれることがあるため、そちらも宛先として扱う
        self._role_ids = {str(r) for r in (role_ids or set())}
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
            if not is_mentioned(msg, self._bot_id, self._role_ids):
                continue
            if self._allowed and str(author.get("id")) not in self._allowed:
                logger.warning(
                    f"許可されていないユーザーからのDiscordコマンドを拒否しました: "
                    f"user_id={author.get('id')}"
                )
                continue
            command = strip_mention(msg.get("content", ""), self._bot_id, self._role_ids)
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
    role_ids = client.fetch_bot_role_ids(bot_id)
    rc = RemoteControl(client, CommandHandler(handlers),
                       bot_id=bot_id, allowed_user_ids=allowed_user_ids,
                       role_ids=role_ids)
    rc.prime()
    logger.info(f"Discordリモコンを有効化しました（bot={me.get('username')}）")
    return rc
