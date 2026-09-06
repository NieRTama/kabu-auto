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
import time
from typing import Callable, Optional

import requests
from loguru import logger

API_BASE = "https://discord.com/api/v10"
MAX_REPLY_LENGTH = 1900  # Discordの2000字制限に対する余裕

# HTTPの待ち時間の上限（接続, 読み取り）。接続待ちを読み取りより短く取る。
#
# 背景: 2026-09-06 のネットワーク断で fetch_messages が約53秒かかり、30秒間隔の
# ポーリングジョブが APScheduler にスキップされた
# （"skipped: maximum number of running instances reached (1)"）。
#
# requests の timeout は「**接続1回あたり**」の上限であって、呼び出し全体の上限ではない。
# 接続は候補IP（IPv6/IPv4の複数レコード）ごとに順に試されるため、接続側を短くすると
# 到達できないIPで足止めされる時間が縮む。
#
# ただしこれは**発生確率を下げるだけで、超過を防ぐ保証にはならない**:
#   - DNS解決(getaddrinfo)はこの上限の対象外で、OS側の解決待ちは止められない
#   - 候補IPの数は環境依存（IPv6有効時はさらに増える）
#   - poll_once() は取得1回に加えて返信ごとに send() を呼ぶため、全体の上限はさらに緩い
# 超過そのものは起こりうる前提で、**本命の対策は超過に気づけるようにすること**
# （APScheduler の標準logging を loguru へ橋渡しし、スキップが**ログに残る**ようにした。
#   src/core/logger.py の _InterceptHandler。なおスキップは routine にも出るため
#   アラート件数には載せない＝通知は飛ばない。ログを見れば必ず残っている、が担保内容）。
CONNECT_TIMEOUT_SECONDS = 4
READ_TIMEOUT_SECONDS = 10
DEFAULT_TIMEOUT = (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)

# 起動時の1回きりの呼び出し（get_me / fetch_bot_role_ids / prime）専用の上限。
#
# これらは失敗しても再試行が無い。get_me が失敗すると build() が None を返し、
# main.py は discord_poll ジョブを登録しないため、**プロセスの生涯にわたって
# リモコン（外出先からの緊急停止）が使えない**。よってポーリングと同じ短さでは切らない。
#
# 一方で**これ以上長くもしない**。build() は main.py の同期起動パス上にあり
# （scheduler.start() より前）、ここで待つと損切り監視の開始まで遅れる。
# リモコンの可用性のために取引本体の起動を遅らせるのは本末転倒なので、
# 変更前の挙動（timeout=10）と同じ値に据え置く。
STARTUP_TIMEOUT = (10, 10)

# Discord のメッセージIDはスノーフレーク: 上位42bitが 2015-01-01 起点のミリ秒。
# 「起動より前に送られたメッセージか」を、追加のAPI呼び出し無しに判定できる。
DISCORD_EPOCH_MS = 1420070400000


# ホストの時計と Discord サーバの時刻はズレる。厳密比較にすると、端末の時計が
# 進んでいるときに起動**後**に送られた緊急停止を取りこぼし、返信も無いまま消える
# （「halt が効いた」と誤認させる）。数秒の余裕を持たせて安全側に倒す。
# 大きくしすぎると起動直前のコマンド（resume 等）を拾ってしまうので短く保つ。
STARTUP_CUTOFF_MARGIN_MS = 5000

# プロセス起動時刻。**このモジュールの読み込み時**（＝main.py の import 時）に確定する。
# RemoteControl の生成時ではない: build() は get_me / fetch_bot_role_ids を叩いてから
# 生成するため、ネットワーク劣化時は数十秒後になる。生成時刻を基準にすると、その間に
# 送られた緊急停止が「起動前」に分類され、無言で捨てられてしまう。
PROCESS_START_MS = int(time.time() * 1000)


def message_sent_at_ms(message_id) -> Optional[int]:
    """メッセージIDから送信時刻(ミリ秒)を取り出す。解釈できなければ None。"""
    try:
        return (int(message_id) >> 22) + DISCORD_EPOCH_MS
    except (TypeError, ValueError):
        return None


class DiscordBotClient:
    """Discord REST API の薄いラッパー（取得と返信のみ）。"""

    def __init__(self, token: str, channel_id: str, timeout=DEFAULT_TIMEOUT,
                 startup_timeout=STARTUP_TIMEOUT):
        self._token = token
        self._channel_id = channel_id
        self._timeout = timeout
        # 起動時の1回きりの呼び出しは長めに待つ（失敗するとリモコンが生涯無効になる）
        self._startup_timeout = startup_timeout

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bot {self._token}"}

    def get_me(self) -> dict:
        resp = requests.get(f"{API_BASE}/users/@me", headers=self._headers,
                            timeout=self._startup_timeout)
        resp.raise_for_status()
        return resp.json()

    def fetch_messages(self, after: Optional[str] = None, limit: int = 20,
                       startup: bool = False) -> list:
        """新着メッセージを取得する。

        startup=True は起動時の既読化（prime）用。再試行が無いので長めに待つ。
        """
        params = {"limit": limit}
        if after:
            params["after"] = after
        resp = requests.get(
            f"{API_BASE}/channels/{self._channel_id}/messages",
            headers=self._headers, params=params,
            timeout=self._startup_timeout if startup else self._timeout,
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
                              headers=self._headers, timeout=self._startup_timeout)
            ch.raise_for_status()
            guild_id = ch.json().get("guild_id")
            if not guild_id:
                return set()
            me = requests.get(f"{API_BASE}/guilds/{guild_id}/members/{bot_id}",
                              headers=self._headers, timeout=self._startup_timeout)
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
                 role_ids: Optional[set] = None,
                 started_at_ms: Optional[int] = None):
        self._client = client
        self._handler = handler
        self._bot_id = bot_id
        # Discordの補完で「Botのロール」を選ばれることがあるため、そちらも宛先として扱う
        self._role_ids = {str(r) for r in (role_ids or set())}
        self._allowed = {str(u) for u in allowed_user_ids if str(u).strip()}
        self._last_id: Optional[str] = None
        # 起動時の既読化が済んだか。失敗したまま poll_once に入ると、
        # after=None で直近20件を取得して**起動前のコマンドを実行してしまう**
        # （モジュール冒頭の「起動時点より前のメッセージは実行しない」が破れる）。
        self._primed = False
        # prime に失敗したときの切り分けに使う（起動前の分だけを捨てるため）。
        # 既定はプロセス起動時刻。ここで time.time() を呼ぶと、build() の
        # ハンドシェイクに掛かった時間の分だけ基準が後ろへずれる。
        self._started_at_ms = (
            PROCESS_START_MS if started_at_ms is None else started_at_ms
        )

    def prime(self) -> None:
        """起動時に既存メッセージを既読扱いにする（過去コマンドの暴発防止）。"""
        try:
            messages = self._client.fetch_messages(limit=1, startup=True)
            if messages:
                self._last_id = messages[-1].get("id", self._last_id)
            # 既読位置を確定できたとき（または空チャンネル）だけ完了扱いにする。
            # 確定しないまま完了にすると、次の poll_once が after=None で直近20件を
            # 取得し、破棄ロジックを通らずに起動前のコマンドを実行してしまう
            # （poll_once 側の判定と揃える）。
            self._primed = (not messages) or self._last_id is not None
        except Exception as e:
            logger.warning(f"Discordリモコンの初期化に失敗（次回のポーリングで既読化）: {e}")

    def _drop_messages_from_before_startup(self, messages: list) -> list:
        """起動より前に送られたメッセージを既読化して取り除く。

        判定できないID（テスト用の連番など）は安全側に倒して「起動前」とみなす。
        """
        cutoff = self._started_at_ms - STARTUP_CUTOFF_MARGIN_MS
        keep, dropped = [], []
        for msg in messages:
            sent_at = message_sent_at_ms(msg.get("id"))
            if sent_at is not None and sent_at >= cutoff:
                keep.append(msg)
            else:
                dropped.append(msg)
        if dropped:
            # INFO ではなく WARNING。コマンドを送った本人には返信が届かないため、
            # 「送ったのに動いていない」ことに気づける手がかりを残す。
            logger.warning(
                f"起動時の既読化に失敗したため、起動前のDiscordメッセージ"
                f"{len(dropped)}件を実行せず既読化しました"
            )
        return keep

    def poll_once(self) -> int:
        """新着を1回分処理する。処理したコマンド数を返す。"""
        try:
            messages = self._client.fetch_messages(after=self._last_id)
        except Exception as e:
            logger.warning(f"Discordメッセージ取得に失敗（次回再試行）: {e}")
            return 0

        # このバッチは（実行するしないに関わらず）消化する。除外した分で
        # 既読位置が止まると、次回また取得して今度は実行してしまうため、
        # 最後にバッチ末尾まで必ず進める（Discordは時系列順で返す）。
        # 末尾にIDが無い異常データもあり得るので、IDを持つ最後の要素を使う。
        fetched = messages
        batch_last_id = next(
            (m.get("id") for m in reversed(fetched) if m.get("id")), None
        )

        if not self._primed:
            # prime() が失敗していた場合の受け皿。既読位置が無いまま処理に入ると
            # 直近20件をまとめて実行し、再起動のたびに過去の halt 等が暴発する。
            # ただし**捨てるのは起動より前の分だけ**にする。起動直後に送られた
            # 緊急停止まで無言で破棄すると「halt が効いた」と誤認させてしまう。
            messages = self._drop_messages_from_before_startup(fetched)
            # 既読位置を確定できないまま「既読化済み」にすると、次回 after=None で
            # 同じバッチを取り直し、今度は破棄ロジックを通らずに起動前のコマンドを
            # 実行してしまう。確定できたとき（または空バッチ）だけ完了扱いにする。
            self._primed = (not fetched) or batch_last_id is not None

        executed = 0
        for msg in messages:
            self._last_id = msg.get("id", self._last_id)
            author = msg.get("author", {}) or {}
            if author.get("bot"):
                continue  # 自分やほかのBotの発言は無視（無限ループ防止）
            if not is_mentioned(msg, self._bot_id, self._role_ids):
                continue
            # 許可リストが空でも素通りさせない（fail-closed）。`self._allowed and ...`
            # と書くと未設定時に判定ごとスキップされ、同じチャンネルの誰でも
            # halt / resume を実行できてしまう。
            if str(author.get("id")) not in self._allowed:
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

        if batch_last_id is not None:
            self._last_id = batch_last_id
        return executed


def build(token: str, channel_id: str, allowed_user_ids: set,
          handlers: dict) -> Optional[RemoteControl]:
    """設定からリモコンを構築する。未設定・接続失敗なら None（機能無効）。"""
    if not token or not channel_id:
        return None
    # 許可ユーザーが空なら機能ごと無効にする。空のまま動かすと、設定漏れが
    # 「誰でも操作できる」状態として表に出ず、事故になるまで気づけない。
    if not {str(u) for u in (allowed_user_ids or set()) if str(u).strip()}:
        logger.warning(
            "DISCORD_ALLOWED_USER_ID が未設定のためDiscordリモコンを無効にします"
            "（許可ユーザーが空のまま動かすと同じチャンネルの誰でも操作できるため）"
        )
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
