"""異常時アラート通知（複数プロバイダ対応）

LINE Notify は提供元により2025年3月31日に終了したため、本モジュールはこれを廃止し、
Webhook型の通知プロバイダを複数並行で扱える抽象に置き換えた。
新しい通知先を追加する場合は AlertProvider を満たすクラスを実装し、
build_providers() に「設定されていれば追加する」分岐を1つ追加すればよい
（alert() および呼び出し側は無改修で済む）。
"""
import time
from typing import Protocol

import requests
from loguru import logger

from src.core import config as cfg

# 通知送信の再試行。通知は「届かなければ存在しないのと同じ」ため、一時的な失敗を
# 1回で諦めない。2026-09-02 の 8:45 のハートビートは DNS解決失敗
# (getaddrinfo failed) で送信できず、再送も無く消えた。ハートビートは
# 「来ないこと自体が異常」という前提で運用しているため、送信失敗を握り潰すと
# その前提が崩れる（実際この日は障害が起きていたが⚪も🔴も届かなかった）。
ALERT_SEND_ATTEMPTS = 3
ALERT_RETRY_BASE_DELAY = 2.0  # 2秒 → 4秒（最大6秒の遅延で収める）

# Discordのメッセージ本文上限（プレーンテキストの上限）。超過分は安全側で切り詰め、
# 末尾に省略マークを付ける（送信エラーで通知が完全に消えるより、要点が欠けても
# 通知自体が届く方を優先する）。
DISCORD_MAX_CONTENT_LENGTH = 2000
_TRUNCATION_SUFFIX = "…(省略)"


class AlertProvider(Protocol):
    """通知プロバイダの最小インターフェース。"""

    name: str

    def send(self, message: str) -> None:
        """メッセージを送信する。失敗時は例外を投げてよい（alert()側が捕捉する）。"""
        ...


class DiscordWebhookProvider:
    """Discord Webhook へメッセージを送信するプロバイダ。

    認証ヘッダーは不要（Webhook URL自体が秘密情報）。POST <url> に
    {"content": "<text>"} をJSONで送る。
    """

    name = "discord"

    def __init__(self, webhook_url: str, timeout: int = 10):
        self._webhook_url = webhook_url
        self._timeout = timeout

    def send(self, message: str) -> None:
        text = message
        if len(text) > DISCORD_MAX_CONTENT_LENGTH:
            text = text[: DISCORD_MAX_CONTENT_LENGTH - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX
        resp = requests.post(
            self._webhook_url,
            json={"content": text},
            timeout=self._timeout,
        )
        resp.raise_for_status()


def build_providers() -> list[AlertProvider]:
    """config から有効な通知プロバイダの一覧を構築する。

    将来プロバイダを追加する場合はここに分岐を1つ追加するだけでよい
    （例: alerts.slack_webhook_url が設定されていれば SlackWebhookProvider を追加）。
    """
    section = cfg.get_section("alerts")
    providers: list[AlertProvider] = []

    discord_url = section.get("discord_webhook_url", "")
    if discord_url:
        providers.append(DiscordWebhookProvider(discord_url))

    return providers


def _is_retryable(exc: Exception) -> bool:
    """再送で直る見込みがある失敗か。

    接続系（DNS解決失敗・タイムアウト・切断）とサーバー側の一時障害（5xx / 429）は
    再送する価値がある。400・401・404（Webhook URLの誤り・失効・削除）は
    何度送っても直らないので即あきらめ、無駄な遅延を作らない。
    """
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    if status is None:
        return True  # 接続系はレスポンスが無い
    return status >= 500 or status == 429


def _send_one(provider: AlertProvider, message: str, *,
              attempts: int = ALERT_SEND_ATTEMPTS) -> bool:
    """1プロバイダへ送信する。一時的な失敗は指数バックオフで再送する。

    戻り値は成功したか。呼び出し側（alert）は他プロバイダの送信を続けるため、
    ここで例外は投げない。
    """
    last_error: Exception = RuntimeError("未送信")
    for i in range(attempts):
        try:
            provider.send(message)
            if i:
                logger.info(f"通知を再送で送信できました（{provider.name}・{i + 1}回目）")
            return True
        except Exception as e:
            last_error = e
            if not _is_retryable(e) or i == attempts - 1:
                break
            # time.sleep をモジュール経由で呼ぶ（既定引数に束縛するとテストで
            # 差し替えられず、実時間だけ待つテストになる）
            time.sleep(ALERT_RETRY_BASE_DELAY * (2 ** i))
    # URL等の秘密情報を含みうる属性は出さず、プロバイダ名のみログに残す
    logger.error(f"通知送信失敗（{provider.name}）: {last_error}")
    return False


# 通知の重要度。スマホの通知一覧で「対応が要るか」を記号だけで判断できるようにする。
# 段階を増やすと結局読み分けなくなるため、行動が変わる4段階に絞る
# （アラート疲れ対策の定石: すべてのアラートは行動を要求すべき／要求しないものは目印で下げる）。
LEVEL_CRITICAL = "critical"   # 要対応: 再ログイン・未解決注文・整合性違反
LEVEL_WARNING = "warning"     # 注意: 損失上限接近・取引停止中
LEVEL_INFO = "info"           # 実行報告: 約定・接続回復
LEVEL_ROUTINE = "routine"     # 定期: ハートビート・日次/週次レポート

_LEVEL_MARKS = {
    LEVEL_CRITICAL: "🔴",
    LEVEL_WARNING: "🟡",
    LEVEL_INFO: "🟢",
    LEVEL_ROUTINE: "⚪",
}


def alert(title: str, message: str, level: str = LEVEL_CRITICAL) -> None:
    """異常・重要イベントを通知する（公開インターフェース。呼び出し側はこれだけ使う）。

    必ずログへ記録した上で、設定済みの全プロバイダへ送信を試みる。
    1つのプロバイダが失敗しても他のプロバイダへの送信は継続する。

    level は本文先頭に付ける記号を決めるだけで、送信先や可否は変えない
    （既定は critical。従来の呼び出しは引数なしでそのまま動く）。
    """
    logger.warning(f"[ALERT] {title}: {message}")

    providers = build_providers()
    if not providers:
        return

    mark = _LEVEL_MARKS.get(level, _LEVEL_MARKS[LEVEL_CRITICAL])
    text = f"{mark}【kabu-auto】{title}\n{message}"
    for provider in providers:
        _send_one(provider, text)
