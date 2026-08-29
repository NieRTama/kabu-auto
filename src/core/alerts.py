"""異常時アラート通知（複数プロバイダ対応）

LINE Notify は提供元により2025年3月31日に終了したため、本モジュールはこれを廃止し、
Webhook型の通知プロバイダを複数並行で扱える抽象に置き換えた。
新しい通知先を追加する場合は AlertProvider を満たすクラスを実装し、
build_providers() に「設定されていれば追加する」分岐を1つ追加すればよい
（alert() および呼び出し側は無改修で済む）。
"""
from typing import Protocol

import requests
from loguru import logger

from src.core import config as cfg

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


def _send_one(provider: AlertProvider, message: str) -> None:
    try:
        provider.send(message)
    except Exception as e:
        # URL等の秘密情報を含みうる属性は出さず、プロバイダ名のみログに残す
        logger.error(f"通知送信失敗（{provider.name}）: {e}")


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
