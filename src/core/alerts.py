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


def alert(title: str, message: str) -> None:
    """異常・重要イベントを通知する（公開インターフェース。呼び出し側はこれだけ使う）。

    必ずログへ記録した上で、設定済みの全プロバイダへ送信を試みる。
    1つのプロバイダが失敗しても他のプロバイダへの送信は継続する。
    """
    logger.warning(f"[ALERT] {title}: {message}")

    providers = build_providers()
    if not providers:
        return

    text = f"【kabu-auto】{title}\n{message}"
    for provider in providers:
        _send_one(provider, text)
