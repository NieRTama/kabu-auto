"""
Discordへの日次損益レポート自動投稿。

src/core/alerts.py の DiscordWebhookProvider をそのまま再利用する（無料・Webhook URLのみで
完結し、API審査やトークン管理が不要なため）。通知用の alerts.discord_webhook_url とは
別チャンネルを想定し、専用の discord_report.webhook_url を使う（同じURLを設定すれば
同一チャンネルにまとめることもできる）。投稿失敗はログに残すのみでアプリを止めない。
"""
from typing import Optional

from loguru import logger

from src.core import config as cfg
from src.core.alerts import DISCORD_MAX_CONTENT_LENGTH, DiscordWebhookProvider
from src.core.pnl_report import format_report_text


def post_text(text: str) -> bool:
    """テキストをDiscordへ投稿する。無効化・URL未設定・送信失敗時は False を返す。"""
    section = cfg.get_section("discord_report")
    if not section.get("enabled", False):
        logger.info("Discord日次レポートが無効（discord_report.enabled=false）のため投稿をスキップしました")
        return False
    webhook_url = section.get("webhook_url", "")
    if not webhook_url:
        logger.warning(
            "Discord日次レポートのWebhook URLが未設定のため投稿をスキップしました"
            "（docs/日次レポート投稿ガイド.md参照）"
        )
        return False
    try:
        DiscordWebhookProvider(webhook_url).send(text)
        logger.info("Discordへ日次レポートを投稿しました")
        return True
    except Exception as e:
        logger.error(f"Discord日次レポート投稿失敗: {e}")
        return False


def format_for_discord(mode: str, report: dict) -> str:
    """Discord投稿用に整形する（上限2000文字。X(280字)より余裕があるため通常は切り詰め不要）。"""
    text = format_report_text(mode, report)
    if len(text) > DISCORD_MAX_CONTENT_LENGTH:
        text = text[: DISCORD_MAX_CONTENT_LENGTH - 1] + "…"
    return text


def post_daily_report(mode: str, reference_capital: float) -> Optional[str]:
    """日次レポートを集計・整形してDiscordへ投稿する。投稿したテキストを返す（失敗時も返す）。"""
    from src.core.pnl_report import build_report
    report = build_report(reference_capital)
    text = format_for_discord(mode, report)
    post_text(text)
    return text
