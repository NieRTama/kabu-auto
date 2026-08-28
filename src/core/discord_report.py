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


def format_for_discord(mode: str, report: dict, holdings: Optional[dict] = None) -> str:
    """Discord投稿用に整形する（上限2000文字。X(280字)より余裕があるため通常は切り詰め不要）。

    holdings を渡すと保有建玉の評価額・含み損益も併記する。Discordは文字数に
    余裕があるため既定で付ける（Xは280字制限のため実現損益のみに留める）。
    """
    text = format_report_text(mode, report, holdings)
    if len(text) > DISCORD_MAX_CONTENT_LENGTH:
        text = text[: DISCORD_MAX_CONTENT_LENGTH - 1] + "…"
    return text


def post_daily_report(mode: str, reference_capital: float) -> Optional[str]:
    """日次レポートを集計・整形してDiscordへ投稿する。投稿したテキストを返す（失敗時も返す）。

    確定した実現損益（当日/週次/月次/総合）に加え、現在の保有建玉の評価額と
    含み損益も報告する（決済前のポジション状況が実現損益だけでは見えないため）。
    """
    from src.core.pnl_report import build_holdings, build_report
    report = build_report(reference_capital)
    try:
        holdings = build_holdings(reference_capital)
    except Exception as e:
        # 保有状況の取得に失敗しても、確定損益のレポートは送る（部分的な失敗で全部を落とさない）
        logger.warning(f"保有状況の集計に失敗しました（実現損益のみ投稿します）: {e}")
        holdings = None
    text = format_for_discord(mode, report, holdings)
    post_text(text)
    return text
