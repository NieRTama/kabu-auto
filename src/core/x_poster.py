"""
X（旧Twitter）への日次損益レポート自動投稿。

OAuth 1.0a（API Key/Secret + Access Token/Secret の4つ、Read and Write権限必須）を使う。
取得手順は docs/日次レポート投稿ガイド.md を参照。投稿失敗はログに残すのみでアプリを
止めない（src/core/alerts.py の通知プロバイダと同じ「失敗を握り潰して継続」方針）。
"""
from typing import Optional

from loguru import logger

from src.core import config as cfg
from src.core.pnl_report import format_report_text

# Xの投稿文字数上限（無料/未認証アカウント基準）。超過時は安全側で切り詰める。
X_MAX_LENGTH = 280
_TRUNCATION_SUFFIX = "…"


def format_daily_report(mode: str, report: dict) -> str:
    """日次レポートのツイート文を組み立てる（モード・当日/週次/月次/総合・勝率）。

    280文字を超える場合は安全に切り詰める（送信エラーで投稿が完全に消えるより、
    要点が欠けても投稿自体が成功する方を優先。src/core/alerts.py と同じ方針）。
    """
    text = format_report_text(mode, report)
    if len(text) > X_MAX_LENGTH:
        text = text[: X_MAX_LENGTH - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX
    return text


def _get_client():
    """tweepy.Client を構築する。4つの鍵が揃っていなければ None を返す。"""
    section = cfg.get_section("x")
    api_key = section.get("api_key", "")
    api_secret = section.get("api_secret", "")
    access_token = section.get("access_token", "")
    access_token_secret = section.get("access_token_secret", "")
    if not (api_key and api_secret and access_token and access_token_secret):
        return None
    import tweepy
    return tweepy.Client(
        consumer_key=api_key, consumer_secret=api_secret,
        access_token=access_token, access_token_secret=access_token_secret,
    )


def post_tweet(text: str) -> bool:
    """テキストをXへ投稿する。鍵未設定・送信失敗時は False を返し、ログにのみ記録する。"""
    section = cfg.get_section("x")
    if not section.get("enabled", False):
        logger.info("X連携が無効（x.enabled=false）のため投稿をスキップしました")
        return False
    client = _get_client()
    if client is None:
        logger.warning("X APIの認証情報が未設定のため投稿をスキップしました（docs/日次レポート投稿ガイド.md参照）")
        return False
    try:
        client.create_tweet(text=text)
        logger.info("Xへ日次レポートを投稿しました")
        return True
    except Exception as e:
        logger.error(f"X投稿失敗: {e}")
        return False


def post_daily_report(mode: str, reference_capital: float) -> Optional[str]:
    """日次レポートを集計・整形してXへ投稿する。投稿したテキストを返す（失敗時もテキストは返す）。"""
    from src.core.pnl_report import build_report
    report = build_report(reference_capital)
    text = format_daily_report(mode, report)
    post_tweet(text)
    return text
