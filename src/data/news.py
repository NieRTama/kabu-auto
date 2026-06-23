"""
ニュース取得（マクロRSS／個別銘柄）。

- マクロ: 日本語の経済ニュースRSS（config の news.macro_sources）。`feedparser` を遅延 import。
- 個別: yfinance のティッカー `.news`（既存依存）。

各記事は dict で返す:
    {"title": str, "summary": str, "published_at": datetime(JST naive), "url": str}

ネットワーク・パース失敗は握りつぶして空リストを返す（バッチを止めないため）。
公開時刻は JST naive に正規化する（プロジェクト全体の時刻規約に合わせる）。
"""
from datetime import datetime, timezone, timedelta
from typing import Optional

from loguru import logger

from src.core import config as cfg
from src.data.market_data import _to_yf_symbol

_JST = timezone(timedelta(hours=9))


def _to_jst_naive(ts) -> Optional[datetime]:
    """各種タイムスタンプを JST naive な datetime へ正規化する。"""
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            # epoch 秒（yfinance の providerPublishTime 等）
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        elif isinstance(ts, datetime):
            dt = ts
        else:
            return None
    except (ValueError, OSError, OverflowError):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(_JST).replace(tzinfo=None)
    return dt


def fetch_macro_news() -> list[dict]:
    """config の news.macro_sources（RSS）からマクロ経済ニュースを取得する。"""
    news_conf = cfg.get_section("news")
    sources = news_conf.get("macro_sources", []) or []
    if not sources:
        return []
    try:
        import feedparser  # 遅延 import
    except Exception as e:
        logger.warning(f"マクロニュース取得を無効化（feedparser 未導入）: {e}")
        return []

    articles: list[dict] = []
    for url in sources:
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logger.warning(f"RSS取得失敗 {url}: {e}")
            continue
        for entry in getattr(feed, "entries", []):
            published = None
            if getattr(entry, "published_parsed", None):
                # published_parsed は UTC の time.struct_time
                import calendar
                published = _to_jst_naive(calendar.timegm(entry.published_parsed))
            articles.append({
                "title": getattr(entry, "title", "") or "",
                "summary": getattr(entry, "summary", "") or "",
                "published_at": published,
                "url": getattr(entry, "link", "") or "",
            })
    logger.info(f"マクロニュース取得: {len(articles)}件（{len(sources)}ソース）")
    return articles


def fetch_symbol_news(symbol: str) -> list[dict]:
    """yfinance のティッカー `.news` から個別銘柄ニュースを取得する。"""
    try:
        import yfinance as yf  # 既存依存
    except Exception as e:
        logger.warning(f"個別ニュース取得を無効化（yfinance 未導入）: {e}")
        return []
    try:
        raw = yf.Ticker(_to_yf_symbol(symbol)).news or []
    except Exception as e:
        logger.warning(f"個別ニュース取得失敗 {symbol}: {e}")
        return []

    articles: list[dict] = []
    for item in raw:
        # yfinance のスキーマは版により item 直下 or item["content"] に入る
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = content.get("title") or item.get("title", "") or ""
        summary = content.get("summary") or content.get("description") or item.get("summary", "") or ""
        ts = (item.get("providerPublishTime")
              or content.get("pubDate")
              or content.get("displayTime"))
        # pubDate/displayTime は ISO 文字列のことがある
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                ts = None
        url = ""
        if isinstance(content.get("canonicalUrl"), dict):
            url = content["canonicalUrl"].get("url", "")
        url = url or item.get("link", "") or ""
        articles.append({
            "title": title,
            "summary": summary,
            "published_at": _to_jst_naive(ts),
            "url": url,
        })
    logger.info(f"個別ニュース取得 {symbol}: {len(articles)}件")
    return articles
