"""
ニュースセンチメントの集約・永続化・特徴量化。

責務:
  1. 取得（news.py）＋採点（sentiment.py）を日次集約し news_sentiment テーブルへ upsert
     （`update_news_sentiment`、news_update ジョブから呼ぶ）。
  2. DBから単一銘柄用のニュース特徴量フレームを構築（`load_news_frame`）。
  3. テクニカル特徴量フレームへ日付で結合（`join_news_features`、build_features から呼ぶ）。

リーク防止の要:
  - `trading_date_for()` が公開時刻を 16:00 JST カットオフで「遅くとも当日大引け後に
    知り得た取引日」へ正規化する（16:00以降公開＝翌営業日）。
  - 特徴量の移動平均・変化量はすべて後ろ向き（rolling/diff）。日付 D の特徴量は
    D 以前のニュースのみに依存するため、一括結合してもルックアヘッドしない。

本モジュールは torch/transformers に直接依存しない（採点は sentiment.py が遅延 import）。
"""
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from src.core import config as cfg
from src.strategy.indicators import NEWS_FEATURE_COLS

MACRO_SYMBOL = "_MACRO_"


def _cutoff_hour() -> int:
    return int(cfg.get_section("news").get("cutoff_hour", 16))


def trading_date_for(published_at: datetime, cutoff_hour: Optional[int] = None) -> date:
    """公開日時を、そのニュースを反映してよい「取引日」へ正規化する（リーク防止）。

    - カットオフ時刻（既定16:00 JST）以降に公開されたニュースは翌日扱い。
    - 週末（土日）は次の月曜へ繰り上げる（祝日は簡略化のため未対応＝翌営業日に丸める）。
    """
    if cutoff_hour is None:
        cutoff_hour = _cutoff_hour()
    d = published_at.date()
    if published_at.hour >= cutoff_hour:
        d = d + timedelta(days=1)
    while d.weekday() >= 5:  # 5=土, 6=日
        d = d + timedelta(days=1)
    return d


# ─── 取得・採点・保存（news_update ジョブ）──────────────────────────────────

def update_news_sentiment(symbols: list[str]) -> None:
    """ニュースを取得・採点し、日次センチメントを news_sentiment へ upsert する。

    16:05（data_update 後・signal_scan 前）に呼ぶ。前進収集専用（過去バックフィルなし）。
    """
    news_conf = cfg.get_section("news")
    if not news_conf.get("enabled", True):
        return
    from src.data import news as news_fetch

    if news_conf.get("macro_news", True):
        articles = news_fetch.fetch_macro_news()
        _score_and_store("macro", MACRO_SYMBOL, articles)

    if news_conf.get("symbol_news", True):
        for sym in symbols:
            try:
                articles = news_fetch.fetch_symbol_news(sym)
                _score_and_store("symbol", sym, articles)
            except Exception as e:
                logger.error(f"個別ニュース処理失敗 {sym}: {e}")


def _score_and_store(scope: str, symbol: str, articles: list[dict]) -> None:
    """記事群を取引日ごとに集約・採点して upsert する。"""
    from src.strategy import sentiment

    cutoff = _cutoff_hour()
    by_date: dict[date, list[str]] = {}
    for a in articles:
        pub = a.get("published_at")
        if pub is None:
            continue
        td = trading_date_for(pub, cutoff)
        text = f"{a.get('title', '')} {a.get('summary', '')}".strip()
        if text:
            by_date.setdefault(td, []).append(text)

    if not by_date:
        return
    for td, texts in by_date.items():
        scores = sentiment.score_texts(texts)
        avg = sum(scores) / len(scores) if scores else 0.0
        _upsert_sentiment(scope, symbol, td, round(avg, 4), len(texts))
    logger.info(f"ニュースセンチメント保存 scope={scope} symbol={symbol}: {len(by_date)}営業日分")


def _upsert_sentiment(scope: str, symbol: str, d: date,
                      score: float, count: int) -> None:
    from src.data.database import NewsSentiment, get_session
    from sqlalchemy import select

    with get_session() as session:
        existing = session.scalar(
            select(NewsSentiment).where(
                NewsSentiment.scope == scope,
                NewsSentiment.symbol == symbol,
                NewsSentiment.date == d,
            )
        )
        if existing is not None:
            existing.sentiment_score = score
            existing.article_count = count
            existing.source = "rss" if scope == "macro" else "yfinance"
        else:
            session.add(NewsSentiment(
                scope=scope, symbol=symbol, date=d,
                sentiment_score=score, article_count=count,
                source="rss" if scope == "macro" else "yfinance",
            ))
        session.commit()


# ─── 特徴量フレームの構築・結合 ──────────────────────────────────────────────

def _read_daily(scope: str, symbol: str) -> tuple[pd.Series, pd.Series]:
    """news_sentiment から (sentiment, article_count) の日次 Series を読む（date index）。"""
    from src.data.database import NewsSentiment, get_session
    from sqlalchemy import select

    with get_session() as session:
        rows = session.execute(
            select(NewsSentiment.date, NewsSentiment.sentiment_score,
                   NewsSentiment.article_count)
            .where(NewsSentiment.scope == scope, NewsSentiment.symbol == symbol)
            .order_by(NewsSentiment.date)
        ).all()
    if not rows:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    dates = [r[0] for r in rows]
    sent = pd.Series([r[1] or 0.0 for r in rows], index=dates, dtype=float)
    cnt = pd.Series([r[2] or 0 for r in rows], index=dates, dtype=float)
    return sent, cnt


def load_news_frame(symbol: str) -> Optional[pd.DataFrame]:
    """単一銘柄用のニュース特徴量フレーム（date index, 列=NEWS_FEATURE_COLS）を返す。

    マクロ（全体）と個別（symbol）のセンチメントを連続営業日に整列し、欠損日は中立(0)で
    埋めてから後ろ向きの移動平均・変化量を計算する（リーク防止）。ニュースが1件も無ければ None。
    """
    macro_sent, _ = _read_daily("macro", MACRO_SYMBOL)
    sym_sent, sym_cnt = _read_daily("symbol", symbol)
    if macro_sent.empty and sym_sent.empty:
        return None

    all_dates = list(macro_sent.index) + list(sym_sent.index)
    start, end = min(all_dates), max(all_dates)
    idx = pd.bdate_range(start, end)  # 連続営業日

    def _reindex(s: pd.Series) -> pd.Series:
        if s.empty:
            return pd.Series(0.0, index=idx)
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        return s.reindex(idx).fillna(0.0)

    macro = _reindex(macro_sent)
    sym = _reindex(sym_sent)
    cnt = _reindex(sym_cnt)

    df = pd.DataFrame(index=idx)
    df["macro_sent"] = macro
    df["macro_sent_ma3"] = macro.rolling(3, min_periods=1).mean()
    df["macro_sent_ma5"] = macro.rolling(5, min_periods=1).mean()
    df["macro_sent_chg"] = macro.diff().fillna(0.0)
    df["sym_sent"] = sym
    df["sym_sent_ma3"] = sym.rolling(3, min_periods=1).mean()
    df["sym_sent_ma5"] = sym.rolling(5, min_periods=1).mean()
    df["sym_news_count"] = cnt

    df.index = [d.date() for d in df.index]  # date キーに（join 用）
    return df[NEWS_FEATURE_COLS]


def join_news_features(feat_df: pd.DataFrame,
                       news_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """テクニカル特徴量フレームへニュース特徴量を日付で左結合する。

    feat_df は OHLCV 由来の DatetimeIndex。news_df は load_news_frame の出力（date index）。
    ニュースが無い日・期間外は中立(0)で埋める（dropna はしない＝行を落とさない）。
    """
    out = feat_df.copy()
    if news_df is None or news_df.empty:
        for col in NEWS_FEATURE_COLS:
            out[col] = 0.0
        return out

    keys = pd.Index([d.date() for d in pd.to_datetime(out.index)])
    aligned = news_df.reindex(keys)
    for col in NEWS_FEATURE_COLS:
        if col in aligned.columns:
            out[col] = aligned[col].fillna(0.0).to_numpy()
        else:
            out[col] = 0.0
    return out
