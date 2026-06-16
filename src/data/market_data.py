"""
市場データ取得・管理モジュール
日足OHLCVデータをyfinanceで取得し、SQLiteに保存する。
kabuステーションAPIは板情報のリアルタイム取得に使用し、
過去データはyfinanceで補完する（権利修正済み）。
"""
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf
from loguru import logger
from sqlalchemy import select

from src.data.database import OHLCV, get_session


def _to_yf_symbol(symbol: str) -> str:
    """東証銘柄コードをyfinance形式に変換（例: 7203 → 7203.T）"""
    return f"{symbol}.T"


def fetch_ohlcv(symbol: str, start: date, end: date) -> pd.DataFrame:
    """yfinanceから権利修正済みOHLCVを取得する"""
    yf_sym = _to_yf_symbol(symbol)
    df = yf.download(yf_sym, start=start.isoformat(), end=end.isoformat(),
                     auto_adjust=True, progress=False)
    if df.empty:
        logger.warning(f"データ取得なし: {symbol} ({start} ~ {end})")
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).date
    df.index.name = "date"
    return df


def upsert_ohlcv(symbol: str, df: pd.DataFrame) -> int:
    """OHLCVデータをDBにupsertする"""
    if df.empty:
        return 0
    with get_session() as session:
        existing_map = {
            r.date: r
            for r in session.scalars(
                select(OHLCV).where(OHLCV.symbol == symbol)
            ).all()
        }
        count = 0
        for dt, row in df.iterrows():
            if dt in existing_map:
                rec = existing_map[dt]
                rec.open = float(row.get("open", 0))
                rec.high = float(row.get("high", 0))
                rec.low = float(row.get("low", 0))
                rec.close = float(row.get("close", 0))
                rec.volume = int(row.get("volume", 0))
                rec.adjusted_close = float(row.get("close", 0))
            else:
                session.add(OHLCV(
                    symbol=symbol,
                    date=dt,
                    open=float(row.get("open", 0)),
                    high=float(row.get("high", 0)),
                    low=float(row.get("low", 0)),
                    close=float(row.get("close", 0)),
                    volume=int(row.get("volume", 0)),
                    adjusted_close=float(row.get("close", 0)),
                ))
                count += 1
        session.commit()
    return count


def update_symbol(symbol: str, years: int = 3) -> None:
    """銘柄の過去データを更新する"""
    end = date.today()
    start = end - timedelta(days=365 * years)
    df = fetch_ohlcv(symbol, start, end)
    added = upsert_ohlcv(symbol, df)
    logger.info(f"データ更新: {symbol} 追加={added}件")


def load_ohlcv(symbol: str, limit: int = 500) -> pd.DataFrame:
    """DBからOHLCVを読み込みDataFrameで返す（最新limit件を時系列昇順で返す）"""
    with get_session() as session:
        rows = list(reversed(session.scalars(
            select(OHLCV).where(OHLCV.symbol == symbol)
            .order_by(OHLCV.date.desc())
            .limit(limit)
        ).all()))
    if not rows:
        return pd.DataFrame()
    data = [
        {
            "date": r.date,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.adjusted_close or r.close,
            "volume": r.volume,
        }
        for r in rows
    ]
    df = pd.DataFrame(data).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def get_watchlist() -> List[str]:
    """ウォッチリスト銘柄を返す（現在はDBに保有ポジションのある銘柄）"""
    from src.data.database import Position
    with get_session() as session:
        positions = session.scalars(select(Position)).all()
    return [p.symbol for p in positions]
