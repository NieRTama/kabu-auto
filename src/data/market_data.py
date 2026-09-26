"""
市場データ取得・管理モジュール
日足OHLCVデータをyfinanceで取得し、SQLiteに保存する。
kabuステーションAPIは板情報のリアルタイム取得に使用し、
過去データはyfinanceで補完する（権利修正済み）。
"""
import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger
from sqlalchemy import func, select

from src.core import clock
from src.data.bar_status import BarStatus, classify
from src.data.database import OHLCV, CorporateAction, get_session


def _to_yf_symbol(symbol: str) -> str:
    """東証銘柄コードをyfinance形式に変換（例: 7203 → 7203.T）"""
    return f"{symbol}.T"


def fetch_ohlcv(symbol: str, start: date, end: date, retries: int = 2) -> pd.DataFrame:
    """yfinanceからOHLCVを取得する（一時的な通信エラーは指定回数までリトライ）。

    `end` は **その日を含む**。yfinance公式仕様の end は排他境界のため、
    ここで内部的に +1日する。呼び出し側では一切加減算しないこと
    （内部と呼び出し側の双方で1日足す事故を防ぐため、境界の解釈は
    この関数に一元化する）。
    """
    yf_sym = _to_yf_symbol(symbol)
    # yfinance の end は排他境界。引数は「含む」なので +1日して渡す。
    yf_end = end + timedelta(days=1)
    df = pd.DataFrame()
    for attempt in range(retries + 1):
        try:
            df = yf.download(yf_sym, start=start.isoformat(), end=yf_end.isoformat(),
                             auto_adjust=False, progress=False)
            break
        except Exception as e:
            if attempt < retries:
                logger.warning(f"yfinance取得失敗 (リトライ {attempt + 1}/{retries}): {symbol} {e}")
                time.sleep(2)
            else:
                logger.error(f"yfinance取得失敗（リトライ上限到達）: {symbol} {e}")
                return pd.DataFrame()
    if df.empty:
        logger.warning(f"データ取得なし: {symbol} ({start} ~ {end})")
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    # auto_adjust=False では "Adj Close" が来る。列名を DB のカラム名に揃える。
    # 生OHLC（株数・必要資金の算定用）と調整済み終値（特徴量・リターン用）を分けて持つ。
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adjusted_close"})
    if "adjusted_close" not in df.columns:
        # 提供側が調整済み終値を返さない場合は生の終値で埋める（用途の分離は維持する）
        df["adjusted_close"] = df["close"]
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
                rec.adjusted_close = float(row.get("adjusted_close", row.get("close", 0)))
            else:
                session.add(OHLCV(
                    symbol=symbol,
                    date=dt,
                    open=float(row.get("open", 0)),
                    high=float(row.get("high", 0)),
                    low=float(row.get("low", 0)),
                    close=float(row.get("close", 0)),
                    volume=int(row.get("volume", 0)),
                    adjusted_close=float(row.get("adjusted_close", row.get("close", 0))),
                ))
                count += 1
        session.commit()
    return count


def fetch_splits(symbol: str) -> pd.Series:
    """yfinanceから分割イベントを取得する（index=date, value=比率）。

    価格系列の比からは分割を復元できないため、イベントAPIを唯一の権威とする。
    取得に失敗した場合は空のSeriesを返す（分割なしと同義に扱う）。
    """
    try:
        raw = yf.Ticker(_to_yf_symbol(symbol)).splits
    except Exception as e:
        logger.warning(f"分割イベント取得失敗: {symbol} {e}")
        return pd.Series(dtype=float)
    if raw is None or len(raw) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(raw.values, index=pd.to_datetime(raw.index).date, dtype=float)
    s.index.name = "date"
    return s


def upsert_splits(symbol: str, splits: pd.Series) -> int:
    """分割イベントをDBにupsertする。新規追加件数を返す。"""
    if splits is None or len(splits) == 0:
        return 0
    with get_session() as session:
        existing = {
            r.date: r
            for r in session.scalars(
                select(CorporateAction).where(
                    CorporateAction.symbol == symbol,
                    CorporateAction.action_type == "SPLIT",
                )
            ).all()
        }
        added = 0
        for dt, ratio in splits.items():
            if dt in existing:
                existing[dt].ratio = float(ratio)
            else:
                session.add(CorporateAction(
                    symbol=symbol, date=dt,
                    action_type="SPLIT", ratio=float(ratio),
                ))
                added += 1
        session.commit()
    return added


def update_symbol(symbol: str, years: int = 3,
                  now: Optional[datetime] = None) -> BarStatus:
    """銘柄の過去データと分割イベントを更新し、最終足の状態を返す。

    戻り値の BarStatus は「更新した結果、この銘柄の足は基準セッションまで
    追いついているか」を表す。呼び出し側はこれを見て、確定していない足の銘柄を
    新規候補から除外する（保有保護の退出は止めない）。
    """
    now = now or clock.now()
    end = now.date()
    start = end - timedelta(days=365 * years)
    df = fetch_ohlcv(symbol, start, end)
    added = upsert_ohlcv(symbol, df)
    split_added = upsert_splits(symbol, fetch_splits(symbol))

    last_bar = max(df.index) if not df.empty else _last_stored_session(symbol)
    status = classify(symbol, last_bar, now)
    logger.info(
        f"データ更新: {symbol} 追加={added}件 分割={split_added}件 "
        f"最終足={last_bar} 状態={status.state}"
    )
    return status


def _last_stored_session(symbol: str) -> Optional[date]:
    """DBに保存済みの最終営業日（取得が空だったときの判定に使う）。"""
    with get_session() as session:
        return session.scalar(
            select(func.max(OHLCV.date)).where(OHLCV.symbol == symbol)
        )


def load_ohlcv(symbol: str, limit: int = 500,
               price_basis: str = "adjusted") -> pd.DataFrame:
    """DBからOHLCVを読み込みDataFrameで返す（最新limit件を時系列昇順で返す）。

    price_basis:
      "adjusted" … 調整済み終値を close として返す（特徴量・リターン計算用）。
                   open/high/low にも同じ調整比率(adjusted_close/close)を掛けて
                   返すことで、low <= close <= high の不変条件を保つ
                   （auto_adjust=False化により生のOHLと調整済みcloseが別々に
                   保存されるようになったため、この分岐で明示的に揃える）。
      "raw"      … 生の終値を close として返す（株数・単元・必要資金・現金の算定用）

    分けないと、1対2分割で過去価格が半値に調整された銘柄について
    「当時100株買えたか」の判定が変わる。
    """
    if price_basis not in ("adjusted", "raw"):
        raise ValueError(f"price_basis は 'adjusted' か 'raw': {price_basis}")
    with get_session() as session:
        rows = list(reversed(session.scalars(
            select(OHLCV).where(OHLCV.symbol == symbol)
            .order_by(OHLCV.date.desc())
            .limit(limit)
        ).all()))
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        if price_basis == "adjusted":
            adj_close = r.adjusted_close or r.close
            ratio = (adj_close / r.close) if r.close else 1.0
            data.append({
                "date": r.date,
                "open": r.open * ratio,
                "high": r.high * ratio,
                "low": r.low * ratio,
                "close": adj_close,
                "volume": r.volume,
            })
        else:
            data.append({
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            })
    df = pd.DataFrame(data).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def latest_closes(symbols: list[str]) -> dict[str, float]:
    """指定銘柄群の最新終値を1クエリでまとめて取得する。

    保有銘柄ごとに最新OHLCVを個別取得する（N+1クエリ）パターンを避けるための
    ヘルパ。window関数で銘柄ごとの最新行（date降順1位）だけを抜き出す。
    戻り値: {"7203": 2500.0, ...}（データが無い銘柄はキーに含まれない）
    """
    if not symbols:
        return {}
    with get_session() as session:
        rn = func.row_number().over(
            partition_by=OHLCV.symbol, order_by=OHLCV.date.desc()
        ).label("rn")
        subq = (
            select(OHLCV.symbol, OHLCV.close, rn)
            .where(OHLCV.symbol.in_(symbols))
            .subquery()
        )
        rows = session.execute(
            select(subq.c.symbol, subq.c.close).where(subq.c.rn == 1)
        ).all()
    return {r.symbol: r.close for r in rows if r.close is not None}


def lookup_company_name(symbol: str) -> str:
    """yfinanceから銘柄コードに対応する会社名を取得する（取得失敗時は空文字）"""
    try:
        info = yf.Ticker(_to_yf_symbol(symbol)).info
    except Exception as e:
        logger.warning(f"会社名取得失敗: {symbol} {e}")
        return ""
    return info.get("longName") or info.get("shortName") or ""


def lookup_sector(symbol: str) -> str:
    """yfinanceから銘柄コードに対応するセクターを取得する（取得失敗時は空文字）。
    RiskManager.check_sector_concentration() のセクター集中リスク判定に使用する。
    """
    try:
        info = yf.Ticker(_to_yf_symbol(symbol)).info
    except Exception as e:
        logger.warning(f"セクター取得失敗: {symbol} {e}")
        return ""
    return info.get("sector") or ""
