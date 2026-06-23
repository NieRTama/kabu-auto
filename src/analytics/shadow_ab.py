"""
シャドーA/B 比較レポート（前進収集）。

本番（純GBM・価格のみ）の action と、シャドー候補（GBM+LSTMアンサンブル）の
action_shadow を、各シグナル発生日からの N日後フォワードリターンで採点し、
ヒット率を比較する。capital を一切リスクに晒さずに「ニュース＋アンサンブルを
本番採用すべきか」を実データで判断するための材料。

採点ルール（HOLDは除外）:
  - BUY  は forward_return > 0 で的中
  - SELL は forward_return < 0 で的中
"""
from datetime import timedelta
from typing import Optional

from loguru import logger
from sqlalchemy import select

from src.data.database import Signal, get_session
from src.data.market_data import load_ohlcv


def _forward_return(closes_by_date: dict, sig_date, horizon: int,
                    sorted_dates: list) -> Optional[float]:
    """シグナル日の終値から horizon 営業日後の終値までのリターン。"""
    if sig_date not in closes_by_date:
        # シグナル日が休場等で終値が無い場合は、直近過去の営業日に寄せる
        past = [d for d in sorted_dates if d <= sig_date]
        if not past:
            return None
        sig_date = past[-1]
    future = [d for d in sorted_dates if d > sig_date]
    if len(future) < horizon:
        return None
    base = closes_by_date[sig_date]
    target = closes_by_date[future[horizon - 1]]
    if not base:
        return None
    return (target - base) / base


def _is_hit(action: str, fwd: float) -> Optional[bool]:
    if action == "BUY":
        return fwd > 0
    if action == "SELL":
        return fwd < 0
    return None  # HOLD は採点対象外


def compare_shadow_ab(horizon: int = 5, max_signals: int = 5000) -> dict:
    """本番 vs シャドーのフォワード・ヒット率を比較して返す。

    戻り値:
        {
          "horizon": int,
          "evaluated": int,            # 採点できたシグナル数（本番がBUY/SELLのもの）
          "production": {"hits": int, "n": int, "hit_rate": float|None},
          "shadow":     {"hits": int, "n": int, "hit_rate": float|None},
          "divergence": int,          # action != action_shadow の件数
          "shadow_available": int,    # action_shadow が非NULLのシグナル数
        }
    """
    with get_session() as session:
        rows = session.scalars(
            select(Signal)
            .where(Signal.action.in_(["BUY", "SELL"]))
            .order_by(Signal.generated_at.desc())
            .limit(max_signals)
        ).all()
        signals = [(s.symbol, s.generated_at, s.action, s.action_shadow) for s in rows]

    # 銘柄ごとに終値系列をキャッシュ
    closes_cache: dict = {}

    def _closes(symbol: str):
        if symbol not in closes_cache:
            try:
                df = load_ohlcv(symbol, limit=2000)
                cmap = {d.date() if hasattr(d, "date") else d: float(c)
                        for d, c in zip(df.index, df["close"])}
                closes_cache[symbol] = (cmap, sorted(cmap.keys()))
            except Exception as e:
                logger.warning(f"A/B: 終値ロード失敗 {symbol}: {e}")
                closes_cache[symbol] = ({}, [])
        return closes_cache[symbol]

    prod_hits = prod_n = 0
    shadow_hits = shadow_n = 0
    divergence = 0
    shadow_available = 0
    evaluated = 0

    for symbol, gen_at, action, action_shadow in signals:
        cmap, sdates = _closes(symbol)
        if not cmap:
            continue
        sig_date = gen_at.date() if hasattr(gen_at, "date") else gen_at
        fwd = _forward_return(cmap, sig_date, horizon, sdates)
        if fwd is None:
            continue
        evaluated += 1
        hit = _is_hit(action, fwd)
        if hit is not None:
            prod_n += 1
            prod_hits += int(hit)
        if action_shadow is not None:
            shadow_available += 1
            if action_shadow != action:
                divergence += 1
            s_hit = _is_hit(action_shadow, fwd)
            if s_hit is not None:
                shadow_n += 1
                shadow_hits += int(s_hit)

    def _rate(h, n):
        return round(h / n, 4) if n else None

    return {
        "horizon": horizon,
        "evaluated": evaluated,
        "production": {"hits": prod_hits, "n": prod_n, "hit_rate": _rate(prod_hits, prod_n)},
        "shadow": {"hits": shadow_hits, "n": shadow_n, "hit_rate": _rate(shadow_hits, shadow_n)},
        "divergence": divergence,
        "shadow_available": shadow_available,
    }
