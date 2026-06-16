"""
シグナル生成: ルールベース＋ML スコアを統合して売買判断を行う
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger

from src.core import config as cfg
from src.strategy.indicators import compute_indicators
from src.strategy import ml_model


@dataclass
class Signal:
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    rule_score: float
    ml_score: float
    combined_score: float


def _rule_score(df: pd.DataFrame) -> float:
    """ルールベースのスコアを計算する（-1.0〜+1.0）"""
    if len(df) < 2:
        return 0.0
    conf = cfg.get_section("strategy")
    short = conf.get("ma_short", 5)
    mid = conf.get("ma_mid", 25)
    rsi_os = conf.get("rsi_oversold", 30)
    rsi_ob = conf.get("rsi_overbought", 70)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0.0

    # MAクロス
    if latest[f"ma{short}"] > latest[f"ma{mid}"] and prev[f"ma{short}"] <= prev[f"ma{mid}"]:
        score += 0.4  # ゴールデンクロス
    elif latest[f"ma{short}"] < latest[f"ma{mid}"] and prev[f"ma{short}"] >= prev[f"ma{mid}"]:
        score -= 0.4  # デッドクロス

    # RSI
    rsi = latest.get("rsi", 50)
    if rsi < rsi_os:
        score += 0.3
    elif rsi > rsi_ob:
        score -= 0.3

    # ボリンジャーバンド
    bb_upper = latest.get("bb_upper")
    bb_lower = latest.get("bb_lower")
    close = latest["close"]
    if bb_upper is not None and bb_lower is not None:
        if close < bb_lower:
            score += 0.2
        elif close > bb_upper:
            score -= 0.2

    # MACD
    if latest.get("macd_hist", 0) > 0 and prev.get("macd_hist", 0) <= 0:
        score += 0.1
    elif latest.get("macd_hist", 0) < 0 and prev.get("macd_hist", 0) >= 0:
        score -= 0.1

    return max(-1.0, min(1.0, score))


def generate(symbol: str, df: pd.DataFrame,
             model: Optional[object] = None) -> Signal:
    """シグナルを生成する"""
    conf = cfg.get_section("strategy")
    ml_weight = conf.get("ml_weight", 0.5)
    rule_weight = conf.get("rule_weight", 0.5)
    buy_thr = conf.get("buy_threshold", 0.6)
    sell_thr = conf.get("sell_threshold", -0.6)

    df = compute_indicators(df)
    rule_s = _rule_score(df)

    ml_s = 0.0
    if model is not None:
        try:
            proba = ml_model.predict_proba(model, df)
            ml_s = (proba - 0.5) * 2  # 0.5基準で-1〜+1に変換
        except Exception as e:
            logger.warning(f"ML推論失敗 ({symbol}): {e}")

    combined = rule_s * rule_weight + ml_s * ml_weight

    if combined >= buy_thr:
        action = "BUY"
    elif combined <= sell_thr:
        action = "SELL"
    else:
        action = "HOLD"

    return Signal(
        symbol=symbol,
        action=action,
        rule_score=rule_s,
        ml_score=ml_s,
        combined_score=combined,
    )
