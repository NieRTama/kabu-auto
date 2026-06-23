"""
シグナル生成: ルールベース＋ML スコアを統合して売買判断を行う
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger

from src.core import config as cfg
from src.strategy.indicators import build_features
from src.strategy import ml_model


@dataclass
class Signal:
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    rule_score: float
    ml_score: float
    combined_score: float


def compute_rule_score(df: pd.DataFrame) -> float:
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


def _ensemble_proba(model, lstm, df: pd.DataFrame,
                    news_df: Optional[pd.DataFrame], symbol: str) -> float:
    """GBM（必須）とLSTM（任意）の上昇確率を重み付き平均する。

    lstm が渡されれば ensemble、None なら純GBM（フラグの解釈は呼び出し側が行い、
    アンサンブルしたいときだけ lstm を渡す）。LSTM推論に失敗した場合も純GBMへ
    フォールバックする（LSTMの欠如は既定状態であり異常ではない）。
    """
    gbm_proba = ml_model.predict_proba(model, df, news_df=news_df)
    if lstm is None:
        return gbm_proba
    try:
        from src.strategy import lstm_model as lstm_mod
        lstm_proba = lstm_mod.predict_proba(lstm, df, news_df=news_df)
    except Exception as e:
        logger.warning(f"LSTM推論失敗 ({symbol})→純GBM: {e}")
        return gbm_proba
    conf = cfg.get_section("strategy")
    w_gbm = float(conf.get("ensemble_gbm_weight", 0.6))
    w_lstm = float(conf.get("ensemble_lstm_weight", 0.4))
    total = w_gbm + w_lstm
    if total <= 0:
        return gbm_proba
    return (w_gbm * gbm_proba + w_lstm * lstm_proba) / total


def generate(symbol: str, df: pd.DataFrame,
             model: Optional[object] = None,
             lstm_model: Optional[object] = None,
             news_df: Optional[pd.DataFrame] = None) -> Signal:
    """シグナルを生成する。

    lstm_model を渡すとアンサンブル（GBM＋LSTM）、None なら純GBM。news_df は
    use_news_features が有効なときだけ build_features 内で結合される。フラグの解釈は
    呼び出し側（signal_scan）が行い、本番／シャドーで lstm_model・news_df の有無を
    切り替える。既定（lstm_model=None・news_df=None）では従来どおり純GBM・価格のみで
    完全に同一挙動。
    """
    conf = cfg.get_section("strategy")
    ml_weight = conf.get("ml_weight", 0.5)
    rule_weight = conf.get("rule_weight", 0.5)
    buy_thr = conf.get("buy_threshold", 0.6)
    sell_thr = conf.get("sell_threshold", -0.6)

    df = build_features(df, news_df=news_df)
    rule_s = compute_rule_score(df)

    ml_s = 0.0
    if model is not None:
        try:
            proba = _ensemble_proba(model, lstm_model, df, news_df, symbol)
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
