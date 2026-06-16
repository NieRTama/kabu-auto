"""テクニカル指標計算モジュール（pandas-ta使用）"""
import pandas as pd
import pandas_ta as ta

from src.core import config as cfg


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCVにテクニカル指標を付加して返す"""
    conf = cfg.get_section("strategy")
    df = df.copy()

    short = conf.get("ma_short", 5)
    mid = conf.get("ma_mid", 25)
    long_ = conf.get("ma_long", 75)
    rsi_p = conf.get("rsi_period", 14)
    bb_p = conf.get("bb_period", 20)
    bb_std = conf.get("bb_std", 2.0)

    df[f"ma{short}"] = ta.sma(df["close"], length=short)
    df[f"ma{mid}"] = ta.sma(df["close"], length=mid)
    df[f"ma{long_}"] = ta.sma(df["close"], length=long_)

    df["rsi"] = ta.rsi(df["close"], length=rsi_p)

    bb = ta.bbands(df["close"], length=bb_p, std=bb_std)
    if bb is not None:
        df["bb_lower"] = bb.iloc[:, 0]  # BBL
        df["bb_mid"] = bb.iloc[:, 1]    # BBM
        df["bb_upper"] = bb.iloc[:, 2]  # BBU

    macd = ta.macd(df["close"])
    if macd is not None:
        df["macd"] = macd.iloc[:, 0]
        df["macd_hist"] = macd.iloc[:, 1]    # MACDh
        df["macd_signal"] = macd.iloc[:, 2]  # MACDs

    df["returns"] = df["close"].pct_change()
    df["volume_ma20"] = ta.sma(df["volume"].astype(float), length=20)
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """ML用の特徴量を作成して返す（ラベルは付与しない）。

    ラベリングは labeling.py のトリプルバリア法で別途行う。
    特徴量は過去データのみから計算されるため、最新行も保持される
    （予測時に当日の特徴量を使えるよう dropna は特徴量列のみで行う）。
    """
    df = compute_indicators(df)
    conf = cfg.get_section("strategy")
    short = conf.get("ma_short", 5)
    mid = conf.get("ma_mid", 25)
    long_ = conf.get("ma_long", 75)

    df["ma_cross_sm"] = df[f"ma{short}"] - df[f"ma{mid}"]
    df["ma_cross_ml"] = df[f"ma{mid}"] - df[f"ma{long_}"]
    bb_width = (df["bb_upper"] - df["bb_lower"]).clip(lower=1e-4)
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / bb_width
    df["price_momentum_5"] = df["close"].pct_change(5)
    df["price_momentum_20"] = df["close"].pct_change(20)

    return df.dropna(subset=FEATURE_COLS)


FEATURE_COLS = [
    "ma_cross_sm", "ma_cross_ml", "rsi", "macd", "macd_hist",
    "bb_pct", "volume_ratio", "price_momentum_5", "price_momentum_20",
    "returns",
]
