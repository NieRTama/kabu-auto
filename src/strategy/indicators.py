"""テクニカル指標計算モジュール（pandas のみ使用）"""
import pandas as pd

from src.core import config as cfg


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()


def _rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=length - 1, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(com=length - 1, min_periods=length).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _bbands(series: pd.Series, length: int, std: float):
    mid = series.rolling(window=length).mean()
    sigma = series.rolling(window=length).std(ddof=0)
    return mid - std * sigma, mid, mid + std * sigma


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, hist, signal_line


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

    df[f"ma{short}"] = _sma(df["close"], short)
    df[f"ma{mid}"] = _sma(df["close"], mid)
    df[f"ma{long_}"] = _sma(df["close"], long_)

    df["rsi"] = _rsi(df["close"], rsi_p)

    df["bb_lower"], df["bb_mid"], df["bb_upper"] = _bbands(df["close"], bb_p, bb_std)

    df["macd"], df["macd_hist"], df["macd_signal"] = _macd(df["close"])

    df["returns"] = df["close"].pct_change(fill_method=None)
    df["volume_ma20"] = _sma(df["volume"].astype(float), 20)
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
    df["price_momentum_5"] = df["close"].pct_change(5, fill_method=None)
    df["price_momentum_20"] = df["close"].pct_change(20, fill_method=None)

    return df.dropna(subset=FEATURE_COLS)


FEATURE_COLS = [
    "ma_cross_sm", "ma_cross_ml", "rsi", "macd", "macd_hist",
    "bb_pct", "volume_ratio", "price_momentum_5", "price_momentum_20",
    "returns",
]
