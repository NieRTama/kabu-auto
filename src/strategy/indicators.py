"""テクニカル指標計算モジュール（pandas のみ使用）"""
from typing import Optional

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


def build_features(df: pd.DataFrame, news_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """ML用の特徴量を作成して返す（ラベルは付与しない）。

    ラベリングは labeling.py のトリプルバリア法で別途行う。
    特徴量は過去データのみから計算されるため、最新行も保持される
    （予測時に当日の特徴量を使えるよう dropna は特徴量列のみで行う）。

    news_df を渡し、かつ strategy.use_news_features が有効な場合のみ、ニュース
    センチメント特徴量（news_features.py）を日付で結合する。dropna は必ず
    基本テクニカル列（BASE_TECHNICAL_COLS）のみで行い、ニュース列は結合後に
    中立値(0)で穴埋めする。これにより、ニュースが未収集の初期履歴や、ニュースを
    渡さないバックテストの一括計算でも行が落ちない（リーク・回帰防止）。
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

    # dropna は基本テクニカル列のみ（ニュース列で落とさない）
    df = df.dropna(subset=BASE_TECHNICAL_COLS)

    if _news_enabled():
        # フラグ有効時はニュース列を必ず付与する（news_df が無ければ中立0で埋める）。
        # active_feature_cols() が常にニュース列を要求するため、列の欠落で
        # 学習・推論時に KeyError を起こさないようにするのが目的。
        from src.strategy.news_features import join_news_features
        df = join_news_features(df, news_df)

    return df


# ── 特徴量列の定義 ──────────────────────────────────────────────────
# 基本テクニカル列（凍結）。dropna はこの集合のみで行う。
BASE_TECHNICAL_COLS = [
    "ma_cross_sm", "ma_cross_ml", "rsi", "macd", "macd_hist",
    "bb_pct", "volume_ratio", "price_momentum_5", "price_momentum_20",
    "returns",
]

# ニュースセンチメント特徴量列（フェーズ4で news_features.py が定義・付与）。
# strategy.use_news_features が有効なときのみ active_feature_cols() に加わる。
NEWS_FEATURE_COLS = [
    "macro_sent", "macro_sent_ma3", "macro_sent_ma5", "macro_sent_chg",
    "sym_sent", "sym_sent_ma3", "sym_sent_ma5", "sym_news_count",
]

# 後方互換用エイリアス（旧称 FEATURE_COLS は基本テクニカル列を指す）
FEATURE_COLS = BASE_TECHNICAL_COLS


def _news_enabled() -> bool:
    return bool(cfg.get_section("strategy").get("use_news_features", False))


def active_feature_cols() -> list:
    """学習・推論で実際に使う特徴量列を返す。

    strategy.use_news_features が有効なときのみニュース特徴量列を加える。
    既定（OFF）では基本テクニカル列のみを返すため、既存モデル・既存テストと
    完全に同一の挙動になる。フラグとモデルの desync は ml_model.load() の
    feature 整合ガードで検知する。
    """
    if _news_enabled():
        return BASE_TECHNICAL_COLS + NEWS_FEATURE_COLS
    return list(BASE_TECHNICAL_COLS)
