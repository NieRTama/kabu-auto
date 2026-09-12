"""テクニカル指標計算モジュール（pandas のみ使用）"""
import pandas as pd

from src.core import config as cfg


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()


def _rsi(series: pd.Series, length: int) -> pd.Series:
    """RSI。端点（片側の変動が0の場合）を仕様として明示する。

    従来は loss を NaN に置換していたため、単調上昇の系列で末尾RSIが
    NaN になり、build_features() の dropna で行ごと落ちていた（レビューF06）。
    端点は次のとおり定義する。

      loss == 0 かつ gain > 0 … 100（下げが一度も無い）
      gain == 0 かつ loss > 0 … 0  （上げが一度も無い）
      両方 0                 … 50 （まったく動いていない＝中立）

    助走期間（ewm の min_periods 未満）は NaN のままにする。
    """
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=length - 1, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(com=length - 1, min_periods=length).mean()
    # loss==0 かつ gain>0 なら rs=inf となり 100-(100/inf)=100 に落ちる。
    # 両方0のときだけ 0/0=NaN になるので、中立の50で明示的に埋める。
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.mask((gain == 0) & (loss == 0), 50.0)


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


def _add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """指標から派生する特徴量列を付加する（欠損行の扱いは呼び出し側が決める）。"""
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
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """ML用の特徴量を作成して返す（ラベルは付与しない）。

    **挙動を変更しないこと。** legacy経路（src/backtest/engine.py・
    src/strategy/signal.py）がこの戻り値に依存しており、特に engine.py は
    「この関数が落とした行＝助走期間」という前提で
    `if dt not in featured_df.index` により推論をスキップしている。
    dropna をやめるとそのガードが無効化され、NaN行が推論へ流れる。

    日付を保持したまま欠損をマスクで扱いたい場合は build_feature_frame() を使う。
    """
    return _add_feature_columns(df).dropna(subset=FEATURE_COLS)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量を、日付インデックスを保持したまま返す（v2経路用）。

    build_features() は欠損行を落として返すため、呼び出し側が
    reset_index(drop=True) して行番号を「N営業日」として数えると、
    途中に欠損があった分だけ実日数とずれる（レビューF06後半）。
    本APIは行を落とさず、学習・判定に使ってよい行かを feature_valid 列で示す。
    時間軸の連続性は市場系列側で保ち、採否はマスクで管理する。
    """
    out = _add_feature_columns(df)
    out["feature_valid"] = out[FEATURE_COLS].notna().all(axis=1)
    return out


FEATURE_COLS = [
    "ma_cross_sm", "ma_cross_ml", "rsi", "macd", "macd_hist",
    "bb_pct", "volume_ratio", "price_momentum_5", "price_momentum_20",
    "returns",
]
