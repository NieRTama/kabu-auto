"""
pytest 設定・共通フィクスチャ

テスト環境では pandas_ta が利用できないため sys.modules にスタブを注入する。
本番環境では requirements.txt の pandas-ta==0.3.14b0 が使われる。
"""
import sys
import types
from unittest.mock import MagicMock

# ─── pandas_ta スタブ ──────────────────────────────────────────────────────
# pandas_ta が未インストール環境（CI/テスト）向けのスタブ
if "pandas_ta" not in sys.modules:
    import pandas as pd
    import numpy as np

    stub = types.ModuleType("pandas_ta")

    def _bbands(series, length=20, std=2.0, **kwargs):
        m = series.rolling(length).mean()
        s = series.rolling(length).std()
        return pd.DataFrame({
            "BBL": m - std * s,
            "BBM": m,
            "BBU": m + std * s,
        })

    def _macd(series, **kwargs):
        ema12 = series.ewm(span=12).mean()
        ema26 = series.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        hist = macd - signal
        return pd.DataFrame({"MACD": macd, "MACDh": hist, "MACDs": signal})

    def _sma(series, length=14, **kwargs):
        return series.rolling(length).mean()

    def _rsi(series, length=14, **kwargs):
        return pd.Series(50.0, index=series.index)

    stub.bbands = _bbands
    stub.macd = _macd
    stub.sma = _sma
    stub.rsi = _rsi
    sys.modules["pandas_ta"] = stub
