"""日足取得の境界・2系列保存・更新結果のテスト

背景: yfinance の end は排他境界のため、営業日Tの引け後に実行しても
Tの足が取得できなかった（F01）。境界の解釈を1箇所に固定する。
"""
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from src.core import config as cfg
from src.data import database as db
from src.data import market_data


def _fake_yf_frame(days: list[date]) -> pd.DataFrame:
    """yf.download が返す形（auto_adjust=False 相当）を模す。"""
    return pd.DataFrame(
        {
            "Open": [1000.0] * len(days),
            "High": [1010.0] * len(days),
            "Low": [990.0] * len(days),
            "Close": [1005.0] * len(days),
            "Adj Close": [1002.0] * len(days),
            "Volume": [100000] * len(days),
        },
        index=pd.to_datetime(days),
    )


class TestFetchBoundary:
    def test_end_is_inclusive_one_day_added_internally(self):
        """呼び出し側は end を含む日付として渡す。+1日は関数内部だけで行う。"""
        captured = {}

        def fake_download(sym, **kwargs):
            captured.update(kwargs)
            return _fake_yf_frame([date(2026, 9, 10)])

        with patch.object(market_data.yf, "download", side_effect=fake_download):
            market_data.fetch_ohlcv("7203", date(2026, 9, 1), date(2026, 9, 10))

        # yfinance へは排他境界（+1日）が渡る
        assert captured["end"] == "2026-09-11"
        assert captured["start"] == "2026-09-01"

    def test_returned_frame_includes_end_date(self):
        with patch.object(market_data.yf, "download",
                          return_value=_fake_yf_frame([date(2026, 9, 9), date(2026, 9, 10)])):
            df = market_data.fetch_ohlcv("7203", date(2026, 9, 1), date(2026, 9, 10))
        assert df.index[-1] == date(2026, 9, 10)


class TestRawAndAdjustedSeparation:
    def test_fetch_keeps_raw_ohlc_and_adjusted_close(self):
        """auto_adjust=False で取得し、生OHLCと調整済み終値を別々に持つ"""
        captured = {}

        def fake_download(sym, **kwargs):
            captured.update(kwargs)
            return _fake_yf_frame([date(2026, 9, 10)])

        with patch.object(market_data.yf, "download", side_effect=fake_download):
            df = market_data.fetch_ohlcv("7203", date(2026, 9, 1), date(2026, 9, 10))

        assert captured["auto_adjust"] is False
        assert df["close"].iloc[0] == 1005.0          # 生の終値
        assert df["adjusted_close"].iloc[0] == 1002.0  # 調整済み終値


class TestLoadPriceBasis:
    def test_raw_and_adjusted_return_different_close(self, tmp_path):
        cfg.load("config.yaml")
        cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
        db.init()

        df = pd.DataFrame(
            {
                "open": [1000.0], "high": [1010.0], "low": [990.0],
                "close": [1005.0], "adjusted_close": [502.5], "volume": [100000],
            },
            index=[date(2026, 9, 10)],
        )
        df.index.name = "date"
        market_data.upsert_ohlcv("7203", df)

        raw = market_data.load_ohlcv("7203", price_basis="raw")
        adj = market_data.load_ohlcv("7203", price_basis="adjusted")
        assert raw["close"].iloc[0] == 1005.0
        assert adj["close"].iloc[0] == 502.5

    def test_default_is_adjusted(self, tmp_path):
        cfg.load("config.yaml")
        cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
        db.init()

        df = pd.DataFrame(
            {
                "open": [1000.0], "high": [1010.0], "low": [990.0],
                "close": [1005.0], "adjusted_close": [502.5], "volume": [100000],
            },
            index=[date(2026, 9, 10)],
        )
        df.index.name = "date"
        market_data.upsert_ohlcv("7203", df)

        assert market_data.load_ohlcv("7203")["close"].iloc[0] == 502.5
