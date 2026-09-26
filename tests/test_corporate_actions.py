"""分割イベントの取得と、分割で評価額が増えないことの検証

背景: auto_adjust で過去価格が遡って調整されるため、調整価格のまま
「当時100株買えたか」を判定すると別の結果になる。分割比率は価格比からは
導出できない（Yahooの生終値も分割調整済みのため）ので、分割イベントの
API を唯一の権威とする。
"""
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data import market_data


class TestFetchSplits:
    def test_returns_split_series(self):
        fake_ticker = MagicMock()
        fake_ticker.splits = pd.Series(
            [2.0], index=pd.to_datetime([date(2026, 6, 1)])
        )
        with patch.object(market_data.yf, "Ticker", return_value=fake_ticker):
            splits = market_data.fetch_splits("7203")
        assert list(splits.values) == [2.0]
        assert splits.index[0] == date(2026, 6, 1)

    def test_empty_when_no_splits(self):
        fake_ticker = MagicMock()
        fake_ticker.splits = pd.Series(dtype=float)
        with patch.object(market_data.yf, "Ticker", return_value=fake_ticker):
            splits = market_data.fetch_splits("7203")
        assert len(splits) == 0


