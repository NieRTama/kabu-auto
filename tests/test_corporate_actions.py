"""分割イベントの取得と、分割で評価額が増えないことの検証

背景: auto_adjust で過去価格が遡って調整されるため、調整価格のまま
「当時100株買えたか」を判定すると別の結果になる。分割比率は価格比からは
導出できない（Yahooの生終値も分割調整済みのため）ので、分割イベントの
API を唯一の権威とする。
"""
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core import config as cfg
from src.data import database as db
from src.data import market_data


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


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


class TestSplitFactor:
    def test_factor_is_product_of_splits_in_range(self, isolated_db):
        market_data.upsert_splits("7203", pd.Series(
            [2.0, 3.0], index=[date(2026, 6, 1), date(2026, 7, 1)]
        ))
        assert market_data.split_factor_between(
            "7203", date(2026, 5, 1), date(2026, 8, 1)) == 6.0

    def test_factor_is_one_when_no_split_in_range(self, isolated_db):
        market_data.upsert_splits("7203", pd.Series(
            [2.0], index=[date(2026, 6, 1)]
        ))
        assert market_data.split_factor_between(
            "7203", date(2026, 7, 1), date(2026, 8, 1)) == 1.0


class TestSplitInvariant:
    def test_split_alone_does_not_change_valuation(self, isolated_db):
        """1対2分割で株数は2倍・価格は半値になり、評価額は変わらない"""
        market_data.upsert_splits("7203", pd.Series(
            [2.0], index=[date(2026, 6, 1)]
        ))
        factor = market_data.split_factor_between(
            "7203", date(2026, 5, 1), date(2026, 7, 1))

        before_qty, before_price = 100, 1000.0
        after_qty = int(before_qty * factor)
        after_price = before_price / factor

        assert after_qty == 200
        assert after_price == 500.0
        assert after_qty * after_price == before_qty * before_price
