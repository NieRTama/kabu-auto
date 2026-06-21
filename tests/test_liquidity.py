"""
流動性フィルタ（レビュー P0-6）のテスト
"""
from datetime import datetime

import pandas as pd
import pytz

from src.core.scheduler import TradingScheduler
from src.risk import liquidity

TZ = pytz.timezone("Asia/Tokyo")


def _df(closes, volumes):
    return pd.DataFrame({"close": closes, "volume": volumes})


class TestAverageTurnover:
    def test_mean_of_close_times_volume(self):
        df = _df([100, 200], [10, 20])  # 1000, 4000 -> mean 2500
        assert liquidity.average_turnover(df, window=20) == 2500.0

    def test_empty_df_returns_zero(self):
        assert liquidity.average_turnover(pd.DataFrame(), window=20) == 0.0

    def test_window_limits_rows(self):
        df = _df([100, 100, 100], [1, 1, 100])  # last 1 day -> 100*100=10000
        assert liquidity.average_turnover(df, window=1) == 10000.0


class TestCheckLiquidity:
    def test_disabled_when_threshold_zero(self):
        ok, _ = liquidity.check_liquidity("7203", pd.DataFrame(), {"min_avg_turnover_yen": 0})
        assert ok is True

    def test_blocks_below_threshold(self):
        df = _df([1000] * 20, [100] * 20)  # turnover 100,000
        ok, reason = liquidity.check_liquidity(
            "7203", df, {"min_avg_turnover_yen": 1_000_000})
        assert ok is False
        assert "流動性不足" in reason

    def test_allows_above_threshold(self):
        df = _df([1000] * 20, [100000] * 20)  # turnover 100,000,000
        ok, _ = liquidity.check_liquidity(
            "7203", df, {"min_avg_turnover_yen": 1_000_000})
        assert ok is True


class TestCheckSpread:
    def test_disabled_when_zero(self):
        ok, _ = liquidity.check_spread({"Buy1": {"Price": 100}, "Sell1": {"Price": 200}},
                                       {"max_spread_ratio": 0})
        assert ok is True

    def test_blocks_wide_spread(self):
        board = {"Buy1": {"Price": 100}, "Sell1": {"Price": 110}}  # spread ~9.5%
        ok, reason = liquidity.check_spread(board, {"max_spread_ratio": 0.01})
        assert ok is False
        assert "スプレッド" in reason

    def test_allows_tight_spread(self):
        board = {"Buy1": {"Price": 1000}, "Sell1": {"Price": 1001}}  # ~0.1%
        ok, _ = liquidity.check_spread(board, {"max_spread_ratio": 0.01})
        assert ok is True

    def test_disabled_skips_even_without_board(self):
        ok, _ = liquidity.check_spread(None, {"max_spread_ratio": 0})
        assert ok is True

    def test_blocks_when_no_board_and_enabled(self):
        """有効化時に板情報そのものが無い場合はfail-closedでブロックする"""
        ok, reason = liquidity.check_spread(None, {"max_spread_ratio": 0.01})
        assert ok is False
        assert "板情報" in reason

    def test_blocks_when_quotes_missing_and_enabled(self):
        """有効化時に最良気配が欠落（取引停止等）していればfail-closedでブロックする"""
        # 空dict({})はPythonでfalsyなため「板情報なし」分岐と区別するべく非空dictを使う
        ok, reason = liquidity.check_spread({"Symbol": "7203"}, {"max_spread_ratio": 0.01})
        assert ok is False
        assert "気配" in reason


class TestNearClose:
    def test_disabled_when_zero(self):
        assert TradingScheduler.is_near_close(0) is False

    def test_true_within_window_before_close(self):
        now = TZ.localize(datetime(2026, 6, 22, 15, 25))  # 月曜 15:25, 引け5分前
        assert TradingScheduler.is_near_close(10, now=now) is True

    def test_false_in_morning(self):
        now = TZ.localize(datetime(2026, 6, 22, 9, 10))
        assert TradingScheduler.is_near_close(10, now=now) is False

    def test_true_after_close(self):
        now = TZ.localize(datetime(2026, 6, 22, 16, 0))
        assert TradingScheduler.is_near_close(10, now=now) is True

    def test_false_on_weekend(self):
        now = TZ.localize(datetime(2026, 6, 20, 15, 25))  # 土曜
        assert TradingScheduler.is_near_close(10, now=now) is False
