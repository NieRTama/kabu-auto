"""
X投稿モジュール（テンプレート整形・投稿処理）のテスト
"""
from unittest.mock import MagicMock, patch

import pytest

import src.core.x_poster as x_poster
from src.core.pnl_report import PeriodPnL


def _report(daily_pnl=10000, daily_pct=0.02, weekly_pnl=-3000, monthly_pnl=45000,
           overall_pnl=123456, win_rate_counts=(3, 2)):
    win, loss = win_rate_counts
    return {
        "daily": PeriodPnL("当日", daily_pnl, daily_pct, win, loss),
        "weekly": PeriodPnL("週次", weekly_pnl, None, 1, 1),
        "monthly": PeriodPnL("月次", monthly_pnl, 0.09, 2, 2),
        "overall": PeriodPnL("総合", overall_pnl, 0.2469, 5, 4),
    }


class TestFormatDailyReport:
    def test_includes_mode_and_all_periods(self):
        text = x_poster.format_daily_report("paper", _report())
        assert "ペーパー" in text
        assert "当日" in text and "週次" in text and "月次" in text and "総合" in text

    def test_yen_amount_with_sign(self):
        text = x_poster.format_daily_report("paper", _report(daily_pnl=10000))
        assert "+10,000円" in text

    def test_negative_amount_no_plus(self):
        text = x_poster.format_daily_report("paper", _report(weekly_pnl=-3000))
        assert "-3,000円" in text
        assert "+-3,000円" not in text

    def test_pct_shown_when_available(self):
        text = x_poster.format_daily_report("paper", _report(daily_pct=0.0247))
        assert "+2.5%" in text or "+2.47%" in text  # 表示桁は実装依存だが%は出る

    def test_pct_omitted_when_none(self):
        text = x_poster.format_daily_report("paper", _report())
        # weekly の pct=None の行に "()" が出ないこと
        lines = text.splitlines()
        weekly_line = next(l for l in lines if l.startswith("週次"))
        assert "(" not in weekly_line

    def test_win_rate_shown(self):
        text = x_poster.format_daily_report("paper", _report(win_rate_counts=(3, 2)))
        assert "勝率60%" in text

    def test_truncates_when_over_280_chars(self):
        huge_report = _report()
        huge_report["daily"] = PeriodPnL("当日" * 200, 1, 0.01, 1, 0)
        text = x_poster.format_daily_report("paper", huge_report)
        assert len(text) <= x_poster.X_MAX_LENGTH
        assert text.endswith(x_poster._TRUNCATION_SUFFIX)


class TestGetClient:
    def test_returns_none_when_keys_missing(self):
        with patch.object(x_poster.cfg, "get_section", return_value={}):
            assert x_poster._get_client() is None

    def test_builds_client_when_all_keys_present(self):
        section = {
            "api_key": "k", "api_secret": "s",
            "access_token": "t", "access_token_secret": "ts",
        }
        with patch.object(x_poster.cfg, "get_section", return_value=section):
            client = x_poster._get_client()
        assert client is not None


class TestPostTweet:
    def test_disabled_skips(self):
        with patch.object(x_poster.cfg, "get_section", return_value={"enabled": False}):
            assert x_poster.post_tweet("hello") is False

    def test_enabled_but_no_keys_skips(self):
        with patch.object(x_poster.cfg, "get_section", return_value={"enabled": True}):
            assert x_poster.post_tweet("hello") is False

    def test_enabled_with_keys_calls_create_tweet(self):
        section = {
            "enabled": True, "api_key": "k", "api_secret": "s",
            "access_token": "t", "access_token_secret": "ts",
        }
        mock_client = MagicMock()
        with patch.object(x_poster.cfg, "get_section", return_value=section), \
             patch.object(x_poster, "_get_client", return_value=mock_client):
            result = x_poster.post_tweet("hello")
        assert result is True
        mock_client.create_tweet.assert_called_once_with(text="hello")

    def test_create_tweet_exception_returns_false(self):
        section = {
            "enabled": True, "api_key": "k", "api_secret": "s",
            "access_token": "t", "access_token_secret": "ts",
        }
        mock_client = MagicMock()
        mock_client.create_tweet.side_effect = RuntimeError("boom")
        with patch.object(x_poster.cfg, "get_section", return_value=section), \
             patch.object(x_poster, "_get_client", return_value=mock_client):
            result = x_poster.post_tweet("hello")
        assert result is False


class TestPostDailyReport:
    def test_returns_text_even_when_post_fails(self):
        with patch.object(x_poster, "post_tweet", return_value=False) as mock_post:
            with patch("src.core.pnl_report.build_report", return_value=_report()):
                text = x_poster.post_daily_report("paper", 500_000)
        assert text is not None
        assert "ペーパー" in text
        mock_post.assert_called_once()
