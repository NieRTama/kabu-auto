"""
Discord日次レポート投稿モジュールのテスト
"""
from unittest.mock import MagicMock, patch

import pytest

import src.core.discord_report as discord_report
from src.core.pnl_report import PeriodPnL


def _report():
    return {
        "daily": PeriodPnL("当日", 10000, 0.02, 3, 2),
        "weekly": PeriodPnL("週次", -3000, None, 1, 1),
        "monthly": PeriodPnL("月次", 45000, 0.09, 2, 2),
        "overall": PeriodPnL("総合", 123456, 0.2469, 5, 4),
    }


class TestFormatForDiscord:
    def test_includes_all_periods(self):
        text = discord_report.format_for_discord("paper", _report())
        assert "当日" in text and "週次" in text and "月次" in text and "総合" in text

    def test_within_discord_limit_unchanged(self):
        text = discord_report.format_for_discord("paper", _report())
        assert len(text) < discord_report.DISCORD_MAX_CONTENT_LENGTH
        assert not text.endswith("…")

    def test_truncates_when_over_2000_chars(self):
        huge_report = _report()
        huge_report["daily"] = PeriodPnL("当日" * 1500, 1, 0.01, 1, 0)
        text = discord_report.format_for_discord("paper", huge_report)
        assert len(text) <= discord_report.DISCORD_MAX_CONTENT_LENGTH
        assert text.endswith("…")


class TestPostText:
    def test_disabled_skips(self):
        with patch.object(discord_report.cfg, "get_section", return_value={"enabled": False}):
            assert discord_report.post_text("hello") is False

    def test_enabled_but_no_url_skips(self):
        with patch.object(discord_report.cfg, "get_section", return_value={"enabled": True}):
            assert discord_report.post_text("hello") is False

    def test_enabled_with_url_sends(self):
        section = {"enabled": True, "webhook_url": "https://discord.example/x"}
        mock_provider = MagicMock()
        with patch.object(discord_report.cfg, "get_section", return_value=section), \
             patch.object(discord_report, "DiscordWebhookProvider", return_value=mock_provider):
            result = discord_report.post_text("hello")
        assert result is True
        mock_provider.send.assert_called_once_with("hello")

    def test_send_exception_returns_false(self):
        section = {"enabled": True, "webhook_url": "https://discord.example/x"}
        mock_provider = MagicMock()
        mock_provider.send.side_effect = RuntimeError("boom")
        with patch.object(discord_report.cfg, "get_section", return_value=section), \
             patch.object(discord_report, "DiscordWebhookProvider", return_value=mock_provider):
            result = discord_report.post_text("hello")
        assert result is False


class TestPostDailyReport:
    def test_returns_text_even_when_post_fails(self):
        with patch.object(discord_report, "post_text", return_value=False) as mock_post:
            with patch("src.core.pnl_report.build_report", return_value=_report()):
                text = discord_report.post_daily_report("paper", 500_000)
        assert text is not None
        assert "ペーパー" in text
        mock_post.assert_called_once()
