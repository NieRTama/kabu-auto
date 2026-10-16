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
        import src.core.alerts as alerts_mod
        with patch.object(discord_report.cfg, "get_section", return_value=section), \
             patch.object(discord_report, "DiscordWebhookProvider", return_value=mock_provider), \
             patch.object(alerts_mod.time, "sleep"):
            result = discord_report.post_text("hello")
        assert result is False
        # アラートと同じ再送経路を通ること（一時障害でレポートを黙って落とさない）
        assert mock_provider.send.call_count == alerts_mod.ALERT_SEND_ATTEMPTS


class TestPostDailyReport:
    def test_returns_text_even_when_post_fails(self):
        with patch.object(discord_report, "post_text", return_value=False) as mock_post:
            with patch("src.core.pnl_report.build_report", return_value=_report()):
                text = discord_report.post_daily_report("paper", 500_000)
        assert text is not None
        assert "ペーパー" in text
        mock_post.assert_called_once()


# ─── 総資産の行 ───────────────────────────────────────────────────

ASSETS = {"total": 492380.0, "day_change": 3160.0, "prev_label": "前日比",
          "basis_change": -7620.0, "unpriced": [], "cash_error": None}


class TestTotalAssetsLine:
    def test_format_line(self):
        from src.core.total_assets import format_line
        assert format_line(ASSETS) == "総資産: 492,380円（前日比 +3,160円 ／ 基準資金比 -7,620円）"

    def test_format_line_unavailable_says_why(self):
        from src.core.total_assets import format_line
        line = format_line({"total": None, "cash_error": "401 Unauthorized"})
        assert line == "総資産: 取得不可（401 Unauthorized）"

    def test_daily_report_puts_assets_after_mode(self):
        text = discord_report.format_for_discord("paper", _report(), assets_line="総資産: 1円")
        lines = text.split("\n")
        assert lines[1].startswith("モード:") and lines[2] == "総資産: 1円"

    def test_weekly_summary_includes_assets(self):
        text = discord_report.format_weekly_summary("paper", _report(), assets_line="総資産: 1円")
        assert text.split("\n")[2] == "総資産: 1円"

    def test_without_assets_line_unchanged(self):
        assert "総資産" not in discord_report.format_for_discord("paper", _report())
