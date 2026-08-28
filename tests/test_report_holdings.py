"""
日次レポートの保有状況（評価額・含み損益）のテスト

実現損益（build_report）だけでは決済前の含み損益が見えないため、
現在のポジション状況も併せて報告する。

価格が取得できない銘柄を「無かったこと」にして評価額を過小表示しないこと
（RiskManager._unrealized_pnl_with_gaps と同じ方針）を重点的に検証する。
"""
from unittest.mock import MagicMock, patch

import pytest

from src.core import pnl_report


def _pos(symbol: str, qty: int, avg_cost: float):
    p = MagicMock()
    p.symbol = symbol
    p.quantity = qty
    p.avg_cost = avg_cost
    return p


def _build(positions, closes, reference_capital=0.0):
    with patch.object(pnl_report, "get_session") as sess_mock, \
         patch("src.data.market_data.latest_closes", return_value=closes):
        ctx = MagicMock()
        ctx.__enter__.return_value.scalars.return_value.all.return_value = positions
        sess_mock.return_value = ctx
        return pnl_report.build_holdings(reference_capital)


class TestBuildHoldings:
    def test_empty_when_no_positions(self):
        h = _build([], {})
        assert h["count"] == 0
        assert h["market_value"] == 0
        assert h["unrealized"] == 0

    def test_computes_market_value_and_unrealized_gain(self):
        h = _build([_pos("9432", 100, 160.0)], {"9432": 180.0})
        assert h["count"] == 1
        assert h["cost"] == 16000.0
        assert h["market_value"] == 18000.0
        assert h["unrealized"] == 2000.0

    def test_computes_unrealized_loss(self):
        h = _build([_pos("9432", 100, 160.0)], {"9432": 150.0})
        assert h["unrealized"] == -1000.0

    def test_sums_multiple_positions(self):
        h = _build(
            [_pos("9432", 100, 160.0), _pos("7203", 100, 2000.0)],
            {"9432": 170.0, "7203": 2100.0},
        )
        assert h["count"] == 2
        assert h["market_value"] == 17000.0 + 210000.0
        assert h["unrealized"] == 1000.0 + 10000.0

    def test_pct_uses_reference_capital(self):
        h = _build([_pos("9432", 100, 160.0)], {"9432": 210.0}, reference_capital=500000.0)
        assert h["unrealized"] == 5000.0
        assert h["pct"] == pytest.approx(0.01)

    def test_pct_none_without_reference_capital(self):
        h = _build([_pos("9432", 100, 160.0)], {"9432": 170.0})
        assert h["pct"] is None


class TestUnpricedNotHidden:
    """価格が取れない銘柄を黙って0円扱いして過小表示しないこと"""

    def test_unpriced_symbol_is_reported(self):
        h = _build([_pos("9432", 100, 160.0)], {})
        assert h["unpriced"] == ["9432"]
        assert h["market_value"] == 0, "価格不明分は評価額に含めない"

    def test_priced_and_unpriced_are_separated(self):
        h = _build(
            [_pos("9432", 100, 160.0), _pos("7203", 100, 2000.0)],
            {"9432": 170.0},  # 7203 の価格が無い
        )
        assert h["unpriced"] == ["7203"]
        assert h["market_value"] == 17000.0, "価格が取れた分だけを評価額に含める"
        assert h["count"] == 2, "保有銘柄数には含める（存在は隠さない）"


class TestFormatting:
    def _report(self):
        from src.core.pnl_report import PeriodPnL
        p = PeriodPnL(label="当日", realized_pnl=0.0, pct=None, win_count=0, loss_count=0)
        return {"daily": p, "weekly": p, "monthly": p, "overall": p}

    def test_holdings_omitted_when_not_passed(self):
        """従来どおり実現損益のみ（後方互換）"""
        text = pnl_report.format_report_text("live", self._report())
        assert "評価額" not in text

    def test_shows_market_value_and_unrealized(self):
        h = {"count": 1, "cost": 16000.0, "market_value": 18000.0,
             "unrealized": 2000.0, "pct": 0.004, "unpriced": []}
        text = pnl_report.format_report_text("live", self._report(), h)
        assert "評価額: 18,000円" in text
        assert "取得 16,000円" in text
        assert "含み損益: +2,000円" in text

    def test_negative_unrealized_has_no_plus_sign(self):
        h = {"count": 1, "cost": 16000.0, "market_value": 15000.0,
             "unrealized": -1000.0, "pct": None, "unpriced": []}
        text = pnl_report.format_report_text("live", self._report(), h)
        assert "含み損益: -1,000円" in text

    def test_shows_no_holdings(self):
        h = {"count": 0, "cost": 0, "market_value": 0,
             "unrealized": 0, "pct": None, "unpriced": []}
        assert "保有: なし" in pnl_report.format_report_text("live", self._report(), h)

    def test_warns_about_unpriced_symbols(self):
        h = {"count": 2, "cost": 16000.0, "market_value": 17000.0,
             "unrealized": 1000.0, "pct": None, "unpriced": ["7203"]}
        text = pnl_report.format_report_text("live", self._report(), h)
        assert "価格取得不可" in text and "7203" in text


class TestDiscordIntegration:
    def test_post_includes_holdings(self):
        import src.core.discord_report as mod
        from src.core.pnl_report import PeriodPnL
        p = PeriodPnL(label="当日", realized_pnl=0.0, pct=None, win_count=0, loss_count=0)
        holdings = {"count": 1, "cost": 16000.0, "market_value": 18000.0,
                    "unrealized": 2000.0, "pct": None, "unpriced": []}
        with patch.object(mod, "build_report", create=True), \
             patch("src.core.pnl_report.build_report",
                   return_value={"daily": p, "weekly": p, "monthly": p, "overall": p}), \
             patch("src.core.pnl_report.build_holdings", return_value=holdings), \
             patch.object(mod, "post_text") as post_mock:
            text = mod.post_daily_report("live", 0.0)
        assert "評価額" in text
        post_mock.assert_called_once()

    def test_holdings_failure_still_posts_realized(self):
        """保有状況の集計に失敗しても実現損益は投稿する（部分失敗で全部を落とさない）"""
        import src.core.discord_report as mod
        from src.core.pnl_report import PeriodPnL
        p = PeriodPnL(label="当日", realized_pnl=0.0, pct=None, win_count=0, loss_count=0)
        with patch("src.core.pnl_report.build_report",
                   return_value={"daily": p, "weekly": p, "monthly": p, "overall": p}), \
             patch("src.core.pnl_report.build_holdings", side_effect=RuntimeError("db down")), \
             patch.object(mod, "post_text") as post_mock:
            text = mod.post_daily_report("live", 0.0)
        assert "日次レポート" in text
        post_mock.assert_called_once()
