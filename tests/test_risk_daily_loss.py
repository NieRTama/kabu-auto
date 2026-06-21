"""
RiskManager 当日損失上限のテスト

High #5: record_loss() / is_daily_loss_limit_reached() / can_place_order() 統合
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import src.risk.manager as mod


@contextmanager
def _make_risk(max_daily_loss: int = 30000, daily_order_limit: int = 100):
    cfg_mock = MagicMock()
    cfg_mock.get_section.return_value = {
        "max_daily_loss": max_daily_loss,
        "daily_order_limit": daily_order_limit,
        "max_positions": 5,
    }

    @contextmanager
    def fake_session():
        session = MagicMock()
        session.scalar.return_value = 0
        yield session

    with patch.object(mod, "cfg", cfg_mock), \
         patch.object(mod, "get_session", fake_session):
        yield mod.RiskManager()


class TestRecordLoss:
    def test_negative_pnl_accumulates(self):
        """損失(負のpnl)は _daily_loss_yen に加算される"""
        with _make_risk() as risk:
            risk.record_loss(-5000)
            risk.record_loss(-3000)
            assert risk._daily_loss_yen == 8000.0

    def test_positive_pnl_not_accumulated(self):
        """利益(正のpnl)は損失カウンタに加算されない"""
        with _make_risk() as risk:
            risk.record_loss(10000)
            assert risk._daily_loss_yen == 0.0

    def test_zero_pnl_not_accumulated(self):
        """pnl=0 は損失カウンタに影響しない"""
        with _make_risk() as risk:
            risk.record_loss(0)
            assert risk._daily_loss_yen == 0.0

    def test_mixed_pnl(self):
        """損失と利益が混在しても損失分のみ加算される"""
        with _make_risk() as risk:
            risk.record_loss(-10000)
            risk.record_loss(5000)
            risk.record_loss(-8000)
            assert risk._daily_loss_yen == 18000.0


class TestDailyLossLimitCheck:
    def test_under_limit_returns_false(self):
        """損失が上限未満なら over=False を返す"""
        with _make_risk(max_daily_loss=30000) as risk:
            risk.record_loss(-29999)
            over, _ = risk.is_daily_loss_limit_reached()
            assert over is False

    def test_at_limit_returns_true(self):
        """損失が上限に達したら (True, message) を返す"""
        with _make_risk(max_daily_loss=30000) as risk:
            risk.record_loss(-30000)
            over, reason = risk.is_daily_loss_limit_reached()
            assert over is True
            assert "30,000" in reason

    def test_over_limit_returns_true(self):
        """損失が上限を超えた場合も True を返す"""
        with _make_risk(max_daily_loss=30000) as risk:
            risk.record_loss(-50000)
            over, _ = risk.is_daily_loss_limit_reached()
            assert over is True

    def test_zero_limit_disables_check(self):
        """max_daily_loss=0 なら損失上限チェック無効"""
        with _make_risk(max_daily_loss=0) as risk:
            risk.record_loss(-999999)
            over, _ = risk.is_daily_loss_limit_reached()
            assert over is False


class TestCanPlaceOrderWithLossLimit:
    def test_loss_limit_blocks_order(self):
        """当日損失上限到達時は can_place_order() が (False, reason) を返す"""
        with _make_risk(max_daily_loss=30000) as risk:
            risk.record_loss(-30000)
            ok, reason = risk.can_place_order()
            assert ok is False
            assert "損失" in reason

    def test_order_count_limit_still_works(self):
        """注文数上限チェックも引き続き機能する"""
        with _make_risk(max_daily_loss=30000, daily_order_limit=5) as risk:
            risk._daily_order_count = 5
            ok, reason = risk.can_place_order()
            assert ok is False
            assert "注文上限" in reason

    def test_no_limit_reached_returns_true(self):
        """どちらの上限も未達なら (True, '') を返す"""
        with _make_risk(max_daily_loss=30000) as risk:
            risk.record_loss(-1000)
            ok, reason = risk.can_place_order()
            assert ok is True
            assert reason == ""


class TestTotalDrawdownLimit:
    def _pos(self, symbol, qty, avg_cost):
        p = MagicMock()
        p.symbol = symbol
        p.quantity = qty
        p.avg_cost = avg_cost
        return p

    def test_unrealized_loss_pushes_over_limit(self):
        """実現損失が上限未満でも、含み損を足すと合計ドローダウン上限で発注が止まる"""
        with _make_risk(max_daily_loss=30000) as risk:
            risk.record_loss(-10000)
            snap = mod.RiskSnapshot(
                positions=[self._pos("7203", 100, 3000)],
                closes={"7203": 2700},  # 含み損 = (2700-3000)*100 = -30000
            )
            over, reason = risk.is_total_loss_limit_reached(snap)
            assert over is True
            assert "合計ドローダウン" in reason
            ok, block = risk.can_place_order(snap)
            assert ok is False

    def test_unrealized_profit_does_not_offset_realized(self):
        """含み益は実現損失を相殺しない（安全側）"""
        with _make_risk(max_daily_loss=30000) as risk:
            risk.record_loss(-5000)
            snap = mod.RiskSnapshot(
                positions=[self._pos("7203", 100, 3000)],
                closes={"7203": 4000},  # 含み益 +100000
            )
            over, _ = risk.is_total_loss_limit_reached(snap)
            assert over is False

    def test_unpriced_symbol_excluded_and_flagged(self):
        """終値が取れない銘柄は合計から除外され、unpriced_symbols() で検知できる"""
        with _make_risk(max_daily_loss=30000) as risk:
            snap = mod.RiskSnapshot(
                positions=[self._pos("9999", 100, 3000)],
                closes={},  # 終値取得不可
            )
            assert risk.unrealized_pnl(snap) == 0.0
            assert risk.unpriced_symbols(snap) == ["9999"]

    def test_no_positions_equals_realized(self):
        """建玉が無ければ合計ドローダウン = 実現損失"""
        with _make_risk(max_daily_loss=30000) as risk:
            risk.record_loss(-30000)
            snap = mod.RiskSnapshot(positions=[], closes={})
            assert risk.current_total_drawdown(snap) == 30000.0


class TestResetDailyCounters:
    def test_reset_clears_loss(self):
        """reset_daily_counters() は _daily_loss_yen をリセットする"""
        with _make_risk() as risk:
            risk.record_loss(-20000)
            risk.reset_daily_counters()
            assert risk._daily_loss_yen == 0.0

    def test_reset_clears_order_count(self):
        """reset_daily_counters() は _daily_order_count もリセットする"""
        with _make_risk() as risk:
            risk._daily_order_count = 50
            risk.reset_daily_counters()
            assert risk._daily_order_count == 0

    def test_after_reset_orders_allowed(self):
        """リセット後は損失上限到達していても発注可能に戻る"""
        with _make_risk(max_daily_loss=30000) as risk:
            risk.record_loss(-50000)
            ok, _ = risk.can_place_order()
            assert ok is False
            risk.reset_daily_counters()
            ok, _ = risk.can_place_order()
            assert ok is True
