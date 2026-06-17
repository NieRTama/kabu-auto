"""
RiskManager.restore_daily_state() のテスト

プロセス再起動で日次カウンタが消えると当日損失上限・注文上限のセーフティが
無効化されるため、起動時に当日（JST）の約定TradeからDBで再構築することを検証する。
"""
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import src.risk.manager as mod


def _make_risk_with_trades(trades):
    cfg_mock = MagicMock()
    cfg_mock.get_section.return_value = {"max_daily_loss": 30000, "daily_order_limit": 100}

    @contextmanager
    def fake_session():
        session = MagicMock()
        scalars_result = MagicMock()
        scalars_result.all.return_value = trades
        session.scalars.return_value = scalars_result
        yield session

    with patch.object(mod, "cfg", cfg_mock), \
         patch.object(mod, "get_session", fake_session):
        risk = mod.RiskManager()
        risk.restore_daily_state()
        return risk


def _trade(filled_at, pnl):
    t = MagicMock()
    t.filled_at = filled_at
    t.pnl = pnl
    return t


class TestRestoreDailyState:
    def test_counts_today_orders_and_losses(self):
        now = datetime.now()
        risk = _make_risk_with_trades([
            _trade(now, -1000),   # 損失
            _trade(now, 500),     # 利益（損失カウンタには入らない）
            _trade(now, -2000),   # 損失
        ])
        assert risk._daily_order_count == 3
        assert risk._daily_loss_yen == 3000.0

    def test_ignores_other_days(self):
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        risk = _make_risk_with_trades([
            _trade(yesterday, -5000),  # 前日 → 無視
            _trade(now, -200),         # 当日
        ])
        assert risk._daily_order_count == 1
        assert risk._daily_loss_yen == 200.0

    def test_ignores_unfilled_trades(self):
        risk = _make_risk_with_trades([
            _trade(None, -9999),   # 未約定（filled_at=None）→ 無視
        ])
        assert risk._daily_order_count == 0
        assert risk._daily_loss_yen == 0.0

    def test_restored_loss_blocks_new_orders(self):
        """復元した当日損失が上限超なら can_place_order が発注を止める"""
        now = datetime.now()
        risk = _make_risk_with_trades([_trade(now, -40000)])
        ok, reason = risk.can_place_order()
        assert ok is False
        assert "損失" in reason
