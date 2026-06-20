"""
取引根拠記録のテスト（Phase 5 / 7.6）
- buy/sell/sell_market が rationale を Trade に記録すること
- 退出系 sell_market は reason から根拠を自動補完すること
- services の _signal_rationale がスコアから根拠文を作ること
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

import src.execution.order_manager as mod
from src.services.trading import _signal_rationale


def _cfg(mode="paper"):
    m = MagicMock()
    m.get_section.side_effect = lambda s: {
        "trading": {"mode": mode, "order_timeout_seconds": 300, "daily_order_limit": 100},
        "kabu_station": {"password": "pw"},
    }.get(s, {})
    return m


@contextmanager
def _make_om(mode="paper"):
    cfg_mock = _cfg(mode)
    client = MagicMock()
    client.send_order.return_value = {"Result": 0, "OrderId": "LIVE-1"}
    risk = MagicMock()
    risk.can_place_order.return_value = (True, "")
    session = MagicMock()
    session.scalar.return_value = None
    added = []
    session.add.side_effect = lambda obj: added.append(obj)

    @contextmanager
    def ctx():
        yield session

    with patch.object(mod, "cfg", cfg_mock), patch.object(mod, "get_session", ctx), \
         patch.object(mod, "latest_closes", lambda syms: {s: 1000.0 for s in syms}):
        om = mod.OrderManager(client, risk)
        yield om, added


class TestRationaleRecording:
    def test_paper_buy_records_rationale(self):
        with _make_om("paper") as (om, added):
            om.buy("7203", 1000.0, 100, sector="Tech", rationale="BUY score=0.15")
        trade = added[0]
        assert trade.rationale == "BUY score=0.15"

    def test_paper_sell_records_rationale(self):
        with _make_om("paper") as (om, added):
            om.sell("7203", 1100.0, 100, rationale="SELL score=-0.15")
        assert added[0].rationale == "SELL score=-0.15"

    def test_stop_loss_market_autofills_rationale(self):
        with _make_om("paper") as (om, added):
            om.sell_market("7203", 100, reason="stop_loss")
        assert "損切り" in added[0].rationale

    def test_emergency_market_autofills_rationale(self):
        with _make_om("paper") as (om, added):
            om.sell_market("7203", 100, reason="emergency")
        assert "緊急決済" in added[0].rationale

    def test_explicit_rationale_overrides_exit_default(self):
        with _make_om("paper") as (om, added):
            om.sell_market("7203", 100, reason="stop_loss", rationale="手動損切り")
        assert added[0].rationale == "手動損切り"

    def test_dry_run_records_rationale(self):
        with _make_om("dry_run") as (om, added):
            om.buy("7203", 1000.0, 100, rationale="BUY score=0.2")
        assert added[0].rationale == "BUY score=0.2"


class TestSignalRationale:
    def test_formats_scores(self):
        sig = SimpleNamespace(action="BUY", combined_score=0.153, rule_score=0.12, ml_score=0.18)
        r = _signal_rationale(sig)
        assert "BUY" in r and "0.153" in r and "rule=0.120" in r and "ml=0.180" in r

    def test_handles_none_scores(self):
        sig = SimpleNamespace(action="SELL", combined_score=-0.1, rule_score=None, ml_score=None)
        r = _signal_rationale(sig)
        assert "rule=—" in r and "ml=—" in r
