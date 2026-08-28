"""
broker_auth（kabuステーションのログイン認証切れ）のテスト

kabuステーションの認証はPCを起動したままでも日をまたぐと切れる
（アプリ自身が "Code 10016: ログイン認証の有効期間が切れました" を持つ）。
2026-08-26/27 は毎朝8:30のトークン更新が401で失敗した後リトライが無く、
終日1,300回超の401を出しながら「発注可」を表示し続ける抜け殻状態だった。

ここでは「認証切れの間は新規発注を止める／退出は止めない」ことを検証する。
"""
from unittest.mock import MagicMock, patch

import pytest

from src.core import broker_auth


@pytest.fixture(autouse=True)
def _reset_state():
    broker_auth.reset()
    yield
    broker_auth.reset()


class TestState:
    def test_initially_valid(self):
        assert broker_auth.is_expired() is False

    def test_mark_expired_sets_state_with_detail_and_timestamp(self):
        broker_auth.mark_expired("401 Unauthorized")
        assert broker_auth.is_expired() is True
        s = broker_auth.get_state()
        assert s["expired"] is True
        assert "401" in s["detail"]
        assert s["since"] is not None

    def test_mark_valid_clears_state(self):
        broker_auth.mark_expired("401")
        broker_auth.mark_valid()
        assert broker_auth.is_expired() is False
        assert broker_auth.get_state()["since"] is None

    def test_repeated_mark_expired_logs_once(self):
        """毎朝の失敗後リトライのたびにCRITICALを重ねない（ログのノイズ抑制）"""
        with patch("src.core.broker_auth.logger") as log_mock:
            broker_auth.mark_expired("401")
            broker_auth.mark_expired("401")
            broker_auth.mark_expired("401")
        assert log_mock.critical.call_count == 1

    def test_since_is_preserved_across_repeated_marks(self):
        broker_auth.mark_expired("401")
        first = broker_auth.get_state()["since"]
        broker_auth.mark_expired("401 again")
        assert broker_auth.get_state()["since"] == first, "検知時刻は最初のものを保つ"


# ─── 発注ゲートとの結線 ──────────────────────────────────────────────


def _risk():
    import src.risk.manager as risk_mod
    with patch.object(risk_mod, "cfg") as cfg_mock:
        cfg_mock.get_section.return_value = {"daily_order_limit": 100, "max_daily_loss": 0}
        r = risk_mod.RiskManager()
    return r


class TestOrderGate:
    def test_blocks_new_orders_when_expired(self):
        r = _risk()
        with patch.object(r, "_count_unresolved_orders", return_value=0), \
             patch.object(r, "is_total_loss_limit_reached", return_value=(False, "")):
            ok_before, _ = r.can_place_order()
            broker_auth.mark_expired("401 Unauthorized")
            ok_after, reason = r.can_place_order()
        assert ok_before is True
        assert ok_after is False
        assert "認証" in reason or "ログイン" in reason

    def test_allows_orders_again_after_recovery(self):
        r = _risk()
        with patch.object(r, "_count_unresolved_orders", return_value=0), \
             patch.object(r, "is_total_loss_limit_reached", return_value=(False, "")):
            broker_auth.mark_expired("401")
            assert r.can_place_order()[0] is False
            broker_auth.mark_valid()
            assert r.can_place_order()[0] is True


class TestExitNotBlocked:
    def test_sell_market_exit_bypasses_auth_gate(self):
        """認証切れでも退出（損切り・トレーリング・緊急）は止めない。

        can_place_order() を通さない reason 経路であることをソースで担保する
        （既存の kill switch / 損失上限バイパスと同じ思想）。
        """
        import inspect
        import src.execution.order_manager as mod
        src = inspect.getsource(mod.OrderManager.sell_market)
        assert 'is_exit = reason in ("stop_loss", "trailing_stop", "emergency")' in src
        assert "if not is_exit:" in src, (
            "退出系が can_place_order() ゲートをバイパスする構造が壊れている"
        )
