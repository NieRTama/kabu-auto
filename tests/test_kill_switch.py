"""
取引停止スイッチ（kill switch）のテスト（Phase 3 / 7.1）

- halt モジュールの永続化（engage/release/load）
- RiskManager.can_place_order() が停止中に新規発注を弾くこと
- 損切り/緊急決済（sell_market の reason バイパス）は停止中も止まらないこと
- OrderManager.halt_trading()/resume_trading() の挙動（未約定BUYキャンセル・未解決ガード）
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import src.core.halt as halt
import src.risk.manager as risk_mod
import src.execution.order_manager as om_mod
from src.execution import order_status as st


@pytest.fixture
def halt_file(tmp_path):
    """各テストを独立した一時ファイルの停止状態で動かす。"""
    path = tmp_path / "trading_halt.json"
    halt.load(str(path))
    yield path
    # 後始末: モジュールグローバルの停止状態が他テストへ漏れないよう、
    # 存在しないパスを読み直して非停止の初期状態に戻す
    halt.load(str(tmp_path / "nonexistent_halt.json"))


class TestHaltModule:
    def test_default_not_halted(self, halt_file):
        assert halt.is_halted() is False
        assert halt.get_state()["halted"] is False

    def test_engage_sets_flag_and_persists(self, halt_file):
        halt.engage("テスト停止")
        assert halt.is_halted() is True
        assert halt.get_state()["reason"] == "テスト停止"
        # 別ファイルから読み直しても停止状態が維持される（永続化）
        halt.load(str(halt_file))
        assert halt.is_halted() is True

    def test_release_clears_flag(self, halt_file):
        halt.engage("x")
        halt.release()
        assert halt.is_halted() is False
        halt.load(str(halt_file))
        assert halt.is_halted() is False

    def test_engage_default_reason(self, halt_file):
        halt.engage()
        assert halt.get_state()["reason"] == "手動停止"


@contextmanager
def _make_risk():
    cfg_mock = MagicMock()
    cfg_mock.get_section.return_value = {
        "max_daily_loss": 30000, "daily_order_limit": 100, "max_positions": 5,
    }

    @contextmanager
    def fake_session():
        session = MagicMock()
        session.scalar.return_value = 0
        yield session

    with patch.object(risk_mod, "cfg", cfg_mock), \
         patch.object(risk_mod, "get_session", fake_session):
        yield risk_mod.RiskManager()


class TestRiskManagerHaltGate:
    def test_halt_blocks_new_orders(self, halt_file):
        with _make_risk() as risk:
            halt.engage("緊急停止")
            ok, reason = risk.can_place_order()
            assert ok is False
            assert "取引停止中" in reason
            assert "緊急停止" in reason

    def test_resume_allows_orders(self, halt_file):
        with _make_risk() as risk:
            halt.engage("x")
            assert risk.can_place_order()[0] is False
            halt.release()
            assert risk.can_place_order()[0] is True


@contextmanager
def _make_order_mgr(open_buy_ids=None, unresolved=0):
    cfg_mock = MagicMock()
    cfg_mock.get_section.return_value = {"mode": "live", "order_timeout_seconds": 300}
    client = MagicMock()
    risk = MagicMock()
    with patch.object(om_mod, "cfg", cfg_mock):
        om = om_mod.OrderManager(client, risk)
    yield om


class TestOrderManagerKillSwitch:
    def test_halt_trading_cancels_pending_buys(self, halt_file):
        with _make_order_mgr() as om:
            om.cancel_all_pending_buys = MagicMock(return_value=2)
            om.close_all_positions = MagicMock()
            with patch.object(om_mod, "alert"):
                result = om.halt_trading("異常検知", close_positions=False)
            assert halt.is_halted() is True
            om.cancel_all_pending_buys.assert_called_once()
            om.close_all_positions.assert_not_called()
            assert result["cancelled_buys"] == 2

    def test_halt_trading_can_close_positions(self, halt_file):
        with _make_order_mgr() as om:
            om.cancel_all_pending_buys = MagicMock(return_value=0)
            om.close_all_positions = MagicMock()
            with patch.object(om_mod, "alert"):
                om.halt_trading("x", close_positions=True)
            om.close_all_positions.assert_called_once()

    def test_resume_blocked_when_unresolved(self, halt_file):
        with _make_order_mgr() as om:
            halt.engage("x")
            om._count_unresolved = MagicMock(return_value=3)
            result = om.resume_trading()
            assert result["ok"] is False
            assert "未解決" in result["reason"]
            assert halt.is_halted() is True  # 解除されない

    def test_resume_succeeds_when_clean(self, halt_file):
        with _make_order_mgr() as om:
            halt.engage("x")
            om._count_unresolved = MagicMock(return_value=0)
            result = om.resume_trading()
            assert result["ok"] is True
            assert halt.is_halted() is False
