"""
異常検知・アラートのテスト（Phase 5 / 7.5）
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import src.core.health as health


@pytest.fixture(autouse=True)
def _reset(tmp_path):
    health.reset()
    from src.core import halt
    halt.load(str(tmp_path / "halt.json"))  # 非停止
    yield
    health.reset()
    halt.load(str(tmp_path / "halt.json"))


@contextmanager
def _patch_unresolved(count):
    session = MagicMock()
    session.scalar.return_value = count

    @contextmanager
    def ctx():
        yield session

    with patch.object(health, "get_session", ctx):
        yield


def _risk(loss=0.0, limit=30000):
    r = MagicMock()
    r.current_daily_loss.return_value = loss
    r.daily_loss_limit.return_value = limit
    return r


class TestCheckAnomalies:
    def test_no_anomalies(self):
        with _patch_unresolved(0):
            assert health.check_anomalies(_risk()) == []

    def test_unresolved_orders_critical(self):
        with _patch_unresolved(2):
            items = health.check_anomalies(_risk())
        keys = {i["key"]: i for i in items}
        assert keys["unresolved_orders"]["level"] == health.CRITICAL

    def test_daily_loss_warn_at_80pct(self):
        with _patch_unresolved(0):
            items = health.check_anomalies(_risk(loss=24000, limit=30000))  # 80%
        assert any(i["key"] == "daily_loss_warn" for i in items)

    def test_daily_loss_limit_critical(self):
        with _patch_unresolved(0):
            items = health.check_anomalies(_risk(loss=30000, limit=30000))
        assert any(i["key"] == "daily_loss_limit" and i["level"] == health.CRITICAL
                   for i in items)

    def test_loss_disabled_when_limit_zero(self):
        with _patch_unresolved(0):
            items = health.check_anomalies(_risk(loss=99999, limit=0))
        assert not any("daily_loss" in i["key"] for i in items)

    def test_halt_warning(self):
        from src.core import halt
        halt.engage("test")
        try:
            with _patch_unresolved(0):
                items = health.check_anomalies(_risk())
            assert any(i["key"] == "halted" for i in items)
        finally:
            halt.release()


class TestRunAndAlert:
    def test_alerts_only_on_new(self):
        with _patch_unresolved(2):
            with patch.object(health, "alert") as mock_alert:
                health.run_and_alert(_risk())
                assert mock_alert.call_count == 1  # unresolved を1回通知
                health.run_and_alert(_risk())  # 同じ異常 → 再通知しない
                assert mock_alert.call_count == 1

    def test_recovery_logged_and_realert_possible(self):
        with patch.object(health, "alert") as mock_alert:
            with _patch_unresolved(2):
                health.run_and_alert(_risk())
            assert mock_alert.call_count == 1
            with _patch_unresolved(0):
                health.run_and_alert(_risk())  # 解消
            with _patch_unresolved(2):
                health.run_and_alert(_risk())  # 再発 → 再通知
            assert mock_alert.call_count == 2
