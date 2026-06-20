"""
ブローカー建玉ドリフト検知のテスト（再レビュー P0-3）

OrderManager.reconcile_positions_with_broker() が DBの Position と
ブローカー /positions の実保有数を照合し、ズレがあれば kill switch を作動させて
新規発注を停止することを検証する。
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import src.core.config as cfg
import src.core.halt as halt
import src.data.database as db
import src.execution.order_manager as mod
from src.data.database import Position, get_session


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    halt.load(str(tmp_path / "trading_halt.json"))
    try:
        yield tmp_path
    finally:
        db._engine = None
        db._Session = None
        halt.load(str(tmp_path / "nonexistent_halt.json"))


def _cfg(mode: str) -> MagicMock:
    m = MagicMock()
    m.get_section.side_effect = lambda s: {
        "trading": {"mode": mode, "order_timeout_seconds": 300, "daily_order_limit": 100},
        "kabu_station": {"password": "pw"},
    }.get(s, {})
    return m


@contextmanager
def _make_om(mode: str):
    cfg_mock = _cfg(mode)
    client = MagicMock()
    risk = MagicMock()
    with patch.object(mod, "cfg", cfg_mock):
        om = mod.OrderManager(client, risk)
        yield om, client


def _add_position(symbol: str, qty: int) -> None:
    with get_session() as s:
        s.add(Position(symbol=symbol, quantity=qty, avg_cost=1000.0))
        s.commit()


class TestReconcilePositionsWithBroker:
    def test_no_drift_when_matching(self, isolated_db):
        _add_position("7203", 100)
        with _make_om("live") as (om, client):
            client.get_positions.return_value = [{"Symbol": "7203", "LeavesQty": 100}]
            result = om.reconcile_positions_with_broker()
        assert result["ok"] is True
        assert result["drift"] == []
        assert halt.is_halted() is False

    def test_db_has_position_broker_does_not_triggers_halt(self, isolated_db):
        _add_position("7203", 100)
        with _make_om("live") as (om, client):
            client.get_positions.return_value = []
            with patch.object(mod, "alert") as alert_mock:
                result = om.reconcile_positions_with_broker()
        assert result["ok"] is False
        assert result["drift"][0]["symbol"] == "7203"
        assert halt.is_halted() is True
        alert_mock.assert_called_once()

    def test_broker_has_position_db_does_not_triggers_halt(self, isolated_db):
        with _make_om("live") as (om, client):
            client.get_positions.return_value = [{"Symbol": "7203", "LeavesQty": 50}]
            with patch.object(mod, "alert"):
                result = om.reconcile_positions_with_broker()
        assert result["ok"] is False
        assert halt.is_halted() is True

    def test_quantity_mismatch_triggers_halt(self, isolated_db):
        _add_position("7203", 100)
        with _make_om("live") as (om, client):
            client.get_positions.return_value = [{"Symbol": "7203", "LeavesQty": 80}]
            with patch.object(mod, "alert"):
                result = om.reconcile_positions_with_broker()
        assert result["ok"] is False
        drift = result["drift"][0]
        assert drift["db_qty"] == 100 and drift["broker_qty"] == 80
        assert halt.is_halted() is True

    def test_does_not_re_engage_already_halted(self, isolated_db):
        """既にhalt中なら理由を上書きして二重にengageしない（既存の停止理由を保持）"""
        halt.engage("既存の停止理由")
        _add_position("7203", 100)
        with _make_om("live") as (om, client):
            client.get_positions.return_value = []
            with patch.object(mod, "alert"):
                om.reconcile_positions_with_broker()
        assert halt.get_state()["reason"] == "既存の停止理由"

    def test_paper_mode_does_nothing(self, isolated_db):
        with _make_om("paper") as (om, client):
            result = om.reconcile_positions_with_broker()
        assert result == {"ok": True, "drift": []}
        client.get_positions.assert_not_called()

    def test_dry_run_mode_does_nothing(self, isolated_db):
        with _make_om("dry_run") as (om, client):
            result = om.reconcile_positions_with_broker()
        assert result == {"ok": True, "drift": []}
        client.get_positions.assert_not_called()

    def test_get_positions_failure_does_not_raise(self, isolated_db):
        with _make_om("live") as (om, client):
            client.get_positions.side_effect = RuntimeError("network")
            result = om.reconcile_positions_with_broker()
        assert result["ok"] is False
        assert halt.is_halted() is False  # API失敗自体はドリフト確定ではないので停止しない
