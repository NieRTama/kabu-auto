"""
Phase 3 ダッシュボードAPIのテスト
- /api/status の運用情報拡張（mode/endpoint/LAN公開/発注可否）
- /api/halt （kill switch）GET/POST/DELETE
- /api/approvals （semi_live 承認キュー）
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import src.core.config as cfg
import src.dashboard.app as dash


@pytest.fixture
def client(tmp_path):
    cfg.load("config.yaml")
    dash._auth_required = False
    dash._emergency_token = "test-emg-token"
    saved_om = dash._order_manager
    yield TestClient(dash.app)
    dash._order_manager = saved_om


class TestStatusEnrichment:
    def test_status_includes_mode_and_lan(self, client):
        dash._system_status["mode"] = "paper"
        dash._order_manager = MagicMock()
        dash._order_manager.status_snapshot.return_value = {
            "mode": "paper", "can_place_order": True, "block_reason": "",
            "halt": {"halted": False}, "unresolved_orders": 0, "pending_approvals": 0,
        }
        body = client.get("/api/status").json()
        assert body["mode"] == "paper"
        assert "mode_description" in body
        assert "lan_exposed" in body
        assert body["places_real_orders"] is False
        assert body["can_place_order"] is True


class TestHaltEndpoints:
    def test_get_halt_status(self, client, tmp_path):
        from src.core import halt
        halt.load(str(tmp_path / "halt.json"))  # 非停止の独立状態で開始
        body = client.get("/api/halt").json()
        assert body["halted"] is False

    def test_engage_requires_emergency_token(self, client):
        dash._order_manager = MagicMock()
        r = client.post("/api/halt", json={"reason": "x"})
        assert r.status_code == 422 or r.status_code == 403  # ヘッダー欠落

    def test_engage_with_token(self, client):
        om = MagicMock()
        om.halt_trading.return_value = {"state": {"halted": True}, "cancelled_buys": 0,
                                        "closed_positions": False}
        dash._order_manager = om
        r = client.post("/api/halt", json={"reason": "異常検知"},
                        headers={"X-Emergency-Token": "test-emg-token"})
        assert r.status_code == 200
        om.halt_trading.assert_called_once()

    def test_release_blocked_returns_409(self, client):
        om = MagicMock()
        om.resume_trading.return_value = {"ok": False, "reason": "未解決の注文があります",
                                          "state": {"halted": True}}
        dash._order_manager = om
        r = client.delete("/api/halt", headers={"X-Emergency-Token": "test-emg-token"})
        assert r.status_code == 409


class TestApprovalEndpoints:
    def test_list_approvals(self, client):
        om = MagicMock()
        om.list_pending_approvals.return_value = [
            {"id": 1, "symbol": "7203", "side": "BUY", "order_type": "LIMIT",
             "price": 1000, "quantity": 100, "sector": "Tech", "created_at": None},
        ]
        dash._order_manager = om
        body = client.get("/api/approvals").json()
        assert len(body) == 1
        assert body[0]["symbol"] == "7203"

    def test_approve(self, client):
        om = MagicMock()
        om.approve_order.return_value = {"ok": True, "order_id": "LIVE-1", "reason": ""}
        dash._order_manager = om
        r = client.post("/api/approvals/1/approve")
        assert r.status_code == 200
        om.approve_order.assert_called_once_with(1)

    def test_approve_failure_returns_400(self, client):
        om = MagicMock()
        om.approve_order.return_value = {"ok": False, "order_id": None, "reason": "発注失敗"}
        dash._order_manager = om
        r = client.post("/api/approvals/1/approve")
        assert r.status_code == 400

    def test_reject(self, client):
        om = MagicMock()
        om.reject_order.return_value = {"ok": True, "reason": ""}
        dash._order_manager = om
        r = client.post("/api/approvals/1/reject")
        assert r.status_code == 200
        om.reject_order.assert_called_once_with(1)
