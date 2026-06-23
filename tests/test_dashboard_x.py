"""
X連携・基準資金設定のダッシュボードAPIのテスト
"""
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import src.core.config as cfg
import src.core.reference_capital as ref_capital
import src.data.database as db
import src.dashboard.app as dash
from src.data.database import Trade, get_session


@pytest.fixture
def isolated(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    ref_capital.load(str(tmp_path / "reference_capital.json"))
    dash._auth_required = False
    try:
        yield tmp_path
    finally:
        db._engine = None
        db._Session = None


class TestReferenceCapitalApi:
    def test_get_returns_defaults(self, isolated):
        client = TestClient(dash.app)
        body = client.get("/api/reference_capital").json()
        assert body["values"] == {"live": 0.0, "dry_run": 0.0, "semi_live": 0.0}

    def test_post_sets_value(self, isolated):
        client = TestClient(dash.app)
        r = client.post("/api/reference_capital", json={"mode": "live", "amount": 300000})
        assert r.status_code == 200
        assert r.json()["values"]["live"] == 300000.0
        # 反映確認
        body = client.get("/api/reference_capital").json()
        assert body["values"]["live"] == 300000.0

    def test_post_paper_rejected(self, isolated):
        client = TestClient(dash.app)
        r = client.post("/api/reference_capital", json={"mode": "paper", "amount": 500000})
        assert r.status_code == 400

    def test_post_negative_rejected(self, isolated):
        client = TestClient(dash.app)
        r = client.post("/api/reference_capital", json={"mode": "live", "amount": -1})
        assert r.status_code == 400


class TestXTestPostApi:
    def test_returns_text_and_length(self, isolated):
        with get_session() as session:
            session.add(Trade(
                order_id="t1", symbol="7203", side="SELL", quantity=100, price=1000,
                filled_price=1000, filled_quantity=100, pnl=10000, status="FILLED",
                filled_at=datetime.now(),
            ))
            session.commit()
        client = TestClient(dash.app)
        r = client.post("/api/x/test_post")
        assert r.status_code == 200
        body = r.json()
        assert "text" in body and "length" in body
        assert body["length"] == len(body["text"])
        assert "kabu-auto" in body["text"]

    def test_does_not_actually_post(self, isolated):
        """test_post はx.enabledに関わらず実際の投稿APIを呼ばない"""
        with patch("src.core.x_poster.post_tweet") as mock_post:
            client = TestClient(dash.app)
            client.post("/api/x/test_post")
        mock_post.assert_not_called()


class TestDiscordReportTestPostApi:
    def test_returns_text_and_length(self, isolated):
        with get_session() as session:
            session.add(Trade(
                order_id="t1", symbol="7203", side="SELL", quantity=100, price=1000,
                filled_price=1000, filled_quantity=100, pnl=10000, status="FILLED",
                filled_at=datetime.now(),
            ))
            session.commit()
        client = TestClient(dash.app)
        r = client.post("/api/discord_report/test_post")
        assert r.status_code == 200
        body = r.json()
        assert "text" in body and "length" in body
        assert body["length"] == len(body["text"])
        assert "kabu-auto" in body["text"]

    def test_does_not_actually_post(self, isolated):
        """test_post はdiscord_report.enabledに関わらず実際の投稿APIを呼ばない"""
        with patch("src.core.discord_report.post_text") as mock_post:
            client = TestClient(dash.app)
            client.post("/api/discord_report/test_post")
        mock_post.assert_not_called()
