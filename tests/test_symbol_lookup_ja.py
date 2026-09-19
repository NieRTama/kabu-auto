"""
GET /api/symbol_lookup/{code} の日本語名優先ロジックのテスト

2026-09-19: yfinanceの`.info`だけを使っていたため、日本株でも英語名しか
返らず、ウォッチリスト追加フォームの自動入力経由で英語名（例:
"SoftBank Corp."）がそのまま登録されていた実害があった。kabuステーション
（国内証券会社のAPI）に接続できていれば、そちらの日本語銘柄名を優先する。
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.dashboard.app as dash


@pytest.fixture
def client():
    dash._auth_required = False
    dash._kabu_client = None
    try:
        yield TestClient(dash.app)
    finally:
        dash._kabu_client = None


class TestBrokerNamePreferred:
    def test_uses_broker_display_name_when_connected(self, client):
        broker = MagicMock()
        broker.get_symbol.return_value = {"DisplayName": "ソフトバンク", "SymbolName": "ソフトバンク（株）"}
        dash.set_kabu_client(broker)
        with patch("src.data.market_data.lookup_company_name") as yf_mock:
            r = client.get("/api/symbol_lookup/9434")
        assert r.status_code == 200
        assert r.json()["name"] == "ソフトバンク"
        yf_mock.assert_not_called()  # ブローカーで取れたのでyfinanceは呼ばない

    def test_falls_back_to_symbol_name_when_display_name_missing(self, client):
        broker = MagicMock()
        broker.get_symbol.return_value = {"SymbolName": "ソフトバンク（株）"}
        dash.set_kabu_client(broker)
        r = client.get("/api/symbol_lookup/9434")
        assert r.json()["name"] == "ソフトバンク（株）"


class TestYfinanceFallback:
    def test_falls_back_to_yfinance_when_not_connected(self, client):
        assert dash._kabu_client is None
        with patch("src.data.market_data.lookup_company_name", return_value="SoftBank Corp.") as yf_mock:
            r = client.get("/api/symbol_lookup/9434")
        assert r.json()["name"] == "SoftBank Corp."
        yf_mock.assert_called_once()

    def test_falls_back_to_yfinance_when_broker_call_raises(self, client):
        broker = MagicMock()
        broker.get_symbol.side_effect = Exception("接続エラー")
        dash.set_kabu_client(broker)
        with patch("src.data.market_data.lookup_company_name", return_value="SoftBank Corp.") as yf_mock:
            r = client.get("/api/symbol_lookup/9434")
        assert r.json()["name"] == "SoftBank Corp."
        yf_mock.assert_called_once()

    def test_falls_back_to_yfinance_when_broker_returns_empty(self, client):
        broker = MagicMock()
        broker.get_symbol.return_value = {}
        dash.set_kabu_client(broker)
        with patch("src.data.market_data.lookup_company_name", return_value="SoftBank Corp.") as yf_mock:
            r = client.get("/api/symbol_lookup/9434")
        assert r.json()["name"] == "SoftBank Corp."
        yf_mock.assert_called_once()
