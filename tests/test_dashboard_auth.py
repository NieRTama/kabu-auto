"""
ダッシュボードのアクセス認証（C-1）のテスト

販売に向けたセキュリティ強化として、ダッシュボードに無認証でアクセスできない
ことを確認する。認証はトークン（X-API-Token ヘッダー / ?token= クエリ / Cookie）で行う。
既定（localhost かつトークン未設定）では認証なしで動く後方互換も確認する。
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import src.core.auth as auth_store
import src.dashboard.app as dash


@pytest.fixture
def auth_enabled(tmp_path):
    """認証を有効化した状態を作る。Authファイル・セッションも隔離し、テスト後に戻す。"""
    orig_required = dash._auth_required
    orig_token = dash._dashboard_token
    dash._auth_required = True
    dash._dashboard_token = "testtoken123"
    auth_store.load(str(tmp_path / "auth.json"))  # 未設定状態から開始
    auth_store._sessions.clear()
    try:
        yield "testtoken123"
    finally:
        dash._auth_required = orig_required
        dash._dashboard_token = orig_token
        auth_store._sessions.clear()


class TestDashboardAuth:
    def test_no_token_is_unauthorized(self, auth_enabled):
        client = TestClient(dash.app)
        r = client.get("/api/status")
        assert r.status_code == 401

    def test_wrong_token_is_unauthorized(self, auth_enabled):
        client = TestClient(dash.app)
        r = client.get("/api/status", headers={"X-API-Token": "wrong"})
        assert r.status_code == 401

    def test_header_token_allows_access(self, auth_enabled):
        client = TestClient(dash.app)
        r = client.get("/api/status", headers={"X-API-Token": auth_enabled})
        assert r.status_code == 200

    def test_query_token_allows_and_sets_cookie(self, auth_enabled):
        client = TestClient(dash.app)
        r = client.get(f"/api/status?token={auth_enabled}")
        assert r.status_code == 200
        # 以後の fetch 用に Cookie が払い出される
        assert "kabu_token" in r.cookies

    def test_mutating_endpoint_also_protected(self, auth_enabled):
        client = TestClient(dash.app)
        r = client.post("/api/risk_profile", json={"name": "low_risk"})
        assert r.status_code == 401

    def test_openapi_protected_when_auth_enabled(self, auth_enabled):
        """認証有効時は /openapi.json も保護され、API仕様を露出しない（M-D）"""
        client = TestClient(dash.app)
        assert client.get("/openapi.json").status_code == 401
        assert client.get("/docs").status_code == 401

    def test_cookie_persists_across_requests(self, auth_enabled):
        """?token= で一度通すと Cookie が保存され、以後トークン無しでも通る"""
        client = TestClient(dash.app)
        first = client.get(f"/api/status?token={auth_enabled}")
        assert first.status_code == 200
        # 同じclientは払い出されたCookieを保持するので、token無しでも通る
        second = client.get("/api/status")
        assert second.status_code == 200


class TestAuthDisabledByDefault:
    def test_localhost_without_token_disables_auth(self):
        """host=127.0.0.1 かつトークン未設定なら認証なし（ローカル専用の後方互換）"""
        fake = {"dashboard": {"host": "127.0.0.1", "api_token": "", "port": 8080}}
        with patch.object(dash.cfg, "get_section", lambda s: fake.get(s, {})):
            with patch.dict("os.environ", {}, clear=False):
                # KABU_DASHBOARD_TOKEN を確実に未設定にする
                import os
                os.environ.pop("KABU_DASHBOARD_TOKEN", None)
                dash.init_auth()
        try:
            assert dash._auth_required is False
            client = TestClient(dash.app)
            assert client.get("/api/status").status_code == 200
        finally:
            dash._auth_required = False
            dash._dashboard_token = None


class TestInitAuth:
    def test_configured_token_enables_auth(self):
        fake = {"dashboard": {"host": "127.0.0.1", "api_token": "mytoken", "port": 8080}}
        with patch.object(dash.cfg, "get_section", lambda s: fake.get(s, {})):
            dash.init_auth()
        try:
            assert dash._auth_required is True
            assert dash._dashboard_token == "mytoken"
        finally:
            dash._auth_required = False
            dash._dashboard_token = None

    def test_lan_host_without_token_autogenerates(self):
        """host=0.0.0.0（LAN公開）でトークン未設定なら自動生成して認証を強制する"""
        fake = {"dashboard": {"host": "0.0.0.0", "api_token": "", "port": 8080}}
        import os
        os.environ.pop("KABU_DASHBOARD_TOKEN", None)
        with patch.object(dash.cfg, "get_section", lambda s: fake.get(s, {})):
            dash.init_auth()
        try:
            assert dash._auth_required is True
            assert dash._dashboard_token  # 自動生成された非空トークン
        finally:
            dash._auth_required = False
            dash._dashboard_token = None

    def test_wildcard_host_console_message_shows_real_lan_ip_not_0000(self, capsys):
        """host=0.0.0.0 のとき、コンソール表示URLは到達不能な 0.0.0.0 ではなく実際のLAN IPにする"""
        fake = {"dashboard": {"host": "0.0.0.0", "api_token": "", "port": 8080}}
        import os
        os.environ.pop("KABU_DASHBOARD_TOKEN", None)
        with patch.object(dash.cfg, "get_section", lambda s: fake.get(s, {})):
            with patch.object(dash, "_get_lan_ip", return_value="192.168.1.50"):
                dash.init_auth()
        try:
            out = capsys.readouterr().out
            assert "http://192.168.1.50:8080/" in out
            assert "http://0.0.0.0:8080/" not in out
        finally:
            dash._auth_required = False
            dash._dashboard_token = None

    def test_specific_lan_ip_host_used_as_is(self):
        """host が具体的なLAN IPの場合は、それをそのまま表示する（0.0.0.0変換は不要）"""
        fake = {"dashboard": {"host": "192.168.1.50", "api_token": "", "port": 8080}}
        import os
        os.environ.pop("KABU_DASHBOARD_TOKEN", None)
        with patch.object(dash.cfg, "get_section", lambda s: fake.get(s, {})):
            with patch.object(dash, "_get_lan_ip") as mock_lan_ip:
                dash.init_auth()
                mock_lan_ip.assert_not_called()
        dash._auth_required = False
        dash._dashboard_token = None


class TestLoginFlow:
    def test_auth_status_exempt_and_reports_unconfigured(self, auth_enabled):
        """/api/auth_status は認証なしでアクセスでき、初回は configured:false"""
        client = TestClient(dash.app)
        r = client.get("/api/auth_status")
        assert r.status_code == 200
        body = r.json()
        assert body["configured"] is False
        assert body["auth_required"] is True

    def test_login_page_reachable_without_auth(self, auth_enabled):
        client = TestClient(dash.app)
        # /login は素通しパス（認証必須でもアクセスできる）
        assert client.get("/login").status_code == 200

    def test_html_navigation_redirects_to_login(self, auth_enabled):
        client = TestClient(dash.app)
        r = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/login"

    def test_setup_then_access_granted(self, auth_enabled):
        client = TestClient(dash.app)
        r = client.post("/api/setup", json={"username": "trader", "password": "s3cretpw!"})
        assert r.status_code == 200
        assert "kabu_session" in r.cookies
        # セッションCookieを保持したまま保護APIにアクセスできる
        assert client.get("/api/status").status_code == 200

    def test_setup_twice_rejected(self, auth_enabled):
        client = TestClient(dash.app)
        client.post("/api/setup", json={"username": "trader", "password": "s3cretpw!"})
        r = client.post("/api/setup", json={"username": "x", "password": "y2345678"})
        assert r.status_code == 400

    def test_login_wrong_and_correct(self, auth_enabled):
        # 事前に作成（別clientで）
        setup_client = TestClient(dash.app)
        setup_client.post("/api/setup", json={"username": "trader", "password": "s3cretpw!"})

        client = TestClient(dash.app)  # セッションCookieを持たない新規client
        bad = client.post("/api/login", json={"username": "trader", "password": "wrong"})
        assert bad.status_code == 401
        good = client.post("/api/login", json={"username": "trader", "password": "s3cretpw!"})
        assert good.status_code == 200
        assert "kabu_session" in good.cookies
        assert client.get("/api/status").status_code == 200

    def test_logout_invalidates_session(self, auth_enabled):
        client = TestClient(dash.app)
        client.post("/api/setup", json={"username": "trader", "password": "s3cretpw!"})
        assert client.get("/api/status").status_code == 200
        client.post("/api/logout")
        # ログアウト後はセッションが無効化され、再びアクセス不可
        r = client.get("/api/status")
        assert r.status_code == 401

    def test_api_token_still_works_alongside_login(self, auth_enabled):
        """ログイン導入後も X-API-Token（curl用）は併用できる"""
        client = TestClient(dash.app)
        r = client.get("/api/status", headers={"X-API-Token": auth_enabled})
        assert r.status_code == 200
