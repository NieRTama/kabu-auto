"""
ウォッチリスト重複防止のテスト（Phase 4 / 6.9）

- 同一リスト内の同じ銘柄コードは DuplicateError
- '7203' と '7203.T'（yfinanceサフィックス）は同一銘柄として重複判定される
- 別リストには同じ銘柄を追加できる
- ダッシュボードAPIは重複時に 409 を返す
"""
import pytest
from fastapi.testclient import TestClient

import src.core.watchlist as wl


@pytest.fixture
def isolated(tmp_path):
    wl.load(str(tmp_path / "watchlists.json"), legacy_path=str(tmp_path / "no_legacy.json"))
    return tmp_path


class TestNormalizeSuffix:
    def test_strips_dot_t_suffix(self, isolated):
        assert wl.normalize_code("7203.T") == "7203"
        assert wl.normalize_code("7203.t") == "7203"

    def test_fullwidth_and_suffix(self, isolated):
        assert wl.normalize_code("７２０３.T") == "7203"

    def test_plain_code_unchanged(self, isolated):
        assert wl.normalize_code("7203") == "7203"


class TestAddDedup:
    def test_duplicate_same_code_raises(self, isolated):
        wl.add("7203", "トヨタ")
        with pytest.raises(wl.DuplicateError):
            wl.add("7203", "トヨタ再")

    def test_duplicate_via_dot_t_raises(self, isolated):
        wl.add("7203", "トヨタ")
        with pytest.raises(wl.DuplicateError):
            wl.add("7203.T", "トヨタ")

    def test_duplicate_via_fullwidth_raises(self, isolated):
        wl.add("7203", "トヨタ")
        with pytest.raises(wl.DuplicateError):
            wl.add("７２０３", "トヨタ")

    def test_different_code_ok(self, isolated):
        wl.add("7203", "トヨタ")
        wl.add("8306", "三菱UFJ")
        assert set(wl.get_codes()) == {"7203", "8306"}

    def test_same_code_different_list_ok(self, isolated):
        wl.add("7203", "トヨタ")
        wl.create_list("別リスト")
        wl.add("7203", "トヨタ")  # 別リストなのでOK
        assert wl.get_codes() == ["7203"]

    def test_duplicate_error_is_value_error(self, isolated):
        """DuplicateError は ValueError のサブクラス（既存の except ValueError 互換）"""
        assert issubclass(wl.DuplicateError, ValueError)


class TestDashboardDedup409:
    @pytest.fixture
    def client(self, isolated, monkeypatch):
        import src.dashboard.app as dash
        import src.core.config as cfg
        cfg.load("config.yaml")
        dash._auth_required = False
        # add 経由の過去データ取得・セクター取得は外部通信なのでスタブ化
        import src.data.market_data as md
        monkeypatch.setattr(md, "update_symbol", lambda *a, **k: None)
        monkeypatch.setattr(md, "lookup_sector", lambda *a, **k: "")
        return TestClient(dash.app)

    def test_add_duplicate_returns_409(self, client):
        r1 = client.post("/api/watchlist", json={"code": "7203", "name": "トヨタ"})
        assert r1.status_code == 200
        r2 = client.post("/api/watchlist", json={"code": "7203", "name": "トヨタ"})
        assert r2.status_code == 409

    def test_add_duplicate_dot_t_returns_409(self, client):
        client.post("/api/watchlist", json={"code": "7203", "name": "トヨタ"})
        r = client.post("/api/watchlist", json={"code": "7203.T", "name": "トヨタ"})
        assert r.status_code == 409
