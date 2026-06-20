"""
カスタムリスクプロファイルのテスト（Phase 4 / 6.2-6.8・11.2・11.6）
- 共有バリデーション（無効値は弾く・丸めない）
- create / update / delete / clone
- export / import / compare / history
- ダッシュボード: 実発注モードでの選択は二重確認
"""
import pytest
from fastapi.testclient import TestClient

import src.core.config as cfg
import src.core.risk_profile as rp


def _valid_params(**over) -> dict:
    p = {
        "max_position_ratio": 0.2,
        "stop_loss_pct": -0.05,
        "max_positions": 5,
        "max_sector_ratio": 0.4,
        "max_daily_loss": 20000,
        "buy_threshold": 0.1,
        "sell_threshold": -0.1,
    }
    p.update(over)
    return p


@pytest.fixture(autouse=True)
def _setup(tmp_path):
    cfg.load("config.yaml")
    rp.load(str(tmp_path / "risk_profile.json"))
    yield


class TestValidation:
    def test_valid_passes(self):
        assert rp.validate_profile(_valid_params())["max_positions"] == 5

    def test_position_ratio_out_of_range_fails(self):
        with pytest.raises(ValueError):
            rp.validate_profile(_valid_params(max_position_ratio=1.5))

    def test_positive_stop_loss_fails(self):
        with pytest.raises(ValueError):
            rp.validate_profile(_valid_params(stop_loss_pct=0.05))

    def test_negative_buy_threshold_fails(self):
        with pytest.raises(ValueError):
            rp.validate_profile(_valid_params(buy_threshold=-0.1))

    def test_positive_sell_threshold_fails(self):
        with pytest.raises(ValueError):
            rp.validate_profile(_valid_params(sell_threshold=0.1))

    def test_missing_key_fails(self):
        p = _valid_params()
        del p["max_positions"]
        with pytest.raises(ValueError):
            rp.validate_profile(p)

    def test_unknown_key_fails(self):
        with pytest.raises(ValueError):
            rp.validate_profile(_valid_params(bogus=1))

    def test_zero_max_positions_fails(self):
        with pytest.raises(ValueError):
            rp.validate_profile(_valid_params(max_positions=0))


class TestCustomCRUD:
    def test_create_and_select(self):
        rp.create_custom("攻め", _valid_params(max_position_ratio=0.3))
        assert "攻め" in rp.get_profiles()
        rp.set_active("攻め")
        assert rp.get_active() == "攻め"
        assert cfg.get_section("trading")["max_position_ratio"] == 0.3

    def test_cannot_use_builtin_name(self):
        with pytest.raises(ValueError):
            rp.create_custom("low_risk", _valid_params())

    def test_create_invalid_rejected(self):
        with pytest.raises(ValueError):
            rp.create_custom("だめ", _valid_params(max_daily_loss=-1))

    def test_update_active_reapplies(self):
        rp.create_custom("p1", _valid_params(max_position_ratio=0.2))
        rp.set_active("p1")
        rp.update_custom("p1", _valid_params(max_position_ratio=0.25))
        assert cfg.get_section("trading")["max_position_ratio"] == 0.25

    def test_cannot_update_builtin(self):
        with pytest.raises(ValueError):
            rp.update_custom("low_risk", _valid_params())

    def test_delete_custom(self):
        rp.create_custom("p2", _valid_params())
        rp.delete_custom("p2")
        assert "p2" not in rp.get_profiles()

    def test_cannot_delete_active(self):
        rp.create_custom("p3", _valid_params())
        rp.set_active("p3")
        with pytest.raises(ValueError):
            rp.delete_custom("p3")

    def test_clone_builtin(self):
        rp.clone("high_risk", "high_copy")
        profiles = rp.get_profiles()
        assert profiles["high_copy"] == profiles["high_risk"]


class TestImportExportCompare:
    def test_export(self):
        data = rp.export_profile("low_risk")
        assert data["name"] == "low_risk"
        assert "max_position_ratio" in data["params"]

    def test_import_creates_custom(self):
        rp.import_profile("取込", _valid_params())
        assert "取込" in rp.get_profiles()

    def test_import_invalid_rejected(self):
        with pytest.raises(ValueError):
            rp.import_profile("不正", _valid_params(stop_loss_pct=1.0))

    def test_import_overwrite(self):
        rp.import_profile("x", _valid_params(max_positions=3))
        rp.import_profile("x", _valid_params(max_positions=7), overwrite=True)
        assert rp.get_profiles()["x"]["max_positions"] == 7

    def test_compare_diff(self):
        result = rp.compare("low_risk", "high_risk")
        assert "max_position_ratio" in result["diff"]
        assert result["diff"]["max_position_ratio"][0] != result["diff"]["max_position_ratio"][1]


class TestHistoryAndPersistence:
    def test_history_records_switch(self):
        rp.set_active("high_risk")
        hist = rp.get_history()
        assert hist[0]["action"] == "switch"
        assert hist[0]["to"] == "high_risk"

    def test_custom_persisted_and_reloaded(self, tmp_path):
        path = str(tmp_path / "rp2.json")
        rp.load(path)
        rp.create_custom("永続", _valid_params(max_positions=8))
        rp.load(path)  # 再読込
        assert "永続" in rp.get_profiles()
        assert rp.get_profiles()["永続"]["max_positions"] == 8


class TestDashboardLiveConfirmation:
    @pytest.fixture
    def client(self, tmp_path):
        import src.dashboard.app as dash
        dash._auth_required = False
        rp.load(str(tmp_path / "rp_dash.json"))
        return TestClient(dash.app), dash

    def test_paper_mode_no_confirmation_needed(self, client):
        c, dash = client
        dash._system_status["mode"] = "paper"
        r = c.post("/api/risk_profile", json={"name": "high_risk"})
        assert r.status_code == 200

    def test_live_mode_requires_confirm(self, client):
        c, dash = client
        dash._system_status["mode"] = "live"
        r = c.post("/api/risk_profile", json={"name": "high_risk"})
        assert r.status_code == 409
        r2 = c.post("/api/risk_profile", json={"name": "high_risk", "confirm": True})
        assert r2.status_code == 200
        dash._system_status["mode"] = "paper"  # 後始末

    def test_create_custom_via_api(self, client):
        c, dash = client
        r = c.post("/api/risk_profiles", json={"name": "API作成", "params": _valid_params()})
        assert r.status_code == 200
        r2 = c.post("/api/risk_profiles", json={"name": "ダメ", "params": _valid_params(max_positions=0)})
        assert r2.status_code == 400
