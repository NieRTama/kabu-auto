"""
リスクプロファイル切替機能のテスト

config の risk_profiles（low_risk / high_risk の2択）を読み、アクティブプロファイルの
値が trading / strategy セクションへ反映されること、永続化・再読込で維持されることを検証する。
"""
import pytest

import src.core.config as cfg
import src.core.risk_profile as rp


class TestRiskProfile:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        # 各テストで実 config.yaml を読み直し、状態をリセットする
        cfg.load("config.yaml")
        rp.load(str(tmp_path / "risk_profile.json"))
        yield

    def test_default_is_low_risk(self):
        """安全側のローリスクが既定であること（live運用前提のデフォルト方針）"""
        assert rp.get_active() == "low_risk"
        assert cfg.get_section("trading")["max_position_ratio"] == 0.12
        assert cfg.get_section("strategy")["buy_threshold"] == 0.32

    def test_set_high_risk_applies_to_config(self):
        rp.set_active("high_risk")
        assert rp.get_active() == "high_risk"
        assert cfg.get_section("trading")["max_position_ratio"] == 0.35
        assert cfg.get_section("trading")["stop_loss_pct"] == -0.10
        assert cfg.get_section("trading")["max_positions"] == 4
        assert cfg.get_section("trading")["max_daily_loss"] == 60000
        assert cfg.get_section("strategy")["buy_threshold"] == 0.18
        assert cfg.get_section("strategy")["sell_threshold"] == -0.18

    def test_set_low_risk_applies_to_config(self):
        rp.set_active("low_risk")
        assert cfg.get_section("trading")["max_position_ratio"] == 0.12
        assert cfg.get_section("trading")["stop_loss_pct"] == -0.04
        assert cfg.get_section("strategy")["sell_threshold"] == -0.32

    def test_high_risk_is_riskier_than_low_risk(self):
        """high_risk は low_risk より一貫してリスクが大きい方向に設定されていること"""
        profiles = rp.get_profiles()
        low, high = profiles["low_risk"], profiles["high_risk"]
        assert high["max_position_ratio"] > low["max_position_ratio"]
        assert high["stop_loss_pct"] < low["stop_loss_pct"]  # より深い損切り許容
        assert high["max_daily_loss"] > low["max_daily_loss"]
        assert high["buy_threshold"] < low["buy_threshold"]  # 閾値が低い=取引頻度が高い

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError):
            rp.set_active("does_not_exist")

    def test_persisted_and_reloaded(self, tmp_path):
        path = str(tmp_path / "rp_persist.json")
        rp.load(path)
        rp.set_active("high_risk")
        # 別インスタンスの再読込を模して load し直すと high_risk が復元される
        cfg.load("config.yaml")  # config を素の値に戻してから
        name = rp.load(path)
        assert name == "high_risk"
        assert cfg.get_section("trading")["max_position_ratio"] == 0.35

    def test_switch_changes_riskmanager_effective_values(self):
        """RiskManager は trading 辞書の参照を保持するため、切替が即反映される"""
        import src.risk.manager as risk_mod
        risk = risk_mod.RiskManager()
        # low_risk: 12% → 100万の12% = 12万 → 1000円株なら 100株
        qty_low = risk.calc_position_size("7203", 1000.0, 1_000_000)
        rp.set_active("high_risk")  # 35% → 35万 → 300株
        qty_high = risk.calc_position_size("7203", 1000.0, 1_000_000)
        assert qty_low == 100
        assert qty_high == 300
