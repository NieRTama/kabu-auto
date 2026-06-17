"""
リスクプロファイル切替機能のテスト

config の risk_profiles を読み、アクティブプロファイルの値が trading / strategy
セクションへ反映されること、永続化・再読込で維持されることを検証する。
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

    def test_default_is_balanced(self):
        assert rp.get_active() == "balanced"
        assert cfg.get_section("trading")["max_position_ratio"] == 0.25
        assert cfg.get_section("strategy")["buy_threshold"] == 0.25

    def test_set_aggressive_applies_to_config(self):
        rp.set_active("aggressive")
        assert rp.get_active() == "aggressive"
        assert cfg.get_section("trading")["max_position_ratio"] == 0.35
        assert cfg.get_section("trading")["stop_loss_pct"] == -0.10
        assert cfg.get_section("trading")["max_positions"] == 6
        assert cfg.get_section("trading")["max_daily_loss"] == 50000
        assert cfg.get_section("strategy")["buy_threshold"] == 0.20
        assert cfg.get_section("strategy")["sell_threshold"] == -0.20

    def test_set_conservative_applies_to_config(self):
        rp.set_active("conservative")
        assert cfg.get_section("trading")["max_position_ratio"] == 0.10
        assert cfg.get_section("trading")["stop_loss_pct"] == -0.05
        assert cfg.get_section("strategy")["sell_threshold"] == -0.30

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError):
            rp.set_active("does_not_exist")

    def test_persisted_and_reloaded(self, tmp_path):
        path = str(tmp_path / "rp_persist.json")
        rp.load(path)
        rp.set_active("aggressive")
        # 別インスタンスの再読込を模して load し直すと aggressive が復元される
        cfg.load("config.yaml")  # config を素の値に戻してから
        name = rp.load(path)
        assert name == "aggressive"
        assert cfg.get_section("trading")["max_position_ratio"] == 0.35

    def test_switch_changes_riskmanager_effective_values(self):
        """RiskManager は trading 辞書の参照を保持するため、切替が即反映される"""
        import src.risk.manager as risk_mod
        risk = risk_mod.RiskManager()
        # balanced: 25% → 100万の25% = 25万 → 1000円株なら 200株
        qty_balanced = risk.calc_position_size("7203", 1000.0, 1_000_000)
        rp.set_active("conservative")  # 10% → 10万 → 100株
        qty_conservative = risk.calc_position_size("7203", 1000.0, 1_000_000)
        assert qty_balanced == 200
        assert qty_conservative == 100
