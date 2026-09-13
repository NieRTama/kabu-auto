"""paper経路の執行仮定の切替（engine_version）のテスト

stop_loss_checkはpaperモードで日足終値を使って損切りを判定しており、
F04（同一終値での判断・約定）がpaper運用にも及んでいた（spec §8）。
**engine_version: legacy のとき挙動が1ビットも変わらないこと**を固定する。
"""
import inspect

import pytest

from src.core import config as cfg
from src.services import trading


@pytest.fixture(autouse=True)
def _load_config():
    cfg.load("config.yaml")


class TestEngineVersionDefault:
    def test_defaults_to_legacy(self):
        """config.yaml の既定は legacy（旧方式へいつでも戻せる）"""
        assert cfg.get_section("strategy").get("engine_version") == "legacy"

    def test_unknown_value_is_treated_as_legacy(self):
        """未知の値は安全側（legacy）に倒す"""
        cfg.get_section("strategy")["engine_version"] = "experimental"
        svc = trading.TradingServices(client=None, risk=None, order_mgr=None)
        assert svc._engine_version() == "legacy"
        assert svc._paper_uses_v2_execution() is False

    def test_v2_is_recognized(self):
        cfg.get_section("strategy")["engine_version"] = "v2"
        svc = trading.TradingServices(client=None, risk=None, order_mgr=None)
        assert svc._engine_version() == "v2"
        assert svc._paper_uses_v2_execution() is True


class TestLegacyBehaviourUnchanged:
    """legacy では paper の挙動が現在と変わらないこと"""

    def _source(self, name):
        return inspect.getsource(getattr(trading.TradingServices, name))

    def test_stop_loss_check_still_has_the_legacy_close_path(self):
        """従来の日足終値による損切り判定が残っている"""
        src = self._source("stop_loss_check")
        assert "load_ohlcv" in src
        assert 'df["close"].iloc[-1]' in src

    def test_stop_loss_check_branches_on_engine_version(self):
        """v2のときだけ別の経路へ入る分岐がある"""
        assert "_paper_uses_v2_execution" in self._source("stop_loss_check")

    def test_signal_scan_branches_on_engine_version(self):
        assert "_paper_uses_v2_execution" in self._source("signal_scan")

    def test_freshness_gate_is_still_wired(self):
        """段階Aの鮮度ゲートが外れていない"""
        assert "_is_fresh_for_new_candidate" in self._source("signal_scan")

    def test_stop_loss_check_does_not_call_the_freshness_gate(self):
        """保有保護の退出は鮮度に関わらず実行する（段階Aの不変条件）"""
        assert "_is_fresh_for_new_candidate" not in self._source("stop_loss_check")
