"""
モード別基準資金（X日次レポートの%算出用）のテスト
"""
import pytest

import src.core.reference_capital as ref_capital


@pytest.fixture(autouse=True)
def _setup(tmp_path):
    ref_capital.load(str(tmp_path / "reference_capital.json"))
    yield


class TestDefaults:
    def test_all_non_paper_modes_default_zero(self):
        values = ref_capital.get_all()
        assert values == {"live": 0.0, "dry_run": 0.0, "semi_live": 0.0}

    def test_get_unset_mode_returns_zero(self):
        assert ref_capital.get("live") == 0.0


class TestSetValue:
    def test_set_and_get(self):
        ref_capital.set_value("live", 300_000)
        assert ref_capital.get("live") == 300_000.0

    def test_paper_rejected(self):
        with pytest.raises(ValueError, match="paper"):
            ref_capital.set_value("paper", 500_000)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="未知のモード"):
            ref_capital.set_value("nonexistent", 100_000)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="0以上"):
            ref_capital.set_value("live", -1)

    def test_persists_across_reload(self, tmp_path):
        path = str(tmp_path / "rc2.json")
        ref_capital.load(path)
        ref_capital.set_value("dry_run", 250_000)
        ref_capital.load(path)
        assert ref_capital.get("dry_run") == 250_000.0


class TestPercentBasis:
    def test_paper_uses_initial_capital_arg(self):
        assert ref_capital.percent_basis("paper", paper_initial_capital=500_000) == 500_000.0

    def test_paper_without_arg_is_zero(self):
        assert ref_capital.percent_basis("paper") == 0.0

    def test_live_uses_configured_value(self):
        ref_capital.set_value("live", 1_000_000)
        assert ref_capital.percent_basis("live") == 1_000_000.0

    def test_live_unset_is_zero(self):
        assert ref_capital.percent_basis("live") == 0.0
