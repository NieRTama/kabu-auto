"""
買える株数が0になったときの説明文のテスト

2026-09-03、買付余力が484,005円あるにもかかわらず朝の買い候補8銘柄すべてが
見送られた。ログには一律「余力不足: 3563」としか出ておらず、資金が足りないと
読めてしまう。実際の原因は**単元(100株)の必要額が1銘柄あたりの上限を超えていた**
ことで、打つ手（比率の見直し）は入金とはまったく違う。

原因が「枠」なのか「現金」なのかを、ログだけで判断できるようにする。
"""
import pytest

import src.core.config as cfg
import src.data.database as db
from src.risk.manager import LOT_SIZE, RiskManager


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    try:
        yield
    finally:
        db._engine = None
        db._Session = None


def _risk(ratio: float) -> RiskManager:
    r = RiskManager()
    cfg.get_section("trading")["max_position_ratio"] = ratio
    return r


class TestPositionBudget:
    def test_budget_is_ratio_of_cash(self):
        assert _risk(0.50).position_budget(484_005.0) == pytest.approx(242_002.5)

    def test_budget_shrinks_with_remaining_cash(self):
        """枠は「残余力に対する比率」。購入のたびに小さくなる（呼び出し側が cash を減算する）"""
        r = _risk(0.50)
        first = r.position_budget(484_005.0)
        second = r.position_budget(484_005.0 - first)
        assert second == pytest.approx(first / 2)


class TestZeroSizeExplanation:
    def test_says_lot_exceeds_budget_not_insufficient_cash(self):
        """9/3 の実例（3563・株価5,909円）。上限超過と分かる文面にする"""
        r = _risk(0.50)
        msg = r._explain_zero_size("3563", 5_909.0, 484_005.0)
        assert "単元" in msg
        assert "上限" in msg
        assert "余力不足" not in msg, "現金不足と誤読される表現を使わない"

    def test_includes_actual_numbers(self):
        """必要額・上限・株価が数字で入る（見ただけで比率を決められるように）"""
        msg = _risk(0.50)._explain_zero_size("3563", 5_909.0, 484_005.0)
        assert "590,900" in msg          # 単元の必要額 5,909 × 100
        assert "242,002" in msg          # 1銘柄上限 484,005 × 0.50
        assert "5,909" in msg            # 株価
        assert "50%" in msg              # 上限比率

    def test_price_unavailable_is_distinguished(self):
        msg = _risk(0.50)._explain_zero_size("9999", 0.0, 484_005.0)
        assert "価格" in msg

    def test_validate_buy_uses_the_explanation(self):
        """結線の検証。validate_buy が返す理由がそのままログに出る"""
        ok, reason = _risk(0.50).validate_buy("3563", 5_909.0, 484_005.0)
        assert ok is False
        assert "単元" in reason and "上限" in reason


class TestSizingBoundary:
    """0.50 で何が買えるようになるかを境界で固定する（設定変更の意図を残す）"""

    def test_lot_just_within_budget_is_buyable(self):
        r = _risk(0.50)
        budget = r.position_budget(484_005.0)          # 242,002.5
        price = budget / LOT_SIZE - 1                  # 単元がぎりぎり収まる株価
        assert r.calc_position_size("X", price, 484_005.0) == LOT_SIZE

    def test_lot_just_over_budget_is_not_buyable(self):
        r = _risk(0.50)
        price = r.position_budget(484_005.0) / LOT_SIZE + 1
        assert r.calc_position_size("X", price, 484_005.0) == 0

    def test_old_ratio_could_not_buy_median_priced_stock(self):
        """0.28 では単元価格の中央値(175,900円)に届かなかったことを記録する"""
        assert _risk(0.28).calc_position_size("X", 1_759.0, 484_005.0) == 0
        assert _risk(0.50).calc_position_size("X", 1_759.0, 484_005.0) == LOT_SIZE
