"""
1銘柄あたりの上限が既存保有分の評価額を差し引いていなかった問題のテスト

後場スロット(afternoon_execution)で「前日までの保有銘柄への買い増し」を許可すると、
position_budget（残余力×上限比率）が既存保有を見ていないため、上限いっぱい保有した
銘柄へさらに満額買い増そうとしてしまう。calc_position_size は「1銘柄上限 - 既存保有の
評価額（現在値ベース）」を残り枠として使う。
"""
import pytest

import src.core.config as cfg
import src.data.database as db
from src.data.database import Position, get_session
from src.risk.manager import LOT_SIZE, RiskManager


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    try:
        yield tmp_path
    finally:
        db._engine = None
        db._Session = None


def _add_position(symbol, quantity, avg_cost, sector=""):
    with get_session() as session:
        session.add(Position(symbol=symbol, quantity=quantity, avg_cost=avg_cost, sector=sector))
        session.commit()


def _risk(ratio: float) -> RiskManager:
    r = RiskManager()
    r._conf = {"max_position_ratio": ratio}
    return r


class TestHeldValueReducesRemainingBudget:
    def test_unheld_symbol_uses_full_budget(self, isolated_db):
        """未保有なら従来どおり枠を丸ごと使える（回帰防止）"""
        r = _risk(0.50)
        # 100万×50%=50万 / (1000円×100株)=5単位=500株
        assert r.calc_position_size("7203", 1000.0, 1_000_000.0) == 500

    def test_existing_holding_reduces_the_remaining_budget(self, isolated_db):
        """既に保有している分の評価額（現在値ベース）だけ枠が減る"""
        _add_position("7203", 200, avg_cost=900.0)  # 現在値1000円で20万円分保有
        r = _risk(0.50)
        # 上限50万円 - 既存保有20万円 = 残り30万円 / (1000×100) = 3単位=300株
        assert r.calc_position_size("7203", 1000.0, 1_000_000.0) == 300

    def test_fully_invested_symbol_cannot_buy_more(self, isolated_db):
        """既に上限額を丸ごと保有していれば追加購入は0"""
        _add_position("7203", 500, avg_cost=1000.0)  # 現在値1000円で50万円分＝上限そのもの
        r = _risk(0.50)
        assert r.calc_position_size("7203", 1000.0, 1_000_000.0) == 0

    def test_other_symbols_are_not_affected(self, isolated_db):
        """他銘柄の保有は、この銘柄の枠に影響しない（銘柄ごとの上限であるため）"""
        _add_position("6758", 500, avg_cost=1000.0)  # 別銘柄で上限いっぱい保有
        r = _risk(0.50)
        assert r.calc_position_size("7203", 1000.0, 1_000_000.0) == 500

    def test_uses_current_price_not_avg_cost_for_held_value(self, isolated_db):
        """保有評価額は現在値ベース。簿価(avg_cost)基準にすると残り枠を過大評価する"""
        _add_position("7203", 200, avg_cost=500.0)  # 簿価は10万円だが現在値は1000円
        r = _risk(0.50)
        # 現在値ベースの評価額20万円で計算: 上限50万-20万=30万→300株
        # （avg_cost基準なら 50万-10万=40万→400株になってしまう）
        assert r.calc_position_size("7203", 1000.0, 1_000_000.0) == 300


class TestZeroSizeExplanationWithHolding:
    def test_explains_the_cap_is_consumed_by_existing_holding(self, isolated_db):
        _add_position("7203", 500, avg_cost=1000.0)
        r = _risk(0.50)
        msg = r._explain_zero_size("7203", 1000.0, 1_000_000.0)
        assert "既存保有" in msg
        assert "単元" in msg and "上限" in msg

    def test_price_unavailable_still_wins_over_holding_message(self, isolated_db):
        _add_position("9999", 100, avg_cost=100.0)
        r = _risk(0.50)
        msg = r._explain_zero_size("9999", 0.0, 1_000_000.0)
        assert "価格" in msg
