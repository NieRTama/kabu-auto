"""
シャドーA/B 比較レポートのテスト。

採点ルール（HOLD除外）: BUY は forward_return>0 で的中、SELL は <0 で的中。
本番とシャドーで action が異なるケースでヒット率が正しく分離されることを検証する。
"""
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

import src.core.config as cfg
import src.data.database as db
from src.data.database import Signal, get_session


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


def _add(symbol, action, action_shadow, gen_at):
    with get_session() as session:
        session.add(Signal(symbol=symbol, action=action, action_shadow=action_shadow,
                           generated_at=gen_at, rule_score=0.0, ml_score=0.0,
                           combined_score=0.0, combined_score_shadow=0.0))
        session.commit()


def _rising_closes():
    # 単調増加の終値（上昇相場）。BUYが的中、SELLが外れる。
    idx = pd.bdate_range("2026-01-01", periods=30)
    closes = pd.Series(range(100, 130), index=idx, dtype=float)
    return pd.DataFrame({"close": closes}, index=idx)


class TestShadowAB:
    def test_production_buy_hits_in_rising_market(self, isolated_db):
        from src.analytics.shadow_ab import compare_shadow_ab
        # 本番BUY / シャドーSELL を上昇相場の序盤に置く
        _add("7203", "BUY", "SELL", datetime(2026, 1, 5, 16, 20))
        with patch("src.analytics.shadow_ab.load_ohlcv", return_value=_rising_closes()):
            res = compare_shadow_ab(horizon=5)
        assert res["evaluated"] == 1
        # 上昇相場: 本番BUYは的中(1/1)、シャドーSELLは外れ(0/1)
        assert res["production"]["hit_rate"] == 1.0
        assert res["shadow"]["hit_rate"] == 0.0
        assert res["divergence"] == 1
        assert res["shadow_available"] == 1

    def test_skips_when_insufficient_forward_data(self, isolated_db):
        from src.analytics.shadow_ab import compare_shadow_ab
        # 終値系列より後 → horizon 先が無く採点不能
        _add("7203", "BUY", None, datetime(2026, 12, 1, 16, 20))
        with patch("src.analytics.shadow_ab.load_ohlcv", return_value=_rising_closes()):
            res = compare_shadow_ab(horizon=5)
        assert res["evaluated"] == 0
        assert res["production"]["hit_rate"] is None

    def test_no_shadow_still_scores_production(self, isolated_db):
        from src.analytics.shadow_ab import compare_shadow_ab
        _add("7203", "BUY", None, datetime(2026, 1, 5, 16, 20))
        with patch("src.analytics.shadow_ab.load_ohlcv", return_value=_rising_closes()):
            res = compare_shadow_ab(horizon=5)
        assert res["production"]["n"] == 1
        assert res["shadow"]["n"] == 0
        assert res["shadow_available"] == 0
