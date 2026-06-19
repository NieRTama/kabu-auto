"""
RiskManager.check_sector_concentration() の金額ベース判定テスト

経緯: 旧実装は「同一セクターの保有銘柄数の割合」で判定していたため、
小口1銘柄・大型1銘柄のような建玉金額が大きく異なるケースを区別できなかった。
quantity × 最新終値（無ければavg_cost）のエクスポージャー比率に変更したことを検証する。
"""
from datetime import date

import pandas as pd
import pytest

import src.core.config as cfg
import src.data.database as db
from src.data.database import Position, get_session
from src.data.market_data import latest_closes, upsert_ohlcv
from src.risk.manager import RiskManager


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


def _add_position(symbol: str, quantity: int, avg_cost: float, sector: str) -> None:
    with get_session() as session:
        session.add(Position(symbol=symbol, quantity=quantity, avg_cost=avg_cost, sector=sector))
        session.commit()


def _set_close(symbol: str, close: float) -> None:
    df = pd.DataFrame(
        {"open": [close], "high": [close], "low": [close], "close": [close], "volume": [1000]},
        index=pd.to_datetime([date(2026, 1, 1)]),
    )
    df.index.name = "date"
    upsert_ohlcv(symbol, df)


class TestLatestCloses:
    def test_returns_latest_close_per_symbol(self, isolated_db):
        _set_close("AAA", 1234.0)
        assert latest_closes(["AAA"]) == {"AAA": 1234.0}

    def test_missing_symbol_omitted(self, isolated_db):
        _set_close("AAA", 100.0)
        result = latest_closes(["AAA", "ZZZ"])
        assert "ZZZ" not in result

    def test_empty_list_returns_empty_dict(self, isolated_db):
        assert latest_closes([]) == {}


class TestSectorConcentrationValueBased:
    def test_small_lot_vs_large_lot_same_sector_count_different_value(self, isolated_db):
        """銘柄数では両方「1銘柄」でも、建玉金額が大きく異なれば判定が変わるべき"""
        # セクターA: 小口1銘柄（100株×1000円=10万円）
        _add_position("1111", quantity=100, avg_cost=1000.0, sector="SectorA")
        _set_close("1111", 1000.0)
        # セクターB: 大型1銘柄（1000株×5000円=500万円）
        _add_position("2222", quantity=1000, avg_cost=5000.0, sector="SectorB")
        _set_close("2222", 5000.0)

        risk = RiskManager()
        risk._conf = {"max_sector_ratio": 0.40}

        # 旧実装なら銘柄数ベースで SectorA も SectorB も 1/2=50% で同じ判定になり、
        # どちらも上限超え扱いになってしまう。金額ベースでは大きく異なるはず。
        ok_a, _ = risk.check_sector_concentration("SectorA")
        ok_b, reason_b = risk.check_sector_concentration("SectorB")

        # SectorA（10万円 / 510万円 ≈ 2%）は上限40%未満 → OK
        assert ok_a is True
        # SectorB（500万円 / 510万円 ≈ 98%）は上限40%超 → NG
        assert ok_b is False
        assert "SectorB" in reason_b

    def test_uses_avg_cost_when_no_ohlcv_available(self, isolated_db):
        """最新終値が取得できない銘柄は avg_cost で代用する（クラッシュしないことを確認）"""
        _add_position("3333", quantity=100, avg_cost=1000.0, sector="SectorC")
        # OHLCVデータを投入しない → latest_closesは空 → avg_costにフォールバック

        risk = RiskManager()
        risk._conf = {"max_sector_ratio": 0.40}
        ok, reason = risk.check_sector_concentration("SectorC")
        # 唯一の保有なので集中率100% >= 上限40% → NG（avg_costへのフォールバックで
        # 例外なく計算できることがこのテストの主目的）
        assert ok is False
        assert "SectorC" in reason

    def test_no_positions_returns_ok(self, isolated_db):
        risk = RiskManager()
        risk._conf = {"max_sector_ratio": 0.40}
        ok, reason = risk.check_sector_concentration("AnySector")
        assert ok is True
        assert reason == ""

    def test_below_threshold_is_ok(self, isolated_db):
        """均等な3銘柄に分散していれば、どのセクターも閾値未満でOK"""
        for i, sym in enumerate(["4001", "4002", "4003"]):
            _add_position(sym, quantity=100, avg_cost=1000.0, sector=f"Sector{i}")
            _set_close(sym, 1000.0)

        risk = RiskManager()
        risk._conf = {"max_sector_ratio": 0.40}
        ok, _ = risk.check_sector_concentration("Sector0")
        assert ok is True  # 1/3 ≈ 33% < 40%
