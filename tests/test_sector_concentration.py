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


class TestDenominatorIsTotalCapital:
    """分母は総資金（買付余力＋建玉評価額）であること。

    2026-09-04 に判明した実害の回帰防止。投資済み額を分母にしていたため、
    保有が9432の17,260円だけの状態で147,500円の銘柄を買おうとすると
    89.5% と判定されて却下されていた。買った瞬間にそのセクターが大半を
    占めるのは当然で、**保有が少ないほど必ず超過する**。保有ゼロなら常に100%で、
    システムは最初の1銘柄を永久に買えなかった。
    """

    def _risk(self, ratio=0.55):
        risk = RiskManager()
        risk._conf = {"max_sector_ratio": ratio}
        return risk

    def test_first_buy_from_empty_portfolio_is_allowed(self, isolated_db):
        """保有ゼロでも、資金に対して十分小さい1件目は買える（デッドロックの解消）"""
        ok, reason = self._risk().check_sector_concentration(
            "SectorX", candidate_notional=147_500.0, cash_balance=484_005.0)
        assert ok is True, reason

    def test_first_buy_is_blocked_without_cash_balance(self, isolated_db):
        """余力を渡さなければ従来どおり厳しく判定する（fail-safeのフォールバック）"""
        ok, _ = self._risk().check_sector_concentration(
            "SectorX", candidate_notional=147_500.0)
        assert ok is False

    def test_small_holding_does_not_block_new_sector(self, isolated_db):
        """実際に起きたケース: 9432を17,260円分だけ持つ状態で3166(147,500円)を買う"""
        _add_position("9432", quantity=100, avg_cost=160.0, sector="Communication Services")
        _set_close("9432", 172.6)
        ok, reason = self._risk().check_sector_concentration(
            "Consumer Cyclical", candidate_notional=147_500.0, cash_balance=484_005.0)
        # 147,500 / (484,005 + 17,260) = 29.4% < 55%
        assert ok is True, reason

    def test_oversized_buy_is_still_blocked(self, isolated_db):
        """資金の大半を1セクターに投じる注文は、総資金基準でも正しく弾く（緩めすぎない）"""
        ok, reason = self._risk().check_sector_concentration(
            "SectorX", candidate_notional=400_000.0, cash_balance=484_005.0)
        # 400,000 / 484,005 = 82.6% >= 55%
        assert ok is False
        assert "82%" in reason or "83%" in reason

    def test_existing_same_sector_holding_counts(self, isolated_db):
        """既存の同一セクター建玉は分子に積み上がる（分母を変えても集中は検知する）"""
        _add_position("5001", quantity=100, avg_cost=2_000.0, sector="SectorY")  # 20万円
        _set_close("5001", 2_000.0)
        risk = self._risk()
        # 既存20万 + 候補15万 = 35万 / (30万 + 20万) = 70% >= 55%
        ok, _ = risk.check_sector_concentration(
            "SectorY", candidate_notional=150_000.0, cash_balance=300_000.0)
        assert ok is False
        # 別セクターなら 15万 / 50万 = 30% で通る
        ok2, _ = risk.check_sector_concentration(
            "SectorZ", candidate_notional=150_000.0, cash_balance=300_000.0)
        assert ok2 is True

    def test_reason_shows_amounts(self, isolated_db):
        """却下理由に金額と比率が入る（なぜ弾かれたかログだけで分かるように）"""
        _, reason = self._risk().check_sector_concentration(
            "SectorX", candidate_notional=400_000.0, cash_balance=484_005.0)
        assert "400,000" in reason and "484,005" in reason


class TestValidateBuyPassesCash:
    def test_validate_buy_forwards_cash_balance(self, isolated_db):
        """結線の検証: validate_buy が余力をセクター判定へ渡すこと。

        渡し忘れるとフォールバック側（投資済み額が分母）になり、
        保有が少ない間ずっと買えないままになる。
        """
        risk = RiskManager()
        risk._conf = {
            "max_sector_ratio": 0.55, "max_position_ratio": 0.50,
            "max_positions": 8, "max_daily_loss": 0,
        }
        ok, reason = risk.validate_buy("3166", 1_475.0, 484_005.0, sector="Consumer Cyclical")
        assert ok is True, reason
