"""
RiskManager の未約定引当・未解決注文ガードのテスト（再レビュー B-3 対応）

- calc_position_size が未約定BUYの引当を実効余力から差し引く
- check_sector_concentration が未約定BUYを集中度に加味する
- can_place_order が UNKNOWN / CANCEL_FAILED 注文の存在時に発注を止める
"""
import pytest

import src.core.config as cfg
import src.data.database as db
import src.execution.order_status as st
from src.data.database import Position, Trade, get_session
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


def _add_position(symbol, quantity, avg_cost, sector=""):
    with get_session() as session:
        session.add(Position(symbol=symbol, quantity=quantity, avg_cost=avg_cost, sector=sector))
        session.commit()


def _add_trade(symbol, side, status, quantity, price, filled_quantity=None, sector=None):
    with get_session() as session:
        session.add(Trade(
            order_id=f"{symbol}-{side}-{status}-{quantity}-{sector}",
            symbol=symbol, side=side, status=status, sector=sector,
            quantity=quantity, price=price, filled_quantity=filled_quantity,
        ))
        session.commit()


class TestReservedCashInPositionSize:
    def test_pending_buy_reduces_available_cash(self, isolated_db):
        risk = RiskManager()
        risk._conf = {"max_position_ratio": 0.20}
        # 引当なし: 100万 × 20% / (1000×100) = 2単位 = 200株
        assert risk.calc_position_size("7203", 1000.0, 1_000_000) == 200
        # 未約定BUY 500株×1000円=50万を引当 → 実効余力50万 → 100株
        _add_trade("6758", "BUY", st.PENDING, 500, 1000.0)
        assert risk.calc_position_size("7203", 1000.0, 1_000_000) == 100

    def test_filled_buy_not_reserved(self, isolated_db):
        """約定済みBUYは引当に含めない（余力は別途wallet/簿価で反映済み）"""
        risk = RiskManager()
        risk._conf = {"max_position_ratio": 0.20}
        _add_trade("6758", "BUY", st.FILLED, 500, 1000.0, filled_quantity=500)
        assert risk.calc_position_size("7203", 1000.0, 1_000_000) == 200


class TestUnresolvedOrderGate:
    def test_unknown_order_blocks(self, isolated_db):
        risk = RiskManager()
        risk._conf = {"daily_order_limit": 100, "max_daily_loss": 0}
        _add_trade("6758", "BUY", st.UNKNOWN, 100, 1000.0)
        ok, reason = risk.can_place_order()
        assert ok is False
        assert "未解決" in reason

    def test_cancel_failed_blocks(self, isolated_db):
        risk = RiskManager()
        risk._conf = {"daily_order_limit": 100, "max_daily_loss": 0}
        _add_trade("6758", "SELL", st.CANCEL_FAILED, 100, 1000.0)
        ok, _ = risk.can_place_order()
        assert ok is False

    def test_clean_state_allows(self, isolated_db):
        risk = RiskManager()
        risk._conf = {"daily_order_limit": 100, "max_daily_loss": 0}
        ok, reason = risk.can_place_order()
        assert ok is True
        assert reason == ""


class TestReservedInSectorConcentration:
    def test_pending_buy_pushes_sector_over_limit(self, isolated_db):
        """保有のみでは均等(33%)でOKだが、未約定BUYを加味すると上限超でNG"""
        for sym, sec in [("1111", "A"), ("2222", "B"), ("3333", "C")]:
            _add_position(sym, 100, 1000.0, sector=sec)
        risk = RiskManager()
        risk._conf = {"max_sector_ratio": 0.40}
        # 引当なし → SectorA は 33% < 40% → OK
        ok, _ = risk.check_sector_concentration("A")
        assert ok is True
        # SectorA銘柄(1111)の未約定BUY 100株×1000=10万を加味 → A=20万/40万=50% → NG
        _add_trade("1111", "BUY", st.PENDING, 100, 1000.0)
        ok, reason = risk.check_sector_concentration("A")
        assert ok is False
        assert "A" in reason


class TestTradeSectorForNewSymbol:
    """再レビュー P1-3: Position未作成の新規銘柄でも、Trade.sectorから
    未約定BUYのセクター引当が正しく解決できることの検証"""

    def test_new_symbol_pending_buy_counted_via_trade_sector(self, isolated_db):
        """9999は保有も無い完全新規銘柄。Trade.sector='D'だけでセクター引当が機能する"""
        _add_position("1111", 100, 1000.0, sector="A")  # 既存保有（10万円）
        _add_trade("9999", "BUY", st.PENDING, 100, 1000.0, sector="D")  # 新規10万円

        risk = RiskManager()
        risk._conf = {"max_sector_ratio": 0.40}
        # 総額20万のうちセクターD=10万 → 50% ≥ 40% → NG
        ok, reason = risk.check_sector_concentration("D")
        assert ok is False
        assert "D" in reason

    def test_new_symbol_without_trade_sector_not_misattributed(self, isolated_db):
        """sector未記録（旧データ等）の新規銘柄は合計には入るがどのセクターにも属さない"""
        _add_trade("9999", "BUY", st.PENDING, 100, 1000.0, sector=None)
        risk = RiskManager()
        risk._conf = {"max_sector_ratio": 0.40}
        ok, _ = risk.check_sector_concentration("D")
        assert ok is True  # Dセクターには何も計上されない
