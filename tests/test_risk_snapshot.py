"""
RiskSnapshot のテスト（Phase 5 / P2-6）

スナップショット経由の判定が、個別DB問い合わせ版と同じ結果になること、
validate_buy がスナップショットを1回だけ構築して各チェックへ渡すことを検証する。
"""
from unittest.mock import patch

import pytest

import src.core.config as cfg
import src.data.database as db
from src.data.database import Position, Trade, get_session
from src.risk.manager import RiskManager, RiskSnapshot


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


def _add_position(symbol, qty, avg_cost, sector=""):
    with get_session() as s:
        s.add(Position(symbol=symbol, quantity=qty, avg_cost=avg_cost, sector=sector))
        s.commit()


def _add_open_buy(symbol, qty, price, sector=None):
    with get_session() as s:
        s.add(Trade(order_id=f"o-{symbol}", symbol=symbol, side="BUY", quantity=qty,
                    price=price, status="PENDING", sector=sector))
        s.commit()


class TestBuildSnapshot:
    def test_collects_positions_and_open_buys(self, isolated_db):
        _add_position("7203", 100, 1000, "Auto")
        _add_open_buy("6758", 100, 2000, "Tech")
        risk = RiskManager()
        snap = risk.build_snapshot()
        assert {p.symbol for p in snap.positions} == {"7203"}
        assert {t.symbol for t in snap.open_buys} == {"6758"}
        assert snap.pos_sector.get("7203") == "Auto"

    def test_unresolved_counted(self, isolated_db):
        with get_session() as s:
            s.add(Trade(order_id="u1", symbol="7203", side="BUY", quantity=100,
                        price=1000, status="UNKNOWN"))
            s.commit()
        risk = RiskManager()
        assert risk.build_snapshot().unresolved_count == 1


class TestSnapshotEquivalence:
    def test_check_max_positions_same_with_and_without_snapshot(self, isolated_db):
        _add_position("a", 100, 1000)
        _add_position("b", 100, 1000)
        _add_open_buy("c", 100, 1000)
        risk = RiskManager()
        snap = risk.build_snapshot()
        # max_positions=6 (low_risk default) なので候補追加でも通る
        without = risk.check_max_positions(candidate_symbol="d")
        with_snap = risk.check_max_positions(candidate_symbol="d", snapshot=snap)
        assert without == with_snap

    def test_reserved_same_with_and_without_snapshot(self, isolated_db):
        _add_open_buy("c", 100, 1500, sector="Tech")
        risk = RiskManager()
        snap = risk.build_snapshot()
        assert risk._reserved_buy_by_sector() == risk._reserved_buy_by_sector(snap)


class TestValidateBuyUsesSingleSnapshot:
    def test_build_snapshot_called_once(self, isolated_db):
        risk = RiskManager()
        real_build = risk.build_snapshot
        with patch.object(risk, "build_snapshot", wraps=real_build) as spy:
            risk.validate_buy("7203", 1000.0, 1_000_000.0, sector="Auto")
        spy.assert_called_once()
