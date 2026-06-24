"""
FIFOロット台帳（src/execution/lots.py）のテスト（Phase 5 / 4.2）

- 単一/複数ロットの記録と消費
- 部分消費・複数BUYの平均化なし（FIFO=個別ロット単位で損益確定）
- ロット跨ぎSELL（複数ロットから消費）
- 在庫不足SELL（保有ロット不足）
- Positionの再構成（残存ロットからの quantity/avg_cost）
"""
from datetime import datetime

import pytest
from sqlalchemy import select

import src.core.config as cfg
import src.data.database as db
from src.data.database import Fill, Position, get_session
from src.execution import lots


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


def _dt(h: int) -> datetime:
    return datetime(2026, 6, 20, h, 0)


class TestRecordBuyFill:
    def test_creates_lot_with_full_remaining(self, isolated_db):
        with get_session() as s:
            fill = lots.record_buy_fill(s, broker_order_id=1, symbol="7203",
                                        qty=100, price=1000.0, filled_at=_dt(9))
            s.commit()
            assert fill.remaining_qty == 100
            assert fill.side == "BUY"

    def test_multiple_buys_create_separate_lots(self, isolated_db):
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 100, 1000.0, _dt(9))
            lots.record_buy_fill(s, 2, "7203", 50, 1100.0, _dt(10))
            s.commit()
            all_fills = s.query(Fill).filter(Fill.symbol == "7203", Fill.side == "BUY").all()
            assert len(all_fills) == 2


class TestConsumeFifo:
    def test_single_lot_full_consume(self, isolated_db):
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 100, 1000.0, _dt(9))
            s.commit()
        with get_session() as s:
            fill, realized, consumed = lots.consume_fifo(s, 2, "7203", 100, 1100.0, _dt(11))
            s.commit()
        assert consumed == 100
        assert realized == pytest.approx(10000.0)  # (1100-1000)*100
        assert fill.side == "SELL"

    def test_partial_consume_leaves_remaining(self, isolated_db):
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 100, 1000.0, _dt(9))
            s.commit()
        with get_session() as s:
            _, realized, consumed = lots.consume_fifo(s, 2, "7203", 40, 1100.0, _dt(11))
            s.commit()
        assert consumed == 40
        assert realized == pytest.approx((1100 - 1000) * 40)
        with get_session() as s:
            lot = s.query(Fill).filter(Fill.symbol == "7203", Fill.side == "BUY").one()
            assert lot.remaining_qty == 60

    def test_fifo_consumes_oldest_lot_first(self, isolated_db):
        """古いロット（先に買った分）から優先的に消費されること"""
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 100, 1000.0, _dt(9))   # 古い: 1000円
            lots.record_buy_fill(s, 2, "7203", 100, 2000.0, _dt(10))  # 新しい: 2000円
            s.commit()
        with get_session() as s:
            _, realized, consumed = lots.consume_fifo(s, 3, "7203", 100, 1500.0, _dt(11))
            s.commit()
        assert consumed == 100
        # 1000円ロットが先に使われる → (1500-1000)*100 = 50000
        assert realized == pytest.approx(50000.0)
        with get_session() as s:
            old_lot = s.scalar(
                select(Fill).where(Fill.broker_order_id == 1)
            )
            new_lot = s.scalar(
                select(Fill).where(Fill.broker_order_id == 2)
            )
            assert old_lot.remaining_qty == 0
            assert new_lot.remaining_qty == 100  # 未消費

    def test_sell_spans_multiple_lots(self, isolated_db):
        """1回のSELLが複数のBUYロットをまたいで消費すること"""
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 50, 1000.0, _dt(9))
            lots.record_buy_fill(s, 2, "7203", 50, 2000.0, _dt(10))
            s.commit()
        with get_session() as s:
            _, realized, consumed = lots.consume_fifo(s, 3, "7203", 80, 1500.0, _dt(11))
            s.commit()
        # 50株@1000 + 30株@2000 を消費
        # realized = (1500-1000)*50 + (1500-2000)*30 = 25000 - 15000 = 10000
        assert consumed == 80
        assert realized == pytest.approx(10000.0)

    def test_insufficient_lots_partial_consume(self, isolated_db):
        """保有ロット不足のSELLは消費できた分だけ処理し、不足を呼び出し元へ伝える"""
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 30, 1000.0, _dt(9))
            s.commit()
        with get_session() as s:
            _, realized, consumed = lots.consume_fifo(s, 2, "7203", 100, 1100.0, _dt(11))
            s.commit()
        assert consumed == 30  # 保有分のみ
        assert realized == pytest.approx((1100 - 1000) * 30)

    def test_no_lots_returns_zero(self, isolated_db):
        with get_session() as s:
            _, realized, consumed = lots.consume_fifo(s, 1, "9999", 100, 1000.0, _dt(9))
            s.commit()
        assert consumed == 0
        assert realized == 0.0

    def test_different_symbols_isolated(self, isolated_db):
        """別銘柄のロットは消費対象にならない"""
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 100, 1000.0, _dt(9))
            s.commit()
        with get_session() as s:
            _, realized, consumed = lots.consume_fifo(s, 2, "6758", 50, 1000.0, _dt(10))
            s.commit()
        assert consumed == 0


class TestRebuildPosition:
    def test_creates_position_from_remaining_lots(self, isolated_db):
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 100, 1000.0, _dt(9))
            lots.rebuild_position(s, "7203")
            s.commit()
        with get_session() as s:
            pos = s.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.quantity == 100
            assert pos.avg_cost == 1000.0

    def test_avg_cost_reflects_remaining_lots_only(self, isolated_db):
        """一部売却後のavg_costは残存ロットのみの加重平均になること"""
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 50, 1000.0, _dt(9))
            lots.record_buy_fill(s, 2, "7203", 50, 2000.0, _dt(10))
            s.commit()
        with get_session() as s:
            # 50株@1000 を売却（FIFOで古い方から消費）
            lots.consume_fifo(s, 3, "7203", 50, 1500.0, _dt(11))
            lots.rebuild_position(s, "7203")
            s.commit()
        with get_session() as s:
            pos = s.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.quantity == 50
            assert pos.avg_cost == pytest.approx(2000.0)  # 残るのは2000円ロットのみ

    def test_fully_sold_position_has_zero_quantity(self, isolated_db):
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 100, 1000.0, _dt(9))
            lots.rebuild_position(s, "7203")
            s.commit()
        with get_session() as s:
            lots.consume_fifo(s, 2, "7203", 100, 1200.0, _dt(11))
            lots.rebuild_position(s, "7203")
            s.commit()
        with get_session() as s:
            pos = s.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.quantity == 0

    def test_sector_set_on_creation(self, isolated_db):
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 100, 1000.0, _dt(9))
            lots.rebuild_position(s, "7203", sector="Auto")
            s.commit()
        with get_session() as s:
            pos = s.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.sector == "Auto"

    def test_sector_backfilled_if_missing(self, isolated_db):
        """既存Positionにsectorが無ければ補完する（既存値は上書きしない）"""
        with get_session() as s:
            s.add(Position(symbol="7203", quantity=0, avg_cost=0.0, sector=""))
            lots.record_buy_fill(s, 1, "7203", 100, 1000.0, _dt(9))
            lots.rebuild_position(s, "7203", sector="Auto")
            s.commit()
        with get_session() as s:
            pos = s.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.sector == "Auto"

    def test_no_lots_and_no_position_returns_none(self, isolated_db):
        with get_session() as s:
            result = lots.rebuild_position(s, "9999")
            assert result is None


class TestPeakPriceLifecycle:
    """トレーリングストップ用 Position.peak_price の初期化・リセット"""

    def test_new_position_initializes_peak_to_avg_cost(self, isolated_db):
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 100, 1000.0, _dt(9))
            lots.rebuild_position(s, "7203")
            s.commit()
        with get_session() as s:
            pos = s.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.peak_price == 1000.0

    def test_full_close_resets_peak_to_none(self, isolated_db):
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 100, 1000.0, _dt(9))
            lots.rebuild_position(s, "7203")
            s.commit()
        with get_session() as s:
            from src.data.database import Position as P
            pos = s.scalar(select(P).where(P.symbol == "7203"))
            pos.peak_price = 1500.0  # 値動き中にピークが更新された想定
            s.commit()
        with get_session() as s:
            lots.consume_fifo(s, 2, "7203", 100, 1400.0, _dt(10))
            lots.rebuild_position(s, "7203")
            s.commit()
        with get_session() as s:
            pos = s.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.quantity == 0
            assert pos.peak_price is None

    def test_reentry_after_full_close_reinitializes_peak(self, isolated_db):
        """完全決済→再エントリーで、古いピークを引き継がず新しいavg_costから再スタートする"""
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 100, 1000.0, _dt(9))
            lots.rebuild_position(s, "7203")
            s.commit()
        with get_session() as s:
            lots.consume_fifo(s, 2, "7203", 100, 1500.0, _dt(10))  # 完全決済（ピーク→None）
            lots.rebuild_position(s, "7203")
            s.commit()
        with get_session() as s:
            lots.record_buy_fill(s, 3, "7203", 100, 800.0, _dt(11))  # 再エントリー（新値で）
            lots.rebuild_position(s, "7203")
            s.commit()
        with get_session() as s:
            pos = s.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.peak_price == 800.0  # 古いピーク(1500/1000)を引き継いでいない

    def test_partial_sell_keeps_peak(self, isolated_db):
        """部分決済（quantity>0が継続）ではピークをリセットしない"""
        with get_session() as s:
            lots.record_buy_fill(s, 1, "7203", 100, 1000.0, _dt(9))
            lots.rebuild_position(s, "7203")
            s.commit()
        with get_session() as s:
            from src.data.database import Position as P
            pos = s.scalar(select(P).where(P.symbol == "7203"))
            pos.peak_price = 1500.0
            s.commit()
        with get_session() as s:
            lots.consume_fifo(s, 2, "7203", 40, 1400.0, _dt(10))  # 100→60（部分決済）
            lots.rebuild_position(s, "7203")
            s.commit()
        with get_session() as s:
            pos = s.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.quantity == 60
            assert pos.peak_price == 1500.0  # 保持される
