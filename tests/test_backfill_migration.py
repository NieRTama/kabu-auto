"""
schema_version v2 バックフィルマイグレーションのテスト（Phase 5 / 4.2）

v1相当のDB（trades・positionsのみ、OrderIntent/Fillが空）に対し、
- 全TradeにOrderIntentが生成・紐付けられること
- 約定済みTradeからFillが生成され、SELLはFIFOで実現損益を再計算すること
- Positionが残存ロットから再構成されること
- schema_version が 2 になること
- 既に移行済みの状態で再実行しても重複生成されないこと（冪等性）
"""
from datetime import date, datetime

import pytest
from sqlalchemy import select, text

import src.core.config as cfg
import src.data.database as db
from src.data.database import Fill, OrderIntent, Position, Trade, get_session


@pytest.fixture
def v1_db(tmp_path):
    """schema_version=1相当（trades/positionsのみ）のDBを素朴に作る。

    db.init() は常に現在のSCHEMA_VERSION(=2)まで移行してしまうため、ここでは
    init() 後に schema_version を1へ戻し、intent_id/Fill関連を一旦クリアしてから
    _run_migrations を単体で呼び直し、v2移行の挙動だけを検証する。
    """
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    with get_session() as s:
        s.execute(text("UPDATE schema_version SET version=1"))
        s.commit()
    try:
        yield tmp_path
    finally:
        db._engine = None
        db._Session = None


def _seed_legacy_trades():
    """旧モデル（intent_id無し・Fill無し）相当のTrade/Positionを直接投入する。"""
    with get_session() as s:
        # BUY 100株@1000 → 全約定
        s.add(Trade(order_id="PAPER-BUY-7203-aaa", symbol="7203", side="BUY",
                    quantity=100, price=1000.0, filled_price=1000.0, filled_quantity=100,
                    status="FILLED", filled_at=datetime(2026, 6, 1, 9, 0)))
        # SELL 100株@1100 → 全約定・旧コードの平均単価会計でpnl=10000のはず
        s.add(Trade(order_id="PAPER-SELLM-7203-bbb", symbol="7203", side="SELL",
                    quantity=100, price=0.0, filled_price=1100.0, filled_quantity=100,
                    status="FILLED", filled_at=datetime(2026, 6, 2, 9, 0), pnl=10000.0))
        # 未約定の古いPENDING注文（fillなし）
        s.add(Trade(order_id="LIVE-PENDING-1", symbol="6758", side="BUY",
                    quantity=50, price=2000.0, status="PENDING"))
        # 拒否された注文
        s.add(Trade(order_id="REJECTED-9999-ccc", symbol="9999", side="BUY",
                    quantity=10, price=500.0, status="REJECTED"))
        s.commit()


class TestBackfillMigration:
    def test_all_trades_get_intent(self, v1_db):
        _seed_legacy_trades()
        db._run_migrations(db._engine)
        with get_session() as s:
            trades = s.scalars(select(Trade)).all()
            assert len(trades) == 4
            for t in trades:
                assert t.intent_id is not None
                intent = s.scalar(select(OrderIntent).where(OrderIntent.id == t.intent_id))
                assert intent is not None
                assert intent.symbol == t.symbol

    def test_rejected_intent_status(self, v1_db):
        _seed_legacy_trades()
        db._run_migrations(db._engine)
        with get_session() as s:
            t = s.scalar(select(Trade).where(Trade.order_id == "REJECTED-9999-ccc"))
            intent = s.scalar(select(OrderIntent).where(OrderIntent.id == t.intent_id))
            assert intent.status == "REJECTED"

    def test_pending_intent_status(self, v1_db):
        _seed_legacy_trades()
        db._run_migrations(db._engine)
        with get_session() as s:
            t = s.scalar(select(Trade).where(Trade.order_id == "LIVE-PENDING-1"))
            intent = s.scalar(select(OrderIntent).where(OrderIntent.id == t.intent_id))
            assert intent.status == "SUBMITTED"

    def test_fills_generated_for_filled_trades(self, v1_db):
        _seed_legacy_trades()
        db._run_migrations(db._engine)
        with get_session() as s:
            fills = s.scalars(select(Fill).where(Fill.symbol == "7203")).all()
            assert len(fills) == 2  # BUY 1件 + SELL 1件
            buy_fill = next(f for f in fills if f.side == "BUY")
            sell_fill = next(f for f in fills if f.side == "SELL")
            assert buy_fill.fill_qty == 100
            assert sell_fill.fill_qty == 100

    def test_sell_pnl_recomputed_via_fifo(self, v1_db):
        """FIFO再計算後もこのケースでは同額(10000)になること（単一ロットのため平均法と一致）"""
        _seed_legacy_trades()
        db._run_migrations(db._engine)
        with get_session() as s:
            t = s.scalar(select(Trade).where(Trade.order_id == "PAPER-SELLM-7203-bbb"))
            assert t.pnl == pytest.approx(10000.0)

    def test_position_rebuilt_from_remaining_lots(self, v1_db):
        """全量売却済みなのでPositionは存在しないか quantity=0 になること"""
        _seed_legacy_trades()
        db._run_migrations(db._engine)
        with get_session() as s:
            pos = s.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos is None or pos.quantity == 0

    def test_partial_position_remains_after_backfill(self, v1_db):
        """一部のみ売却した場合、残存ロットからPositionが正しく再構成されること"""
        with get_session() as s:
            s.add(Trade(order_id="PAPER-BUY-6758-x", symbol="6758", side="BUY",
                        quantity=200, price=500.0, filled_price=500.0, filled_quantity=200,
                        status="FILLED", filled_at=datetime(2026, 6, 1, 9, 0)))
            s.add(Trade(order_id="PAPER-SELL-6758-y", symbol="6758", side="SELL",
                        quantity=80, price=600.0, filled_price=600.0, filled_quantity=80,
                        status="FILLED", filled_at=datetime(2026, 6, 2, 9, 0), pnl=8000.0))
            s.commit()
        db._run_migrations(db._engine)
        with get_session() as s:
            pos = s.scalar(select(Position).where(Position.symbol == "6758"))
            assert pos.quantity == 120
            assert pos.avg_cost == pytest.approx(500.0)

    def test_schema_version_becomes_2(self, v1_db):
        _seed_legacy_trades()
        db._run_migrations(db._engine)
        assert db.get_schema_version() == 2

    def test_idempotent_rerun_does_not_duplicate(self, v1_db):
        """移行を2回走らせても、intentやFillが重複生成されないこと"""
        _seed_legacy_trades()
        db._run_migrations(db._engine)
        with get_session() as s:
            intent_count_1 = len(s.scalars(select(OrderIntent)).all())
            fill_count_1 = len(s.scalars(select(Fill)).all())
        # 強制的にもう一度 v2 を流す（schema_versionを1に戻して再実行）
        with get_session() as s:
            s.execute(text("UPDATE schema_version SET version=1"))
            s.commit()
        db._run_migrations(db._engine)
        with get_session() as s:
            intent_count_2 = len(s.scalars(select(OrderIntent)).all())
            fill_count_2 = len(s.scalars(select(Fill)).all())
        assert intent_count_1 == intent_count_2
        assert fill_count_1 == fill_count_2
