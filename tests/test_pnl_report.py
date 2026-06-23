"""
日次/週次/月次/総合の損益サマリ集計（pnl_report.py）のテスト
"""
from datetime import date, datetime

import pytest

import src.core.config as cfg
import src.data.database as db
from src.core.pnl_report import build_report
from src.data.database import Trade, get_session


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


def _add_trade(order_id, pnl, filled_at, status="FILLED"):
    with get_session() as session:
        session.add(Trade(
            order_id=order_id, symbol="7203", side="SELL", quantity=100, price=1000,
            filled_price=1000, filled_quantity=100, pnl=pnl, status=status,
            filled_at=filled_at,
        ))
        session.commit()


class TestBuildReport:
    def test_daily_only_includes_today(self, isolated_db):
        today = date(2026, 6, 22)  # 月曜
        _add_trade("t1", 10000, datetime(2026, 6, 22, 10, 0))   # 当日
        _add_trade("t2", 5000, datetime(2026, 6, 19, 10, 0))    # 先週金曜（前週）
        report = build_report(reference_capital=0, today=today)
        assert report["daily"].realized_pnl == 10000
        assert report["weekly"].realized_pnl == 10000  # 月曜始まりなので先週金曜は含まない

    def test_weekly_includes_monday_to_today(self, isolated_db):
        today = date(2026, 6, 24)  # 水曜
        _add_trade("t1", 1000, datetime(2026, 6, 22, 10, 0))  # 月曜（週初）
        _add_trade("t2", 2000, datetime(2026, 6, 23, 10, 0))  # 火曜
        _add_trade("t3", 3000, datetime(2026, 6, 24, 10, 0))  # 当日
        report = build_report(reference_capital=0, today=today)
        assert report["weekly"].realized_pnl == 6000
        assert report["daily"].realized_pnl == 3000

    def test_monthly_includes_month_start_to_today(self, isolated_db):
        today = date(2026, 6, 24)
        _add_trade("t1", 1000, datetime(2026, 6, 1, 10, 0))   # 月初
        _add_trade("t2", 2000, datetime(2026, 5, 31, 10, 0))  # 前月（除外）
        report = build_report(reference_capital=0, today=today)
        assert report["monthly"].realized_pnl == 1000

    def test_overall_includes_all_history(self, isolated_db):
        today = date(2026, 6, 24)
        _add_trade("t1", 1000, datetime(2026, 1, 1, 10, 0))
        _add_trade("t2", 2000, datetime(2026, 6, 24, 10, 0))
        report = build_report(reference_capital=0, today=today)
        assert report["overall"].realized_pnl == 3000

    def test_dry_run_excluded(self, isolated_db):
        today = date(2026, 6, 24)
        _add_trade("t1", 99999, datetime(2026, 6, 24, 10, 0), status="DRY_RUN")
        report = build_report(reference_capital=0, today=today)
        assert report["daily"].realized_pnl == 0

    def test_pct_none_when_reference_capital_zero(self, isolated_db):
        today = date(2026, 6, 24)
        _add_trade("t1", 1000, datetime(2026, 6, 24, 10, 0))
        report = build_report(reference_capital=0, today=today)
        assert report["daily"].pct is None

    def test_pct_computed_when_reference_capital_set(self, isolated_db):
        today = date(2026, 6, 24)
        _add_trade("t1", 50000, datetime(2026, 6, 24, 10, 0))
        report = build_report(reference_capital=500_000, today=today)
        assert report["daily"].pct == 0.1

    def test_win_rate(self, isolated_db):
        today = date(2026, 6, 24)
        _add_trade("t1", 1000, datetime(2026, 6, 24, 9, 0))
        _add_trade("t2", -500, datetime(2026, 6, 24, 10, 0))
        _add_trade("t3", 2000, datetime(2026, 6, 24, 11, 0))
        report = build_report(reference_capital=0, today=today)
        assert report["daily"].win_count == 2
        assert report["daily"].loss_count == 1
        assert report["daily"].win_rate == pytest.approx(0.667, abs=0.001)

    def test_no_trades_win_rate_is_none(self, isolated_db):
        today = date(2026, 6, 24)
        report = build_report(reference_capital=0, today=today)
        assert report["daily"].win_rate is None
