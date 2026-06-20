"""
バックテスト履歴管理のテスト（Phase 4 / 6.1）
- ページング（15件・新しい順・total/total_pages）
- 単体/複数削除（取引明細も同時削除・トランザクション）
- アーカイブ/解除（既定一覧から除外）
"""
from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient

import src.core.config as cfg
import src.data.database as db
import src.dashboard.app as dash
from src.data.database import BacktestRun, BacktestTradeRecord, get_session


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    dash._auth_required = False
    try:
        yield tmp_path
    finally:
        db._engine = None
        db._Session = None


def _add_run(symbol="7203", archived=0, n_trades=0) -> int:
    with get_session() as session:
        run = BacktestRun(
            symbol=symbol, start_date=date(2025, 1, 1), end_date=date(2025, 6, 1),
            initial_capital=500000, final_capital=550000, total_return=0.1,
            max_drawdown=-0.05, sharpe_ratio=1.2, win_rate=0.6, trade_count=n_trades,
            use_ml=0, created_at=datetime(2025, 6, 1, 12, 0), archived=archived,
        )
        session.add(run)
        session.commit()
        run_id = run.id
        for i in range(n_trades):
            session.add(BacktestTradeRecord(
                run_id=run_id, symbol=symbol, entry_date=date(2025, 1, 2),
                entry_price=1000, exit_price=1100, quantity=100, pnl=10000,
            ))
        session.commit()
    return run_id


class TestPaging:
    def test_paging_newest_first(self, isolated_db):
        ids = [_add_run() for _ in range(20)]
        client = TestClient(dash.app)
        body = client.get("/api/backtest/runs?page=1&page_size=15").json()
        assert body["total"] == 20
        assert body["total_pages"] == 2
        assert len(body["runs"]) == 15
        # 新しい順（id降順）
        assert body["runs"][0]["id"] == ids[-1]

    def test_second_page(self, isolated_db):
        for _ in range(20):
            _add_run()
        client = TestClient(dash.app)
        body = client.get("/api/backtest/runs?page=2&page_size=15").json()
        assert len(body["runs"]) == 5


class TestArchive:
    def test_archived_excluded_by_default(self, isolated_db):
        _add_run()
        rid = _add_run()
        client = TestClient(dash.app)
        client.post(f"/api/backtest/{rid}/archive")
        body = client.get("/api/backtest/runs").json()
        assert body["total"] == 1
        assert all(not r["archived"] for r in body["runs"])

    def test_include_archived(self, isolated_db):
        rid = _add_run()
        client = TestClient(dash.app)
        client.post(f"/api/backtest/{rid}/archive")
        body = client.get("/api/backtest/runs?include_archived=true").json()
        assert body["total"] == 1

    def test_unarchive(self, isolated_db):
        rid = _add_run()
        client = TestClient(dash.app)
        client.post(f"/api/backtest/{rid}/archive")
        client.post(f"/api/backtest/{rid}/unarchive")
        body = client.get("/api/backtest/runs").json()
        assert body["total"] == 1

    def test_archive_missing_404(self, isolated_db):
        client = TestClient(dash.app)
        assert client.post("/api/backtest/999/archive").status_code == 404


class TestDelete:
    def test_delete_single_removes_run_and_trades(self, isolated_db):
        rid = _add_run(n_trades=3)
        client = TestClient(dash.app)
        r = client.delete(f"/api/backtest/{rid}")
        assert r.status_code == 200
        # 実行も明細も消えている
        with get_session() as session:
            assert session.get(BacktestRun, rid) is None
            from sqlalchemy import select, func
            cnt = session.scalar(
                select(func.count(BacktestTradeRecord.id))
                .where(BacktestTradeRecord.run_id == rid)
            )
            assert cnt == 0

    def test_delete_missing_404(self, isolated_db):
        client = TestClient(dash.app)
        assert client.delete("/api/backtest/999").status_code == 404

    def test_bulk_delete(self, isolated_db):
        ids = [_add_run() for _ in range(3)]
        client = TestClient(dash.app)
        r = client.post("/api/backtest/delete", json={"ids": ids})
        assert r.json()["deleted"] == 3
        body = client.get("/api/backtest/runs").json()
        assert body["total"] == 0
