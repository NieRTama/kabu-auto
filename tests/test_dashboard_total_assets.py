"""/api/total_assets（総資産 = 現金 + 保有評価額）のテスト。"""
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.dashboard.app as dash
from src.core import total_assets
from src.data.database import Position, Trade, get_session
from src.data.market_data import upsert_ohlcv


@pytest.fixture
def client(isolated_db, tmp_path, monkeypatch):
    dash._auth_required = False
    # 履歴ファイルは本番の data/ へ書かせない
    monkeypatch.setattr(total_assets, "HISTORY_PATH", tmp_path / "hist.json")
    monkeypatch.setattr(dash, "_order_manager", None)
    monkeypatch.setattr(dash, "_kabu_client", None)
    monkeypatch.setattr(total_assets, "_wallet_cache", None)
    monkeypatch.setitem(dash._system_status, "mode", "paper")
    return TestClient(dash.app)


def _pos(symbol, qty, avg_cost, close=None):
    with get_session() as s:
        s.add(Position(symbol=symbol, quantity=qty, avg_cost=avg_cost))
        s.commit()
    if close is not None:
        df = pd.DataFrame({"open": [close], "high": [close], "low": [close], "close": [close],
                           "volume": [1000]}, index=pd.to_datetime([date(2026, 1, 1)]))
        df.index.name = "date"
        upsert_ohlcv(symbol, df)


class FakeClient:
    def __init__(self, cash=None, fail=False):
        self.cash, self.fail, self.calls = cash, fail, 0

    def get_wallet(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("401 Unauthorized")
        return {"StockAccountWallet": self.cash}


def test_paper_cash_is_initial_plus_realized_minus_cost(client):
    _pos("7203", 100, 1000.0, close=1100.0)
    with get_session() as s:
        s.add(Trade(order_id="x", symbol="9999", side="SELL", quantity=100, pnl=5000.0, status="FILLED"))
        s.commit()
    d = client.get("/api/total_assets").json()
    assert d["cash"] == 500_000 + 5000 - 100_000
    assert d["market_value"] == 110_000
    assert d["total"] == 405_000 + 110_000
    assert d["unrealized"] == 10_000
    assert d["cash_source"] == "paper"


def test_live_uses_broker_wallet_and_caches_it(client, monkeypatch):
    monkeypatch.setitem(dash._system_status, "mode", "live")
    fake = FakeClient(cash=300_000)
    monkeypatch.setattr(dash, "_kabu_client", fake)
    _pos("7203", 100, 1000.0, close=1200.0)
    d = client.get("/api/total_assets").json()
    assert d["cash"] == 300_000 and d["total"] == 420_000
    client.get("/api/total_assets")
    assert fake.calls == 1  # 30秒以内の再取得はキャッシュ（API呼び出しを増やさない）


def test_live_wallet_failure_gives_null_total_not_wrong_number(client, monkeypatch):
    monkeypatch.setitem(dash._system_status, "mode", "live")
    monkeypatch.setattr(dash, "_kabu_client", FakeClient(fail=True))
    _pos("7203", 100, 1000.0, close=1200.0)
    d = client.get("/api/total_assets").json()
    assert d["cash"] is None and d["total"] is None
    assert d["market_value"] == 120_000
    assert "401" in d["cash_error"]


def test_unpriced_position_is_excluded_and_listed(client):
    _pos("9999", 100, 1000.0)  # 価格データなし
    d = client.get("/api/total_assets").json()
    assert d["market_value"] == 0
    assert d["unpriced"] == ["9999"]


# ─── 前日比（data/total_assets_history.json への記録） ─────────────────

@pytest.fixture
def history(client):
    return total_assets.HISTORY_PATH


def test_first_day_has_no_previous_and_records_today(client, history, monkeypatch):
    monkeypatch.setattr(dash.clock, "today", lambda: date(2026, 10, 15))
    d = client.get("/api/total_assets").json()
    assert d["prev_total"] is None and d["day_change"] is None
    import json
    assert json.loads(history.read_text())["2026-10-15"] == d["total"]


def test_day_change_against_previous_business_day(client, history, monkeypatch):
    history.write_text('{"2026-10-14": 400000.0}')
    monkeypatch.setattr(dash.clock, "today", lambda: date(2026, 10, 15))
    d = client.get("/api/total_assets").json()  # paper: 現金50万・保有なし
    assert d["prev_total"] == 400000 and d["day_change"] == 100000
    assert d["prev_label"] == "前日比"


def test_gap_in_records_labels_with_date(client, history, monkeypatch):
    history.write_text('{"2026-10-09": 510000.0}')  # 10/12(祝)・10/13・10/14 の記録なし
    monkeypatch.setattr(dash.clock, "today", lambda: date(2026, 10, 15))
    d = client.get("/api/total_assets").json()
    assert d["day_change"] == -10000 and d["prev_label"] == "10/9比"


def test_previous_business_day_skips_weekend(client, history, monkeypatch):
    history.write_text('{"2026-10-16": 490000.0}')  # 金曜
    monkeypatch.setattr(dash.clock, "today", lambda: date(2026, 10, 19))  # 月曜
    assert client.get("/api/total_assets").json()["prev_label"] == "前日比"


# ─── 最大ドローダウン（資産ベース） ─────────────────────────────────

def test_drawdown_is_against_equity_not_profit_peak():
    # 実データ（2026-09〜10）の累積損益。旧実装は利益の山2,405円を分母にして -600.6% と出していた
    cum = [80, 1935, 2405, 2400, -5050, -11750, -12040]
    pct, yen = dash._compute_max_drawdown(cum, 500_000)
    assert yen == -14445
    assert pct == round(-14445 / 502405, 4)  # 約 -2.88%


def test_drawdown_without_capital_gives_amount_only():
    assert dash._compute_max_drawdown([100, -900], 0) == (None, -1000)


def test_drawdown_counts_loss_from_start():
    # 最初から負け続けても 0% にならない（旧実装は利益の山が0以下だと常に0）
    pct, yen = dash._compute_max_drawdown([-1000, -3000], 100_000)
    assert yen == -3000 and pct == -0.03
