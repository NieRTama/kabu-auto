"""再起動で鮮度の記録（_bar_states）が消えても、DBの最終足から復元できることのテスト。

2026-10-22: 16:00 のデータ更新（旧プロセス）の後、16:08 に再起動したため新プロセスの
_bar_states が空のまま 16:20 の signal_scan が走り、全59銘柄が鮮度不足として除外された。
"""
from datetime import date, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.services.trading as trading
from src.core import clock
from src.data import bar_status
from src.data.market_data import upsert_ohlcv
from src.risk.manager import RiskManager

NOW = datetime(2026, 10, 22, 16, 25)  # 木曜・大引け後（当日の足が確定しているべき時刻）


def _bar(symbol: str, d: date) -> None:
    df = pd.DataFrame({"open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0],
                       "volume": [1000]}, index=pd.to_datetime([d]))
    df.index.name = "date"
    upsert_ohlcv(symbol, df)


@pytest.fixture
def svc(isolated_db, monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: NOW)
    return trading.TradingServices(client=None, risk=RiskManager(), order_mgr=None)


def test_restores_freshness_from_db_after_restart(svc):
    assert bar_status.as_of_session(NOW) == date(2026, 10, 22)
    _bar("1111", date(2026, 10, 22))   # 当日の足あり
    _bar("2222", date(2026, 10, 21))   # 前日まで
    assert svc._bar_states == {} and svc._data_batch_id is None  # 再起動直後の状態

    svc._restore_bar_states_from_db(["1111", "2222", "3333"])

    assert svc._is_fresh_for_new_candidate("1111") is True
    assert svc._is_fresh_for_new_candidate("2222") is False   # stale
    assert svc._is_fresh_for_new_candidate("3333") is False   # 足が無い（missing）
    assert svc._data_batch_id.startswith("restored-")


def test_signal_scan_restores_when_states_empty(svc, monkeypatch):
    called = {}
    monkeypatch.setattr(trading.TradingServices, "_restore_bar_states_from_db",
                        lambda self, codes: called.setdefault("codes", codes))
    monkeypatch.setattr(trading.watchlist_store, "get_all_codes", lambda: ["1111"])
    monkeypatch.setattr(trading.market_calendar, "is_holiday", lambda d: False)
    try:
        svc.signal_scan()
    except Exception:
        pass  # 以降のスキャン本体は本テストの対象外
    assert called["codes"] == ["1111"]


def test_manual_signal_scan_endpoint(isolated_db, monkeypatch):
    import src.dashboard.app as dash
    dash._auth_required = False
    ran = []
    monkeypatch.setattr(dash, "_signal_scan_fn", lambda: ran.append(1))
    r = TestClient(dash.app).post("/api/signal_scan")
    assert r.status_code == 200 and ran == [1]


def test_manual_signal_scan_endpoint_unavailable(isolated_db, monkeypatch):
    import src.dashboard.app as dash
    dash._auth_required = False
    monkeypatch.setattr(dash, "_signal_scan_fn", None)
    assert TestClient(dash.app).post("/api/signal_scan").status_code == 503
