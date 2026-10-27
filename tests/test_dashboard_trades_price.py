"""/api/trades の価格表示のテスト。

2026-10-27: 退出の売り（成行）は price=0 で記録され、実単価は filled_price に入る。
取引履歴が price だけを返していたため ¥0 と表示され、リターン%も計算されなかった。
"""
from fastapi.testclient import TestClient

import src.dashboard.app as dash
from src.data.database import Trade, get_session


def test_market_sell_shows_filled_price_and_limit_order_keeps_price(isolated_db, monkeypatch):
    monkeypatch.setattr(dash, "_auth_required", False)
    with get_session() as s:
        s.add(Trade(order_id="m", symbol="3489", side="SELL", quantity=100, price=0.0,
                    filled_price=751.8, filled_quantity=100, pnl=1390.0, status="FILLED"))
        s.add(Trade(order_id="l", symbol="3660", side="BUY", quantity=200, price=467.0,
                    status="PENDING"))
        s.commit()
    rows = {r["order_id"]: r for r in TestClient(dash.app).get("/api/trades").json()}
    assert rows["m"]["price"] == 751.8, "成行の売りは約定単価を返すこと（¥0にしない）"
    assert rows["l"]["price"] == 467.0, "未約定は注文価格のまま"
