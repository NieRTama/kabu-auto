"""data/stop_loss.pause がある間は損切りと保有期限の売りをせず、建値ストップ・トレーリングは残す。"""
from datetime import datetime, timedelta
from unittest.mock import patch

import src.risk.manager as risk_mod
from src.data.database import Fill, Position, get_session

NOW = datetime(2026, 11, 10, 10, 0)


def _hold(avg_cost=1000.0, peak=None, days_ago=20):
    with get_session() as s:
        s.add(Fill(broker_order_id="b1", symbol="7203", side="BUY", fill_qty=100, fill_price=avg_cost,
                   filled_at=NOW - timedelta(days=days_ago), source="test", remaining_qty=100))
        s.add(Position(symbol="7203", quantity=100, avg_cost=avg_cost, peak_price=peak or avg_cost))
        s.commit()


def _exit(price):
    risk = risk_mod.RiskManager()
    risk._conf = {"stop_loss_pct": -0.15, "breakeven_trigger_pct": 0.02, "trailing_stop_pct": 0.04,
                  "time_stop_days": 14, "time_stop_min_gain_pct": 0.02}
    with patch("src.risk.manager.clock.now", return_value=NOW):
        return risk.evaluate_exit("7203", price)


def test_stop_loss_and_time_stop_paused(isolated_db):
    _hold()
    pause = risk_mod.STOP_LOSS_PAUSE
    assert _exit(800.0) == (True, "stop_loss")   # -20%: 置く前は損切り
    assert _exit(960.0) == (True, "time_stop")   # -4%・20日: 置く前は保有期限
    pause.write_text("")
    assert _exit(800.0) == (False, "")
    assert _exit(960.0) == (False, "")
    pause.unlink()
    assert _exit(800.0) == (True, "stop_loss")   # 消せば再開


def test_trailing_still_works_while_paused(isolated_db):
    _hold(peak=1100.0)  # ピーク +10% で建値ストップ・トレーリングが有効
    risk_mod.STOP_LOSS_PAUSE.write_text("")
    assert _exit(1050.0) == (True, "trailing_stop")  # 1100×0.96=1056 を下回った
