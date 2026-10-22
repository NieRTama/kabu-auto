"""
保有期限（time_stop）と一部利確（partial_take_profit）のテスト

- time_stop: 保有開始（残ロットのうち最古のBUY約定日）から time_stop_days 以上経っても
  含み益率が time_stop_min_gain_pct 未満なら全量退出する（動かない銘柄で資金を寝かせない）。
- partial_take_profit: 含み益率が partial_take_profit_pct 以上で、まだ一部利確していなければ
  保有の partial_take_profit_ratio を単元(100株)単位で切り捨てて売る。1サイクル1回だけ。
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from sqlalchemy import select

import src.core.scheduler as scheduler_mod
import src.risk.manager as risk_mod
from src.data.database import Fill, Position, get_session
from src.execution.lots import rebuild_position
from src.services.trading import TradingServices

NOW = datetime(2026, 10, 21, 10, 0)


def _add_holding(symbol="7203", avg_cost=1000.0, quantity=100, bought_days_ago=0,
                 partial_taken=False):
    with get_session() as session:
        session.add(Fill(broker_order_id="b1", symbol=symbol, side="BUY", fill_qty=quantity,
                         fill_price=avg_cost, filled_at=NOW - timedelta(days=bought_days_ago),
                         source="test", remaining_qty=quantity))
        session.add(Position(symbol=symbol, quantity=quantity, avg_cost=avg_cost,
                             peak_price=avg_cost, partial_taken=partial_taken))
        session.commit()


def _risk(**extra):
    risk = risk_mod.RiskManager()
    risk._conf = {"stop_loss_pct": -0.08, "trailing_stop_pct": 0.0,
                  "breakeven_trigger_pct": 0.0, **extra}
    return risk


def _exit(risk, symbol, price):
    with patch("src.risk.manager.clock.now", return_value=NOW):
        return risk.evaluate_exit(symbol, price)


class TestTimeStop:
    def test_stale_flat_position_exits(self, isolated_db):
        _add_holding(bought_days_ago=15)
        risk = _risk(time_stop_days=14, time_stop_min_gain_pct=0.02)
        assert _exit(risk, "7203", 1010.0) == (True, "time_stop")  # +1%

    def test_stale_losing_position_exits(self, isolated_db):
        _add_holding(bought_days_ago=15)
        risk = _risk(time_stop_days=14, time_stop_min_gain_pct=0.02)
        assert _exit(risk, "7203", 960.0) == (True, "time_stop")  # -4%（損切り線の手前）

    def test_stale_but_rising_position_is_kept(self, isolated_db):
        _add_holding(bought_days_ago=15)
        risk = _risk(time_stop_days=14, time_stop_min_gain_pct=0.02)
        assert _exit(risk, "7203", 1030.0) == (False, "")  # +3%

    def test_young_position_is_kept(self, isolated_db):
        _add_holding(bought_days_ago=13)
        risk = _risk(time_stop_days=14, time_stop_min_gain_pct=0.02)
        assert _exit(risk, "7203", 1000.0) == (False, "")

    def test_disabled_when_zero(self, isolated_db):
        _add_holding(bought_days_ago=100)
        risk = _risk(time_stop_days=0)
        assert _exit(risk, "7203", 1000.0) == (False, "")

    def test_holding_start_uses_remaining_lots_not_stale_opened_at(self, isolated_db):
        # Position.opened_at は再エントリーで更新されず古いまま残ることがある
        # （9434: 9/08の行のまま10/13に買い直し）。残ロットの約定日で判定する。
        _add_holding(bought_days_ago=5)
        with get_session() as session:
            pos = session.scalar(select(Position).where(Position.symbol == "7203"))
            pos.opened_at = NOW - timedelta(days=60)
            session.commit()
        risk = _risk(time_stop_days=14, time_stop_min_gain_pct=0.02)
        assert _exit(risk, "7203", 1000.0) == (False, "")

    def test_stop_loss_takes_priority(self, isolated_db):
        _add_holding(bought_days_ago=30)
        risk = _risk(time_stop_days=14, time_stop_min_gain_pct=0.02)
        assert _exit(risk, "7203", 900.0) == (True, "stop_loss")


class TestPartialTakeQty:
    def _qty(self, quantity, price, ratio=0.5, pct=0.05, taken=False):
        _add_holding(quantity=quantity, partial_taken=taken)
        risk = _risk(partial_take_profit_pct=pct, partial_take_profit_ratio=ratio)
        return risk.partial_take_qty("7203", price)

    def test_half_of_300_rounds_down_to_100(self, isolated_db):
        assert self._qty(300, 1050.0) == 100

    def test_half_of_400_is_200(self, isolated_db):
        assert self._qty(400, 1060.0) == 200

    def test_single_lot_cannot_be_split(self, isolated_db):
        assert self._qty(100, 1100.0) == 0

    def test_below_trigger(self, isolated_db):
        assert self._qty(300, 1040.0) == 0

    def test_only_once_per_cycle(self, isolated_db):
        assert self._qty(300, 1100.0, taken=True) == 0

    def test_disabled_when_zero(self, isolated_db):
        assert self._qty(300, 1100.0, pct=0.0) == 0

    def test_mark_then_reset_on_full_close_and_reentry(self, isolated_db):
        _add_holding(quantity=300)
        risk = _risk(partial_take_profit_pct=0.05, partial_take_profit_ratio=0.5)
        risk.mark_partial_taken("7203")
        assert risk.partial_take_qty("7203", 1100.0) == 0
        with get_session() as session:
            # 全量決済 → 残ロット0
            for f in session.scalars(select(Fill)).all():
                f.remaining_qty = 0
            rebuild_position(session, "7203")
            session.commit()
            pos = session.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.partial_taken is False
            # 再エントリーで保有開始日も更新される
            session.add(Fill(broker_order_id="b2", symbol="7203", side="BUY", fill_qty=300,
                             fill_price=1000.0, filled_at=NOW, source="test", remaining_qty=300))
            with patch("src.execution.lots.clock.now", return_value=NOW):
                rebuild_position(session, "7203")
            session.commit()
            pos = session.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.opened_at == NOW


class TestStopLossCheckWiring:
    def _run(self, exit_ret=(False, ""), partial_qty=0, order_id="o1"):
        client, risk, order_mgr = MagicMock(), MagicMock(), MagicMock()
        risk.evaluate_exit.return_value = exit_ret
        risk.partial_take_qty.return_value = partial_qty
        order_mgr.sell_market.return_value = order_id
        with patch("src.services.trading.cfg") as cfg_mock:
            cfg_mock.get_section.return_value = {"mode": "live"}
            svc = TradingServices(client, risk, order_mgr)
        client.get_board.return_value = {"CurrentPrice": 1100.0}
        with patch.object(scheduler_mod.TradingScheduler, "is_market_open", return_value=True), \
             patch("src.services.trading.watchlist_store") as watchlist_mock, \
             patch("src.services.trading._held_symbols", return_value=[]), \
             patch("src.services.trading._get_position_qty", return_value=300), \
             patch("src.services.trading.alert") as alert_mock:
            watchlist_mock.get_codes.return_value = ["7203"]
            svc.stop_loss_check()
        return risk, order_mgr, alert_mock

    def test_time_stop_sells_all(self):
        _, order_mgr, alert_mock = self._run(exit_ret=(True, "time_stop"))
        order_mgr.sell_market.assert_called_once_with("7203", 300, reason="time_stop")
        assert "保有期限" in alert_mock.call_args[0][0]

    def test_partial_sells_part_and_marks(self):
        risk, order_mgr, alert_mock = self._run(partial_qty=100)
        order_mgr.sell_market.assert_called_once_with("7203", 100, reason="partial_take_profit")
        risk.mark_partial_taken.assert_called_once_with("7203")
        assert "一部利確" in alert_mock.call_args[0][0]

    def test_partial_rejected_does_not_mark(self):
        risk, order_mgr, _ = self._run(partial_qty=100, order_id=None)
        risk.mark_partial_taken.assert_not_called()

    def test_full_exit_skips_partial(self):
        risk, _, _ = self._run(exit_ret=(True, "stop_loss"), partial_qty=100)
        risk.partial_take_qty.assert_not_called()
