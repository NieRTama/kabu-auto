"""
トレーリングストップ・ブレークイーブンロジック（RiskManager.evaluate_exit）のテスト

含み益のピークを追跡し、(1) 基準の損切りライン、(2) ブレークイーブン到達後の
取得単価保全、(3) ピークからのトレーリングストップ、のうち最も安全なラインで
退出判定する。breakeven_trigger_pct/trailing_stop_pctが0なら従来の損切りのみと
完全に一致することも確認する。
"""
from unittest.mock import MagicMock, patch

import pytest

import src.core.config as cfg
import src.core.scheduler as scheduler_mod
import src.data.database as db
import src.risk.manager as risk_mod
from src.data.database import Position, get_session
from src.services.trading import TradingServices


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


def _add_position(symbol="7203", avg_cost=1000.0, peak_price=None, quantity=100):
    with get_session() as session:
        session.add(Position(symbol=symbol, quantity=quantity, avg_cost=avg_cost,
                             peak_price=peak_price))
        session.commit()


def _risk(stop_loss_pct=-0.07, trailing_stop_pct=0.0, breakeven_trigger_pct=0.0):
    risk = risk_mod.RiskManager()
    risk._conf = {
        "stop_loss_pct": stop_loss_pct,
        "trailing_stop_pct": trailing_stop_pct,
        "breakeven_trigger_pct": breakeven_trigger_pct,
    }
    return risk


class TestPureStopLossBackwardCompat:
    """trailing/breakevenが0(未設定)なら旧should_stop_lossと同じ挙動"""

    def test_no_position_returns_false(self, isolated_db):
        risk = _risk()
        assert risk.evaluate_exit("9999", 1000.0) == (False, "")

    def test_above_stop_line_no_exit(self, isolated_db):
        _add_position(avg_cost=1000.0, peak_price=1000.0)
        risk = _risk(stop_loss_pct=-0.07)
        should_exit, reason = risk.evaluate_exit("7203", 950.0)  # -5%
        assert should_exit is False

    def test_breaches_stop_line_exits_as_stop_loss(self, isolated_db):
        _add_position(avg_cost=1000.0, peak_price=1000.0)
        risk = _risk(stop_loss_pct=-0.07)
        should_exit, reason = risk.evaluate_exit("7203", 920.0)  # -8%
        assert should_exit is True
        assert reason == "stop_loss"


class TestPeakTracking:
    def test_peak_price_updates_on_new_high(self, isolated_db):
        _add_position(avg_cost=1000.0, peak_price=1000.0)
        risk = _risk()
        risk.evaluate_exit("7203", 1200.0)
        with get_session() as session:
            from sqlalchemy import select
            pos = session.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.peak_price == 1200.0

    def test_peak_price_does_not_decrease(self, isolated_db):
        _add_position(avg_cost=1000.0, peak_price=1200.0)
        risk = _risk()
        risk.evaluate_exit("7203", 1100.0)  # 現在値はピークより低い
        with get_session() as session:
            from sqlalchemy import select
            pos = session.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.peak_price == 1200.0  # ピークは下がらない

    def test_peak_initialized_from_avg_cost_when_none(self, isolated_db):
        _add_position(avg_cost=1000.0, peak_price=None)
        risk = _risk()
        risk.evaluate_exit("7203", 900.0)
        with get_session() as session:
            from sqlalchemy import select
            pos = session.scalar(select(Position).where(Position.symbol == "7203"))
            assert pos.peak_price == 1000.0  # avg_costとmax(900)→1000のまま


class TestBreakeven:
    def test_not_armed_below_trigger(self, isolated_db):
        """ピーク含み益がbreakeven_trigger_pct未満なら、従来の損切りラインのみ"""
        _add_position(avg_cost=1000.0, peak_price=1020.0)  # +2%（トリガー3%未満）
        risk = _risk(stop_loss_pct=-0.07, breakeven_trigger_pct=0.03, trailing_stop_pct=0.05)
        should_exit, reason = risk.evaluate_exit("7203", 950.0)  # -5%（旧損切りラインの内側）
        assert should_exit is False

    def test_armed_protects_breakeven(self, isolated_db):
        """ピーク+3%以上に達した後は、取得単価を下回ったら即退出（旧損切りラインより内側でも）"""
        _add_position(avg_cost=1000.0, peak_price=1050.0)  # +5%（トリガー3%超え）
        risk = _risk(stop_loss_pct=-0.07, breakeven_trigger_pct=0.03, trailing_stop_pct=0.0)
        should_exit, reason = risk.evaluate_exit("7203", 990.0)  # -1%（取得単価未満）
        assert should_exit is True
        assert reason == "trailing_stop"

    def test_armed_but_still_above_breakeven_no_exit(self, isolated_db):
        _add_position(avg_cost=1000.0, peak_price=1050.0)
        risk = _risk(stop_loss_pct=-0.07, breakeven_trigger_pct=0.03, trailing_stop_pct=0.0)
        should_exit, reason = risk.evaluate_exit("7203", 1010.0)  # 取得単価より上
        assert should_exit is False


class TestTrailing:
    def test_trailing_line_above_breakeven_triggers_earlier(self, isolated_db):
        """ピークが十分伸びていれば、トレーリングラインが取得単価より高くなり先に発動する"""
        _add_position(avg_cost=1000.0, peak_price=1200.0)  # +20%
        risk = _risk(stop_loss_pct=-0.07, breakeven_trigger_pct=0.03, trailing_stop_pct=0.05)
        # トレーリングライン = 1200 * (1-0.05) = 1140（取得単価1000より高い）
        should_exit, reason = risk.evaluate_exit("7203", 1130.0)
        assert should_exit is True
        assert reason == "trailing_stop"

    def test_above_trailing_line_no_exit(self, isolated_db):
        _add_position(avg_cost=1000.0, peak_price=1200.0)
        risk = _risk(stop_loss_pct=-0.07, breakeven_trigger_pct=0.03, trailing_stop_pct=0.05)
        should_exit, reason = risk.evaluate_exit("7203", 1150.0)  # トレーリングライン1140より上
        assert should_exit is False

    def test_trailing_line_tracks_new_peak_in_same_call(self, isolated_db):
        """現在値が新ピークを更新した場合、そのピークに基づくトレーリングラインで判定する"""
        _add_position(avg_cost=1000.0, peak_price=1000.0)
        risk = _risk(stop_loss_pct=-0.07, breakeven_trigger_pct=0.03, trailing_stop_pct=0.05)
        # 現在値1300が新ピーク。トレーリングライン=1300*0.95=1235 < 1300なので退出しない
        should_exit, _ = risk.evaluate_exit("7203", 1300.0)
        assert should_exit is False


class TestStopLossCheckWiring:
    """stop_loss_check() がevaluate_exit()の結果をsell_market()へ正しく橋渡しすることの検証"""

    def _services(self, evaluate_exit_return):
        client, risk, order_mgr = MagicMock(), MagicMock(), MagicMock()
        risk.evaluate_exit.return_value = evaluate_exit_return
        with patch("src.services.trading.cfg") as cfg_mock:
            cfg_mock.get_section.return_value = {"mode": "paper"}
            svc = TradingServices(client, risk, order_mgr)
        return svc, risk, order_mgr

    def _run(self, svc, price=900.0):
        with patch.object(scheduler_mod.TradingScheduler, "is_market_open", return_value=True), \
             patch("src.services.trading.watchlist_store") as watchlist_mock, \
             patch("src.services.trading._get_position_qty", return_value=100), \
             patch("src.services.trading.load_ohlcv") as load_ohlcv_mock, \
             patch("src.services.trading.alert") as alert_mock:
            watchlist_mock.get_codes.return_value = ["7203"]
            df_mock = MagicMock()
            df_mock.__len__.return_value = 1
            df_mock.__getitem__.return_value.iloc.__getitem__.return_value = price
            load_ohlcv_mock.return_value = df_mock
            svc.stop_loss_check()
            return alert_mock

    def test_market_closed_skips_entirely(self):
        svc, risk, order_mgr = self._services((True, "stop_loss"))
        with patch.object(scheduler_mod.TradingScheduler, "is_market_open", return_value=False):
            svc.stop_loss_check()
        risk.evaluate_exit.assert_not_called()
        order_mgr.sell_market.assert_not_called()

    def test_no_exit_does_not_sell(self):
        svc, risk, order_mgr = self._services((False, ""))
        self._run(svc)
        order_mgr.sell_market.assert_not_called()

    def test_stop_loss_reason_sells_with_stop_loss_reason_and_alert(self):
        svc, risk, order_mgr = self._services((True, "stop_loss"))
        alert_mock = self._run(svc, price=900.0)
        order_mgr.sell_market.assert_called_once_with("7203", 100, reason="stop_loss")
        title = alert_mock.call_args[0][0]
        assert title == "損切り実行"

    def test_trailing_stop_reason_sells_with_trailing_stop_reason_and_alert(self):
        svc, risk, order_mgr = self._services((True, "trailing_stop"))
        alert_mock = self._run(svc, price=1100.0)
        order_mgr.sell_market.assert_called_once_with("7203", 100, reason="trailing_stop")
        title = alert_mock.call_args[0][0]
        assert "トレーリングストップ" in title
