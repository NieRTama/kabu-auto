"""paper経路の執行仮定の切替（engine_version）のテスト

stop_loss_checkはpaperモードで日足終値を使って損切りを判定しており、
F04（同一終値での判断・約定）がpaper運用にも及んでいた（spec §8）。
**engine_version: legacy のとき挙動が1ビットも変わらないこと**を固定する。

Task 7 レビュー是正（Important指摘2件）:
  - 指摘1: signal_scanのv2分岐が保存したシグナルを、_execute_pending_signals
    （morning_execution経由）が拾えるようにする（呼び出し元ゲートがmodeのみを
    見てengine_versionを見ていなかったため、v2+paperでは永久に拾われなかった）。
  - 指摘2: stop_loss_checkのv2分岐で、退出判断を即sell_market()に直結させず、
    signal_scanのSELL経路と同じ合流先（Signal保存）に載せ、翌営業日の
    morning_executionに実際の退出発注を委ねる。
"""
import inspect
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

import src.core.config as cfg
import src.core.scheduler as scheduler_mod
import src.data.database as db
from src.data.database import Signal, get_session
from src.services import trading


@pytest.fixture(autouse=True)
def _load_config():
    cfg.load("config.yaml")


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


class TestEngineVersionDefault:
    def test_defaults_to_legacy(self):
        """config.yaml の既定は legacy（旧方式へいつでも戻せる）"""
        assert cfg.get_section("strategy").get("engine_version") == "legacy"

    def test_unknown_value_is_treated_as_legacy(self):
        """未知の値は安全側（legacy）に倒す"""
        cfg.get_section("strategy")["engine_version"] = "experimental"
        svc = trading.TradingServices(client=None, risk=None, order_mgr=None)
        assert svc._engine_version() == "legacy"
        assert svc._paper_uses_v2_execution() is False

    def test_v2_is_recognized(self):
        cfg.get_section("strategy")["engine_version"] = "v2"
        svc = trading.TradingServices(client=None, risk=None, order_mgr=None)
        assert svc._engine_version() == "v2"
        assert svc._paper_uses_v2_execution() is True


class TestLegacyBehaviourUnchanged:
    """legacy では paper の挙動が現在と変わらないこと"""

    def _source(self, name):
        return inspect.getsource(getattr(trading.TradingServices, name))

    def test_stop_loss_check_still_has_the_legacy_close_path(self):
        """従来の日足終値による損切り判定が残っている"""
        src = self._source("stop_loss_check")
        assert "load_ohlcv" in src
        assert 'df["close"].iloc[-1]' in src

    def test_stop_loss_check_branches_on_engine_version(self):
        """v2のときだけ別の経路へ入る分岐がある"""
        assert "_paper_uses_v2_execution" in self._source("stop_loss_check")

    def test_signal_scan_branches_on_engine_version(self):
        assert "_paper_uses_v2_execution" in self._source("signal_scan")

    def test_freshness_gate_is_still_wired(self):
        """段階Aの鮮度ゲートが外れていない"""
        assert "_is_fresh_for_new_candidate" in self._source("signal_scan")

    def test_stop_loss_check_does_not_call_the_freshness_gate(self):
        """保有保護の退出は鮮度に関わらず実行する（段階Aの不変条件）"""
        assert "_is_fresh_for_new_candidate" not in self._source("stop_loss_check")


# ─── 指摘1: v2+paper でも _execute_pending_signals がゲートで return しない ──


class TestExecutePendingSignalsGateV2Paper:
    """signal_scan がv2で保存したシグナルを、翌営業日の morning_execution
    （_execute_pending_signals）が実際に拾って発注処理まで進むこと。
    """

    def _make_services(self, mode, engine_version, board_price=1000.0):
        cfg.get_section("strategy")["engine_version"] = engine_version
        client = MagicMock()
        client.get_board.return_value = {"CurrentPrice": board_price}
        client.get_wallet.return_value = {"StockAccountWallet": 500_000.0}
        risk = MagicMock()
        risk.validate_buy.return_value = (True, "")
        risk.calc_position_size.return_value = 100
        order_mgr = MagicMock()
        order_mgr.sell.return_value = "SELL-ORD"
        order_mgr.buy.return_value = "BUY-ORD"
        svc = trading.TradingServices(client, risk, order_mgr)
        svc.trading_conf = {"mode": mode}
        return svc, client, risk, order_mgr

    def _seed_sell_signal(self, symbol="7203"):
        with get_session() as session:
            session.add(Signal(symbol=symbol, action="SELL",
                               rule_score=0.0, ml_score=0.0, combined_score=0.0))
            session.commit()

    def test_v2_paper_reaches_order_flow_via_morning_execution(self, isolated_db):
        """v2 + paper: ゲートで return されず、保存済みSELLシグナルに基づき
        実際に order_mgr.sell() まで到達する"""
        svc, client, risk, order_mgr = self._make_services("paper", "v2")
        self._seed_sell_signal("7203")
        with patch.object(scheduler_mod.TradingScheduler, "is_market_open", return_value=True), \
             patch("src.services.trading._get_position_qty", return_value=100), \
             patch("src.services.trading.alert"):
            svc.morning_execution()
        order_mgr.sell.assert_called_once()
        assert order_mgr.sell.call_args[0][0] == "7203"

    def test_legacy_paper_still_returns_at_gate(self, isolated_db):
        """回帰防止: legacy + paper は従来通りゲートで即returnし、
        is_market_open すら呼ばれず、保存済みシグナルも拾わない"""
        svc, client, risk, order_mgr = self._make_services("paper", "legacy")
        self._seed_sell_signal("7203")
        with patch.object(scheduler_mod.TradingScheduler, "is_market_open") as market_mock, \
             patch("src.services.trading._get_position_qty", return_value=100):
            svc.morning_execution()
        market_mock.assert_not_called()
        order_mgr.sell.assert_not_called()

    def test_v2_paper_market_closed_still_returns(self, isolated_db):
        """v2 + paper でもゲート自体は通過するだけで、市場が閉じていれば
        通常通り何もしない（ゲート変更が他の安全条件をバイパスしないこと）"""
        svc, client, risk, order_mgr = self._make_services("paper", "v2")
        self._seed_sell_signal("7203")
        with patch.object(scheduler_mod.TradingScheduler, "is_market_open", return_value=False), \
             patch("src.services.trading._get_position_qty", return_value=100):
            svc.morning_execution()
        order_mgr.sell.assert_not_called()


# ─── 指摘2: stop_loss_check のv2分岐は即sell_marketを呼ばず翌営業日に委ねる ──


class TestStopLossCheckDefersInV2:
    def _services(self, engine_version, evaluate_exit_return=(True, "stop_loss")):
        cfg.get_section("strategy")["engine_version"] = engine_version
        client, risk, order_mgr = MagicMock(), MagicMock(), MagicMock()
        risk.evaluate_exit.return_value = evaluate_exit_return
        svc = trading.TradingServices(client, risk, order_mgr)
        svc.trading_conf = {"mode": "paper"}
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

    def test_v2_does_not_call_sell_market(self, isolated_db):
        svc, risk, order_mgr = self._services("v2")
        self._run(svc)
        order_mgr.sell_market.assert_not_called()

    def test_v2_saves_sell_signal_to_db(self, isolated_db):
        svc, risk, order_mgr = self._services("v2")
        self._run(svc)
        with get_session() as session:
            sig = session.scalar(select(Signal).where(Signal.symbol == "7203"))
            assert sig is not None
            assert sig.action == "SELL"

    def test_v2_alert_mentions_next_session_execution(self, isolated_db):
        """通知文言が『即実行した』ではなく『翌営業日に執行予定』へ更新されていること"""
        svc, risk, order_mgr = self._services("v2")
        alert_mock = self._run(svc)
        alert_mock.assert_called_once()
        title, body = alert_mock.call_args[0][0], alert_mock.call_args[0][1]
        assert "翌営業日" in title or "翌営業日" in body

    def test_legacy_still_calls_sell_market_immediately(self, isolated_db):
        """回帰防止: legacy は従来通り即座に sell_market を呼ぶ"""
        svc, risk, order_mgr = self._services("legacy")
        self._run(svc)
        order_mgr.sell_market.assert_called_once_with("7203", 100, reason="stop_loss")

    def test_v2_trailing_stop_reason_also_defers(self, isolated_db):
        """退出理由がtrailing_stopの場合もv2では即sell_marketを呼ばない"""
        svc, risk, order_mgr = self._services("v2", evaluate_exit_return=(True, "trailing_stop"))
        self._run(svc, price=1100.0)
        order_mgr.sell_market.assert_not_called()
        with get_session() as session:
            sig = session.scalar(select(Signal).where(Signal.symbol == "7203"))
            assert sig is not None
            assert sig.action == "SELL"

    def test_v2_signal_is_picked_up_by_next_morning_execution(self, isolated_db):
        """v2で保存されたSELLシグナルが、翌営業日のmorning_executionで
        実際に売り注文（order_mgr.sell）まで進むこと"""
        svc, risk, order_mgr = self._services("v2")
        self._run(svc)  # stop_loss_check がSignalをDBに保存する

        # 翌営業日: 別インスタンスで morning_execution を実行
        client2 = MagicMock()
        client2.get_board.return_value = {"CurrentPrice": 950.0}
        client2.get_wallet.return_value = {"StockAccountWallet": 500_000.0}
        order_mgr2 = MagicMock()
        order_mgr2.sell.return_value = "SELL-ORD"
        svc2 = trading.TradingServices(client2, risk, order_mgr2)
        svc2.trading_conf = {"mode": "paper"}
        with patch.object(scheduler_mod.TradingScheduler, "is_market_open", return_value=True), \
             patch("src.services.trading._get_position_qty", return_value=100), \
             patch("src.services.trading.alert"):
            svc2.morning_execution()
        order_mgr2.sell.assert_called_once()
        assert order_mgr2.sell.call_args[0][0] == "7203"
