"""
afternoon_execution（後場スロット）の検証テスト

2026-08-25 取引頻度向上のため追加。朝(morning_execution)に見送った銘柄を
後場(12:35)に拾い直すジョブで、既存の morning_execution とロジックを共有する
（src.services.trading.TradingServices._execute_pending_signals）。

検証observategory:
  - スケジューラに 12:35・平日で登録されていること
  - main.py から afternoon_execution がコールバック登録されていること
  - skip_existing=True により、既に保有中の銘柄へはBUYしない（買い増ししない）こと
  - skip_existing=True でも SELL（保有分の手仕舞い）は従来どおり行われること
  - morning_execution（skip_existing=False）は従来どおり保有有無に関係なくBUY評価すること
    （後場だけの挙動変更であり、朝の挙動を変えていないことの回帰防止）
"""
from unittest.mock import MagicMock, patch

import src.core.scheduler as scheduler_mod
from src.services.trading import TradingServices


# ─── スケジューラ登録 ────────────────────────────────────────────────────


class TestAfternoonExecutionScheduling:
    def _start_and_capture(self):
        sched = scheduler_mod.TradingScheduler()
        for name in (
            "morning_execution", "afternoon_execution",
        ):
            sched.register(name, MagicMock())

        calls = {}
        with patch.object(sched._scheduler, "add_job") as mock_add_job, \
             patch.object(sched._scheduler, "start"):
            sched.start()
            for call in mock_add_job.call_args_list:
                kwargs = call.kwargs
                calls[kwargs["id"]] = (
                    kwargs.get("hour"), kwargs.get("minute"), kwargs.get("day_of_week"),
                )
        return calls

    def test_afternoon_execution_registered_at_1235_weekdays(self):
        calls = self._start_and_capture()
        assert "afternoon_execution" in calls, "afternoon_execution ジョブが登録されていない"
        hour, minute, dow = calls["afternoon_execution"]
        assert (hour, minute) == (12, 35)
        assert dow == "mon-fri"

    def test_afternoon_execution_runs_after_morning_execution(self):
        calls = self._start_and_capture()
        m_hour, m_minute, _ = calls["morning_execution"]
        a_hour, a_minute, _ = calls["afternoon_execution"]
        assert (a_hour * 60 + a_minute) > (m_hour * 60 + m_minute)

    def test_omitted_when_not_registered(self):
        """register() されていなければ add_job は afternoon_execution 分を呼ばない"""
        sched = scheduler_mod.TradingScheduler()
        with patch.object(sched._scheduler, "add_job") as mock_add_job, \
             patch.object(sched._scheduler, "start"):
            sched.start()
            ids = [c.kwargs.get("id") for c in mock_add_job.call_args_list]
        assert "afternoon_execution" not in ids


# ─── main.py 結線（ソース検証） ──────────────────────────────────────────
# main.py は uvicorn 等の重い依存をトップレベルでimportするため、他の結線テスト
# （test_risk_wiring.py 等）と同様にソーステキスト検証で行う。


class TestAfternoonExecutionWiring:
    def test_afternoon_execution_registered_with_scheduler(self):
        with open("main.py", encoding="utf-8") as f:
            main_src = f.read()
        assert 'scheduler.register("afternoon_execution", services.afternoon_execution)' in main_src, (
            "afternoon_execution がスケジューラに登録されていない"
        )


# ─── 実行ロジック（TradingServices を実インスタンス化して検証）──────────────


def _make_signal(symbol: str, action: str):
    sig = MagicMock()
    sig.symbol = symbol
    sig.action = action
    sig.combined_score = 0.2
    sig.rule_score = 0.2
    sig.ml_score = 0.2
    return sig


class TestSkipExistingBehavior:
    """skip_existing の有無で BUY 対象がどう変わるかを、実装コード
    （_execute_pending_signals）を実際に実行して検証する。"""

    def _services(self):
        client, risk, order_mgr = MagicMock(), MagicMock(), MagicMock()
        client.get_wallet.return_value = {"StockAccountWallet": 500_000.0}
        client.get_board.return_value = {"CurrentPrice": 1000.0, "Sell1": {"Price": 1000.0},
                                          "Buy1": {"Price": 1000.0}}
        risk.validate_buy.return_value = (True, "")
        risk.calc_position_size.return_value = 100
        with patch("src.services.trading.cfg") as cfg_mock:
            cfg_mock.get_section.return_value = {
                "mode": "live", "no_new_buy_minutes_before_close": 0,
            }
            svc = TradingServices(client, risk, order_mgr)
        return svc, risk, order_mgr

    def _run(self, svc, method_name: str, signals: list, held_qty: dict):
        with patch.object(scheduler_mod.TradingScheduler, "is_market_open", return_value=True), \
             patch.object(scheduler_mod.TradingScheduler, "is_near_close", return_value=False), \
             patch("src.services.trading._select_latest_signals", return_value=signals), \
             patch("src.services.trading._get_position_qty",
                   side_effect=lambda sym: held_qty.get(sym, 0)), \
             patch("src.services.trading.watchlist_store") as watchlist_mock, \
             patch("src.services.trading.load_ohlcv"), \
             patch("src.services.trading.liquidity") as liquidity_mock:
            watchlist_mock.get_sectors.return_value = {}
            liquidity_mock.check_liquidity.return_value = (True, "")
            liquidity_mock.check_spread.return_value = (True, "")
            getattr(svc, method_name)()

    def test_afternoon_skips_buy_for_already_held_symbol(self):
        """後場: 既に保有中の銘柄はBUY対象から除外される（買い増ししない）"""
        svc, risk, order_mgr = self._services()
        self._run(svc, "afternoon_execution",
                   signals=[_make_signal("7203", "BUY")],
                   held_qty={"7203": 100})
        order_mgr.buy.assert_not_called()

    def test_afternoon_buys_unheld_symbol(self):
        """後場: 未保有の銘柄は通常どおりBUY評価される（朝の見送りを拾い直す）"""
        svc, risk, order_mgr = self._services()
        self._run(svc, "afternoon_execution",
                   signals=[_make_signal("7203", "BUY")],
                   held_qty={"7203": 0})
        order_mgr.buy.assert_called_once()

    def test_afternoon_still_sells_held_symbol(self):
        """後場: skip_existing はBUYのみに作用し、SELL（保有分の手仕舞い）は従来どおり行う"""
        svc, risk, order_mgr = self._services()
        self._run(svc, "afternoon_execution",
                   signals=[_make_signal("7203", "SELL")],
                   held_qty={"7203": 100})
        order_mgr.sell.assert_called_once()

    def test_morning_still_buys_held_symbol(self):
        """朝: 保有有無に関係なくBUY評価する（後場だけの挙動変更であることの回帰防止）"""
        svc, risk, order_mgr = self._services()
        self._run(svc, "morning_execution",
                   signals=[_make_signal("7203", "BUY")],
                   held_qty={"7203": 100})
        order_mgr.buy.assert_called_once()
