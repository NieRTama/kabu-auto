"""
注文照合(reconcile_orders)の定期実行に関するテスト（再レビュー P0-1対応）

- スケジューラに interval ジョブとして登録されること
- TradingServices.reconcile_orders() が市場時間外は何もせず、市場時間中のみ
  OrderManager.reconcile_open_orders() を呼ぶこと
"""
from unittest.mock import MagicMock, patch

import src.core.scheduler as scheduler_mod
from src.services.trading import TradingServices


class TestReconcileSchedulerRegistration:
    def test_reconcile_orders_registered_as_interval_job(self):
        sched = scheduler_mod.TradingScheduler()
        sched.register("reconcile_orders", MagicMock())
        with patch.object(sched._scheduler, "add_job") as mock_add_job, \
             patch.object(sched._scheduler, "start"):
            sched.start()
        calls = {c.kwargs["id"]: c for c in mock_add_job.call_args_list}
        assert "reconcile_orders" in calls
        call = calls["reconcile_orders"]
        assert call.args[1] == "interval"
        assert call.kwargs.get("seconds", 0) > 0

    def test_omitted_when_not_registered(self):
        """登録しなければジョブは追加されない（既存のオプショナル登録パターンと同様）"""
        sched = scheduler_mod.TradingScheduler()
        with patch.object(sched._scheduler, "add_job") as mock_add_job, \
             patch.object(sched._scheduler, "start"):
            sched.start()
        ids = {c.kwargs["id"] for c in mock_add_job.call_args_list}
        assert "reconcile_orders" not in ids


class TestTradingServicesReconcileOrders:
    def _services(self, market_open: bool):
        client, risk, order_mgr = MagicMock(), MagicMock(), MagicMock()
        with patch.object(scheduler_mod.TradingScheduler, "is_market_open",
                          return_value=market_open), \
             patch("src.services.trading.cfg") as cfg_mock:
            cfg_mock.get_section.return_value = {}
            svc = TradingServices(client, risk, order_mgr)
        return svc, order_mgr

    def test_calls_reconcile_when_market_open(self):
        svc, order_mgr = self._services(market_open=True)
        with patch.object(scheduler_mod.TradingScheduler, "is_market_open", return_value=True):
            svc.reconcile_orders()
        order_mgr.reconcile_open_orders.assert_called_once()

    def test_skips_when_market_closed(self):
        svc, order_mgr = self._services(market_open=False)
        with patch.object(scheduler_mod.TradingScheduler, "is_market_open", return_value=False):
            svc.reconcile_orders()
        order_mgr.reconcile_open_orders.assert_not_called()

    def test_exception_is_swallowed(self):
        svc, order_mgr = self._services(market_open=True)
        order_mgr.reconcile_open_orders.side_effect = RuntimeError("boom")
        with patch.object(scheduler_mod.TradingScheduler, "is_market_open", return_value=True):
            svc.reconcile_orders()  # raiseしない
