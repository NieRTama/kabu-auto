"""
エラー率監視（error_rate）と生存確認（heartbeat）のテスト

2026-08 に起きた2件の障害（ログイン認証切れ・トレーリングストップのKeyError）は
どちらも既存の異常検知（未解決注文・損失上限）をすり抜けた。共通点は
「取引の結果」は見ているが「システムが機能しているか」を見ていなかったこと。

- error_rate: 「壊れたらエラーが増える」という普遍則で未知の障害を捉える
- heartbeat : 通知が来ないこと自体を異常として扱えるようにする能動的な生存信号

時刻は注入して実時間に依存させない。
"""
from unittest.mock import MagicMock, patch

import pytest

from src.core import broker_auth, error_rate, heartbeat


@pytest.fixture(autouse=True)
def _reset():
    error_rate.reset()
    broker_auth.reset()
    yield
    error_rate.reset()
    broker_auth.reset()


class TestErrorRateWindow:
    def test_counts_events_in_window(self):
        for i in range(5):
            error_rate.record(f"err{i}", now=100.0 + i)
        snap = error_rate.snapshot(now=110.0, window_seconds=900)
        assert snap["count"] == 5

    def test_drops_events_outside_window(self):
        error_rate.record("old", now=0.0)
        error_rate.record("recent", now=1000.0)
        snap = error_rate.snapshot(now=1000.0, window_seconds=900)
        assert snap["count"] == 1, "窓の外の古いエラーは数えない"
        assert snap["latest"] == "recent"

    def test_empty_when_no_errors(self):
        snap = error_rate.snapshot(now=100.0)
        assert snap["count"] == 0
        assert snap["latest"] == ""

    def test_latest_message_is_reported(self):
        error_rate.record("first", now=1.0)
        error_rate.record("last", now=2.0)
        assert error_rate.snapshot(now=3.0)["latest"] == "last"

    def test_reset_clears(self):
        error_rate.record("x", now=1.0)
        error_rate.reset()
        assert error_rate.snapshot(now=2.0)["count"] == 0


class TestErrorRateSink:
    @staticmethod
    def _message(text: str, level: str = "ERROR"):
        msg = MagicMock()
        level_obj = MagicMock()
        level_obj.name = level
        msg.record = {"message": text, "level": level_obj}
        return msg

    def test_sink_records_message(self):
        """loguru のシンクとして呼ばれたらカウントされる"""
        clock = {"t": 50.0}
        sink = error_rate.make_sink(clock=lambda: clock["t"])
        sink(self._message("決済チェックエラー: 6753 'trailing_stop'"))
        snap = error_rate.snapshot(now=50.0)
        assert snap["count"] == 1
        assert "trailing_stop" in snap["latest"]

    def test_sink_separates_warning_from_error(self):
        """WARNING は別枠で数える（閾値が違うため混ぜない）"""
        sink = error_rate.make_sink(clock=lambda: 50.0)
        sink(self._message("建玉照合失敗（次回再試行）: 401", level="WARNING"))
        sink(self._message("損切りチェックエラー: 9432 401"))
        snap = error_rate.snapshot(now=50.0)
        assert snap["count"] == 1
        assert snap["warning_count"] == 1
        assert "建玉照合失敗" in snap["latest_warning"]
        assert "損切りチェック" in snap["latest"]


class TestHealthIntegration:
    """health.check_anomalies がエラー多発を異常として拾うこと"""

    def _risk(self):
        r = MagicMock()
        # 損失上限は0（無効）にして、このテストではエラー率だけを見る
        r.daily_loss_limit.return_value = 0
        r.get_daily_loss.return_value = 0
        r.current_total_drawdown.return_value = 0
        r.unrealized_pnl.return_value = 0
        r.unpriced_symbols.return_value = []
        return r

    def _run(self, threshold=10, count=0, warn_threshold=50, warn_count=0):
        from src.core import health
        for i in range(count):
            error_rate.record(f"401 Unauthorized #{i}", now=1000.0)
        for i in range(warn_count):
            error_rate.record(f"建玉照合失敗（次回再試行）: 401 #{i}",
                              now=1000.0, level="WARNING")
        with patch.object(health, "cfg") as cfg_mock, \
             patch.object(health, "get_session") as sess_mock, \
             patch.object(health.time, "monotonic", return_value=1000.0), \
             patch.object(health.halt, "is_halted", return_value=False):
            cfg_mock.get_section.side_effect = lambda s: {
                "trading": {"max_daily_loss": 0},
                "runtime": {"error_rate_threshold": threshold,
                            "error_rate_warning_threshold": warn_threshold,
                            "error_rate_window_seconds": 900},
            }.get(s, {})
            ctx = MagicMock()
            ctx.__enter__.return_value.scalar.return_value = 0
            sess_mock.return_value = ctx
            return health.check_anomalies(self._risk())

    def test_flags_when_over_threshold(self):
        items = self._run(threshold=10, count=12)
        keys = {i["key"] for i in items}
        assert "error_rate_high" in keys

    def test_warning_flood_is_detected(self):
        """2026-09-02 の実測比率の再現。

        認証断のあいだ15秒毎の建玉照合が WARNING を出し続け、15分あたり約60件に
        達していた。一方 ERROR は約3.5件で閾値10に届かず、**497回失敗しているのに
        「エラーは少ない」と判定**されていた。WARNING を別枠で数えて検知する。
        """
        items = self._run(threshold=10, count=3, warn_threshold=50, warn_count=60)
        keys = {i["key"] for i in items}
        assert "error_rate_high" not in keys, "ERRORは閾値未満のまま（実測どおり）"
        assert "warning_rate_high" in keys

    def test_normal_warning_volume_is_ignored(self):
        """通常運転で出る程度の警告では発報しない（誤検知させない）"""
        items = self._run(threshold=10, count=1, warn_threshold=50, warn_count=10)
        assert "warning_rate_high" not in {i["key"] for i in items}

    def test_warning_check_disabled_when_zero(self):
        items = self._run(threshold=10, count=0, warn_threshold=0, warn_count=200)
        assert "warning_rate_high" not in {i["key"] for i in items}

    def test_error_alert_takes_precedence(self):
        """ERROR多発時は警告の重複通知を出さない（同じ障害を2通で騒がない）"""
        items = self._run(threshold=10, count=12, warn_threshold=50, warn_count=60)
        keys = {i["key"] for i in items}
        assert "error_rate_high" in keys
        assert "warning_rate_high" not in keys

    def test_silent_when_under_threshold(self):
        items = self._run(threshold=10, count=3)
        assert "error_rate_high" not in {i["key"] for i in items}

    def test_disabled_when_threshold_zero(self):
        items = self._run(threshold=0, count=100)
        assert "error_rate_high" not in {i["key"] for i in items}


class TestHeartbeatMessage:
    def _snap(self, can_order=True, block="", unresolved=0):
        return {"can_place_order": can_order, "block_reason": block,
                "unresolved_orders": unresolved}

    def test_reports_healthy_state(self):
        msg = heartbeat.build_message("live", self._snap(), position_count=3)
        assert "発注: 可" in msg
        assert "認証: 有効" in msg
        assert "3件" in msg

    def test_reports_auth_expired(self):
        broker_auth.mark_expired("401")
        msg = heartbeat.build_message("live", self._snap(False, "認証切れ"), 1)
        assert "切れ（要ログイン）" in msg
        assert "発注: 不可" in msg

    def test_includes_block_reason(self):
        msg = heartbeat.build_message("live", self._snap(False, "kill switch ON"), 0)
        assert "kill switch ON" in msg

    def test_send_swallows_alert_failure(self):
        """通知に失敗しても例外を投げない（運用を止めない）"""
        with patch.object(heartbeat, "alert", side_effect=RuntimeError("discord down")):
            heartbeat.send("live", self._snap(), 0)  # 例外が出なければ成功


class TestHeartbeatScheduling:
    def test_registered_before_market_open(self):
        import src.core.scheduler as sched_mod
        sched = sched_mod.TradingScheduler()
        sched.register("heartbeat", MagicMock())
        calls = {}
        with patch.object(sched._scheduler, "add_job") as add_job, \
             patch.object(sched._scheduler, "start"):
            sched.start()
            for c in add_job.call_args_list:
                calls[c.kwargs["id"]] = (c.kwargs.get("hour"), c.kwargs.get("minute"),
                                         c.kwargs.get("day_of_week"))
        assert "heartbeat" in calls
        hour, minute, dow = calls["heartbeat"]
        assert dow == "mon-fri"
        assert hour * 60 + minute < 9 * 60, "場が開く9:00より前に送る"
