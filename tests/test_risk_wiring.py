"""
リスク管理の結線・スケジューラ順序の検証テスト

レビューで発見された下記2点の再発防止:
  - signal_scan が data_update より前の時刻に登録され、前日終値で
    シグナルを生成してしまっていた問題
  - validate_buy() / reset_daily_counters() が実装されているのに
    main.py の実行パスから一度も呼ばれていなかった問題
"""
import json
from unittest.mock import MagicMock, patch

import pytest

import src.core.scheduler as scheduler_mod
import src.core.watchlist as watchlist_mod


# ─── スケジューラのジョブ登録順序 ────────────────────────────────────────


class TestSchedulerJobOrdering:
    def _start_and_capture(self):
        """TradingScheduler.start() を実行し、add_job呼び出し引数を {job_id: (hour, minute, day_of_week)} で返す"""
        sched = scheduler_mod.TradingScheduler()
        for name in (
            "risk_reset", "token_refresh", "data_update", "db_backup",
            "ml_retrain", "stop_loss_check", "signal_scan", "morning_execution",
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

    def test_risk_reset_job_registered(self):
        """risk_reset ジョブが登録され、平日朝（取引開始前）に実行される"""
        calls = self._start_and_capture()
        assert "risk_reset" in calls, "risk_reset ジョブが登録されていない"
        hour, minute, dow = calls["risk_reset"]
        assert dow == "mon-fri"
        assert hour < 9, "取引開始(9:00)より前にリセットされるべき"

    def test_signal_scan_runs_after_data_update(self):
        """signal_scan は data_update より後の時刻に実行される（前日終値バグの再発防止）"""
        calls = self._start_and_capture()
        du_hour, du_minute, _ = calls["data_update"]
        ss_hour, ss_minute, _ = calls["signal_scan"]
        du_total = du_hour * 60 + du_minute
        ss_total = ss_hour * 60 + ss_minute
        assert ss_total > du_total, (
            f"signal_scan({ss_hour}:{ss_minute:02d}) は "
            f"data_update({du_hour}:{du_minute:02d}) より後でなければならない"
        )


# ─── main.py のリスク結線（ソース検証） ──────────────────────────────────
# main.py は uvicorn 等の重い依存をトップレベルでimportするため、
# 他のテスト（test_live_mode_guard.py）と同様にソーステキスト検証で行う。


class TestMainPyRiskWiring:
    @pytest.fixture(autouse=True)
    def _source(self):
        with open("main.py", encoding="utf-8") as f:
            self.src = f.read()

    def test_reset_daily_counters_registered_with_scheduler(self):
        assert 'scheduler.register("risk_reset", risk.reset_daily_counters)' in self.src, (
            "reset_daily_counters() がスケジューラに登録されていない"
        )

    def test_validate_buy_used_in_signal_scan_and_morning_execution(self):
        assert self.src.count("risk.validate_buy(") >= 2, (
            "validate_buy() が signal_scan / morning_execution の両方から呼ばれていない"
        )

    def test_sector_passed_to_order_manager_buy(self):
        assert "order_mgr.buy(" in self.src
        assert "sector=sector" in self.src, (
            "buy() にセクター情報が渡されていない（セクター集中チェックが機能しない）"
        )


# ─── watchlist のセクター管理 ────────────────────────────────────────────


class TestWatchlistSector:
    def _isolated_path(self, tmp_path):
        path = tmp_path / "watchlist.json"
        watchlist_mod.load(str(path))
        return path

    def test_new_entry_has_empty_sector(self, tmp_path):
        self._isolated_path(tmp_path)
        watchlist_mod.add("7203", "トヨタ自動車")
        assert watchlist_mod.get_sectors() == {"7203": ""}

    def test_update_sector_sets_value(self, tmp_path):
        self._isolated_path(tmp_path)
        watchlist_mod.add("7203", "トヨタ自動車")
        watchlist_mod.update_sector("7203", "Consumer Cyclical")
        assert watchlist_mod.get_sectors() == {"7203": "Consumer Cyclical"}

    def test_update_sector_with_empty_string_is_noop(self, tmp_path):
        """セクター取得失敗時（空文字）は既存値を上書きしない"""
        self._isolated_path(tmp_path)
        watchlist_mod.add("7203", "トヨタ自動車")
        watchlist_mod.update_sector("7203", "Consumer Cyclical")
        watchlist_mod.update_sector("7203", "")
        assert watchlist_mod.get_sectors() == {"7203": "Consumer Cyclical"}

    def test_persisted_to_disk(self, tmp_path):
        path = self._isolated_path(tmp_path)
        watchlist_mod.add("7203", "トヨタ自動車")
        watchlist_mod.update_sector("7203", "Consumer Cyclical")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["sector"] == "Consumer Cyclical"
