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
            "reconcile_orders",
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


# ─── リスク結線（ソース検証） ────────────────────────────────────────────
# main.py / services は uvicorn 等の重い依存をトップレベルでimportするため、
# 他のテスト（test_live_mode_guard.py）と同様にソーステキスト検証で行う。
# 取引ロジックは src/services/trading.py に切り出したため、結線はそちらを検証する。


class TestRiskWiring:
    @pytest.fixture(autouse=True)
    def _source(self):
        with open("main.py", encoding="utf-8") as f:
            self.main_src = f.read()
        with open("src/services/trading.py", encoding="utf-8") as f:
            self.svc_src = f.read()

    def test_reset_daily_counters_registered_with_scheduler(self):
        assert 'scheduler.register("risk_reset", risk.reset_daily_counters)' in self.main_src, (
            "reset_daily_counters() がスケジューラに登録されていない"
        )

    def test_validate_buy_used_in_signal_scan_and_morning_execution(self):
        assert self.svc_src.count("validate_buy(") >= 2, (
            "validate_buy() が signal_scan / morning_execution の両方から呼ばれていない"
        )

    def test_sector_passed_to_order_manager_buy(self):
        assert "order_mgr.buy(" in self.svc_src
        assert "sector=sector" in self.svc_src, (
            "buy() にセクター情報が渡されていない（セクター集中チェックが機能しない）"
        )


# ─── watchlist のセクター管理 ────────────────────────────────────────────


class TestWatchlistSector:
    def _isolated_path(self, tmp_path):
        path = tmp_path / "watchlists.json"
        # legacy_path も tmp_path 配下の存在しないパスにし、実行カレントディレクトリの
        # 本物の watchlist.json を誤って取り込まないようにする
        watchlist_mod.load(str(path), legacy_path=str(tmp_path / "no_legacy.json"))
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
        active = data["lists"][data["active"]]
        assert active[0]["sector"] == "Consumer Cyclical"


# ─── 複数ウォッチリスト管理 ────────────────────────────────────────────


class TestWatchlistMultiList:
    def _isolated_path(self, tmp_path):
        path = tmp_path / "watchlists.json"
        watchlist_mod.load(str(path), legacy_path=str(tmp_path / "no_legacy.json"))
        return path

    def test_starts_with_single_default_list(self, tmp_path):
        self._isolated_path(tmp_path)
        assert watchlist_mod.get_list_names() == [watchlist_mod.DEFAULT_LIST_NAME]
        assert watchlist_mod.get_active_list_name() == watchlist_mod.DEFAULT_LIST_NAME

    def test_create_list_switches_active(self, tmp_path):
        self._isolated_path(tmp_path)
        watchlist_mod.create_list("高配当株")
        assert "高配当株" in watchlist_mod.get_list_names()
        assert watchlist_mod.get_active_list_name() == "高配当株"

    def test_create_duplicate_name_raises(self, tmp_path):
        self._isolated_path(tmp_path)
        watchlist_mod.create_list("高配当株")
        with pytest.raises(ValueError):
            watchlist_mod.create_list("高配当株")

    def test_lists_are_independent(self, tmp_path):
        """別リストへ切り替えると銘柄も完全に分離されている"""
        self._isolated_path(tmp_path)
        watchlist_mod.add("7203", "トヨタ自動車")
        watchlist_mod.create_list("高配当株")
        watchlist_mod.add("8306", "三菱UFJ")
        assert watchlist_mod.get_codes() == ["8306"]
        watchlist_mod.switch_active(watchlist_mod.DEFAULT_LIST_NAME)
        assert watchlist_mod.get_codes() == ["7203"]

    def test_delete_active_list_falls_back(self, tmp_path):
        self._isolated_path(tmp_path)
        watchlist_mod.create_list("高配当株")  # アクティブが高配当株になる
        watchlist_mod.delete_list("高配当株")
        assert watchlist_mod.get_active_list_name() == watchlist_mod.DEFAULT_LIST_NAME
        assert "高配当株" not in watchlist_mod.get_list_names()

    def test_cannot_delete_last_list(self, tmp_path):
        self._isolated_path(tmp_path)
        with pytest.raises(ValueError):
            watchlist_mod.delete_list(watchlist_mod.DEFAULT_LIST_NAME)

    def test_rename_list_updates_active_pointer(self, tmp_path):
        self._isolated_path(tmp_path)
        watchlist_mod.rename_list(watchlist_mod.DEFAULT_LIST_NAME, "コア銘柄")
        assert watchlist_mod.get_active_list_name() == "コア銘柄"
        assert watchlist_mod.DEFAULT_LIST_NAME not in watchlist_mod.get_list_names()

    def test_export_returns_active_entries(self, tmp_path):
        self._isolated_path(tmp_path)
        watchlist_mod.add("7203", "トヨタ自動車")
        exported = watchlist_mod.export_list()
        assert exported["name"] == watchlist_mod.DEFAULT_LIST_NAME
        assert exported["entries"][0]["code"] == "7203"

    def test_import_creates_new_list_and_switches_active(self, tmp_path):
        self._isolated_path(tmp_path)
        names = watchlist_mod.import_list("輸入リスト", [{"code": "9984", "name": "ソフトバンクG"}])
        assert "輸入リスト" in names
        assert watchlist_mod.get_active_list_name() == "輸入リスト"
        assert watchlist_mod.get_codes() == ["9984"]

    def test_import_existing_name_without_overwrite_raises(self, tmp_path):
        self._isolated_path(tmp_path)
        with pytest.raises(ValueError):
            watchlist_mod.import_list(watchlist_mod.DEFAULT_LIST_NAME, [{"code": "9984"}])

    def test_import_existing_name_with_overwrite_replaces(self, tmp_path):
        self._isolated_path(tmp_path)
        watchlist_mod.add("7203", "トヨタ自動車")
        watchlist_mod.import_list(
            watchlist_mod.DEFAULT_LIST_NAME, [{"code": "9984", "name": "ソフトバンクG"}], overwrite=True
        )
        assert watchlist_mod.get_codes() == ["9984"]

    def test_import_rejects_non_list_entries(self, tmp_path):
        """entries がリストでない不正な取込は弾く（C-1/H-1 入力検証）"""
        self._isolated_path(tmp_path)
        with pytest.raises(ValueError):
            watchlist_mod.import_list("不正", {"code": "9984"})

    def test_import_rejects_non_dict_entry(self, tmp_path):
        self._isolated_path(tmp_path)
        with pytest.raises(ValueError):
            watchlist_mod.import_list("不正", ["9984", "7203"])

    def test_import_rejects_too_many_entries(self, tmp_path):
        self._isolated_path(tmp_path)
        huge = [{"code": str(1000 + i)} for i in range(watchlist_mod.MAX_ENTRIES_PER_LIST + 1)]
        with pytest.raises(ValueError):
            watchlist_mod.import_list("巨大", huge)

    def test_import_rejects_malformed_code(self, tmp_path):
        self._isolated_path(tmp_path)
        with pytest.raises(ValueError):
            watchlist_mod.import_list("不正コード", [{"code": "abc;DROP TABLE"}])

    def test_import_dedupes_and_skips_empty(self, tmp_path):
        self._isolated_path(tmp_path)
        watchlist_mod.import_list("整理", [
            {"code": "9984"}, {"code": "9984"}, {"code": ""}, {"code": "7203"},
        ])
        assert watchlist_mod.get_codes() == ["9984", "7203"]

    def test_import_rejects_unicode_code(self, tmp_path):
        """NFKC正規化後もASCII以外の文字が残るコードは弾く（M-C）"""
        self._isolated_path(tmp_path)
        with pytest.raises(ValueError):
            watchlist_mod.import_list("不正", [{"code": "株式４"}])

    def test_add_normalizes_fullwidth_code(self, tmp_path):
        """全角数字コードはNFKCで半角化されて登録される"""
        self._isolated_path(tmp_path)
        watchlist_mod.add("７２０３", "トヨタ自動車")
        assert watchlist_mod.get_codes() == ["7203"]

    def test_add_rejects_malformed_code(self, tmp_path):
        """単体追加(add)でも不正な銘柄コードを弾く（M-C）"""
        self._isolated_path(tmp_path)
        with pytest.raises(ValueError):
            watchlist_mod.add("abc;DROP TABLE")

    def test_get_all_codes_unions_all_lists_deduped(self, tmp_path):
        """get_all_codes はアクティブ以外のリストも含め重複除去して返す（MLモデル学習データ確保用）"""
        self._isolated_path(tmp_path)
        watchlist_mod.add("7203", "トヨタ自動車")
        watchlist_mod.create_list("高配当株")
        watchlist_mod.add("8306", "三菱UFJ")
        watchlist_mod.add("7203", "トヨタ自動車")  # 別リストにも同コード（重複）
        codes = watchlist_mod.get_all_codes()
        assert sorted(codes) == ["7203", "8306"]

    def test_legacy_single_list_file_migrates(self, tmp_path):
        """旧形式（フラットなリストのみ）の watchlist.json から自動移行する"""
        legacy = tmp_path / "watchlist.json"
        with open(legacy, "w", encoding="utf-8") as f:
            json.dump([{"code": "7203", "name": "トヨタ自動車", "sector": "Consumer Cyclical"}], f)
        new_path = tmp_path / "watchlists.json"
        watchlist_mod.load(str(new_path), legacy_path=str(legacy))
        assert watchlist_mod.get_codes() == ["7203"]
        assert watchlist_mod.get_active_list_name() == watchlist_mod.DEFAULT_LIST_NAME
        assert new_path.exists(), "移行後は新形式ファイルが書き出されるべき"
