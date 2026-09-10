"""シグナルのデータ基準日記録と、鮮度による新規候補の除外

背景: 生成日時しか持たないため、古い足から作られたシグナルを区別できなかった。
また確定していない足でも新規候補として保存されていた（F01）。
"""
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

from src.core import clock
from src.core import config as cfg
from src.data import database as db
from src.data.bar_status import BarStatus
from src.data.database import Signal, get_session
from src.services import trading
from src.strategy.signal import Signal as TradeSignal


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


class TestSignalDataAsOf:
    def test_data_as_of_is_saved_separately_from_generated_at(self, isolated_db):
        sig = TradeSignal(symbol="7203", action="BUY", rule_score=0.3,
                          ml_score=0.1, combined_score=0.4)
        trading._save_signal(sig, data_as_of=date(2026, 9, 10))

        with get_session() as session:
            row = session.scalar(select(Signal))
        assert row.data_as_of == date(2026, 9, 10)
        # 生成日時は保存時刻（clock.now）で、データ基準日とは独立に決まる
        assert row.generated_at is not None
        assert row.generated_at.date() == clock.today()

    def test_data_as_of_is_nullable_for_legacy_rows(self, isolated_db):
        sig = TradeSignal(symbol="7203", action="HOLD", rule_score=0.0,
                          ml_score=0.0, combined_score=0.0)
        trading._save_signal(sig)

        with get_session() as session:
            row = session.scalar(select(Signal))
        assert row.data_as_of is None


def _status(symbol: str, state: str) -> BarStatus:
    return BarStatus(
        symbol=symbol,
        last_bar_session=date(2026, 9, 10) if state == "fresh" else date(2026, 9, 1),
        observed_at=datetime(2026, 9, 10, 16, 20),
        is_final=state in ("fresh", "stale"),
        state=state,
    )


class TestFreshnessGate:
    def test_stale_symbol_is_excluded_from_new_candidates(self, isolated_db):
        """確定していない/古い足の銘柄は新規候補にしない"""
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        svc._bar_states = {"7203": _status("7203", "stale")}

        assert svc._is_fresh_for_new_candidate("7203") is False

    def test_fresh_symbol_is_allowed(self, isolated_db):
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        svc._bar_states = {"7203": _status("7203", "fresh")}

        with patch.object(trading.clock, "now",
                          return_value=datetime(2026, 9, 10, 16, 20)):
            assert svc._is_fresh_for_new_candidate("7203") is True

    def test_unknown_symbol_is_not_fresh(self, isolated_db):
        """更新結果が無い銘柄は fresh と判定しない（安全側）"""
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        svc._bar_states = {}

        assert svc._is_fresh_for_new_candidate("7203") is False

    def test_data_update_records_states(self, isolated_db):
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        with patch.object(trading, "update_symbol",
                          return_value=_status("7203", "fresh")), \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]):
            states = svc.data_update()

        assert states["7203"].state == "fresh"
        assert svc._bar_states["7203"].state == "fresh"

    def test_failed_update_is_recorded_as_missing(self, isolated_db):
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        with patch.object(trading, "update_symbol",
                          side_effect=RuntimeError("network down")), \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]):
            states = svc.data_update()

        assert states["7203"].state == "missing"
        assert svc._is_fresh_for_new_candidate("7203") is False

    def test_fresh_flag_is_revalidated_against_current_time(self, isolated_db):
        """_bar_statesが前営業日分の"fresh"のまま残っていても、現在時刻の基準
        セッションと一致しなければ新規候補にしない（部分更新中の素通し防止）"""
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        # 前営業日(2026-09-09)分の"fresh"が残っている状態を模す
        stale_but_marked_fresh = BarStatus(
            symbol="7203",
            last_bar_session=date(2026, 9, 9),
            observed_at=datetime(2026, 9, 9, 16, 0),
            is_final=True,
            state="fresh",
        )
        svc._bar_states = {"7203": stale_but_marked_fresh}

        with patch.object(trading.clock, "now",
                          return_value=datetime(2026, 9, 10, 16, 20)):
            assert svc._is_fresh_for_new_candidate("7203") is False


class TestDataBatchId:
    def test_data_update_issues_batch_id(self, isolated_db):
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        with patch.object(trading, "update_symbol",
                          return_value=_status("7203", "fresh")), \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]):
            svc.data_update()

        assert svc._data_batch_id is not None
        assert len(svc._data_batch_id) > 0

    def test_batch_id_changes_between_updates(self, isolated_db):
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        with patch.object(trading, "update_symbol",
                          return_value=_status("7203", "fresh")), \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]):
            svc.data_update()
            first = svc._data_batch_id
            svc.data_update()
            second = svc._data_batch_id

        assert first != second

    def test_batch_id_is_none_before_first_update(self, isolated_db):
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        assert svc._data_batch_id is None


class TestZeroFreshAlert:
    def test_alerts_when_all_symbols_excluded(self, isolated_db):
        """全銘柄が鮮度不足で除外されたらWARNINGアラートを出す"""
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        svc._bar_states = {}  # 空 = 全銘柄が鮮度不足

        with patch.object(trading.watchlist_store, "get_codes",
                          return_value=["7203", "9984"]), \
             patch.object(trading.TradingScheduler, "is_maintenance_window",
                          return_value=False), \
             patch.object(trading.clock, "today", return_value=date(2026, 9, 10)), \
             patch.object(trading, "alert") as mock_alert:
            svc.signal_scan()

        assert mock_alert.called
        args, kwargs = mock_alert.call_args
        assert kwargs.get("level") == trading.LEVEL_WARNING or \
               (len(args) >= 3 and args[2] == trading.LEVEL_WARNING)


class TestGateWiring:
    """鮮度ゲートの結線をソース検証で固定する（signal_scanは呼ぶ、stop_loss_checkは呼ばない）"""

    def _source(self, name):
        import inspect
        return inspect.getsource(getattr(trading.TradingServices, name))

    def test_signal_scan_calls_freshness_gate(self):
        assert "_is_fresh_for_new_candidate" in self._source("signal_scan")

    def test_stop_loss_check_does_not_call_freshness_gate(self):
        """保有保護の退出は鮮度に関わらず実行する。ゲートを絶対に持ち込まない"""
        assert "_is_fresh_for_new_candidate" not in self._source("stop_loss_check")
        assert "_bar_states" not in self._source("stop_loss_check")
