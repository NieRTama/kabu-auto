"""
WebSocket再接続ループのログ間引き（KabuClient._ws_log）のテスト

kabuステーション未起動時・場外時間帯は接続と切断を延々繰り返すため、毎回INFOを
記録するとログが埋まり本物の異常が埋もれる（2026-08-28 実際に発生）。
最初の数回だけ記録し、以後は5分に1回へ落とす。
"""
from unittest.mock import MagicMock, patch

import src.api.kabu_client as mod


def _client():
    with patch.object(mod, "cfg") as cfg_mock:
        cfg_mock.get_section.return_value = {"base_url": "http://localhost:18080/kabusapi"}
        return mod.KabuClient()


class TestWsLogThrottle:
    def test_first_n_attempts_are_logged(self):
        c = _client()
        with patch.object(mod, "logger") as log_mock:
            for _ in range(mod.KabuClient._WS_LOG_FIRST_N):
                c._ws_log("接続中")
        assert log_mock.info.call_count == mod.KabuClient._WS_LOG_FIRST_N

    def test_subsequent_attempts_are_suppressed(self):
        c = _client()
        with patch.object(mod, "logger") as log_mock, \
             patch.object(mod.time, "monotonic", return_value=1000.0):
            for _ in range(50):
                c._ws_log("接続中")
        # 最初のN回だけ。以降は同一時刻なので間引かれる
        assert log_mock.info.call_count == mod.KabuClient._WS_LOG_FIRST_N

    def test_logs_again_after_interval(self):
        c = _client()
        now = {"t": 0.0}
        with patch.object(mod, "logger") as log_mock, \
             patch.object(mod.time, "monotonic", side_effect=lambda: now["t"]):
            for _ in range(10):
                c._ws_log("接続中")
            first = log_mock.info.call_count
            now["t"] += mod.KabuClient._WS_LOG_INTERVAL_SEC + 1
            c._ws_log("接続中")
            assert log_mock.info.call_count == first + 1, "間隔経過後は再び1回記録する"

    def test_reset_on_connection_established(self):
        """接続確立でカウンタが戻り、次に切れたときは再び数回記録される"""
        c = _client()
        with patch.object(mod, "logger"), patch.object(mod.time, "monotonic", return_value=1000.0):
            for _ in range(50):
                c._ws_log("接続中")
        with patch.object(mod, "logger"):
            c._on_ws_open(MagicMock())
        assert c._ws_attempts == 0
        with patch.object(mod, "logger") as log_mock, \
             patch.object(mod.time, "monotonic", return_value=1000.0):
            for _ in range(mod.KabuClient._WS_LOG_FIRST_N):
                c._ws_log("接続中")
        assert log_mock.info.call_count == mod.KabuClient._WS_LOG_FIRST_N
