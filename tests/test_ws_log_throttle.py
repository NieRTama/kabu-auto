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
        assert c._ws_log_state == {}
        with patch.object(mod, "logger") as log_mock, \
             patch.object(mod.time, "monotonic", return_value=1000.0):
            for _ in range(mod.KabuClient._WS_LOG_FIRST_N):
                c._ws_log("接続中")
        assert log_mock.info.call_count == mod.KabuClient._WS_LOG_FIRST_N


class TestErrorLogThrottle:
    """WARNING も間引くこと（2026-09-04 の実害の回帰防止）。

    kabuステーションが47分停止した際、`_on_ws_error` が間引き無しで WARNING を
    **522件**出し、当日の警告614件の大半を占めた。前日入れた警告率監視
    （15分50件）を単独で振り切る量で、「未知の障害を捉える」はずの監視が
    既知の接続断で埋まってしまう。
    """

    def test_ws_error_is_throttled(self):
        c = _client()
        with patch.object(mod, "logger") as log_mock, \
             patch.object(mod.time, "monotonic", return_value=1000.0):
            for _ in range(500):
                c._on_ws_error(MagicMock(), "[WinError 10061] 接続拒否")
        assert log_mock.warning.call_count == mod.KabuClient._WS_LOG_FIRST_N

    def test_ws_close_is_throttled(self):
        c = _client()
        with patch.object(mod, "logger") as log_mock, \
             patch.object(mod.time, "monotonic", return_value=1000.0):
            for _ in range(500):
                c._on_ws_close(MagicMock(), None, None)
        assert log_mock.info.call_count == mod.KabuClient._WS_LOG_FIRST_N

    def test_kinds_are_counted_independently(self):
        """接続試行・エラー・切断は1回の失敗で全部起きる。
        状態を共有すると「最初の数回」が種類の数だけ目減りしてしまう。
        """
        c = _client()
        n = mod.KabuClient._WS_LOG_FIRST_N
        with patch.object(mod, "logger") as log_mock, \
             patch.object(mod.time, "monotonic", return_value=1000.0):
            for _ in range(n):
                c._ws_log("接続中")
                c._on_ws_error(MagicMock(), "err")
                c._on_ws_close(MagicMock(), None, None)
        assert log_mock.warning.call_count == n, "エラーは独立して数える"
        assert log_mock.info.call_count == n * 2, "接続中と切断がそれぞれ n 回"

    def test_error_logged_again_after_interval(self):
        """間引いても、間隔経過後は再び記録する（完全に黙らせない）"""
        c = _client()
        now = {"t": 0.0}
        with patch.object(mod, "logger") as log_mock, \
             patch.object(mod.time, "monotonic", side_effect=lambda: now["t"]):
            for _ in range(100):
                c._on_ws_error(MagicMock(), "err")
            first = log_mock.warning.call_count
            now["t"] += mod.KabuClient._WS_LOG_INTERVAL_SEC + 1
            c._on_ws_error(MagicMock(), "err")
            assert log_mock.warning.call_count == first + 1
