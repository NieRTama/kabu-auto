"""
サイレント故障の検知（liveness）のテスト

エラー率監視（error_rate）は「壊れたらエラーが増える」故障を捉えるが、
例外を出さずに静かに止まる故障（ジョブの死・無反応化・空データの返却）は
すり抜ける。そこで正常系の動作（板取得の成功）に印をつけ、場中に一定時間
途絶えたら異常とみなす。

時刻は注入して実時間に依存させない。
"""
from unittest.mock import MagicMock, patch

import pytest

from src.core import liveness


@pytest.fixture(autouse=True)
def _reset():
    liveness.reset()
    yield
    liveness.reset()


class TestMarkAndElapsed:
    def test_none_before_any_success(self):
        assert liveness.seconds_since_alive(now=100.0) is None

    def test_elapsed_since_mark(self):
        liveness.mark_alive(now=100.0)
        assert liveness.seconds_since_alive(now=160.0) == 60.0

    def test_mark_updates_to_latest(self):
        liveness.mark_alive(now=100.0)
        liveness.mark_alive(now=500.0)
        assert liveness.seconds_since_alive(now=560.0) == 60.0


class TestIsSilent:
    def test_silent_when_market_open_and_stale(self):
        liveness.mark_alive(now=0.0)
        assert liveness.is_silent(now=1000.0, market_open=True, threshold_seconds=900) is True

    def test_not_silent_when_recent(self):
        liveness.mark_alive(now=0.0)
        assert liveness.is_silent(now=100.0, market_open=True, threshold_seconds=900) is False

    def test_never_silent_when_market_closed(self):
        """場外に動きが無いのは正常（誤検知させない）"""
        liveness.mark_alive(now=0.0)
        assert liveness.is_silent(now=99999.0, market_open=False, threshold_seconds=900) is False

    def test_not_silent_before_first_success(self):
        """起動直後の未取得状態では検知しない（別途preflightが担保）"""
        assert liveness.is_silent(now=99999.0, market_open=True, threshold_seconds=900) is False

    def test_disabled_when_threshold_zero(self):
        liveness.mark_alive(now=0.0)
        assert liveness.is_silent(now=99999.0, market_open=True, threshold_seconds=0) is False


class TestClientMarksOnSuccess:
    """get_board() の成功が liveness に記録されること（結線の検証）"""

    def test_get_board_marks_alive(self):
        import src.api.kabu_client as mod
        with patch.object(mod, "cfg") as cfg_mock:
            cfg_mock.get_section.return_value = {"base_url": "http://x/kabusapi"}
            client = mod.KabuClient()
        resp = MagicMock()
        resp.json.return_value = {"CurrentPrice": 100}
        with patch.object(mod.requests, "get", return_value=resp), \
             patch.object(mod.time, "monotonic", return_value=777.0):
            client.get_board("7203")
        assert liveness.seconds_since_alive(now=777.0) == 0.0

    def test_failed_get_board_does_not_mark(self):
        """取得失敗時は「生きている」印を付けない（故障を隠さない）"""
        import src.api.kabu_client as mod
        with patch.object(mod, "cfg") as cfg_mock:
            cfg_mock.get_section.return_value = {"base_url": "http://x/kabusapi"}
            client = mod.KabuClient()
        resp = MagicMock()
        resp.raise_for_status.side_effect = RuntimeError("401")
        with patch.object(mod.requests, "get", return_value=resp):
            with pytest.raises(RuntimeError):
                client.get_board("7203")
        assert liveness.seconds_since_alive(now=100.0) is None


class TestHealthIntegration:
    def _risk(self):
        r = MagicMock()
        r.daily_loss_limit.return_value = 0
        r.current_total_drawdown.return_value = 0
        r.unrealized_pnl.return_value = 0
        r.unpriced_symbols.return_value = []
        return r

    def _run(self, market_open: bool, stale_seconds: float):
        from src.core import health
        liveness.mark_alive(now=0.0)
        with patch.object(health, "cfg") as cfg_mock, \
             patch.object(health, "get_session") as sess_mock, \
             patch.object(health.time, "monotonic", return_value=stale_seconds), \
             patch.object(health.halt, "is_halted", return_value=False), \
             patch.object(health.TradingScheduler, "is_market_open", return_value=market_open):
            cfg_mock.get_section.side_effect = lambda s: {
                "trading": {},
                "runtime": {"error_rate_threshold": 0, "liveness_silence_seconds": 900},
            }.get(s, {})
            ctx = MagicMock()
            ctx.__enter__.return_value.scalar.return_value = 0
            sess_mock.return_value = ctx
            return {i["key"] for i in health.check_anomalies(self._risk())}

    def test_flags_silence_during_market_hours(self):
        assert "liveness_silent" in self._run(market_open=True, stale_seconds=1000.0)

    def test_silent_check_skipped_when_market_closed(self):
        assert "liveness_silent" not in self._run(market_open=False, stale_seconds=99999.0)

    def test_no_flag_when_recently_alive(self):
        assert "liveness_silent" not in self._run(market_open=True, stale_seconds=100.0)
