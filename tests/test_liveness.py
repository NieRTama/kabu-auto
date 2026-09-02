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

    def test_not_silent_at_first_observation_before_any_success(self):
        """未取得のまま開場を初観測した時点では検知しない（起動直後の即発報を避ける）"""
        assert liveness.is_silent(now=99999.0, market_open=True, threshold_seconds=900) is False

    def test_silent_after_threshold_even_without_any_success(self):
        """一度も成功していなくても、開場から閾値を超えたら検知する。

        以前は `_last_success is None` で無条件に False を返しており、
        **板取得が最初から最後まで失敗するプロセスでは監視が永久に無効**だった。
        """
        liveness.is_silent(now=0.0, market_open=True, threshold_seconds=900)  # 開場観測
        assert liveness.is_silent(now=899.0, market_open=True, threshold_seconds=900) is False
        assert liveness.is_silent(now=900.0, market_open=True, threshold_seconds=900) is True

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
        with patch.object(mod.requests, "request", return_value=resp), \
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
        with patch.object(mod.requests, "request", return_value=resp):
            with pytest.raises(RuntimeError):
                client.get_board("7203")
        assert liveness.seconds_since_alive(now=100.0) is None


class TestNeverConnectedProcess:
    """引け後に起動し、翌日一度も取得に成功しないプロセス（2026-09-02 の事故の再現）。

    9/1 19:07（引け後）に起動 → 翌9/2 8:30のトークン更新は成功 → 9:00にセッション断
    → 以後 get_board が全て401。板取得の成功が1度も無いため `_last_success` は None のまま
    で、`is_silent()` は終日 False を返し続けた。497回の401が出ても無言だった。
    """

    def _evening_start_then_next_morning(self):
        """引け後の起動 → 夜間 → 翌朝の開場、までを再現する（成功は一度も無い）。"""
        # 19:07 起動。health_check は23:00まで15分毎に走り、閉場を観測する
        for t in range(0, 14000, 900):
            liveness.is_silent(now=t, market_open=False, threshold_seconds=900)
            liveness.mark_closed(now=t)
        # 23:00〜翌8:00 は health_check が走らない（実運用と同じ）

    def test_silence_is_detected_next_morning(self):
        self._evening_start_then_next_morning()
        open_at = 50000.0
        # 9:00 開場を初観測。この時点ではまだ発報しない
        assert liveness.is_silent(now=open_at, market_open=True,
                                 threshold_seconds=900) is False
        # 9:15 閾値超過。ここで気づけていれば当日中に手当てできた
        assert liveness.is_silent(now=open_at + 900, market_open=True,
                                 threshold_seconds=900) is True

    def test_alert_says_never_succeeded_not_zero_minutes(self):
        """通知文が「0分間成功していません」にならないこと。

        health は `seconds_since_alive() or 0` としていたため、未成功のまま
        発報すると「0分間取得が成功していません」という意味の通らない文面になり、
        深刻さが伝わらなかった。未成功は未成功として書く。
        """
        from src.core import health
        risk = MagicMock()
        risk.daily_loss_limit.return_value = 0
        risk.current_total_drawdown.return_value = 0
        risk.unrealized_pnl.return_value = 0
        risk.unpriced_symbols.return_value = []

        self._evening_start_then_next_morning()
        open_at = 50000.0
        liveness.is_silent(now=open_at, market_open=True, threshold_seconds=900)
        with patch.object(health, "cfg") as cfg_mock, \
             patch.object(health, "get_session") as sess_mock, \
             patch.object(health.time, "monotonic", return_value=open_at + 900), \
             patch.object(health.halt, "is_halted", return_value=False), \
             patch.object(health.TradingScheduler, "is_market_open", return_value=True):
            cfg_mock.get_section.side_effect = lambda s: {
                "trading": {},
                "runtime": {"error_rate_threshold": 0, "error_rate_warning_threshold": 0,
                            "liveness_silence_seconds": 900},
            }.get(s, {})
            ctx = MagicMock()
            ctx.__enter__.return_value.scalar.return_value = 0
            sess_mock.return_value = ctx
            items = health.check_anomalies(risk)
        msg = next(i["message"] for i in items if i["key"] == "liveness_silent")
        assert "一度も" in msg
        assert "0分間" not in msg

    def test_recovery_clears_the_alert(self):
        """認証が戻って取得に成功したら発報は止まる"""
        self._evening_start_then_next_morning()
        open_at = 50000.0
        liveness.is_silent(now=open_at, market_open=True, threshold_seconds=900)
        assert liveness.is_silent(now=open_at + 900, market_open=True,
                                 threshold_seconds=900) is True
        liveness.mark_alive(now=open_at + 1000)
        assert liveness.is_silent(now=open_at + 1100, market_open=True,
                                 threshold_seconds=900) is False


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


class TestLunchBreakFalsePositive:
    """昼休みを「途絶」と数えない（2026-08-31 に実際に発生した誤検知の回帰防止）。

    前場最終取得(11:25) → 昼休み(11:30-12:30) → 後場開始(12:30) の時点で
    「65分間データ取得なし」と通知された。65分 = 昼休み60分 + ジョブ間隔5分 で、
    実際にはシステムは正常に動いていた。
    """

    def test_closed_period_does_not_accumulate(self):
        liveness.mark_alive(now=0.0)          # 11:25 前場の最終取得
        liveness.mark_closed(now=300.0)       # 11:30 昼休み入り（health_checkが呼ぶ）
        liveness.mark_closed(now=3600.0)      # 12:00 昼休み中
        liveness.mark_closed(now=3900.0)      # 12:29 昼休み中
        # 12:30 後場開始。閉場中は基準が進んでいるので途絶とみなさない
        assert liveness.is_silent(now=3901.0, market_open=True, threshold_seconds=900) is False

    def test_real_silence_after_reopen_is_still_detected(self):
        """昼休み明けに本当に止まっていれば従来どおり検知する（検知力を落とさない）"""
        liveness.mark_alive(now=0.0)
        liveness.mark_closed(now=3600.0)      # 昼休み中に基準が進む
        # 後場開始後、閾値を超えて取得できていない
        assert liveness.is_silent(now=3600.0 + 1000, market_open=True,
                                  threshold_seconds=900) is True

    def test_mark_closed_before_first_success_is_noop(self):
        """1度も成功していない状態で mark_closed しても「生存扱い」にしない"""
        liveness.mark_closed(now=100.0)
        assert liveness.seconds_since_alive(now=200.0) is None
        assert liveness.is_silent(now=99999.0, market_open=True, threshold_seconds=900) is False


class TestHealthCallsMarkClosed:
    def test_market_closed_advances_baseline(self):
        """health.check_anomalies が閉場中に mark_closed を呼ぶこと（結線の検証）"""
        from unittest.mock import MagicMock, patch
        from src.core import health
        risk = MagicMock()
        risk.daily_loss_limit.return_value = 0
        risk.current_total_drawdown.return_value = 0
        risk.unrealized_pnl.return_value = 0
        risk.unpriced_symbols.return_value = []

        liveness.mark_alive(now=0.0)
        with patch.object(health, "cfg") as cfg_mock, \
             patch.object(health, "get_session") as sess_mock, \
             patch.object(health.time, "monotonic", return_value=5000.0), \
             patch.object(health.halt, "is_halted", return_value=False), \
             patch.object(health.TradingScheduler, "is_market_open", return_value=False):
            cfg_mock.get_section.side_effect = lambda s: {
                "trading": {}, "runtime": {"error_rate_threshold": 0,
                                           "liveness_silence_seconds": 900},
            }.get(s, {})
            ctx = MagicMock()
            ctx.__enter__.return_value.scalar.return_value = 0
            sess_mock.return_value = ctx
            health.check_anomalies(risk)
        # 閉場中の呼び出しで基準が 5000 まで進んでいる
        assert liveness.seconds_since_alive(now=5000.0) == 0.0


class TestOvernightFalsePositive:
    """夜間をまたいだ誤検知の防止（2026-09-01 に「1050分間取得なし」で発報）。

    mark_closed() は health_check（平日8:00-23:00）からしか呼ばれないため、
    23:00〜翌8:00の9時間は基準が進まず、翌朝の開場時に積み上がった経過時間で
    誤検知していた。呼び出し側に依存しきらないよう、開場直後は判定を猶予する。
    """

    def _overnight(self):
        """15:30引け → 23:00までhealth_check → 夜間は呼ばれない、を再現。

        health_check は閉場中も is_silent(market_open=False) を通るため、
        そこで閉場が観測される（実運用と同じ順序で再現する）。
        """
        liveness.mark_alive(now=0)
        liveness.is_silent(now=100, market_open=True, threshold_seconds=900)
        for t in range(3600, 27000, 900):
            liveness.is_silent(now=t, market_open=False, threshold_seconds=900)
            liveness.mark_closed(now=t)

    def test_no_alert_right_after_market_opens(self):
        self._overnight()
        assert liveness.is_silent(now=63000, market_open=True,
                                  threshold_seconds=900) is False

    def test_no_alert_during_grace_period(self):
        self._overnight()
        liveness.is_silent(now=63000, market_open=True, threshold_seconds=900)
        # 開場から8分（グレース15分の途中）
        assert liveness.is_silent(now=63480, market_open=True,
                                  threshold_seconds=900) is False

    def test_detects_real_silence_after_grace(self):
        """グレース明け後に本当に取得できていなければ検知する（検知力を落とさない）"""
        self._overnight()
        liveness.is_silent(now=63000, market_open=True, threshold_seconds=900)
        assert liveness.is_silent(now=64000, market_open=True,
                                  threshold_seconds=900) is True

    def test_grace_resets_after_market_closes(self):
        """閉場を挟むとグレースが張り直される（昼休み明けにも効く）"""
        liveness.mark_alive(now=0)
        liveness.is_silent(now=100, market_open=True, threshold_seconds=900)   # 開場
        liveness.is_silent(now=200, market_open=False, threshold_seconds=900)  # 閉場
        # 再開場の直後は、前回成功から長時間空いていても検知しない
        assert liveness.is_silent(now=100000, market_open=True,
                                  threshold_seconds=900) is False

    def test_normal_intraday_detection_still_works(self):
        """通常の場中（開場から十分経過後）は従来どおり検知する"""
        liveness.mark_alive(now=0)
        liveness.is_silent(now=0, market_open=True, threshold_seconds=900)  # 開場記録
        liveness.mark_alive(now=1000)      # 途中で取得成功
        assert liveness.is_silent(now=1500, market_open=True,
                                  threshold_seconds=900) is False
        assert liveness.is_silent(now=2000, market_open=True,
                                  threshold_seconds=900) is True

    def test_health_notifies_close_transition_to_is_silent(self):
        """health_check が閉場を is_silent にも伝えること（結線の回帰防止）。

        mark_closed() だけでは health_check の稼働時間帯(8:00-23:00)しか
        カバーできず、夜間分が積み上がる。閉場の遷移を is_silent() にも
        渡しておかないと、翌朝の開場でグレースが張られない。
        """
        from unittest.mock import MagicMock, patch
        from src.core import health
        risk = MagicMock()
        risk.daily_loss_limit.return_value = 0
        risk.current_total_drawdown.return_value = 0
        risk.unrealized_pnl.return_value = 0
        risk.unpriced_symbols.return_value = []

        liveness.mark_alive(now=0.0)
        with patch.object(health, "cfg") as cfg_mock, \
             patch.object(health, "get_session") as sess_mock, \
             patch.object(health.time, "monotonic", return_value=1000.0), \
             patch.object(health.halt, "is_halted", return_value=False), \
             patch.object(health.TradingScheduler, "is_market_open", return_value=False), \
             patch.object(health.liveness, "is_silent", wraps=liveness.is_silent) as spy:
            cfg_mock.get_section.side_effect = lambda s: {
                "trading": {}, "runtime": {"error_rate_threshold": 0,
                                           "liveness_silence_seconds": 900},
            }.get(s, {})
            ctx = MagicMock()
            ctx.__enter__.return_value.scalar.return_value = 0
            sess_mock.return_value = ctx
            health.check_anomalies(risk)
        assert spy.called, "閉場中に is_silent が呼ばれていない（グレースが張られない）"
        assert spy.call_args.kwargs["market_open"] is False
