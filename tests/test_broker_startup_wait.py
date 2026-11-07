"""起動時の待機中と、ログインされないまま場が開く朝の通知のテスト。

2026-11-02 実例: 金曜夜の起動で kabuステーションへ接続できず、スケジューラが動かないまま
月曜 12:00 まで待機した。定時再起動も生存通知も無く、通知は金曜夜の1通だけだった。
2026-11-06 実例: 8:35 の通知のあとログインされず、09:20 まで損切りチェックが失敗し続けた。
"""
from datetime import datetime

from src.core import broker_watch as bw

MON = "2026-11-09"
SAT = "2026-11-07"
RT = {"auto_launch_broker": True}


def _at(day, hhmm):
    return datetime.fromisoformat(f"{day} {hhmm}:00")


class Fake:
    def __init__(self):
        self.restarts, self.launches, self.alerts = 0, 0, []

    def restart(self):
        self.restarts += 1
        return True, "kabuステーションを起動しました（本日1回目）"

    def launch(self):
        self.launches += 1
        return False, "kabuステーションは既に起動しています"

    def alert(self, title, message, **kw):
        self.alerts.append(title)


def _refused():
    raise ConnectionError("18080 refused")


def _tick(w, f, when, mono, *, runtime=RT):
    w.tick(now=when, monotonic=mono, runtime=runtime, refresh_token=_refused,
           restart=f.restart, launch=f.launch, alert=f.alert, pause_exists=lambda: False)


class TestStartupWaitWatcher:
    def test_weekend_is_silent_and_monday_restarts_once_and_reminds(self):
        f, w = Fake(), bw.StartupWaitWatcher(monotonic=0.0)
        _tick(w, f, _at(SAT, "09:00"), 1000.0)
        assert f.restarts == 0 and f.alerts == []
        _tick(w, f, _at(MON, "08:10"), 200_000.0)      # 再起動はまだ。再通知はする
        assert f.restarts == 0 and len(f.alerts) == 1
        _tick(w, f, _at(MON, "08:25"), 200_060.0)
        _tick(w, f, _at(MON, "08:31"), 200_420.0)
        assert f.restarts == 1, "再起動は1日1回だけ（ログイン画面を操作中に落とさない）"

    def test_reminder_repeats_every_30_minutes(self):
        f, w = Fake(), bw.StartupWaitWatcher(monotonic=0.0)
        _tick(w, f, _at(MON, "09:00"), 10_000.0)
        n = len(f.alerts)
        _tick(w, f, _at(MON, "09:10"), 10_600.0)
        assert len(f.alerts) == n
        _tick(w, f, _at(MON, "09:30"), 11_800.0)
        assert len(f.alerts) == n + 1

    def test_after_the_window_only_launches_if_missing(self):
        """8:40 以降は強制終了しない（人がパスキーで操作中かもしれない）"""
        f, w = Fake(), bw.StartupWaitWatcher(monotonic=0.0)
        _tick(w, f, _at(MON, "09:30"), 10_000.0)
        assert f.restarts == 0 and f.launches == 1

    def test_no_restart_when_auto_launch_is_off(self):
        f, w = Fake(), bw.StartupWaitWatcher(monotonic=0.0)
        _tick(w, f, _at(MON, "08:30"), 10_000.0, runtime={"auto_launch_broker": False})
        assert f.restarts == 0


class TestDailyRestart:
    def _run(self, *, refresh=_refused, runtime=RT, closed=False, pause=False):
        f = Fake()
        bw.run_daily_restart(runtime=runtime, refresh_token=refresh, restart=f.restart,
                             alert=f.alert, closed_today=closed, pause_exists=lambda: pause)
        return f

    def test_restarts_when_auth_is_gone(self):
        assert self._run().restarts == 1

    def test_keeps_app_when_auth_is_valid(self):
        assert self._run(refresh=lambda: "tok").restarts == 0

    def test_skips_on_holiday_pause_and_when_disabled(self):
        assert self._run(closed=True).restarts == 0
        assert self._run(pause=True).restarts == 0
        assert self._run(runtime={"auto_launch_broker": False}).restarts == 0

    def test_failure_is_notified(self):
        f = Fake()
        f.restart = lambda: (False, "終了に失敗しました（rc=1）")
        bw.run_daily_restart(runtime=RT, refresh_token=_refused, restart=f.restart,
                             alert=f.alert, closed_today=False, pause_exists=lambda: False)
        assert f.alerts == ["kabuステーションを再起動できませんでした"]


class TestLoginReminder:
    def _alerts(self, when, expired=True):
        out = []
        bw.login_reminder_tick(now=when, expired=expired, alert=lambda t, m, **k: out.append(t))
        return out

    def test_reminds_only_when_logged_out_in_the_window(self):
        assert self._alerts(_at(MON, "08:50"))
        assert self._alerts(_at(MON, "15:20"))
        assert not self._alerts(_at(MON, "08:50"), expired=False)
        assert not self._alerts(_at(MON, "08:35"))
        assert not self._alerts(_at(MON, "15:35"))
        assert not self._alerts(_at(SAT, "09:05"))
