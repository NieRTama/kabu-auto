"""認証復帰チェック（broker_watch.recovery_tick）を1日の時系列で回すシナリオテスト。

2026-10-13 の不具合は、判定関数を単体で、しかも異常時の入力の組み合わせだけで
テストしていたため見逃した（「ログイン済みで認証も有効」という毎朝必ず起きる
状態を誰も試していなかった）。ここでは5分ごとのチェックを実際の時刻の並びで
回し、完全自動ログイン（KabuSの強制終了を伴う）が何回呼ばれるかを固定する。
"""
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.core import broker_auth
from src.core import broker_watch as bw

RUNTIME = {"broker_full_login_enabled": True, "auto_launch_broker": False}


class FakeBroker:
    """KabuS の状態だけを持つ偽物。refresh_token は実物と同じく失敗時に例外を投げる。"""

    def __init__(self, *, alive=True, logged_in=False):
        self.alive = alive
        self.logged_in = logged_in
        self.full_logins = 0

    def refresh_token(self):
        if not self.alive:
            raise ConnectionError("18080 refused")
        if not self.logged_in:
            raise RuntimeError("401 Unauthorized")

    def full_login(self):
        self.full_logins += 1
        self.alive = True
        self.logged_in = True
        return True, "完全自動ログインを実行しました"


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    broker_auth.reset()
    # mark_valid 直後の mark_expired を競合として無視する猶予（3秒）を切る。
    # テストは実時間で数ミリ秒しか経たないため、猶予が残ると認証切れを再現できない。
    monkeypatch.setattr(broker_auth, "_RACE_GRACE_SECONDS", 0)
    yield
    broker_auth.reset()


def _tick(broker, hh, mm, *, probe=None, notifier=None):
    # 2026-10-13（火）は営業日（10/12 のスポーツの日の翌日）。
    bw.recovery_tick(
        refresh_token=broker.refresh_token,
        on_recovered=lambda: None,
        probe_running=probe or (lambda: broker.alive),
        now=datetime(2026, 10, 13, hh, mm),
        runtime=RUNTIME,
        launch_broker=lambda: (False, "auto_launch無効"),
        full_login=broker.full_login,
        notifier=notifier or bw.BrokerDownNotifier(),
        alert=MagicMock(),
        monotonic=float(hh * 3600 + mm * 60),
    )


def _every_5min(start, end):
    """(時, 分) を start から end まで5分刻みで返す。"""
    h, m = start
    while (h, m) <= end:
        yield h, m
        m += 5
        if m >= 60:
            h, m = h + 1, m - 60


class TestNormalMorning:
    def test_no_full_login_after_the_0830_job_succeeded(self):
        """2026-10-13 の再現: 08:30の定時ログイン成功後は、5分ごとのチェックが
        何回走っても完全自動ログインを呼ばない（旧実装は08:44・08:54に呼んだ）。"""
        broker = FakeBroker(alive=True, logged_in=False)
        broker_auth.mark_expired("08:00 401")      # 朝の認証切れ
        for h, m in _every_5min((8, 3), (8, 28)):    # 08:30より前は定時ジョブの担当
            _tick(broker, h, m)
        broker.logged_in = True                      # 08:30 定時ジョブがログイン
        for h, m in _every_5min((8, 33), (11, 3)):
            _tick(broker, h, m)
        assert broker.full_logins == 0
        assert broker_auth.is_expired() is False

    def test_no_full_login_before_0840_even_if_expired(self):
        """08:40より前は、認証が切れていても08:30の定時ジョブに任せる"""
        broker = FakeBroker(alive=True, logged_in=False)
        broker_auth.mark_expired("401")
        for h, m in _every_5min((8, 0), (8, 39)):
            _tick(broker, h, m)
        assert broker.full_logins == 0


class TestRealFailuresStillRecover:
    def test_auth_expires_midday_while_alive_recovers_once(self):
        """生存中に本当に認証が切れたら1回だけ復帰させ、以後は呼ばない"""
        broker = FakeBroker(alive=True, logged_in=True)
        for h, m in _every_5min((9, 0), (9, 55)):
            _tick(broker, h, m)
        broker.logged_in = False
        broker_auth.mark_expired("10:00 401")
        for h, m in _every_5min((10, 0), (11, 0)):
            _tick(broker, h, m)
        assert broker.full_logins == 1
        assert broker_auth.is_expired() is False

    def test_process_crash_midday_recovers_once(self):
        """場中のクラッシュ（401ではなく接続エラーで、認証切れとは記録されない）も復帰させる"""
        broker = FakeBroker(alive=True, logged_in=True)
        _tick(broker, 10, 55)
        broker.alive = False
        broker.logged_in = False
        for h, m in _every_5min((11, 0), (11, 30)):
            _tick(broker, h, m)
        assert broker.full_logins == 1


class TestFinalSafetyCheck:
    def test_does_not_kill_when_token_can_be_fetched(self):
        """判定が誤って真になっても、トークンが取れる（壊れていない）なら実行しない。

        例: tasklist の誤判定でプロセスが見えない（alive=False）が、APIは正常。
        判定ロジックとは独立した最終確認で、正常なKabuSの強制終了を防ぐ。
        """
        broker = FakeBroker(alive=True, logged_in=True)
        for h, m in _every_5min((9, 0), (10, 0)):
            _tick(broker, h, m, probe=lambda: False)
        assert broker.full_logins == 0


class TestAuthValidNow:
    def test_true_and_marks_valid_when_token_fetched(self):
        broker_auth.mark_expired("401")
        assert bw.auth_valid_now(lambda: None) is True
        assert broker_auth.is_expired() is False

    def test_false_when_token_fetch_fails(self):
        def fail():
            raise RuntimeError("401")
        assert bw.auth_valid_now(fail) is False


class TestDownNoticeBeforeMorningLogin:
    """2026-10-22 の再現: KabuS が夜のうちに終了していると、08:24 に🔴「起動していません」が
    出ていた。完全自動ログインが有効なら 08:30 の定時ジョブが起動からやり直すため、
    08:40（日中の自動復帰を始める時刻）より前のプロセス断は通知しない。"""

    def _alerts(self, broker, times, notifier):
        sent = []
        for hh, mm in times:
            bw.recovery_tick(
                refresh_token=broker.refresh_token, on_recovered=lambda: None,
                probe_running=lambda: broker.alive, now=datetime(2026, 10, 13, hh, mm),
                runtime=RUNTIME, launch_broker=lambda: (False, "auto_launch無効"),
                full_login=lambda: (False, "失敗"), notifier=notifier,
                alert=lambda *a, **k: sent.append((hh, mm, a[0])), monotonic=float(hh * 3600 + mm * 60),
            )
        return sent

    def test_no_down_notice_before_0840_when_full_login_enabled(self):
        broker = FakeBroker(alive=False)
        sent = self._alerts(broker, [(8, 4), (8, 9), (8, 14), (8, 19), (8, 24), (8, 29), (8, 34)],
                            bw.BrokerDownNotifier())
        assert sent == []

    def test_still_notifies_if_down_after_0840(self):
        broker = FakeBroker(alive=False)
        sent = self._alerts(broker, [(8, 44)], bw.BrokerDownNotifier())
        assert any(t == "kabuステーションが起動していません" for _, _, t in sent)
