"""
broker_wait.wait_for_broker() のテスト

OS起動時の自動実行では kabuステーションがまだ起動・ログインされていないため
必ず接続に失敗する。従来は fail-closed で即中断していたが、ログインされるまで
待機できるようにした（2026-08-28）。

実時間に依存させないよう sleep / monotonic を注入して検証する
（時限爆弾テスト・遅いテストを避ける）。
"""
import pytest

from src.core.broker_wait import wait_for_broker


class _Clock:
    """注入用の疑似時計。sleep した分だけ時刻が進む。"""

    def __init__(self):
        self.now = 0.0
        self.slept = []

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    def monotonic(self):
        return self.now


def _connect_failing_n_times(n: int):
    """最初の n 回は例外、その後成功する connect を返す。"""
    state = {"calls": 0}

    def connect():
        state["calls"] += 1
        if state["calls"] <= n:
            raise ConnectionError("connection refused")
        return "token"

    connect.state = state
    return connect


class TestImmediateSuccess:
    def test_returns_true_without_sleeping(self):
        clock = _Clock()
        connect = _connect_failing_n_times(0)
        ok = wait_for_broker(connect, timeout_seconds=1800,
                             sleep=clock.sleep, monotonic=clock.monotonic)
        assert ok is True
        assert connect.state["calls"] == 1
        assert clock.slept == [], "成功時は待機しない"

    def test_on_wait_start_not_called_when_immediately_connected(self):
        clock = _Clock()
        called = []
        wait_for_broker(_connect_failing_n_times(0), timeout_seconds=1800,
                        on_wait_start=lambda: called.append(True),
                        sleep=clock.sleep, monotonic=clock.monotonic)
        assert called == [], "初回成功時は待機開始通知を出さない"


class TestDisabled:
    def test_timeout_zero_tries_once_only(self):
        """timeout_seconds=0 は従来動作（1回試して失敗なら即False）"""
        clock = _Clock()
        connect = _connect_failing_n_times(5)
        ok = wait_for_broker(connect, timeout_seconds=0,
                             sleep=clock.sleep, monotonic=clock.monotonic)
        assert ok is False
        assert connect.state["calls"] == 1
        assert clock.slept == []


class TestRetryUntilSuccess:
    def test_succeeds_after_retries(self):
        clock = _Clock()
        connect = _connect_failing_n_times(3)
        ok = wait_for_broker(connect, timeout_seconds=1800, initial_interval=30, max_interval=60,
                             sleep=clock.sleep, monotonic=clock.monotonic)
        assert ok is True
        assert connect.state["calls"] == 4  # 初回1 + リトライ3

    def test_interval_backs_off_and_caps(self):
        clock = _Clock()
        connect = _connect_failing_n_times(5)
        wait_for_broker(connect, timeout_seconds=1800, initial_interval=30, max_interval=60,
                        sleep=clock.sleep, monotonic=clock.monotonic)
        # 30 → 60 → 60 ... と倍化して上限で頭打ちになる
        assert clock.slept[0] == 30
        assert clock.slept[1] == 60
        assert all(s <= 60 for s in clock.slept), "max_interval を超えて待たない"

    def test_notifies_wait_start_once_and_connected(self):
        clock = _Clock()
        starts, connects = [], []
        wait_for_broker(_connect_failing_n_times(3), timeout_seconds=1800,
                        on_wait_start=lambda: starts.append(True),
                        on_connected=lambda waited: connects.append(waited),
                        sleep=clock.sleep, monotonic=clock.monotonic)
        assert len(starts) == 1, "待機開始通知は1回だけ"
        assert len(connects) == 1, "接続時の通知は1回"
        assert connects[0] > 0


class TestTimeout:
    def test_returns_false_after_timeout(self):
        clock = _Clock()
        connect = _connect_failing_n_times(10_000)  # 常に失敗
        ok = wait_for_broker(connect, timeout_seconds=300, initial_interval=30, max_interval=60,
                             sleep=clock.sleep, monotonic=clock.monotonic)
        assert ok is False
        assert clock.now >= 300, "タイムアウトまでは待つ"

    def test_does_not_sleep_past_timeout(self):
        """残り時間より長くは眠らない（タイムアウト時刻を大きく超過しない）"""
        clock = _Clock()
        wait_for_broker(_connect_failing_n_times(10_000), timeout_seconds=100,
                        initial_interval=30, max_interval=60,
                        sleep=clock.sleep, monotonic=clock.monotonic)
        assert clock.now == pytest.approx(100, abs=1)


class TestNotificationFailureIsolation:
    def test_wait_start_notification_failure_does_not_abort(self):
        """通知に失敗しても待機・接続は継続する（通知は副次的関心事）"""
        clock = _Clock()

        def boom():
            raise RuntimeError("discord down")

        ok = wait_for_broker(_connect_failing_n_times(2), timeout_seconds=1800,
                             on_wait_start=boom,
                             sleep=clock.sleep, monotonic=clock.monotonic)
        assert ok is True

    def test_connected_notification_failure_does_not_abort(self):
        clock = _Clock()

        def boom(waited):
            raise RuntimeError("discord down")

        ok = wait_for_broker(_connect_failing_n_times(1), timeout_seconds=1800,
                             on_connected=boom,
                             sleep=clock.sleep, monotonic=clock.monotonic)
        assert ok is True
