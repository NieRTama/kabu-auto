"""
認証切れからの自動復帰（auth_recovery）のテスト

2026-08-29 に発覚した設計欠陥の回帰防止:
  当初は token_refresh 内で30分だけブロッキング待機していたため、
  タイムアウトすると翌朝まで復帰しなかった。実際に 08:30 に認証切れ →
  09:00 にタイムアウト → その後ログインしても終日発注不可、という状態が起きた。
  「通知を見てログインしたのに動かない」では通知の意味が無い。

現在は「認証切れの間、時間制限なく定期的に試す」方式。
"""
from unittest.mock import MagicMock, patch

import pytest

from src.core import auth_recovery, broker_auth


@pytest.fixture(autouse=True)
def _reset():
    broker_auth.reset()
    yield
    broker_auth.reset()


class TestSkipWhenValid:
    def test_does_nothing_when_auth_is_valid(self):
        """認証が有効なときはAPIを叩かない（無駄なトークン再発行をしない）"""
        connect = MagicMock()
        assert auth_recovery.attempt_recovery(connect) is False
        connect.assert_not_called()


class TestRecovery:
    def test_recovers_when_connect_succeeds(self):
        broker_auth.mark_expired("401")
        connect = MagicMock()
        assert auth_recovery.attempt_recovery(connect) is True
        assert broker_auth.is_expired() is False
        connect.assert_called_once()

    def test_stays_expired_when_connect_fails(self):
        broker_auth.mark_expired("401")
        connect = MagicMock(side_effect=RuntimeError("401"))
        assert auth_recovery.attempt_recovery(connect) is False
        assert broker_auth.is_expired() is True, "失敗しても認証切れのまま（誤って発注を許可しない）"

    def test_notifies_once_on_recovery(self):
        broker_auth.mark_expired("401")
        notified = []
        auth_recovery.attempt_recovery(MagicMock(), on_recovered=lambda: notified.append(1))
        assert len(notified) == 1

    def test_no_notification_on_failure(self):
        broker_auth.mark_expired("401")
        notified = []
        auth_recovery.attempt_recovery(MagicMock(side_effect=RuntimeError("401")),
                                       on_recovered=lambda: notified.append(1))
        assert notified == []

    def test_notification_failure_does_not_break_recovery(self):
        """通知が失敗しても復帰処理自体は成立させる"""
        broker_auth.mark_expired("401")

        def boom():
            raise RuntimeError("discord down")

        assert auth_recovery.attempt_recovery(MagicMock(), on_recovered=boom) is True
        assert broker_auth.is_expired() is False


class TestRetriesIndefinitely:
    def test_can_recover_after_many_failures(self):
        """何度失敗しても、後で成功すれば復帰できる（タイムアウトで諦めない）。

        これが 2026-08-29 の欠陥そのもの。30分でタイムアウトする実装では
        以降いくらログインしても復帰しなかった。
        """
        broker_auth.mark_expired("401")
        failing = MagicMock(side_effect=RuntimeError("401"))
        # 朝の30分間に相当する回数だけ失敗させる（5分間隔なら6回）
        for _ in range(6):
            assert auth_recovery.attempt_recovery(failing) is False
        assert broker_auth.is_expired() is True

        # 昼過ぎにログインした、というシナリオ
        assert auth_recovery.attempt_recovery(MagicMock()) is True
        assert broker_auth.is_expired() is False

    def test_does_not_retry_after_recovered(self):
        """復帰後は再度切れるまで何もしない"""
        broker_auth.mark_expired("401")
        auth_recovery.attempt_recovery(MagicMock())
        connect = MagicMock()
        auth_recovery.attempt_recovery(connect)
        connect.assert_not_called()


class TestScheduling:
    def test_registered_as_interval_job(self):
        import src.core.scheduler as sched_mod
        sched = sched_mod.TradingScheduler()
        sched.register("auth_recovery_check", MagicMock())
        with patch.object(sched._scheduler, "add_job") as add_job, \
             patch.object(sched._scheduler, "start"):
            sched.start()
            kwargs = {c.kwargs["id"]: c for c in add_job.call_args_list}
        assert "auth_recovery_check" in kwargs
        call = kwargs["auth_recovery_check"]
        assert call.args[1] == "interval", "定期実行でなければ復帰し続けられない"
        assert call.kwargs.get("minutes", 0) > 0

    def test_weekly_report_registered_on_saturday(self):
        import src.core.scheduler as sched_mod
        sched = sched_mod.TradingScheduler()
        sched.register("discord_weekly_report", MagicMock())
        with patch.object(sched._scheduler, "add_job") as add_job, \
             patch.object(sched._scheduler, "start"):
            sched.start()
            kwargs = {c.kwargs["id"]: c.kwargs for c in add_job.call_args_list}
        assert "discord_weekly_report" in kwargs
        assert kwargs["discord_weekly_report"]["day_of_week"] == "sat"


class TestNoBlockingWaitInTokenRefresh:
    def test_token_refresh_does_not_block(self):
        """token_refresh がブロッキング待機を持たないこと（回帰防止）。

        待機を持つと、その間スケジューラのワーカーが占有され、
        かつタイムアウトで諦める旧来の欠陥が復活する。
        """
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        start = src.index("def token_refresh():")
        end = src.index("def auth_recovery_check():")
        block = src[start:end]
        assert "wait_for_broker" not in block, (
            "token_refresh 内でブロッキング待機している（タイムアウトで復帰不能になる）"
        )
