"""kabuステーションのプロセス断を知らせる判断（broker_watch）のテスト。

この判断を main.py のクロージャに直接書いていたときは、テストが
「ソース文字列が含まれるか」しか見られず、休場日・判定不能・再通知間隔・
自動起動の有無といった分岐を一切検証できなかった（実際にレビューで
複数の穴が見つかった）。判断だけを切り出し、ここで実挙動を固定する。
"""
from datetime import datetime

import pytest

from src.core import broker_watch as bw

# 2026-09-07 は月曜（営業日）、2026-09-05 は土曜。
WEEKDAY_MORNING = datetime(2026, 9, 7, 9, 30)
WEEKDAY_NIGHT = datetime(2026, 9, 7, 22, 0)
SATURDAY = datetime(2026, 9, 5, 9, 30)


class TestNotifyWindow:
    """夜間・休場日は鳴らさないこと。

    2026-09-05 に「対応不要な🔴が毎週末届くと本物の🔴を無視する癖がつく」ため
    token_refresh へ休場日ガードを入れたばかり。同じ轍を踏まない。
    """

    def test_weekday_market_hours_is_in_window(self):
        assert bw.in_notify_window(WEEKDAY_MORNING) is True

    def test_weekday_night_is_out_of_window(self):
        assert bw.in_notify_window(WEEKDAY_NIGHT) is False

    def test_holiday_is_out_of_window(self):
        assert bw.in_notify_window(SATURDAY) is False


class TestDownNotification:
    def _notifier(self):
        return bw.BrokerDownNotifier(repeat_seconds=1800)

    def _check(self, n, *, alive, now, window=True, auto_launch=False):
        return n.check(alive=alive, now=now, notify_window=window,
                       auto_launch=auto_launch)

    def test_notifies_when_down(self):
        n = self._notifier()
        notice = self._check(n, alive=False, now=100.0)
        assert notice is not None
        title, body = notice
        assert "起動していません" in title

    def test_silent_while_running(self):
        assert self._check(self._notifier(), alive=True, now=100.0) is None

    def test_silent_outside_the_window(self):
        n = self._notifier()
        assert self._check(n, alive=False, now=100.0, window=False) is None

    def test_repeats_after_the_interval(self):
        n = self._notifier()
        assert self._check(n, alive=False, now=100.0) is not None
        assert self._check(n, alive=False, now=100.0 + 600) is None
        assert self._check(n, alive=False, now=100.0 + 1801) is not None

    def test_recovery_rearms_the_notification(self):
        """復帰したら記録を戻す（次に落ちればすぐ知らせる）。"""
        n = self._notifier()
        self._check(n, alive=False, now=100.0)
        self._check(n, alive=True, now=200.0)          # 復帰
        assert self._check(n, alive=False, now=300.0) is not None

    def test_unknown_does_not_rearm_the_throttle(self):
        """判定できなかった回でスロットルを解除しないこと。

        is_running() は確認失敗時に True を返す（起動判断の安全側）。
        これを復帰と誤解すると、tasklist がたまに失敗するだけで
        通知間隔が縮み、鳴りすぎる。
        """
        n = self._notifier()
        assert self._check(n, alive=False, now=100.0) is not None
        assert self._check(n, alive=None, now=200.0) is None    # 判定不能
        assert self._check(n, alive=False, now=300.0) is None, (
            "判定不能がスロットルを解除している"
        )

    def test_unknown_alone_never_notifies(self):
        n = self._notifier()
        assert self._check(n, alive=None, now=100.0) is None


class TestMessageMatchesConfiguration:
    """文面が実際の構成と食い違わないこと。

    「自動起動は無効にしています」と無条件に書くと、自動起動が有効な構成で
    「手で起動してください」と言った直後に自動起動が走り、二重起動を誘発する。
    """

    def _body(self, *, auto_launch):
        n = bw.BrokerDownNotifier()
        return n.check(alive=False, now=1.0, notify_window=True,
                       auto_launch=auto_launch)[1]

    def test_manual_configuration_tells_the_user_to_launch(self):
        body = self._body(auto_launch=False)
        assert "起動してログイン" in body
        assert "自動起動" in body and "無効" in body

    def test_auto_configuration_does_not_claim_manual_only(self):
        body = self._body(auto_launch=True)
        assert "無効" not in body, "自動起動が有効なのに『無効』と書いている"
        assert "自動起動" in body

    def test_does_not_claim_auth_is_fine(self):
        """「認証切れではなく」と断定しないこと。

        アプリが落ちていれば token_refresh は接続エラーで認証切れも記録する。
        断定すると、直前に出た認証切れ通知と矛盾する文面になる。
        """
        for auto in (True, False):
            assert "認証切れではなく" not in self._body(auto_launch=auto)


class TestDownMessageTellsWhatActuallyHappened:
    """「自動起動を試みます」と言い切らず、実際の結果を伝えること。

    実行ファイル未検出・日次上限・クールダウンで起動できなかった場合に
    logger.info だけで済ませると、30分おきに「試みます」と言い続けながら
    一度も起動せず、利用者は待たされたまま終日取引を失う
    （直近コミット a4a591d で潰した「問題を隠すログ」と同型）。
    """

    def test_manual_configuration(self):
        _, body = bw.down_message(auto_launch=False, launch_result=None)
        assert "起動してログイン" in body
        assert "無効" in body

    def test_reports_launch_failure_with_reason(self):
        _, body = bw.down_message(
            auto_launch=True,
            launch_result=(False, "実行ファイルが見つかりません: C:/x/KabuS.exe"),
        )
        assert "実行ファイルが見つかりません" in body, "失敗の理由が本文に無い"
        assert "手動" in body or "手で" in body, "人がやるべきことが書かれていない"

    def test_reports_launch_success(self):
        _, body = bw.down_message(
            auto_launch=True, launch_result=(True, "kabuステーションを起動しました（本日1回目）"),
        )
        assert "起動しました" in body
        assert "失敗" not in body

    def test_does_not_promise_when_nothing_was_attempted(self):
        _, body = bw.down_message(auto_launch=True, launch_result=None)
        assert "試みます" not in body, "実行していないのに約束している"


class TestNoRedAfterSuccessfulLaunch:
    """自動起動が成功した直後に🔴で追い打ちしないこと。

    直前に🟢「起動しました」を出しているのに、同じ周期で
    🔴「kabuステーションが起動していません」が並ぶと矛盾する。
    対応不要な🔴を鳴らさないという、このモジュールの目的そのものに反する。
    """

    def test_silent_when_launch_succeeded(self):
        n = bw.BrokerDownNotifier()
        notice = n.check(alive=False, now=100.0, notify_window=True,
                         auto_launch=True,
                         launch_result=(True, "kabuステーションを起動しました（本日1回目）"))
        assert notice is None

    def test_still_notifies_when_launch_failed(self):
        n = bw.BrokerDownNotifier()
        notice = n.check(alive=False, now=100.0, notify_window=True,
                         auto_launch=True,
                         launch_result=(False, "実行ファイルが見つかりません"))
        assert notice is not None
        assert "実行ファイルが見つかりません" in notice[1]

    def test_success_does_not_consume_the_throttle_slot(self):
        """成功で黙った回が、次の本当の異常の通知を遅らせないこと。"""
        n = bw.BrokerDownNotifier(repeat_seconds=1800)
        n.check(alive=False, now=100.0, notify_window=True, auto_launch=True,
                launch_result=(True, "起動しました"))
        notice = n.check(alive=False, now=200.0, notify_window=True,
                         auto_launch=True, launch_result=(False, "上限に達しています"))
        assert notice is not None
