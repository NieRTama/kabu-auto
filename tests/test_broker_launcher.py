"""
kabuステーションの起動制御（broker_launcher）のテスト

2026-08-31 にkabuステーションがクラッシュし、翌朝まで kabu-auto が待機状態の
ままだった。復旧には「アプリを起動する」物理操作が必要で、外出先からは何も
できなかった。プロセスの起動だけを自動化する（認証は認証アプリで人が行う）。

実際にプロセスを起動してしまわないよう、subprocess は必ずモックする。
"""
import itertools
import threading
import time
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core import broker_launcher as bl


#: 既定の時計。呼び出しのたびに大きく進み、クールダウンに引っかからない。
_SHARED_CLOCK = itertools.count(0, 10_000)


@pytest.fixture(autouse=True)
def _reset():
    bl.reset()
    yield
    bl.reset()


def _launch(*, running=False, exists=True, max_attempts=3, popen_error=None, now=None):
    """launch() を安全に実行する（実プロセスは起動しない）。

    now を渡さない場合は、呼び出しごとに時刻が大きく進む時計を使う。
    こうしないと「起動直後クールダウン」に引っかかり、回数上限など別の観点を
    検証しているテストがクールダウンのせいで落ちてしまう。
    """
    popen = MagicMock()
    if popen_error:
        popen.side_effect = popen_error
    # 既定の時計はモジュール共有にする。呼び出しごとに作り直すと時刻が 0 に戻り、
    # 「前回起動より前」になってクールダウン判定が壊れる。
    clock = now or (lambda: float(next(_SHARED_CLOCK)))
    with patch.object(bl, "probe_running", return_value=running), \
         patch.object(bl, "_now", clock), \
         patch.object(bl.os.path, "isfile", return_value=exists), \
         patch.object(bl.subprocess, "Popen", popen) as p:
        ok, detail = bl.launch("C:/dummy/KabuS.exe", max_attempts_per_day=max_attempts)
    return ok, detail, popen


class TestAlreadyRunning:
    def test_does_not_launch_when_running(self):
        """多重起動しない（既に起動していれば何もしない）"""
        ok, detail, popen = _launch(running=True)
        assert ok is False
        assert "既に起動" in detail
        popen.assert_not_called()

    def test_attempt_not_counted_when_already_running(self):
        _launch(running=True)
        assert bl.attempts_today() == 0, "起動していない試行は数えない"


class TestMissingExecutable:
    def test_fails_when_exe_not_found(self):
        ok, detail, popen = _launch(exists=False)
        assert ok is False
        assert "見つかりません" in detail
        popen.assert_not_called()


class TestSuccessfulLaunch:
    def test_launches_and_counts(self):
        ok, detail, popen = _launch()
        assert ok is True
        popen.assert_called_once()
        assert bl.attempts_today() == 1

    def test_uses_given_path(self):
        _, _, popen = _launch()
        assert popen.call_args[0][0] == ["C:/dummy/KabuS.exe"]

    def test_popen_failure_is_reported_not_raised(self):
        ok, detail, _ = _launch(popen_error=OSError("access denied"))
        assert ok is False
        assert "起動に失敗" in detail


class TestDailyLimit:
    def test_stops_after_limit(self):
        """証券会社側の障害時に無意味な起動を繰り返さない"""
        for _ in range(3):
            assert _launch(max_attempts=3)[0] is True
        ok, detail, popen = _launch(max_attempts=3)
        assert ok is False
        assert "上限" in detail
        popen.assert_not_called()

    def test_zero_means_unlimited(self):
        for _ in range(10):
            assert _launch(max_attempts=0)[0] is True
        assert bl.attempts_today() == 10

    def test_counter_resets_on_new_day(self):
        for _ in range(3):
            _launch(max_attempts=3)
        assert _launch(max_attempts=3)[0] is False
        # 日付が変わればリセットされる
        with patch.object(bl, "_today", return_value=date(2099, 1, 1)):
            assert _launch(max_attempts=3)[0] is True


class TestIsRunning:
    """tasklist の成否も判定に含める（returncode 非ゼロ＝判定不能）。"""

    def _result(self, stdout, returncode=0):
        result = MagicMock()
        result.stdout = stdout
        result.returncode = returncode
        return result

    def test_detects_running_process(self):
        r = self._result("KabuS.exe   24472 Console   1   150,000 K")
        with patch.object(bl.subprocess, "run", return_value=r):
            assert bl.is_running() is True

    def test_detects_absent_process(self):
        r = self._result("情報: 指定条件に一致するタスクは実行されていません。")
        with patch.object(bl.subprocess, "run", return_value=r):
            assert bl.is_running() is False

    def test_assumes_running_on_check_failure(self):
        """確認に失敗したら「起動中」とみなす（多重起動を避ける安全側）"""
        with patch.object(bl.subprocess, "run", side_effect=OSError("boom")):
            assert bl.is_running() is True


MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


def _main_src() -> str:
    """main.py の中身。相対パスで開くとルート以外からの pytest で落ちる。"""
    return MAIN_PY.read_text(encoding="utf-8")


class TestWiring:
    def _main_src(self):
        return _main_src()

    def test_discord_launch_command_registered(self):
        # 登録は {name: 関数} と {name: (関数, 説明)} の両形式を許す
        import re
        assert re.search(r'"launch":\s*\(?_cmd_launch', self._main_src()), (
            "launch コマンドが登録されていない"
        )

    def test_auto_launch_on_auth_recovery(self):
        """プロセスが落ちているとき自動起動を試みる結線があること。

        生存確認は probe_running()（判定不能を None で返す3値）を使う。
        is_running() は「起動してよいか」の判断用に不明を True へ倒しているため、
        通知や自動起動の判断には流用しない。
        """
        src = self._main_src()
        assert "auto_launch_broker" in src
        assert "broker_launcher.probe_running()" in src

    def test_auth_is_not_automated(self):
        """認証（2段階認証）は自動化しない方針が守られていること"""
        import inspect
        src = inspect.getsource(bl)
        for forbidden in ("password", "PASSWORD", "SendKeys", "keybd_event", "pyautogui"):
            assert forbidden not in src, f"認証を自動化する実装が入っている: {forbidden}"


class TestRelaunchCooldown:
    """起動直後の再起動を抑止すること。

    KabuS.exe を Popen してから tasklist に現れるまで数秒かかる。その間は
    is_running() が False を返し続けるため、近接した2回目の呼び出し
    （5分間隔ジョブ ＋ Discord の launch コマンド等）で二重起動になる。
    """

    def test_second_launch_right_after_is_suppressed(self):
        t = [1000.0]
        ok1, _, _ = _launch(now=lambda: t[0])
        assert ok1 is True
        t[0] += 5                     # 5秒後（まだプロセス一覧に現れていない）
        ok2, detail, popen2 = _launch(now=lambda: t[0])
        assert ok2 is False
        assert "起動直後" in detail
        popen2.assert_not_called()

    def test_launch_allowed_again_after_cooldown(self):
        t = [1000.0]
        _launch(now=lambda: t[0])
        t[0] += bl.RELAUNCH_COOLDOWN_SECONDS + 1
        ok, _, popen = _launch(now=lambda: t[0])
        assert ok is True
        popen.assert_called_once()

    def test_cooldown_does_not_consume_an_attempt(self):
        t = [1000.0]
        _launch(now=lambda: t[0])
        t[0] += 5
        _launch(now=lambda: t[0])
        assert bl.attempts_today() == 1, "起動しなかった呼び出しは回数に数えない"


class TestConcurrentLaunch:
    """同時に呼ばれても1つしか起動しないこと（結果の担保）。

    定期チェックと Discord の launch コマンドが近接すると、両方が「未起動」と
    判定して2回起動する（修正前の実装で実際に「本日1回目」「本日2回目」が出る）。
    止めているのはクールダウンで、ロック内での確認はその前段の多重防御。
    """

    def test_only_one_process_is_started(self):
        popen = MagicMock()
        started = []

        def slow_probe(process_name="KabuS"):
            time.sleep(0.15)          # tasklist 相当の所要時間
            return False

        def worker():
            started.append(bl.launch("C:/dummy/KabuS.exe")[0])

        with patch.object(bl, "probe_running", side_effect=slow_probe), \
             patch.object(bl.os.path, "isfile", return_value=True), \
             patch.object(bl.subprocess, "Popen", popen):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=10)

        assert popen.call_count == 1, f"同時呼び出しで {popen.call_count} 回起動した"
        assert sorted(started) == [False, True]


class TestReloginGuidance:
    """認証切れの通知が「既存の窓でログインする」ことを案内すること。

    kabuステーションは**プロセスが生きたまま認証だけ**が日次で切れる。
    通知が「ログインしてください」だけだと、利用者は落ちていると誤解して
    2つ目を起動してしまう（2026-09-07 に実際に発生した二重起動の直接原因）。
    """

    def _alert_body(self) -> str:
        """通知の組み立て箇所（案内文は alert() 呼び出しの手前で作る）。"""
        src = _main_src()
        i = src.index("kabuステーションの再ログインが必要です")
        return src[max(0, i - 1200):i + 1200]

    def test_branches_on_process_liveness(self):
        """アプリの生死で文面を変えること。

        「新しく起動しないでください」と無条件に言うと、本当にクラッシュした
        とき（2026-08-31）に「何もするな」と伝えることになり終日止まる。
        判定には probe_running()（判定不能を None で返す方）を使う。
        """
        assert "broker_launcher.probe_running()" in self._alert_body()

    def test_has_a_message_for_the_crashed_case(self):
        assert "自体が起動していません" in self._alert_body()

    def test_handles_the_unknown_case(self):
        assert "確認できませんでした" in self._alert_body()

    def test_mentions_the_app_may_already_be_running(self):
        body = self._alert_body()
        assert "起動したまま" in body or "起動済み" in body, (
            "アプリが起動したままである可能性を伝えていない"
        )

    def test_tells_not_to_start_a_second_instance(self):
        body = self._alert_body()
        assert "二重起動" in body or "新しく起動しない" in body, (
            "2つ目を起動しないよう案内していない"
        )


class TestRunningCheckPrecedence:
    def test_running_wins_over_missing_executable(self):
        """起動中なら、パスが見つからなくても「既に起動しています」と答えること。

        先に実行ファイルの有無を見ると、既定以外の場所にインストールされた環境で
        「実行ファイルが見つかりません」と返る。二重起動を止めるための変更が、
        逆に利用者を手動起動へ誘導してしまう。
        """
        ok, detail, popen = _launch(running=True, exists=False)
        assert ok is False
        assert "既に起動" in detail
        popen.assert_not_called()


class TestBrokerDownIsReported:
    """プロセス断の検知を認証状態に依存させないこと（結線）。

    場中のクラッシュは接続エラーになるだけで 401 ではないため
    （kabu_client は401のときだけ mark_expired する）、認証切れの判定の
    後ろに置くと検知にも自動起動にも到達しない。
    通知するかどうかの判断自体は tests/test_broker_watch.py が直接検証する。
    """

    def _recovery_body(self) -> str:
        src = _main_src()
        i = src.index("def auth_recovery_check")
        return src[i:src.index("def db_backup", i)]

    def test_liveness_is_not_gated_by_auth_state(self):
        body = self._recovery_body()
        assert "not broker_auth.is_expired()" not in body, (
            "認証切れゲートが残っており、場中クラッシュでは生存確認に到達しない"
        )

    def test_uses_the_tested_decision_module(self):
        body = self._recovery_body()
        assert "broker_watch.in_notify_window" in body
        assert "broker_down.check(" in body

    def test_uses_the_tri_state_probe(self):
        """起動判断用の is_running() を通知判断に流用しないこと。"""
        body = self._recovery_body()
        assert "broker_launcher.probe_running()" in body


class TestManualLaunchIsNotRateLimited:
    """人が明示的に頼んだ起動は日次上限で止めないこと。

    自動起動を無効にした構成では Discord の launch が唯一の起動経路になる。
    上限3回はもともと「自動起動の暴走」を止めるためのもので、
    人の明示操作まで縛ると、クラッシュ時に終日起動できなくなる。
    """

    def test_manual_launch_bypasses_the_daily_limit(self):
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        i = src.index("def _cmd_launch")
        assert "manual=True" in src[i:i + 400], (
            "Discord の launch が日次上限を回避していない"
        )


class TestProbeRunningTriState:
    """判定できなかった場合を「起動中」とも「未起動」とも混同しないこと。

    is_running() は確認失敗時に True を返す（多重起動を避ける安全側）。
    これは起動判断としては正しいが、案内文や通知の判断に流用すると
    本当に落ちているときに誤誘導する。逆に、失敗を False と読むと
    2つ目を起動してしまう。
    """

    def _run(self, *, stdout, returncode=0):
        result = MagicMock()
        result.stdout = stdout
        result.returncode = returncode
        return patch.object(bl.subprocess, "run", return_value=result)

    def test_returns_true_when_present(self):
        with self._run(stdout="KabuS.exe  1 Console  1  1 K"):
            assert bl.probe_running() is True

    def test_returns_false_when_absent(self):
        with self._run(stdout="情報: 指定条件に一致するタスクは実行されていません。"):
            assert bl.probe_running() is False

    def test_returns_none_when_the_call_raises(self):
        with patch.object(bl.subprocess, "run", side_effect=OSError("boom")):
            assert bl.probe_running() is None

    def test_returns_none_when_tasklist_exits_nonzero(self):
        """RPC unavailable 等で非ゼロ終了すると stdout は空になる。

        これを「未起動」と読むと、誤った🔴が飛ぶうえ、自動起動が有効な構成では
        2つ目の KabuS を起動する（この修正が防ごうとしているものそのもの）。
        """
        with self._run(stdout="", returncode=1):
            assert bl.probe_running() is None

    def test_is_running_still_biases_to_true_when_unknown(self):
        """起動判断の安全側（多重起動を避ける）は変えない。"""
        with self._run(stdout="", returncode=1):
            assert bl.is_running() is True


class TestManualLaunchEscapeHatch:
    """人が明示的に頼んだ起動には逃げ道を残すこと。

    自動起動を無効にした構成では Discord の `launch` が唯一の起動経路になる。
    tasklist が非ゼロ終了して判定不能になると、is_running() は安全側の True を
    返すため「既に起動しています」で拒否され、何度打っても起動できず終日
    発注不可になる。人は画面を見て判断できるので、判定不能なら通す。
    """

    def _launch_with_probe(self, probe, *, manual):
        popen = MagicMock()
        with patch.object(bl, "probe_running", return_value=probe), \
             patch.object(bl.os.path, "isfile", return_value=True), \
             patch.object(bl, "_now", lambda: float(next(_SHARED_CLOCK))), \
             patch.object(bl.subprocess, "Popen", popen):
            ok, detail = bl.launch("C:/dummy/KabuS.exe", manual=manual)
        return ok, detail, popen

    def test_manual_proceeds_when_probe_is_unknown(self):
        ok, _, popen = self._launch_with_probe(None, manual=True)
        assert ok is True
        popen.assert_called_once()

    def test_automatic_refuses_when_probe_is_unknown(self):
        ok, detail, popen = self._launch_with_probe(None, manual=False)
        assert ok is False
        assert "確認できません" in detail
        popen.assert_not_called()

    def test_neither_launches_when_clearly_running(self):
        for manual in (True, False):
            bl.reset()
            ok, detail, popen = self._launch_with_probe(True, manual=manual)
            assert ok is False, f"manual={manual} で起動中なのに起動した"
            assert "既に起動" in detail
            popen.assert_not_called()

    def test_manual_does_not_consume_the_automatic_budget(self):
        """手動起動は自動起動の日次上限を食わないこと。

        食うと、自動起動が一度も走っていないのに手動3回でその日の自動起動が
        止まる（設定コメントが謳う「上限は自動起動にだけ効く」が破れる）。
        """
        for _ in range(3):
            self._launch_with_probe(False, manual=True)
        assert bl.attempts_today() == 0
