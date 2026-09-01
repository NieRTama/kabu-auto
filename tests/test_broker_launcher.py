"""
kabuステーションの起動制御（broker_launcher）のテスト

2026-08-31 にkabuステーションがクラッシュし、翌朝まで kabu-auto が待機状態の
ままだった。復旧には「アプリを起動する」物理操作が必要で、外出先からは何も
できなかった。プロセスの起動だけを自動化する（認証は認証アプリで人が行う）。

実際にプロセスを起動してしまわないよう、subprocess は必ずモックする。
"""
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.core import broker_launcher as bl


@pytest.fixture(autouse=True)
def _reset():
    bl.reset()
    yield
    bl.reset()


def _launch(*, running=False, exists=True, max_attempts=3, popen_error=None):
    """launch() を安全に実行する（実プロセスは起動しない）。"""
    popen = MagicMock()
    if popen_error:
        popen.side_effect = popen_error
    with patch.object(bl, "is_running", return_value=running), \
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
    def test_detects_running_process(self):
        result = MagicMock()
        result.stdout = "KabuS.exe   24472 Console   1   150,000 K"
        with patch.object(bl.subprocess, "run", return_value=result):
            assert bl.is_running() is True

    def test_detects_absent_process(self):
        result = MagicMock()
        result.stdout = "情報: 指定条件に一致するタスクは実行されていません。"
        with patch.object(bl.subprocess, "run", return_value=result):
            assert bl.is_running() is False

    def test_assumes_running_on_check_failure(self):
        """確認に失敗したら「起動中」とみなす（多重起動を避ける安全側）"""
        with patch.object(bl.subprocess, "run", side_effect=OSError("boom")):
            assert bl.is_running() is True


class TestWiring:
    def _main_src(self):
        with open("main.py", encoding="utf-8") as f:
            return f.read()

    def test_discord_launch_command_registered(self):
        assert '"launch": _cmd_launch' in self._main_src()

    def test_auto_launch_on_auth_recovery(self):
        """認証切れが続くとき自動起動を試みること"""
        src = self._main_src()
        assert "auto_launch_broker" in src
        assert "broker_launcher.is_running()" in src

    def test_auth_is_not_automated(self):
        """認証（2段階認証）は自動化しない方針が守られていること"""
        import inspect
        src = inspect.getsource(bl)
        for forbidden in ("password", "PASSWORD", "SendKeys", "keybd_event", "pyautogui"):
            assert forbidden not in src, f"認証を自動化する実装が入っている: {forbidden}"
