"""
画面セッション切断の検知・自動復帰（session_watch）のテスト

RDP切断でコンソールセッションがDisc状態のまま残ると、GUI自動化（SendKeys）が
静かに失敗する（2026-09-27に実際に完全自動ログインが失敗した実害）。
実際にsubprocessを実行してしまわないよう、必ずモックする。
"""
import subprocess
from unittest.mock import MagicMock, patch

from src.core import session_watch as sw

QUERY_SESSION_SAMPLE = """\
 SESSIONNAME       USERNAME                 ID  STATE   TYPE        DEVICE
 services                                    0  Disc
>                  garnet                    1  Disc
 console                                     2  Conn
"""

QUERY_SESSION_ATTACHED = """\
 SESSIONNAME       USERNAME                 ID  STATE   TYPE        DEVICE
 services                                    0  Disc
 console            garnet                   1  Active
"""

QUERY_SESSION_NO_MATCH = """\
 SESSIONNAME       USERNAME                 ID  STATE   TYPE        DEVICE
 services                                    0  Disc
 console            someone_else             1  Active
"""


def _run_result(stdout, returncode=0):
    return MagicMock(stdout=stdout, stderr="", returncode=returncode)


class TestIsConsoleDisconnected:
    def test_detects_disconnected_session_for_current_user(self):
        with patch.object(sw.os, "environ", {"USERNAME": "garnet"}), \
             patch.object(sw.subprocess, "run", return_value=_run_result(QUERY_SESSION_SAMPLE)):
            assert sw.is_console_disconnected() == "1"

    def test_returns_none_when_session_is_active(self):
        with patch.object(sw.os, "environ", {"USERNAME": "garnet"}), \
             patch.object(sw.subprocess, "run", return_value=_run_result(QUERY_SESSION_ATTACHED)):
            assert sw.is_console_disconnected() is None

    def test_returns_none_when_user_not_found(self):
        with patch.object(sw.os, "environ", {"USERNAME": "garnet"}), \
             patch.object(sw.subprocess, "run", return_value=_run_result(QUERY_SESSION_NO_MATCH)):
            assert sw.is_console_disconnected() is None

    def test_returns_none_on_nonzero_exit(self):
        with patch.object(sw.os, "environ", {"USERNAME": "garnet"}), \
             patch.object(sw.subprocess, "run", return_value=_run_result("", returncode=1)):
            assert sw.is_console_disconnected() is None

    def test_returns_none_when_query_exe_missing(self):
        with patch.object(sw.os, "environ", {"USERNAME": "garnet"}), \
             patch.object(sw.subprocess, "run", side_effect=FileNotFoundError()):
            assert sw.is_console_disconnected() is None

    def test_returns_none_on_timeout(self):
        with patch.object(sw.os, "environ", {"USERNAME": "garnet"}), \
             patch.object(sw.subprocess, "run",
                           side_effect=subprocess.TimeoutExpired(cmd="query", timeout=10)):
            assert sw.is_console_disconnected() is None


class TestReconnectToConsole:
    def test_success(self):
        with patch.object(sw.subprocess, "run", return_value=_run_result("", returncode=0)) as run:
            ok, detail = sw.reconnect_to_console("1")
        assert ok is True
        args = run.call_args.args[0]
        assert "1" in args
        assert "/dest:console" in args

    def test_failure(self):
        with patch.object(sw.subprocess, "run",
                           return_value=_run_result("error", returncode=1)) as run:
            ok, detail = sw.reconnect_to_console("1")
        assert ok is False
        args = run.call_args.args[0]
        assert "1" in args
        assert "/dest:console" in args

    def test_returns_false_on_exception(self):
        with patch.object(sw.subprocess, "run", side_effect=OSError("boom")):
            ok, detail = sw.reconnect_to_console("1")
        assert ok is False
