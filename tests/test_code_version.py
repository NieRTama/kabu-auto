"""稼働中のコードと手元の最新を比べる（code_version）のテスト。

2026-09-30 の修正が 9/28 起動のプロセスに反映されないまま10日間眠り、
10/9 の再起動で初めて有効になって 10/13 に不具合として表に出た。
「直したのに動いていない」を定期報告で自動的に知らせる。
"""
import os
import time

import pytest

from src.core import code_version as cv
from src.core import heartbeat as hb


@pytest.fixture(autouse=True)
def _restore_state():
    saved = dict(cv._state)
    yield
    cv._state.clear()
    cv._state.update(saved)


def _repo(tmp_path):
    (tmp_path / "src" / "core").mkdir(parents=True)
    for rel in ("main.py", "config.yaml", "src/core/a.py"):
        (tmp_path / rel).write_text("x", encoding="utf-8")
    return tmp_path


def _touch_future(path):
    future = time.time() + 60
    os.utime(path, (future, future))


class TestStaleNotice:
    def test_none_right_after_capture(self, tmp_path):
        repo = _repo(tmp_path)
        cv.capture(repo=repo)
        assert cv.stale_notice() is None

    def test_reports_files_changed_after_start(self, tmp_path):
        repo = _repo(tmp_path)
        cv.capture(repo=repo)
        _touch_future(repo / "src" / "core" / "a.py")
        _touch_future(repo / "config.yaml")
        notice = cv.stale_notice()
        assert notice is not None
        assert "src/core/a.py" in notice and "config.yaml" in notice
        assert "再起動" in notice

    def test_ignores_non_python_files_under_src(self, tmp_path):
        """__pycache__ 等の生成物で誤報しない（.py だけを見る）"""
        repo = _repo(tmp_path)
        cv.capture(repo=repo)
        pyc = repo / "src" / "core" / "a.cpython-311.pyc"
        pyc.write_text("x", encoding="utf-8")
        _touch_future(pyc)
        assert cv.stale_notice() is None

    def test_none_when_not_captured(self, monkeypatch):
        monkeypatch.setattr(cv, "_state", {})
        assert cv.stale_notice() is None


class TestHeartbeatShowsVersion:
    def test_message_includes_running_code_and_stale_notice(self, monkeypatch):
        monkeypatch.setattr(cv, "describe", lambda: "abc1234（起動 10/13 22:19）")
        monkeypatch.setattr(cv, "stale_notice", lambda: "⚠ 未反映の変更があります: main.py")
        msg = hb.build_message("live", {"can_place_order": True}, 2)
        assert "abc1234" in msg
        assert "未反映の変更" in msg

    def test_no_notice_line_when_up_to_date(self, monkeypatch):
        monkeypatch.setattr(cv, "describe", lambda: "abc1234（起動 10/13 22:19）")
        monkeypatch.setattr(cv, "stale_notice", lambda: None)
        msg = hb.build_message("live", {"can_place_order": True}, 2)
        assert "未反映" not in msg
