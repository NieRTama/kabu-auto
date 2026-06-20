"""
プロセス多重起動防止ロックのテスト（再レビュー P1-1）

- ロックファイルが無ければ取得できる
- 自プロセスのPIDが既に書かれていても取得できる（再入可能）
- 生きている他プロセスのPIDが書かれていれば取得失敗
- 死んでいるプロセスのPID（stale lock）は上書きして取得できる
- release() は自プロセスが書いたロックのみ削除する
"""
import os

import pytest

import src.core.process_lock as process_lock


@pytest.fixture(autouse=True)
def _reset_module_state():
    yield
    process_lock._path = None


class TestAcquire:
    def test_acquires_when_no_lock_file_exists(self, tmp_path):
        path = tmp_path / "kabu_auto.lock"
        ok, detail = process_lock.acquire(str(path))
        assert ok is True
        assert detail == ""
        assert path.read_text(encoding="utf-8").strip() == str(os.getpid())

    def test_acquires_when_lock_held_by_own_pid(self, tmp_path):
        path = tmp_path / "kabu_auto.lock"
        path.write_text(str(os.getpid()), encoding="utf-8")
        ok, _ = process_lock.acquire(str(path))
        assert ok is True

    def test_fails_when_another_running_process_holds_lock(self, tmp_path, monkeypatch):
        path = tmp_path / "kabu_auto.lock"
        other_pid = os.getpid() + 1
        path.write_text(str(other_pid), encoding="utf-8")
        monkeypatch.setattr(process_lock, "_is_process_running", lambda pid: True)
        ok, detail = process_lock.acquire(str(path))
        assert ok is False
        assert str(other_pid) in detail

    def test_overwrites_stale_lock_from_dead_process(self, tmp_path, monkeypatch):
        path = tmp_path / "kabu_auto.lock"
        dead_pid = os.getpid() + 1
        path.write_text(str(dead_pid), encoding="utf-8")
        monkeypatch.setattr(process_lock, "_is_process_running", lambda pid: False)
        ok, _ = process_lock.acquire(str(path))
        assert ok is True
        assert path.read_text(encoding="utf-8").strip() == str(os.getpid())

    def test_corrupt_lock_file_is_treated_as_stale(self, tmp_path):
        path = tmp_path / "kabu_auto.lock"
        path.write_text("not-a-pid", encoding="utf-8")
        ok, _ = process_lock.acquire(str(path))
        assert ok is True

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "kabu_auto.lock"
        ok, _ = process_lock.acquire(str(path))
        assert ok is True
        assert path.exists()


class TestRelease:
    def test_release_removes_own_lock(self, tmp_path):
        path = tmp_path / "kabu_auto.lock"
        process_lock.acquire(str(path))
        process_lock.release()
        assert not path.exists()

    def test_release_does_not_remove_others_lock(self, tmp_path, monkeypatch):
        path = tmp_path / "kabu_auto.lock"
        other_pid = os.getpid() + 1
        path.write_text(str(other_pid), encoding="utf-8")
        process_lock._path = path
        process_lock.release()
        assert path.exists()

    def test_release_without_acquire_is_noop(self):
        process_lock.release()  # raiseしない


class TestIsProcessRunning:
    def test_current_process_is_running(self):
        assert process_lock._is_process_running(os.getpid()) is True

    def test_zero_or_negative_pid_is_not_running(self):
        assert process_lock._is_process_running(0) is False
        assert process_lock._is_process_running(-1) is False
