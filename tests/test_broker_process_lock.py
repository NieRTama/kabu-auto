"""broker_launcher.py と broker_full_login.py が同一のロックを共有していることのテスト。

## 背景（2026-09-13）

両モジュールはそれぞれ独立した threading.Lock を持っていた。両方の自動実行フラグ
（auto_launch_broker / broker_full_login_enabled）を同時に有効化すると、互いの存在を
知らない2つのロックが、それぞれ安全なつもりで同じ KabuS.exe を同時に起動/再起動
しうるという設計上の隙間があった（統合テストで発見。実害はまだ無い）。

この回帰を防ぐため、両モジュールが同一のロックオブジェクトを参照していることを固定する。
"""
from src.core import broker_full_login, broker_launcher, broker_process_lock


class TestSharedLock:
    def test_broker_launcher_uses_the_shared_lock(self):
        assert broker_launcher._lock is broker_process_lock.lock

    def test_broker_full_login_uses_the_shared_lock(self):
        assert broker_full_login._lock is broker_process_lock.lock

    def test_both_modules_share_the_same_lock_instance(self):
        """両モジュールのロックが同一オブジェクトであること（=同時に持てないこと）を直接確認する。"""
        assert broker_launcher._lock is broker_full_login._lock
