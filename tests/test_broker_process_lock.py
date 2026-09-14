"""broker_launcher.py と broker_full_login.py が同一のロック・実行中状態を
共有していることのテスト。

## 背景（2026-09-13）

両モジュールはそれぞれ独立した threading.Lock を持っていた。両方の自動実行フラグ
（auto_launch_broker / broker_full_login_enabled）を同時に有効化すると、互いの存在を
知らない2つのロックが、それぞれ安全なつもりで同じ KabuS.exe を同時に起動/再起動
しうるという設計上の隙間があった（統合テストで発見。実害はまだ無い）。

**ロックオブジェクトを共有するだけでは解決しなかった**（最初の対応、同日）。
broker_full_login.run() は外部呼び出し（最大180秒）の間ロックを手放す設計のため、
その間に broker_launcher.launch() が同じロックを取得できてしまい、「実行中かどうか」
を判定する状態（クールダウン起点・実行中フラグ）がモジュールごとに別々のままだと
検知できず、実際に再現テストで二重起動が成立した（下記 TestActualCrossModuleExclusion
参照）。broker_process_lock.py にクールダウン起点(last_operation_at)と実行中フラグ
(in_progress)を追加し、両モジュールがそれを参照するよう変更した。
"""
import threading
import time
from unittest.mock import MagicMock, patch

from src.core import broker_full_login, broker_launcher, broker_process_lock


class TestSharedLock:
    def test_broker_launcher_uses_the_shared_lock(self):
        assert broker_launcher._lock is broker_process_lock.lock

    def test_broker_full_login_uses_the_shared_lock(self):
        assert broker_full_login._lock is broker_process_lock.lock

    def test_both_modules_share_the_same_lock_instance(self):
        """両モジュールのロックが同一オブジェクトであること（=同時に持てないこと）を直接確認する。"""
        assert broker_launcher._lock is broker_full_login._lock


class TestActualCrossModuleExclusion:
    """ロックオブジェクトの共有だけでなく、実際に一方の操作中はもう一方が
    KabuS.exeを起動できないことを、両モジュールの公開関数を実際に呼び出して確認する
    （2026-09-13、この検証方法でロック共有のみの対応では防げないことを発見した）。
    """

    def setup_method(self):
        broker_full_login.reset()
        broker_launcher.reset()

    def teardown_method(self):
        broker_full_login.reset()
        broker_launcher.reset()

    def test_launcher_is_rejected_while_full_login_is_in_flight(self):
        """full_login実行中（WSL側スクリプトがKabuS.exeをkillして再起動している
        最中を模擬）に、launcher.launch()がKabuS.exeを別途起動できてしまわないこと。"""
        launcher_result = {}
        full_login_started = threading.Event()
        release_full_login = threading.Event()

        def slow_wsl_run(*args, **kwargs):
            full_login_started.set()
            assert release_full_login.wait(timeout=5), "full_loginが解放されなかった"
            r = MagicMock()
            r.returncode = 0
            r.stdout = ""
            r.stderr = ""
            return r

        def full_login_worker():
            with patch.object(broker_full_login.subprocess, "run", side_effect=slow_wsl_run):
                broker_full_login.run(max_attempts_per_day=0)

        th = threading.Thread(target=full_login_worker)
        th.start()
        assert full_login_started.wait(timeout=5), "full_loginが開始しなかった"

        with patch.object(broker_launcher, "probe_running", return_value=False), \
             patch.object(broker_launcher.os.path, "isfile", return_value=True), \
             patch.object(broker_launcher.subprocess, "Popen") as mock_popen:
            ok, detail = broker_launcher.launch(max_attempts_per_day=0)
            launcher_result["ok"] = ok
            launcher_result["popen_called"] = mock_popen.called

        release_full_login.set()
        th.join(timeout=5)

        assert launcher_result["ok"] is False
        assert launcher_result["popen_called"] is False, (
            "full_login実行中にlauncherがKabuS.exeを別途起動できてしまった（レース再現）"
        )

    def test_full_login_respects_launcher_cooldown_immediately_after_launch(self):
        """launcher.launch()が起動した直後（プロセスがまだtasklistに現れない
        クールダウン中）に、full_login.run()がそれを知らずに追い打ちで
        再起動しないこと（逆方向のレース）。"""
        t = [1000.0]

        with patch.object(broker_launcher, "probe_running", return_value=False), \
             patch.object(broker_launcher.os.path, "isfile", return_value=True), \
             patch.object(broker_launcher.subprocess, "Popen"), \
             patch.object(broker_launcher, "_now", lambda: t[0]):
            ok1, _ = broker_launcher.launch(max_attempts_per_day=0)
        assert ok1 is True

        t[0] += 5  # 起動から5秒後（まだクールダウン中）

        with patch.object(broker_full_login, "_now", lambda: t[0]), \
             patch.object(broker_full_login.subprocess, "run") as mock_run:
            ok2, detail2 = broker_full_login.run(max_attempts_per_day=0)

        assert ok2 is False
        assert mock_run.called is False, (
            "launcherの起動クールダウン中にfull_loginが追い打ちで実行してしまった"
        )
