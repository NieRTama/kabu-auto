"""kabuステーションの完全自動ログイン制御（broker_full_login）のテスト

2026-09-11: 「認証は自動化しない」という既存方針（broker_launcher.py参照）を
転換し、Gmail API経由のワンタイムパスワード自動取得・自動入力
（kabusapi-auto-login-template、WSL2/Docker側に導入）を使って
起動〜ログイン〜2段階認証入力までを完全自動化する。

実際にWSLコマンドを実行してしまわないよう、subprocess は必ずモックする。
"""
import itertools
import threading
import time
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core import broker_full_login as bfl
import src.core.scheduler as scheduler_mod


_SHARED_CLOCK = itertools.count(0, 10_000)


@pytest.fixture(autouse=True)
def _reset():
    bfl.reset()
    yield
    bfl.reset()


def _run(*, returncode=0, stdout="", stderr="", raise_timeout=False,
          raise_error=None, max_attempts=3, now=None, manual=False,
          timeout_seconds=180):
    """run() を安全に実行する（実際のwslコマンドは呼ばない）。"""
    clock = now or (lambda: float(next(_SHARED_CLOCK)))
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr

    def fake_run(*args, **kwargs):
        if raise_timeout:
            raise bfl.subprocess.TimeoutExpired(
                cmd=args[0] if args else "wsl", timeout=kwargs.get("timeout", 0)
            )
        if raise_error:
            raise raise_error
        return result

    mock_run = MagicMock(side_effect=fake_run)
    with patch.object(bfl, "_now", clock), \
         patch.object(bfl.subprocess, "run", mock_run):
        ok, detail = bfl.run(manual=manual, max_attempts_per_day=max_attempts,
                              timeout_seconds=timeout_seconds)
    return ok, detail, mock_run


class TestSuccessfulRun:
    def test_runs_and_counts(self):
        ok, detail, mock_run = _run()
        assert ok is True
        mock_run.assert_called_once()
        assert bfl.attempts_today() == 1

    def test_builds_wsl_command_with_defaults(self):
        _, _, mock_run = _run()
        command = mock_run.call_args[0][0]
        assert command[0] == "wsl"
        assert "-d" in command and "Ubuntu" in command
        joined = " ".join(command)
        assert "kabusapi-auto-login-template" in joined
        assert "run_login_only.sh" in joined

    def test_uses_given_timeout(self):
        _, _, mock_run = _run(timeout_seconds=42)
        assert mock_run.call_args.kwargs["timeout"] == 42


class TestFailure:
    def test_nonzero_returncode_is_reported_not_raised(self):
        ok, detail, _ = _run(returncode=1, stderr="ログイン画面が見つかりません")
        assert ok is False
        assert "失敗" in detail

    def test_nonzero_returncode_falls_back_to_stdout_when_stderr_is_empty(self):
        """WSL側スクリプトは set -e のため診断出力の多くが stdout に出る。
        stderr が空でも失敗理由が失われないこと（Fix 3の回帰防止）。"""
        ok, detail, _ = _run(
            returncode=1, stderr="", stdout="docker compose up --build に失敗しました"
        )
        assert ok is False
        assert "docker compose up --build に失敗しました" in detail

    def test_timeout_is_reported_not_raised(self):
        ok, detail, _ = _run(raise_timeout=True)
        assert ok is False
        assert "タイムアウト" in detail

    def test_timeout_message_warns_about_lingering_wsl_state(self):
        """タイムアウトはWSL側プロセスの停止を保証しないため、手動ログイン前に
        既存試行の残存を確認するよう促す文言を含むこと（Fix 4）。"""
        ok, detail, _ = _run(raise_timeout=True)
        assert ok is False
        assert "WSL" in detail
        assert "確認" in detail

    def test_unexpected_exception_is_reported_not_raised(self):
        ok, detail, _ = _run(raise_error=OSError("wsl.exe not found"))
        assert ok is False
        assert "失敗" in detail

    def test_failure_still_counts_as_an_attempt(self):
        """失敗しても試行回数は消費する（無限リトライで暴走しないよう上限に近づく）"""
        _run(returncode=1)
        assert bfl.attempts_today() == 1


class TestDailyLimit:
    def test_stops_after_limit(self):
        for _ in range(3):
            assert _run(max_attempts=3)[0] is True
        ok, detail, mock_run = _run(max_attempts=3)
        assert ok is False
        assert "上限" in detail
        mock_run.assert_not_called()

    def test_zero_means_unlimited(self):
        for _ in range(5):
            assert _run(max_attempts=0)[0] is True
        assert bfl.attempts_today() == 5

    def test_counter_resets_on_new_day(self):
        for _ in range(3):
            _run(max_attempts=3)
        assert _run(max_attempts=3)[0] is False
        with patch.object(bfl, "_today", return_value=date(2099, 1, 1)):
            assert _run(max_attempts=3)[0] is True


class TestManualBypassesDailyLimit:
    def test_manual_does_not_consume_the_automatic_budget(self):
        for _ in range(5):
            _run(manual=True, max_attempts=3)
        assert bfl.attempts_today() == 0

    def test_manual_still_proceeds_past_automatic_limit(self):
        for _ in range(3):
            _run(max_attempts=3)  # 自動枠を使い切る
        ok, _, mock_run = _run(manual=True, max_attempts=3)
        assert ok is True
        mock_run.assert_called_once()


class TestRerunCooldown:
    """直後の再実行を抑止すること（WSL側スクリプトはKabuS.exeをkillしてから
    再起動するため、近接した2回目が起動直後のプロセスをまた落とすのを防ぐ）。"""

    def test_second_run_right_after_is_suppressed(self):
        t = [1000.0]
        ok1, _, _ = _run(now=lambda: t[0])
        assert ok1 is True
        t[0] += 5
        ok2, detail, mock_run2 = _run(now=lambda: t[0])
        assert ok2 is False
        assert "直前に実行" in detail
        mock_run2.assert_not_called()

    def test_run_allowed_again_after_cooldown(self):
        t = [1000.0]
        _run(now=lambda: t[0])
        t[0] += bfl.RERUN_COOLDOWN_SECONDS + 1
        ok, _, mock_run = _run(now=lambda: t[0])
        assert ok is True
        mock_run.assert_called_once()

    def test_cooldown_does_not_consume_an_attempt(self):
        t = [1000.0]
        _run(now=lambda: t[0])
        t[0] += 5
        _run(now=lambda: t[0])
        assert bfl.attempts_today() == 1


class TestConcurrentRun:
    """同時に呼ばれても1つしか実行しないこと（クールダウンにより2つ目が抑止される）。"""

    def test_only_one_execution_proceeds(self):
        started = []

        def slow_run(*args, **kwargs):
            time.sleep(0.15)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        def worker():
            ok, _ = bfl.run(max_attempts_per_day=0)
            started.append(ok)

        with patch.object(bfl.subprocess, "run", side_effect=slow_run) as mock_run:
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=10)

        assert mock_run.call_count == 1, f"{mock_run.call_count}回実行された"


class TestCooldownAnchorsToCompletionTime:
    """クールダウンは実行開始時刻ではなく完了時刻を基準にすること（Fix 2a）。

    タイムアウト上限いっぱいまでかかった実行の直後は、経過時間の見かけ上
    クールダウンをすり抜けてはならない（そこはWSL側スクリプトが直前に
    KabuS.exeをkillし直したばかりの、最も再実行を避けたいタイミングのため）。
    """

    def test_long_run_still_cools_down_immediately_after_completion(self):
        t = [1000.0]

        def clock():
            return t[0]

        def slow_run(*args, **kwargs):
            # 実行に180秒（タイムアウト上限相当）かかったことを模す。
            # 開始時刻を基準にする旧実装なら、次の呼び出しの経過時間は
            # 見かけ上180秒経過したことになりクールダウンをすり抜ける。
            t[0] += 180
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch.object(bfl, "_now", clock), \
             patch.object(bfl.subprocess, "run", side_effect=slow_run):
            ok1, _ = bfl.run(max_attempts_per_day=0)
        assert ok1 is True

        with patch.object(bfl, "_now", clock), \
             patch.object(bfl.subprocess, "run", side_effect=slow_run) as mock_run2:
            ok2, detail2 = bfl.run(max_attempts_per_day=0)

        assert ok2 is False
        assert "直前に実行" in detail2
        mock_run2.assert_not_called()


class TestLockNotHeldDuringSubprocess:
    """_lock は subprocess.run() の実行中は保持しないこと（Fix 2b）。

    Discordリモコンのポーリング等がその間ブロックされ続けると、
    緊急停止(halt)経路まで最大180秒止まりうる安全上の懸念があるため。
    """

    def test_attempts_today_does_not_block_while_run_is_in_flight(self):
        def slow_run(*args, **kwargs):
            time.sleep(0.3)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        def worker():
            bfl.run(max_attempts_per_day=0)

        with patch.object(bfl.subprocess, "run", side_effect=slow_run):
            th = threading.Thread(target=worker)
            th.start()
            time.sleep(0.05)  # worker が subprocess.run 内でブロック中のはず
            start = time.perf_counter()
            bfl.attempts_today()
            elapsed = time.perf_counter() - start
            th.join(timeout=5)

        assert elapsed < 0.2, (
            f"attempts_today() が {elapsed:.3f}秒ブロックされた"
            "（_lockがsubprocess実行中も保持されている疑い）"
        )


class TestSchedulerWiring:
    def test_broker_full_login_registered_as_cron_job(self):
        sched = scheduler_mod.TradingScheduler()
        sched.register("broker_full_login", MagicMock())
        with patch.object(sched._scheduler, "add_job") as mock_add_job, \
             patch.object(sched._scheduler, "start"):
            sched.start()
        calls = {c.kwargs["id"]: c for c in mock_add_job.call_args_list}
        assert "broker_full_login" in calls
        call = calls["broker_full_login"]
        assert call.args[1] == "cron"
        assert call.kwargs.get("day_of_week") == "mon-fri"
        assert call.kwargs.get("hour") == 6
        assert call.kwargs.get("minute") == 45

    def test_omitted_when_not_registered(self):
        sched = scheduler_mod.TradingScheduler()
        with patch.object(sched._scheduler, "add_job") as mock_add_job, \
             patch.object(sched._scheduler, "start"):
            sched.start()
        ids = {c.kwargs["id"] for c in mock_add_job.call_args_list}
        assert "broker_full_login" not in ids


MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


def _main_src() -> str:
    """main.py の中身。相対パスで開くとルート以外からの pytest で落ちる。"""
    return MAIN_PY.read_text(encoding="utf-8")


class TestMainWiring:
    def test_discord_full_login_command_registered(self):
        import re
        assert re.search(r'"full_login":\s*\(?_cmd_full_login', _main_src()), (
            "full_login コマンドが登録されていない"
        )

    def test_scheduler_job_registered(self):
        assert 'scheduler.register("broker_full_login"' in _main_src()

    def test_holiday_is_skipped(self):
        """休場日に誤発報しないこと（token_refreshと同じ二重ガードの型）。"""
        i = _main_src().index("def broker_full_login_job")
        body = _main_src()[i:i + 800]
        assert "market_calendar.is_holiday" in body

    def test_disabled_by_default_flag_is_checked(self):
        i = _main_src().index("def broker_full_login_job")
        body = _main_src()[i:i + 800]
        assert "broker_full_login_enabled" in body
