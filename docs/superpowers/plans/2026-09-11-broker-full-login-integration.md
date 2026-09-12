# Kabuステーション完全自動ログイン統合 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** kabu-autoの既存「2段階認証は自動化しない」方針を転換し、Kabuステーションの起動〜ログイン〜2段階認証入力までを、既存スケジューラ（平日06:45）と既存Discordコマンド経路の両方から呼べる形で完全自動化する。

**Architecture:** 新モジュール `src/core/broker_full_login.py` が、WSL側の完全自動ログインスクリプト（`kabusapi-auto-login-template/scripts/kabustation/run_login_only.sh`、本計画のTask 4で新規作成）を `subprocess.run(..., timeout=...)` で同期実行する薄いラッパーとして働く。既存の `broker_launcher.py`（プロセス起動のみ）は無改修。`scheduler.py` と `main.py` の両方から呼べるようにし、フラグ既定offで段階導入する。

**Tech Stack:** Python 3 / pytest / APScheduler / WSL2 (Ubuntu) + Docker Engine（環境構築済み） / PowerShell（SendKeys、WSL外側で実行）

**Spec:** [docs/superpowers/specs/2026-09-11-broker-full-login-integration-design.md](../specs/2026-09-11-broker-full-login-integration-design.md)

## Global Constraints

- 既存 `src/core/broker_launcher.py` は変更しない（責務分離。`tests/test_broker_launcher.py::TestWiring::test_auth_is_not_automated` が同モジュールへの認証自動化混入を回帰検知しており、これは新モジュールには適用されないことを利用する）
- 新モジュールはパスワード等の秘密情報を一切扱わない（WSL側スクリプトに完全委譲する）
- 新機能は `runtime.broker_full_login_enabled`（既定 `false`）配下に置き、既存の自動実行経路（`auto_launch_broker` 等）には触れない
- 外部コマンド（`wsl` 呼び出し）は必ずモックしてテストする。実際のWSL/Kabuステーション操作を伴う統合確認はユーザー立ち会いで別途行う（本計画のタスクには含めない）
- 既存コード同様、日本語のdocstring・コメントで「なぜ」を残す

---

### Task 1: `broker_full_login.py` コアロジック

**Files:**
- Create: `src/core/broker_full_login.py`
- Test: `tests/test_broker_full_login.py`

**Interfaces:**
- Produces:
  - `reset() -> None`
  - `attempts_today() -> int`
  - `run(*, manual: bool = False, max_attempts_per_day: int = DEFAULT_MAX_ATTEMPTS_PER_DAY, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS, wsl_distro: str = DEFAULT_WSL_DISTRO, project_dir: str = DEFAULT_PROJECT_DIR, script_path: str = DEFAULT_SCRIPT_PATH) -> tuple[bool, str]`
  - `DEFAULT_MAX_ATTEMPTS_PER_DAY: int = 3`
  - `DEFAULT_TIMEOUT_SECONDS: int = 180`
  - `DEFAULT_WSL_DISTRO: str = "Ubuntu"`
  - `DEFAULT_PROJECT_DIR: str = "~/projects/kabusapi-auto-login-template"`
  - `DEFAULT_SCRIPT_PATH: str = "scripts/kabustation/run_login_only.sh"`
  - `RERUN_COOLDOWN_SECONDS: int = 60`

- [ ] **Step 1: テストファイルを作成し、最初の失敗するテストを書く**

`tests/test_broker_full_login.py`:

```python
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
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `pytest tests/test_broker_full_login.py -v`
Expected: FAIL（`src.core.broker_full_login` モジュールが存在しない、`ModuleNotFoundError`）

- [ ] **Step 3: 最小実装を書く**

`src/core/broker_full_login.py`:

```python
"""kabuステーション（KabuS.exe）の完全自動ログイン制御。

## 背景

kabu-auto は従来、Kabuステーションの起動のみを自動化し、2段階認証
（ワンタイムパスワード入力）は意図的に自動化していなかった
（`broker_launcher.py` 参照。「認証は自動化しない、専用認証アプリでの承認は
人が行う」という方針）。

2026-09-11、Gmail API経由でワンタイムパスワードを自動取得・自動入力する
外部リポジトリ（kabusapi-auto-login-template、WSL2/Docker側に導入済み）を
統合し、起動〜ログイン〜2段階認証入力までの完全自動化に方針転換した。

## 責務

このモジュールは、WSL側の完全自動ログインスクリプト
（kabusapi-auto-login-template/scripts/kabustation/run_login_only.sh）を
呼び出すことだけを担う。ログイン自体のロジック（パスワード入力、Gmail API
連携、SendKeys等）はすべてWSL側スクリプトに任せる（本モジュールは
パスワード等の秘密情報を一切扱わない）。

起動後にAPI接続できたかの確認は、既存の broker_wait / auth_recovery が
行う（責務分離。broker_launcher.py と同じ設計思想）。

同時実行・連続実行を避けるため、broker_launcher.py と同様のクールダウン・
日次試行回数上限を持つ。
"""
import subprocess
import threading
import time
from datetime import date
from typing import Optional

from loguru import logger

DEFAULT_WSL_DISTRO = "Ubuntu"
DEFAULT_PROJECT_DIR = "~/projects/kabusapi-auto-login-template"
DEFAULT_SCRIPT_PATH = "scripts/kabustation/run_login_only.sh"
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_MAX_ATTEMPTS_PER_DAY = 3

# 直前の実行からこの秒数以内の再実行を抑止する（broker_launcher.py と同じ考え方。
# WSL側スクリプトはKabuS.exeを一旦killしてから再起動するため、近接した2回目が
# 走ると起動直後のプロセスをまた落とすことになる）。
RERUN_COOLDOWN_SECONDS = 60

_lock = threading.Lock()
_attempts: int = 0
_attempts_date: Optional[date] = None
_last_run_at: float = 0.0


def _today() -> date:
    from src.core import clock
    return clock.today()


def _now() -> float:
    """クールダウン判定用の時刻（単調増加。テストで差し替える）。"""
    return time.monotonic()


def reset() -> None:
    """テスト用に試行回数とクールダウンを初期化する。"""
    global _attempts, _attempts_date, _last_run_at
    with _lock:
        _attempts = 0
        _attempts_date = None
        _last_run_at = 0.0


def attempts_today() -> int:
    with _lock:
        _roll_over_if_new_day()
        return _attempts


def _roll_over_if_new_day() -> None:
    """呼び出し側で _lock を保持している前提。"""
    global _attempts, _attempts_date
    today = _today()
    if _attempts_date != today:
        _attempts_date = today
        _attempts = 0


def run(*, manual: bool = False,
        max_attempts_per_day: int = DEFAULT_MAX_ATTEMPTS_PER_DAY,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        wsl_distro: str = DEFAULT_WSL_DISTRO,
        project_dir: str = DEFAULT_PROJECT_DIR,
        script_path: str = DEFAULT_SCRIPT_PATH) -> tuple[bool, str]:
    """Kabuステーションの起動〜ログイン〜2段階認証入力までを完全自動化する。

    (実行できたか, 説明) を返す。WSL側スクリプトの完了まで**同期的にブロックする**
    （起動確認は tasklist で済む broker_launcher.launch() と異なり、ログイン完了
    までを1回の呼び出しで待つ設計。呼び出し元は数十秒〜数分のブロックを想定する）。

    manual=True は人が明示的に頼んだ実行（Discordコマンド）。broker_launcher.launch()
    と同じく、日次上限では止めない。
    """
    global _attempts, _last_run_at

    with _lock:
        elapsed = _now() - _last_run_at
        if _last_run_at and elapsed < RERUN_COOLDOWN_SECONDS:
            return False, (
                f"直前に実行しています（{int(elapsed)}秒前）。"
                f"{RERUN_COOLDOWN_SECONDS}秒は再実行しません"
            )

        _roll_over_if_new_day()
        if manual:
            attempt_no = None
        else:
            if max_attempts_per_day > 0 and _attempts >= max_attempts_per_day:
                return False, (
                    f"本日の実行回数が上限({max_attempts_per_day}回)に達しています。"
                    "証券会社側の障害の可能性があるため、手動で確認してください"
                )
            _attempts += 1
            attempt_no = _attempts

        command = [
            "wsl", "-d", wsl_distro, "--", "bash", "-lc",
            f"cd {project_dir} && ./{script_path}",
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            _last_run_at = _now()
            logger.error(f"完全自動ログインがタイムアウトしました（{timeout_seconds}秒）")
            return False, f"完全自動ログインがタイムアウトしました（{timeout_seconds}秒）"
        except Exception as e:
            _last_run_at = _now()
            logger.error(f"完全自動ログインの実行に失敗しました: {e}")
            return False, f"実行に失敗しました: {e}"

        _last_run_at = _now()

        if result.returncode != 0:
            logger.error(
                f"完全自動ログインが失敗しました（rc={result.returncode}）: "
                f"{result.stderr.strip()[:500]}"
            )
            return False, (
                f"完全自動ログインに失敗しました（rc={result.returncode}）: "
                f"{result.stderr.strip()[:300]}"
            )

    how = "手動" if attempt_no is None else f"本日{attempt_no}回目"
    logger.warning(f"完全自動ログインを実行しました（{how}）")
    return True, f"完全自動ログインを実行しました（{how}）"
```

- [ ] **Step 4: テストを実行して成功を確認する**

Run: `pytest tests/test_broker_full_login.py -v`
Expected: PASS

- [ ] **Step 5: 残りのテストケースを追加する**

`tests/test_broker_full_login.py` の `TestSuccessfulRun` クラスの下に追記:

```python
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

    def test_timeout_is_reported_not_raised(self):
        ok, detail, _ = _run(raise_timeout=True)
        assert ok is False
        assert "タイムアウト" in detail

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

        with patch.object(bfl, "_now", lambda: float(next(_SHARED_CLOCK))), \
             patch.object(bfl.subprocess, "run", side_effect=slow_run) as mock_run:
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=10)

        assert mock_run.call_count == 1, f"{mock_run.call_count}回実行された"
```

- [ ] **Step 6: 全テストを実行して成功を確認する**

Run: `pytest tests/test_broker_full_login.py -v`
Expected: PASS（全ケース）

- [ ] **Step 7: コミット**

```bash
git add src/core/broker_full_login.py tests/test_broker_full_login.py
git commit -m "feat(core): Kabuステーション完全自動ログイン制御モジュールを追加

WSL側の完全自動ログインスクリプト（kabusapi-auto-login-template）を
呼び出す薄いラッパー。broker_launcher.py（プロセス起動のみ）とは別モジュールとし、
既存の「認証は自動化しない」責務を保ったまま新責務を分離した。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `scheduler.py` へのcronジョブ追加

**Files:**
- Modify: `src/core/scheduler.py`
- Test: `tests/test_broker_full_login.py`（追記）

**Interfaces:**
- Consumes: `TradingScheduler.register(name: str, callback) -> None`（既存）
- Produces: ジョブID `"broker_full_login"`（cron, 平日06:45）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_broker_full_login.py` の末尾に追記:

```python
import src.core.scheduler as scheduler_mod


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
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `pytest tests/test_broker_full_login.py::TestSchedulerWiring -v`
Expected: FAIL（`test_broker_full_login_registered_as_cron_job` が `"broker_full_login" in calls` で失敗）

- [ ] **Step 3: `scheduler.py` にジョブ登録を追加する**

`src/core/scheduler.py` の `start()` メソッド内、`if "risk_reset" in cb:` ブロックの直前に挿入:

```python
        if "broker_full_login" in cb:
            # Kabuステーションの起動〜ログイン〜2段階認証入力までを完全自動化する
            # （2026-09-11、broker_full_login.py参照。従来「認証は自動化しない」
            # 方針だったが、Gmail API経由のワンタイムパスワード自動入力へ転換した）。
            # 平日06:45（risk_reset(8:25)より前）に実行し、完了後は既存の
            # wait_for_broker_minutes（無制限待機）がAPI接続を引き継ぐ。
            self._scheduler.add_job(
                cb["broker_full_login"], "cron",
                day_of_week="mon-fri", hour=6, minute=45, id="broker_full_login",
            )
        if "risk_reset" in cb:
```

（既存の `if "risk_reset" in cb:` 行はそのまま残し、その手前に新ブロックを挿入する形にする）

- [ ] **Step 4: テストを実行して成功を確認する**

Run: `pytest tests/test_broker_full_login.py -v`
Expected: PASS（全ケース）

- [ ] **Step 5: モジュール冒頭のdocstringにジョブ一覧を追記する**

`src/core/scheduler.py` 冒頭のdocstringに1行追加:

```python
"""
APSchedulerによるジョブスケジューラ
- 毎朝6:45: kabuステーション完全自動ログイン（broker_full_login_enabled=trueのときのみ実働）
- 毎朝8:25: 日次リスクカウンタリセット
```

- [ ] **Step 6: 全テストを実行する**

Run: `pytest tests/test_broker_full_login.py tests/test_reconcile_scheduling.py -v`
Expected: PASS（既存のスケジューラテストも壊れていないこと）

- [ ] **Step 7: コミット**

```bash
git add src/core/scheduler.py tests/test_broker_full_login.py
git commit -m "feat(core): 完全自動ログインをスケジューラへ登録（平日06:45）

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: `main.py` への統合（config・ラッパー関数・Discordコマンド・登録）

**Files:**
- Modify: `main.py`
- Modify: `config.yaml`
- Test: `tests/test_broker_full_login.py`（追記）

**Interfaces:**
- Consumes: `broker_full_login.run(...)`（Task 1）, `cfg.get_section("runtime")`（既存）, `alert(title, message, level)`（既存）, `market_calendar.is_holiday`, `clock.today()`（既存）
- Produces: `main.py` 内の `_full_login_broker(*, manual=False)`, `broker_full_login_job()`, `_cmd_full_login(_args)`。Discordコマンド `"full_login"`。スケジューラ登録 `scheduler.register("broker_full_login", broker_full_login_job)`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_broker_full_login.py` の末尾に追記:

```python
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
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `pytest tests/test_broker_full_login.py::TestMainWiring -v`
Expected: FAIL（4件とも、`main.py` にまだ何も無いため）

- [ ] **Step 3: `config.yaml` に設定項目を追加する**

`config.yaml` の `runtime` セクション、`max_launch_attempts_per_day: 3` の直後に追記:

```yaml
  # ─── 完全自動ログイン（2026-09-11〜）─────────────────────
  # 上記の「認証（2段階認証）は自動化しない」という方針を転換し、Gmail API経由の
  # ワンタイムパスワード自動取得・自動入力（kabusapi-auto-login-template、
  # WSL2/Docker側 ~/projects/kabusapi-auto-login-template に導入済み）を使って
  # Kabuステーションの起動〜ログイン〜2段階認証入力までを完全自動化できるように
  # した。false の間は broker_full_login は一切動かず、従来通り人がログインする。
  # 段階導入のため既定は false。実運用で安全性を確認してから true にする。
  broker_full_login_enabled: false
  # WSL側スクリプトのタイムアウト秒数。ハングしても親プロセス(kabu-auto本体)を
  # 巻き込まない。
  broker_full_login_timeout_seconds: 180
  # 1日の完全自動ログイン試行上限。broker起動と同じ考え方で、無意味な
  # リトライの暴走を防ぐ。**自動実行(平日06:45)にだけ効く**。Discordの
  # full_login（人の明示操作）は上限で止めない。
  broker_full_login_max_attempts_per_day: 3
```

- [ ] **Step 4: `main.py` にimportとラッパー関数を追加する**

`main.py` のimport文（`from src.core import (` ブロック）に `broker_full_login` を追加:

```python
from src.core import (
    config as cfg, logger as log_setup, watchlist as watchlist_store,
    risk_profile as risk_profile_store, halt as halt_store, trading_mode as tm,
    process_lock, reference_capital as reference_capital_store, broker_wait, broker_auth,
    discord_bot, auth_recovery, broker_launcher, broker_full_login, broker_watch, discord_queries,
    discord_slash,
    market_calendar, clock,
)
```

`_launch_broker` 関数（231行目付近）の直後に追記:

```python
    def _full_login_broker(*, manual: bool = False) -> tuple[bool, str]:
        """kabuステーションの起動〜ログイン〜2段階認証入力までを完全自動化する
        （設定を読んで broker_full_login へ委譲）。

        manual=True は人が明示的に頼んだ実行（Discord の full_login）。
        broker_launcher.launch() の manual と同じ考え方で、日次上限では止めない。
        """
        rt = cfg.get_section("runtime")
        return broker_full_login.run(
            manual=manual,
            max_attempts_per_day=int(
                rt.get("broker_full_login_max_attempts_per_day",
                       broker_full_login.DEFAULT_MAX_ATTEMPTS_PER_DAY)
            ),
            timeout_seconds=int(
                rt.get("broker_full_login_timeout_seconds",
                       broker_full_login.DEFAULT_TIMEOUT_SECONDS)
            ),
        )

    def broker_full_login_job():
        """スケジューラから呼ばれる完全自動ログインジョブ（平日06:45）。

        broker_full_login_enabled が false の間は何もしない（段階導入のフラグ）。
        休場日は認証が切れていても異常ではないため実行しない
        （token_refreshと同じ二重ガードの型。2026-09-05に休場日誤発報の実例あり）。
        """
        rt = cfg.get_section("runtime")
        if not bool(rt.get("broker_full_login_enabled", False)):
            return
        if market_calendar.is_holiday(clock.today()):
            logger.info(
                f"完全自動ログイン省略: 本日は休場です（{market_calendar.holiday_name(clock.today())}）"
            )
            return
        ok, detail = _full_login_broker()
        if ok:
            alert(
                "kabuステーションの完全自動ログインを実行しました",
                detail,
                level=alerts_mod.LEVEL_INFO,
            )
        else:
            alert(
                "kabuステーションの完全自動ログインに失敗しました",
                f"{detail}\n手動でログインしてください。",
            )
```

- [ ] **Step 5: Discordコマンドを追加する**

`_cmd_launch` 関数（409行目付近）の直後に追記:

```python
    def _cmd_full_login(_args: str) -> str:
        """kabuステーションの起動〜ログイン〜2段階認証入力まで完全自動で行う。"""
        ok, detail = _full_login_broker(manual=True)
        return detail
```

コマンド辞書 `_discord_handlers`（454行目付近）の `"launch"` の直後に追記:

```python
        "launch": (_cmd_launch, "kabuステーションを起動する（認証は手動）"),
        "full_login": (_cmd_full_login, "kabuステーションの起動〜ログイン〜2段階認証入力まで完全自動化する"),
```

- [ ] **Step 6: スケジューラへの登録を追加する**

`scheduler.register("risk_reset", risk.reset_daily_counters)` の直前に追記:

```python
    scheduler.register("broker_full_login", broker_full_login_job)
    scheduler.register("risk_reset", risk.reset_daily_counters)
```

- [ ] **Step 7: テストを実行して成功を確認する**

Run: `pytest tests/test_broker_full_login.py -v`
Expected: PASS（全ケース）

- [ ] **Step 8: 既存テストスイート全体を実行し、回帰がないことを確認する**

Run: `pytest -q`
Expected: PASS（既存のテストが壊れていないこと。特に `tests/test_broker_launcher.py::TestWiring::test_auth_is_not_automated` が引き続きPASSすること＝新モジュールは`broker_launcher.py`に影響していない）

- [ ] **Step 9: コミット**

```bash
git add main.py config.yaml tests/test_broker_full_login.py
git commit -m "feat: 完全自動ログインをconfig・Discordコマンド・main.pyへ結線

runtime.broker_full_login_enabled（既定false）配下で段階導入。
Discordの full_login コマンドは既定に関わらず手動実行できる
（既存 launch コマンドと同じ設計）。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: WSL側テンプレートに `run_login_only.sh` を追加

**Files:**
- Create (WSL内、kabu-autoのgit管理外): `~/projects/kabusapi-auto-login-template/scripts/kabustation/run_login_only.sh`

**Interfaces:**
- Consumes: 既存の `scripts/kabustation/login_kabustation.ps1`, `scripts/onetime_password/get_onetime_password.py`, `scripts/onetime_password/input_onetime_password.ps1`（すべてテンプレート導入済み）
- Produces: `broker_full_login.py` の `DEFAULT_SCRIPT_PATH` が指す実行可能スクリプト

- [ ] **Step 1: スクリプトを作成する**

WSL（Ubuntu）内で以下を実行し、`~/projects/kabusapi-auto-login-template/scripts/kabustation/run_login_only.sh` を作成する:

```bash
#!/bin/bash
# kabuステーションの起動〜ログイン〜2段階認証入力を行う（kabu-proxy抜き版）。
#
# kabu-auto本体（Windows側で直接動くPython）はKabuステーションAPIへ
# localhost:18080 で直接接続するため、WSLからの中継用プロキシ
# （kabu-proxyコンテナ・Windows側nginx）は不要。元の run_kabustation_api.sh から
# それらの起動を除いたもの（2026-09-11 kabu-auto統合時に追加）。
set -e

docker compose -f docker/docker-compose.yml up --build onetime_password -d

powershell.exe -ExecutionPolicy Bypass ./scripts/kabustation/login_kabustation.ps1

ONETIME_PASSWORD=$(docker compose -f docker/docker-compose.yml exec onetime_password python ./scripts/onetime_password/get_onetime_password.py)
echo "$ONETIME_PASSWORD" | powershell.exe -ExecutionPolicy Bypass ./scripts/onetime_password/input_onetime_password.ps1
```

- [ ] **Step 2: 実行権限を付与し、構文チェックする**

```bash
chmod +x ~/projects/kabusapi-auto-login-template/scripts/kabustation/run_login_only.sh
bash -n ~/projects/kabusapi-auto-login-template/scripts/kabustation/run_login_only.sh
```

Expected: `bash -n` が何も出力せず終了コード0（構文エラーなし）

- [ ] **Step 3: テンプレートリポジトリ側でコミットする（任意・推奨）**

```bash
cd ~/projects/kabusapi-auto-login-template
git add scripts/kabustation/run_login_only.sh
git commit -m "kabu-auto統合用: kabu-proxyを起動しないログイン専用スクリプトを追加"
```

（このリポジトリはユーザー個人の導入であり、kabu-autoのGit管理下ではない。ローカルコミットのみで良く、フォーク元へのプッシュは不要）

---

### Task 5: ドキュメント更新

**Files:**
- Modify: `docs/詳細設計書.md`
- Modify: `docs/概要設計書.md`

**Interfaces:**
- Consumes: Task 1〜4で確定した設計（モジュール名・設定キー・スケジュール時刻）

- [ ] **Step 1: `docs/詳細設計書.md` の既存 `runtime` 設定説明箇所を確認する**

`auto_launch_broker` や `broker_exe_path` の説明がある節を探し、その直後に「完全自動ログイン（broker_full_login）」の節を追記する。内容:
- 背景（2段階認証自動化への方針転換の経緯、2026-09-11）
- `broker_full_login_enabled` / `broker_full_login_timeout_seconds` / `broker_full_login_max_attempts_per_day` の説明
- WSL2/Docker側の `kabusapi-auto-login-template` との連携図（起動元は `~/projects/kabusapi-auto-login-template`）
- Discordコマンド `full_login` の説明

- [ ] **Step 2: `docs/概要設計書.md` に一言サマリを追記する**

「Kabuステーション連携」または同等の節に、完全自動ログイン機能が追加されたことを1〜2文で追記する。

- [ ] **Step 3: Obsidian vaultへ同期する**

[[obsidian_vault_sync]] の取り決めに従い、更新した `README.md` / `詳細設計書.md` / `概要設計書.md` を `Tama vault/Claude/` へコピーする（次回のGitHub push時に実施、と明記されているならそのタイミングで良い）。

- [ ] **Step 4: コミット**

```bash
git add docs/詳細設計書.md docs/概要設計書.md
git commit -m "docs: 完全自動ログイン統合をdesign docへ反映

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## 実装後、ユーザー立ち会いで行うこと（本計画のスコープ外）

- `config.yaml` の `broker_full_login_enabled` を一度手動で `true` にし、Discordの `full_login` コマンドで実際にWSL経由のログインが通ることを確認する（パスワード・Gmail API認証情報が必要なため自動テスト化できない）
- 実運用で数日、`health_check`（15分毎エラー率）・`auth_recovery_check`（5分間隔）が完全自動ログインの再起動時間帯をノイズとして拾っていないか確認する
- 問題なければ平日06:45の自動実行（`broker_full_login_enabled: true`）へ切り替える
- **平日06:45の無人自動実行に移行する前に**、kabu-auto本体を起動しているタスクスケジューラのタスクが「ログオン時のみ実行（対話セッションあり）」になっていることを確認する。WSL側 `login_kabustation.ps1` の SendKeys は対話デスクトップを必要とするため、非対話（サービス相当）実行だと動作しない（最終レビュー Recommendation #4 より）
