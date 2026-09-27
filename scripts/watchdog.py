"""kabu-auto 死活監視（watchdog）。

背景（2026-09-27）: 本体はタスクスケジューラ \\kabu\\kabu-auto がログオン時に1回だけ
起動する。ログオン中に誤ってコンソールを閉じる等で本体が落ちても、トリガーが
再発火しないため再起動されず、気づかれないまま停止し続けた実例（約1時間）がある。
このスクリプトを5分おきのタスク \\kabu\\kabu-auto-watchdog から実行し、
落ちていればタスクを再実行し、Discordへ通知する。

生死判定は「data/kabu_auto.lock が存在し、そのPIDが生きているか」で行う
（process_lock._is_process_running を再利用）。ポート8080の待受で判定しない理由:
ブローカー接続待ち中（起動直後・週末）や、手動起動（switch_mode.ps1）はポートが
開く前でも正常な状態であり、それを「死んでいる」と誤検知して再起動を連打しないため。

再起動は30分に1回までにレート制限する（data/watchdog_state.json）。レート制限中に
まだ死んでいる場合は「復旧しない」通知を1回だけ出す（再起動のたびに1回のみ）。
data/watchdog.pause が存在する間は何もしない（保守作業等で意図的に止めている間の
誤爆防止）。
"""
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # scripts/ から src.core.* をimportできるようにする

LOCK_PATH = REPO_ROOT / "data" / "kabu_auto.lock"
PAUSE_PATH = REPO_ROOT / "data" / "watchdog.pause"
STATE_PATH = REPO_ROOT / "data" / "watchdog_state.json"
TASK_NAME = r"\kabu\kabu-auto"
RESTART_COOLDOWN = timedelta(minutes=30)

ACTION_NONE = "none"
ACTION_RESTART = "restart"
ACTION_ALERT_STUCK = "alert_stuck"


def _lock_pid_alive() -> bool:
    """data/kabu_auto.lock のPIDが生きていれば True。ロックが無ければ False。"""
    from src.core.process_lock import _is_process_running

    if not LOCK_PATH.exists():
        return False
    try:
        pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    return _is_process_running(pid)


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def decide(lock_pid_alive: bool, paused: bool, state: dict, now: datetime) -> str:
    """次に取るべき行動を返す純粋関数（副作用なし）。テストが直接呼ぶ。

    戻り値: "none" | "restart" | "alert_stuck"
    """
    if lock_pid_alive or paused:
        return ACTION_NONE

    last_restart_str = state.get("last_restart")
    if not last_restart_str:
        return ACTION_RESTART

    last_restart = datetime.fromisoformat(last_restart_str)
    if now - last_restart >= RESTART_COOLDOWN:
        return ACTION_RESTART

    if state.get("stuck_alerted"):
        return ACTION_NONE
    return ACTION_ALERT_STUCK


def _restart(now: datetime) -> None:
    subprocess.run(
        ["schtasks", "/run", "/tn", TASK_NAME],
        creationflags=subprocess.CREATE_NO_WINDOW,
        check=False,
    )
    STATE_PATH.write_text(
        json.dumps({"last_restart": now.isoformat(), "stuck_alerted": False}),
        encoding="utf-8",
    )
    from dotenv import load_dotenv

    load_dotenv()
    from src.core import config as cfg
    from src.core.alerts import LEVEL_CRITICAL, alert

    cfg.load(str(REPO_ROOT / "config.yaml"))
    alert(
        "kabu-autoが停止していたため再起動しました",
        "watchdogがdata/kabu_auto.lockの死亡を検知し、タスク\\kabu\\kabu-autoを再実行しました。",
        level=LEVEL_CRITICAL,
    )


def _alert_stuck(state: dict) -> None:
    state = dict(state)
    state["stuck_alerted"] = True
    STATE_PATH.write_text(json.dumps(state), encoding="utf-8")

    from dotenv import load_dotenv

    load_dotenv()
    from src.core import config as cfg
    from src.core.alerts import LEVEL_CRITICAL, alert

    cfg.load(str(REPO_ROOT / "config.yaml"))
    alert(
        "kabu-autoの自動復旧に失敗しています",
        "再起動から30分以内に再度停止を検知しました。手動確認が必要です。",
        level=LEVEL_CRITICAL,
    )


def main() -> None:
    from src.core.clock import now as jst_now

    now = jst_now()
    paused = PAUSE_PATH.exists()
    alive = _lock_pid_alive()
    state = _load_state()

    action = decide(alive, paused, state, now)
    if action == ACTION_RESTART:
        _restart(now)
    elif action == ACTION_ALERT_STUCK:
        _alert_stuck(state)


if __name__ == "__main__":
    main()
