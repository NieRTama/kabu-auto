"""watchdog.decide() の判定ロジックのテスト（2026-09-27）。

- 生きていれば何もしない
- data/watchdog.pause があれば何もしない（生死に関わらず）
- 死んでいて初回なら再起動
- 死んでいて再起動から30分以内ならalert_stuck（未通知の場合のみ）、既に通知済みなら何もしない
- 死んでいて再起動から30分以上経っていれば再度restart
"""
from datetime import datetime, timedelta

from scripts import watchdog


NOW = datetime(2026, 9, 27, 12, 0, 0)


def test_alive_does_nothing():
    assert watchdog.decide(True, False, {}, NOW) == watchdog.ACTION_NONE


def test_paused_does_nothing_even_if_dead():
    assert watchdog.decide(False, True, {}, NOW) == watchdog.ACTION_NONE


def test_dead_first_time_restarts():
    assert watchdog.decide(False, False, {}, NOW) == watchdog.ACTION_RESTART


def test_dead_within_cooldown_alerts_once():
    state = {"last_restart": (NOW - timedelta(minutes=10)).isoformat()}
    assert watchdog.decide(False, False, state, NOW) == watchdog.ACTION_ALERT_STUCK


def test_dead_within_cooldown_already_alerted_does_nothing():
    state = {
        "last_restart": (NOW - timedelta(minutes=10)).isoformat(),
        "stuck_alerted": True,
    }
    assert watchdog.decide(False, False, state, NOW) == watchdog.ACTION_NONE


def test_dead_after_cooldown_restarts_again():
    state = {
        "last_restart": (NOW - timedelta(minutes=31)).isoformat(),
        "stuck_alerted": True,
    }
    assert watchdog.decide(False, False, state, NOW) == watchdog.ACTION_RESTART
