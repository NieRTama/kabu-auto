"""運用概要（ops_summary）のログ解析部分のテスト。"""
from src.core import ops_summary as o

WARN = [
    "2026-10-14 08:35:01.455 | WARNING  | src.core.alerts:alert:130 - [ALERT] kabuステーションの再ログインが必要です: x",
    "2026-10-14 09:00:00.000 | ERROR    | src.services.trading:check:10 - 損切りチェックエラー: 9434 401",
    "2026-10-14 09:00:05.000 | ERROR    | src.services.trading:check:10 - 損切りチェックエラー: 2353 401",
    "2026-10-14 09:22:11.191 | WARNING  | src.core.alerts:alert:130 - [ALERT] kabuステーションの認証が回復しました: y",
    "2026-10-14 12:38:12.036 | WARNING  | src.core.alerts:alert:130 - [ALERT] kabuステーションのログイン待ち: z",
]


def test_auth_downtime_pairs_and_open_span():
    assert o._auth_downtime(o._alerts(WARN)) == ["08:35〜09:22（47分）", "12:38〜（継続中）"]


def test_error_kinds_groups_by_masked_message():
    assert o._error_kinds(WARN) == {"trading: 損切りチェックエラー: # #": 2}
