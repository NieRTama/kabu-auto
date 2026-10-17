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


# ─── 総資産の行 ───────────────────────────────────────────────────

def test_assets_line_from_live_value():
    assert o._assets_line(None, {"total": 1000.0, "day_change": None, "basis_change": None,
                                 "unpriced": []}) == "総資産: 1,000円"


def test_assets_line_from_record_in_separate_process(tmp_path, monkeypatch):
    from datetime import date
    from src.core import total_assets
    p = tmp_path / "h.json"
    p.write_text('{"2026-10-15": 492380.0, "2026-10-16": 495000.0}', encoding="utf-8")
    monkeypatch.setattr(total_assets, "HISTORY_PATH", p)
    line = o._assets_line(date(2026, 10, 16), None)
    assert line == "総資産: 495,000円（本日の記録値 ／ 前日比 +2,620円）"


def test_assets_line_without_record(tmp_path, monkeypatch):
    from datetime import date
    from src.core import total_assets
    monkeypatch.setattr(total_assets, "HISTORY_PATH", tmp_path / "none.json")
    assert o._assets_line(date(2026, 10, 16), None) == "総資産: 本日の記録なし（17:30の日次レポートで記録）"


# ─── Discord 向けの短い版 ───────────────────────────────────────────

def test_short_version_is_brief_and_points_to_file(isolated_db, tmp_path, monkeypatch):
    from datetime import date
    from src.core import total_assets
    monkeypatch.setattr(total_assets, "HISTORY_PATH", tmp_path / "h.json")
    text = o.build(date(2026, 10, 16), full=False,
                   assets={"total": 1000.0, "day_change": None, "basis_change": None, "unpriced": []})
    lines = text.split("\n")
    assert len(lines) <= 8
    assert lines[0] == "【kabu-auto 運用概要 2026-10-16】"
    assert lines[1] == "総資産: 1,000円"
    assert "発注: なし" in text and "取引停止:" in text and "ERROR" in text
    assert lines[-1] == "詳細: log/ops_daily/ops_daily_2026-10-16.md"
    assert "▼" not in text  # 詳細版の節（ALERT全文など）は含めない
