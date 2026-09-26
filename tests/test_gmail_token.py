from datetime import datetime

from src.core import gmail_token

ISSUED = datetime(2026, 9, 25, 8, 48)


def _write(tmp_path):
    path = tmp_path / "issued.txt"
    gmail_token.record_issued(ISSUED, path)
    return path


def test_no_reminder_while_plenty_of_time_left(tmp_path):
    path = _write(tmp_path)
    assert gmail_token.reminder(datetime(2026, 9, 29, 20, 0), path) is None


def test_warning_within_two_days_of_expiry(tmp_path):
    path = _write(tmp_path)
    level, msg = gmail_token.reminder(datetime(2026, 9, 30, 20, 0), path)
    assert level == "warning"
    assert "10/02 08:48" in msg
    assert "gmail_reauth.bat" in msg


def test_critical_after_expiry(tmp_path):
    path = _write(tmp_path)
    level, msg = gmail_token.reminder(datetime(2026, 10, 2, 9, 0), path)
    assert level == "critical"
    assert "gmail_reauth.bat" in msg


def test_warning_when_never_recorded(tmp_path):
    level, msg = gmail_token.reminder(datetime(2026, 9, 30), tmp_path / "missing.txt")
    assert level == "warning"
    assert "gmail_reauth.bat" in msg


def test_full_login_failure_points_to_reauth_script(monkeypatch):
    import subprocess
    from src.core import broker_full_login

    broker_full_login.reset()
    stderr = "x" * 400 + "アクセストークンの更新に失敗しました: invalid_grant\n再度auth_gmailapi.pyで認証してください。"
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr=stderr),
    )
    ok, detail = broker_full_login.run(manual=True)
    assert not ok
    assert "gmail_reauth.bat" in detail
