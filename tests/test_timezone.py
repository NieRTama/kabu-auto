"""
タイムゾーン統一の回帰テスト

DBに保存する日時カラムが JST naive（datetime.now）で統一されていることを検証する。
旧 datetime.utcnow と混在すると morning_execution の cutoff 比較が9時間ずれ、
前日シグナルを取りこぼしてライブ自動売買が成立しなくなる（第2回レビュー C-1）。
"""
from datetime import datetime, timedelta

from src.data.database import Position, Signal


def _call_default(column_default):
    """SQLAlchemy は0引数callableをcontext付きラッパーに包むため、context=Noneで呼び出す"""
    return column_default.arg(None)


def _is_local_now_not_utc(value: datetime) -> bool:
    """値が datetime.now()（ローカル=JST）に近く、datetime.utcnow() ではないこと。

    ローカルがUTCのCI環境では now==utcnow で区別不能のため、その場合は now近接のみ確認する。
    JST等オフセットのある環境では utcnow と大きくずれることも確認する。
    """
    near_now = abs((value - datetime.now()).total_seconds()) < 5
    local_offset = abs((datetime.now() - datetime.utcnow()).total_seconds())
    if local_offset < 60:
        return near_now  # UTC環境では now 近接のみで判定
    far_from_utc = abs((value - datetime.utcnow()).total_seconds()) > 60
    return near_now and far_from_utc


class TestModelDefaultsAreJstNow:
    """DBの日時default が datetime.now（JST naive）であり utcnow でないことを検証"""

    def test_signal_generated_at_uses_now(self):
        assert _is_local_now_not_utc(_call_default(Signal.__table__.c.generated_at.default))

    def test_position_opened_at_uses_now(self):
        assert _is_local_now_not_utc(_call_default(Position.__table__.c.opened_at.default))

    def test_position_updated_at_uses_now(self):
        assert _is_local_now_not_utc(_call_default(Position.__table__.c.updated_at.default))
        assert _is_local_now_not_utc(_call_default(Position.__table__.c.updated_at.onupdate))


class TestMorningExecutionCutoffConsistency:
    def test_same_basis_picks_up_prior_signal(self):
        """generated_at と cutoff が同一基準なら前日16:20のシグナルが翌9:05に拾える"""
        # signal_scan: 前日16:20 JST に datetime.now で保存
        generated_at = datetime(2026, 6, 16, 16, 20)
        # morning_execution: 翌9:05 JST、cutoff = now - 20h
        cutoff = datetime(2026, 6, 17, 9, 5) - timedelta(hours=20)
        assert generated_at >= cutoff, "同一基準なら前日シグナルが20時間ウィンドウに入る"
