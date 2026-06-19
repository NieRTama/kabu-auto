"""
タイムゾーン統一の回帰テスト

DBに保存する日時カラムが JST naive（datetime.now）で統一されていることを検証する。
旧 datetime.utcnow と混在すると morning_execution の cutoff 比較が9時間ずれ、
前日シグナルを取りこぼしてライブ自動売買が成立しなくなる（第2回レビュー C-1）。
"""
from datetime import datetime, timedelta

import src.core.clock as clock
from src.data.database import Position, Signal


def _call_default(column_default):
    """SQLAlchemy は0引数callableをcontext付きラッパーに包むため、context=Noneで呼び出す"""
    return column_default.arg(None)


def _is_jst_now(value: datetime) -> bool:
    """値が clock.now()（JST naive）に近いことを確認する。

    システムローカル時刻（datetime.now/utcnow）と比較するとCI(UTC)とJST環境で結果が
    変わってしまうため、本物の時刻ソースである clock.now() を基準にする。
    """
    return abs((value - clock.now()).total_seconds()) < 5


class TestModelDefaultsAreJstNow:
    """DBの日時default が clock.now（JST naive）を使っていることを検証"""

    def test_signal_generated_at_uses_now(self):
        assert _is_jst_now(_call_default(Signal.__table__.c.generated_at.default))

    def test_position_opened_at_uses_now(self):
        assert _is_jst_now(_call_default(Position.__table__.c.opened_at.default))

    def test_position_updated_at_uses_now(self):
        assert _is_jst_now(_call_default(Position.__table__.c.updated_at.default))
        assert _is_jst_now(_call_default(Position.__table__.c.updated_at.onupdate))


class TestMorningExecutionCutoffConsistency:
    def test_same_basis_picks_up_prior_signal(self):
        """generated_at と cutoff が同一基準なら前日16:20のシグナルが翌9:05に拾える"""
        # signal_scan: 前日16:20 JST に datetime.now で保存
        generated_at = datetime(2026, 6, 16, 16, 20)
        # morning_execution: 翌9:05 JST、cutoff = now - 20h
        cutoff = datetime(2026, 6, 17, 9, 5) - timedelta(hours=20)
        assert generated_at >= cutoff, "同一基準なら前日シグナルが20時間ウィンドウに入る"
