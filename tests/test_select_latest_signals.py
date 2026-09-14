"""
main._select_latest_signals() のテスト

経緯: 旧実装は `generated_at >= now - timedelta(hours=20)` という固定時間窓で
シグナルを拾っていたため、土日・祝日を挟むと前営業日（例: 金曜16:20）のシグナルを
月曜9:05の発注時に取りこぼしていた（20時間を超えるため）。
「最新のシグナル生成日そのもの」を基準にすることで、休場日数に関わらず
前営業日分を正しく拾えることを検証する。
"""
from datetime import datetime, timedelta

import pytest

import src.core.config as cfg
import src.data.database as db
from src.core import clock
from src.data.database import Signal, get_session

import main as main_module


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    try:
        yield tmp_path
    finally:
        db._engine = None
        db._Session = None


def _batch_time(dt: datetime) -> datetime:
    """その日の signal_scan 実行時刻（16:20）に正規化する。

    時刻を固定することで、テスト実行が深夜に走った場合に「+5分」が日付を跨いで
    バッチ判定の前提が変わる、といった実行時刻依存のゆらぎを排除する。
    """
    return dt.replace(hour=16, minute=20, second=0, microsecond=0)


def _add_signal(symbol: str, action: str, generated_at: datetime) -> None:
    with get_session() as session:
        session.add(Signal(
            symbol=symbol, action=action, generated_at=generated_at,
            rule_score=0.0, ml_score=0.0, combined_score=0.0,
        ))
        session.commit()


class TestSelectLatestSignals:
    def test_no_signals_returns_empty(self, isolated_db):
        with get_session() as session:
            result = main_module._select_latest_signals(session)
        assert result == []

    def test_friday_signal_picked_up_on_monday(self, isolated_db):
        """金曜16:20生成のシグナルは、月曜9:05（72時間以上後）でも正しく拾える。

        日時は必ず「実行時刻からの相対」で組み立てる。絶対日付を書くと、実装の陳腐化判定
        （clock.now() と比較して max_age_days 超過なら空リスト）が実時刻基準のため、
        コードを一切変更しなくても時間の経過だけでテストが落ちる（時限爆弾になる）。
        """
        now = clock.now()
        # 週末を挟んだ想定の3日前16:20（signal_scan の実行時刻に合わせる）
        friday_signal_time = _batch_time(now - timedelta(days=3))
        assert (now - friday_signal_time).total_seconds() / 3600 > 20, \
            "前提: 20時間カットオフでは確実に取りこぼす時間差であること"

        _add_signal("7203", "BUY", friday_signal_time)

        with get_session() as session:
            result = main_module._select_latest_signals(session, max_age_days=5)
        assert len(result) == 1
        assert result[0].symbol == "7203"

    def test_dedup_keeps_latest_per_symbol(self, isolated_db):
        """同銘柄に複数シグナルがある場合、最新の1件のみ残す"""
        base = _batch_time(clock.now() - timedelta(days=3))  # 相対日付（絶対日付は時限爆弾）
        _add_signal("7203", "BUY", base)
        _add_signal("7203", "SELL", base + timedelta(minutes=5))

        with get_session() as session:
            result = main_module._select_latest_signals(session)
        assert len(result) == 1
        assert result[0].action == "SELL"  # より新しい方が残る

    def test_old_buy_batch_excluded_when_a_newer_buy_batch_exists(self, isolated_db):
        """古いBUYバッチより新しいBUYバッチが存在すれば、古い方は対象外
        （同じ銘柄群への重複発注を防ぐ）。"""
        now = clock.now()
        _add_signal("7203", "BUY", _batch_time(now - timedelta(days=5)))   # 古いバッチ
        _add_signal("6758", "BUY", _batch_time(now - timedelta(days=3)))   # 最新バッチ

        with get_session() as session:
            result = main_module._select_latest_signals(session)
        symbols = {s.symbol for s in result}
        assert symbols == {"6758"}

    def test_todays_stop_loss_sell_does_not_hide_yesterdays_buy_batch(self, isolated_db):
        """新規Important C: 前営業日のBUYバッチ（signal_scan）が、当日保存された
        損切りSELL（stop_loss_check、別銘柄）によって丸ごと隠れてはいけない。

        修正前は「最新シグナル生成日」をBUY/SELL共通の基準にしていたため、
        当日付のSELLが1件保存されただけで基準日が「今日」にすり替わり、
        前営業日のBUYバッチ（本来9:05のmorning_executionが拾うべきもの）が
        全滅していた（レビュー実測: BUY 3件が消える）。
        """
        yesterday_batch = _batch_time(clock.now() - timedelta(days=1))
        _add_signal("7203", "BUY", yesterday_batch)
        _add_signal("6758", "BUY", yesterday_batch)
        _add_signal("9984", "BUY", yesterday_batch)

        # 当日09:30にstop_loss_checkが7203をSELL
        _add_signal("7203", "SELL", clock.now())

        with get_session() as session:
            result = main_module._select_latest_signals(session)
        by_symbol = {s.symbol: s for s in result}
        assert by_symbol.keys() == {"7203", "6758", "9984"}
        assert by_symbol["7203"].action == "SELL"   # dedupでSELLが優先（Important 4）
        assert by_symbol["6758"].action == "BUY"
        assert by_symbol["9984"].action == "BUY"

    def test_old_buy_batch_not_revived_by_an_unrelated_fresh_sell(self, isolated_db):
        """BUY自体がmax_age_daysより古ければ、他銘柄の新しいSELLがあっても
        対象外のままであること（signal_scanが止まったまま何日も経った
        BUYバッチを誤って蘇らせないための安全策）。"""
        _add_signal("7203", "BUY", _batch_time(clock.now() - timedelta(days=10)))
        _add_signal("6758", "SELL", clock.now())  # 他銘柄の新しいSELL

        with get_session() as session:
            result = main_module._select_latest_signals(session, max_age_days=5)
        by_symbol = {s.symbol: s for s in result}
        assert "7203" not in by_symbol
        assert by_symbol["6758"].action == "SELL"

    def test_hold_signals_excluded(self, isolated_db):
        """HOLDシグナルは対象外（BUY/SELLのみ）"""
        _add_signal("7203", "HOLD", datetime.now())
        with get_session() as session:
            result = main_module._select_latest_signals(session)
        assert result == []

    def test_stale_signals_beyond_max_age_excluded(self, isolated_db):
        """最新シグナルがmax_age_daysを超えて古い場合は空リスト（陳腐化したシグナルでの誤発注防止）"""
        _add_signal("7203", "BUY", datetime.now() - timedelta(days=10))
        with get_session() as session:
            result = main_module._select_latest_signals(session, max_age_days=5)
        assert result == []

    def test_within_max_age_still_included(self, isolated_db):
        _add_signal("7203", "BUY", datetime.now() - timedelta(days=2))
        with get_session() as session:
            result = main_module._select_latest_signals(session, max_age_days=5)
        assert len(result) == 1

    def test_sell_wins_over_more_recent_buy(self, isolated_db):
        """最終ブランチレビュー Important 4: 同日の損切りSELLが、より新しい
        signal_scan由来のBUYで上書きされて消えてはいけない。

        v2+paperでは、日中の stop_loss_check が保存したSELLの後に、同日16:20の
        signal_scan が同一銘柄へBUYを生成することがある。dedupが単純に「生成時刻が
        新しい方」を残すと、保有保護の退出（SELL）が消えて最低1営業日遅れる。
        """
        base = _batch_time(clock.now() - timedelta(days=3))
        _add_signal("7203", "SELL", base)                       # 日中の損切り判定
        _add_signal("7203", "BUY", base + timedelta(minutes=5))  # その後のsignal_scan

        with get_session() as session:
            result = main_module._select_latest_signals(session)
        assert len(result) == 1
        assert result[0].action == "SELL"

    def test_sell_wins_over_more_recent_buy_among_other_symbols(self, isolated_db):
        """他銘柄のdedup（最新1件を残す通常ルール）と共存すること"""
        base = _batch_time(clock.now() - timedelta(days=3))
        _add_signal("7203", "SELL", base)
        _add_signal("7203", "BUY", base + timedelta(minutes=5))
        _add_signal("6758", "BUY", base)
        _add_signal("6758", "BUY", base + timedelta(minutes=5))  # 同銘柄同アクションは最新が残る

        with get_session() as session:
            result = main_module._select_latest_signals(session)
        by_symbol = {s.symbol: s for s in result}
        assert by_symbol["7203"].action == "SELL"
        assert by_symbol["6758"].action == "BUY"
        assert by_symbol["6758"].generated_at == base + timedelta(minutes=5)
