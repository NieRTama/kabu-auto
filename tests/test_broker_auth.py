"""
broker_auth（kabuステーションのログイン認証切れ）のテスト

kabuステーションの認証はPCを起動したままでも日をまたぐと切れる
（アプリ自身が "Code 10016: ログイン認証の有効期間が切れました" を持つ）。
2026-08-26/27 は毎朝8:30のトークン更新が401で失敗した後リトライが無く、
終日1,300回超の401を出しながら「発注可」を表示し続ける抜け殻状態だった。

ここでは「認証切れの間は新規発注を止める／退出は止めない」ことを検証する。
"""
from unittest.mock import MagicMock, patch

import pytest

from src.core import broker_auth


@pytest.fixture(autouse=True)
def _reset_state():
    broker_auth.reset()
    yield
    broker_auth.reset()


class TestState:
    def test_initially_valid(self):
        assert broker_auth.is_expired() is False

    def test_mark_expired_sets_state_with_detail_and_timestamp(self):
        broker_auth.mark_expired("401 Unauthorized")
        assert broker_auth.is_expired() is True
        s = broker_auth.get_state()
        assert s["expired"] is True
        assert "401" in s["detail"]
        assert s["since"] is not None

    def test_mark_valid_clears_state(self):
        broker_auth.mark_expired("401")
        broker_auth.mark_valid()
        assert broker_auth.is_expired() is False
        assert broker_auth.get_state()["since"] is None

    def test_repeated_mark_expired_logs_once(self):
        """毎朝の失敗後リトライのたびにCRITICALを重ねない（ログのノイズ抑制）"""
        with patch("src.core.broker_auth.logger") as log_mock:
            broker_auth.mark_expired("401")
            broker_auth.mark_expired("401")
            broker_auth.mark_expired("401")
        assert log_mock.critical.call_count == 1

    def test_since_is_preserved_across_repeated_marks(self):
        broker_auth.mark_expired("401")
        first = broker_auth.get_state()["since"]
        broker_auth.mark_expired("401 again")
        assert broker_auth.get_state()["since"] == first, "検知時刻は最初のものを保つ"


class TestTokenRefreshRaceGrace:
    """トークン更新直後の一過性401を誤検知しない（2026-09-08 の実害の回帰防止）。

    KabuClient._token はスレッド間で共有され排他制御が無い。auth_recovery_check
    （5分間隔）がトークンを更新した直後、別スレッドの reconcile_positions_with_broker
    （15秒間隔）が更新前のトークンで送信済みだったリクエストが到達し401で拒否される
    ことがある。300秒(5分)は15秒の倍数のため、この競合は周期的に必ず発生する。

    実測（kabuステーション側ログ）: トークン更新の2.2ミリ秒後に別リクエストが
    Code=4001009「APIキー不一致」で拒否された。以前は個々のAPI呼び出しの401を
    無視していたため無害だったが、「場中の401を認証切れとして記録する」ようにした
    2026-09-02 の修正が、この一過性の競合を5分ごとの新規発注停止に変えてしまった。
    """

    def test_expired_within_grace_after_valid_is_ignored(self):
        with patch.object(broker_auth.time, "monotonic", return_value=100.0):
            broker_auth.mark_valid()
        with patch.object(broker_auth.time, "monotonic", return_value=101.0):  # 1秒後
            broker_auth.mark_expired("401 (race)")
        assert broker_auth.is_expired() is False, "猶予中の401は無視する"

    def test_expired_after_grace_elapsed_is_recorded(self):
        """猶予を過ぎれば本物の切れとして通常どおり検知する（検知力を落とさない）"""
        with patch.object(broker_auth.time, "monotonic", return_value=100.0):
            broker_auth.mark_valid()
        with patch.object(broker_auth.time, "monotonic",
                          return_value=100.0 + broker_auth._RACE_GRACE_SECONDS + 0.1):
            broker_auth.mark_expired("401 (real)")
        assert broker_auth.is_expired() is True

    def test_grace_does_not_apply_before_any_success(self):
        """一度も mark_valid していない状態（起動直後の初回401）は猶予なしで検知する"""
        broker_auth.mark_expired("401")
        assert broker_auth.is_expired() is True

    def test_ignored_race_does_not_suppress_log_entirely(self):
        """猶予中でもDEBUGログには残す（完全に消さない）"""
        with patch.object(broker_auth.time, "monotonic", return_value=100.0):
            broker_auth.mark_valid()
        with patch.object(broker_auth.time, "monotonic", return_value=101.0), \
             patch.object(broker_auth, "logger") as log_mock:
            broker_auth.mark_expired("401 (race)")
        assert log_mock.debug.called
        assert log_mock.critical.called is False

    def test_repeated_race_hits_do_not_accumulate_into_expiry(self):
        """猶予中に401が何度来ても、蓄積して切れ扱いにはならない"""
        with patch.object(broker_auth.time, "monotonic", return_value=100.0):
            broker_auth.mark_valid()
        for t in (100.5, 101.0, 101.5, 102.0):
            with patch.object(broker_auth.time, "monotonic", return_value=t):
                broker_auth.mark_expired("401 (race)")
        assert broker_auth.is_expired() is False


# ─── 発注ゲートとの結線 ──────────────────────────────────────────────


def _risk():
    import src.risk.manager as risk_mod
    with patch.object(risk_mod, "cfg") as cfg_mock:
        cfg_mock.get_section.return_value = {"daily_order_limit": 100, "max_daily_loss": 0}
        r = risk_mod.RiskManager()
    return r


class TestOrderGate:
    def test_blocks_new_orders_when_expired(self):
        r = _risk()
        with patch.object(r, "_count_unresolved_orders", return_value=0), \
             patch.object(r, "is_total_loss_limit_reached", return_value=(False, "")):
            ok_before, _ = r.can_place_order()
            broker_auth.mark_expired("401 Unauthorized")
            ok_after, reason = r.can_place_order()
        assert ok_before is True
        assert ok_after is False
        assert "認証" in reason or "ログイン" in reason

    def test_allows_orders_again_after_recovery(self):
        r = _risk()
        with patch.object(r, "_count_unresolved_orders", return_value=0), \
             patch.object(r, "is_total_loss_limit_reached", return_value=(False, "")):
            broker_auth.mark_expired("401")
            assert r.can_place_order()[0] is False
            broker_auth.mark_valid()
            assert r.can_place_order()[0] is True


class TestExitNotBlocked:
    def test_sell_market_exit_bypasses_auth_gate(self):
        """認証切れでも退出（損切り・トレーリング・緊急）は止めない。

        can_place_order() を通さない reason 経路であることをソースで担保する
        （既存の kill switch / 損失上限バイパスと同じ思想）。
        """
        import inspect
        import src.execution.order_manager as mod
        src = inspect.getsource(mod.OrderManager.sell_market)
        assert 'is_exit = reason in ("stop_loss", "trailing_stop", "emergency")' in src
        assert "if not is_exit:" in src, (
            "退出系が can_place_order() ゲートをバイパスする構造が壊れている"
        )
