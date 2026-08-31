"""
認証復帰時のキャッチアップ発注（catchup_execution）のテスト

2026-08-31 の実害:
  08:30 認証切れ → 10:58 ログインで復帰 → しかし 9:05 の morning_execution は
  既に401で失敗済みだったため、その日の約定は0件だった。
  復帰しても「次の定期実行(12:35)まで何もしない」状態だった。

復帰した時点が場中なら、逃した発注をその場で取り返す。ただし朝の発注は
「寄り付き直後の値動きが落ち着いた時間」を狙う設計なので、締切（既定14:00）を
過ぎたら見送る（no_new_buy_minutes_before_close と同じ思想）。
"""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import pytz

import src.core.scheduler as sched_mod
from src.core.scheduler import TradingScheduler
from src.services.trading import TradingServices

TZ = pytz.timezone("Asia/Tokyo")


def _services(deadline=14):
    client, risk, order_mgr = MagicMock(), MagicMock(), MagicMock()
    with patch("src.services.trading.cfg") as cfg_mock:
        cfg_mock.get_section.return_value = {
            "mode": "live", "catchup_deadline_hour": deadline,
        }
        svc = TradingServices(client, risk, order_mgr)
    return svc


def _run(svc, *, market_open: bool, now_hour: int, now_minute: int = 0):
    """指定時刻・市場状態で catchup_execution を実行し、発注本体が呼ばれたかを返す。"""
    fake_now = TZ.localize(datetime(2026, 8, 31, now_hour, now_minute))
    with patch.object(TradingScheduler, "is_market_open", return_value=market_open), \
         patch.object(sched_mod, "datetime") as dt_mock, \
         patch.object(svc, "_execute_pending_signals") as exec_mock:
        dt_mock.now.return_value = fake_now
        svc.catchup_execution()
        return exec_mock


class TestMarketClosed:
    def test_does_nothing_outside_market_hours(self):
        """場外（夜間・休日）に復帰しても発注しない（誤発注防止）"""
        svc = _services()
        exec_mock = _run(svc, market_open=False, now_hour=22)
        exec_mock.assert_not_called()


class TestDeadline:
    def test_executes_before_deadline(self):
        """締切前（10:58 に復帰）なら逃した分を取り返す＝2026-08-31 のケース"""
        svc = _services(deadline=14)
        exec_mock = _run(svc, market_open=True, now_hour=10, now_minute=58)
        exec_mock.assert_called_once()

    def test_skips_after_deadline(self):
        """締切後（14:30）は見送る。引け間際の想定外の約定を避ける"""
        svc = _services(deadline=14)
        exec_mock = _run(svc, market_open=True, now_hour=14, now_minute=30)
        exec_mock.assert_not_called()

    def test_boundary_at_deadline_is_excluded(self):
        """ちょうど14:00は「過ぎている」扱い（is_before は < 比較）"""
        svc = _services(deadline=14)
        exec_mock = _run(svc, market_open=True, now_hour=14, now_minute=0)
        exec_mock.assert_not_called()

    def test_deadline_zero_disables_the_limit(self):
        """0 なら締切なし（場中ならいつでも実行）"""
        svc = _services(deadline=0)
        exec_mock = _run(svc, market_open=True, now_hour=15, now_minute=0)
        exec_mock.assert_called_once()


class TestSafety:
    def test_skips_existing_positions(self):
        """未保有銘柄のみを対象にする（保有銘柄への買い増しをしない）。

        後場スロット(afternoon_execution)と同じ経路を使うことで、
        既存の二重発注ガード(_has_pending_order)もそのまま効く。
        """
        svc = _services()
        exec_mock = _run(svc, market_open=True, now_hour=10)
        assert exec_mock.call_args.kwargs["skip_existing"] is True

    def test_source_is_recorded_distinctly(self):
        """発注のきっかけを区別して記録する（後から追跡できるように）"""
        svc = _services()
        exec_mock = _run(svc, market_open=True, now_hour=10)
        assert exec_mock.call_args.kwargs["source"] == "catchup_execution"


class TestIsBefore:
    """締切判定そのもの（時刻ヘルパー）"""

    def _at(self, h, m=0):
        return TZ.localize(datetime(2026, 8, 31, h, m))

    def test_true_before_target(self):
        assert TradingScheduler.is_before(14, now=self._at(13, 59)) is True

    def test_false_at_target(self):
        assert TradingScheduler.is_before(14, now=self._at(14, 0)) is False

    def test_false_after_target(self):
        assert TradingScheduler.is_before(14, now=self._at(14, 1)) is False

    def test_minute_precision(self):
        assert TradingScheduler.is_before(14, 30, now=self._at(14, 29)) is True
        assert TradingScheduler.is_before(14, 30, now=self._at(14, 30)) is False


class TestWiring:
    def test_catchup_is_called_on_auth_recovery(self):
        """認証復帰のコールバックからキャッチアップが呼ばれること（結線の検証）"""
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        assert "services.catchup_execution()" in src, (
            "認証復帰時にキャッチアップ発注が呼ばれていない"
        )
        # 復帰通知の直後にあること（順序: 通知 → 発注）
        assert src.index("認証が回復しました") < src.index("services.catchup_execution()")
