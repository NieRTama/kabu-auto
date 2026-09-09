"""
_bought_today()（後場スロットの同日二度打ち防止）のテスト

2026-09-09、afternoon_execution の skip_existing の基準を「保有の有無」から
「本日既にBUYが成立/進行中か」へ変更した。これにより前日までの保有銘柄への
買い増しを許可しつつ、朝と後場で同じシグナルを二重約定させないようにする。

_bought_today() は実際のDB（OrderIntent と Trade の結合）に対して直接テストする。
モックした afternoon_execution のテストだけでは、日付フィルタ・状態フィルタ・
JOIN条件そのものの正しさを検証できないため。
"""
from datetime import datetime, timedelta

import pytest

import src.core.config as cfg
import src.data.database as db
import src.execution.order_status as st
from src.core import clock
from src.data.database import OrderIntent, Trade, get_session
from src.services.trading import _bought_today


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


def _add_buy(symbol: str, status: str, *, created_at=None) -> None:
    """OrderIntent+Trade を1組作る（_record_trade の最小相当）。"""
    created_at = created_at or clock.now()
    with get_session() as session:
        intent = OrderIntent(
            symbol=symbol, side="BUY", target_quantity=100,
            order_type="LIMIT", limit_price=1000.0,
            source="test", mode="live", status="SUBMITTED",
            created_at=created_at,
        )
        session.add(intent)
        session.flush()
        session.add(Trade(
            order_id=f"{symbol}-{status}-{created_at.timestamp()}",
            intent_id=intent.id, symbol=symbol, side="BUY",
            quantity=100, price=1000.0, status=status,
        ))
        session.commit()


class TestNoActivityToday:
    def test_unheld_and_untouched_symbol_is_not_bought(self, isolated_db):
        assert _bought_today("7203") is False


class TestSuccessfulOrCanOpenBuyBlocksRetry:
    """実際に成立/進行中の発注は「本日済み」として二度打ちを止める"""

    def test_filled_buy_today_counts(self, isolated_db):
        _add_buy("7203", st.FILLED)
        assert _bought_today("7203") is True

    def test_pending_buy_today_counts(self, isolated_db):
        """未約定でも進行中なら二度打ちを止める（_has_pending_order と多重防御）"""
        _add_buy("7203", st.PENDING)
        assert _bought_today("7203") is True

    def test_partially_filled_counts(self, isolated_db):
        _add_buy("7203", st.PARTIALLY_FILLED)
        assert _bought_today("7203") is True


class TestFailedAttemptAllowsRetry:
    """実際には成立しなかった発注は「未購入」扱いにし、同日の再挑戦を許す"""

    def test_rejected_buy_today_does_not_count(self, isolated_db):
        _add_buy("7203", st.REJECTED)
        assert _bought_today("7203") is False

    def test_cancelled_buy_today_does_not_count(self, isolated_db):
        _add_buy("7203", st.CANCELLED)
        assert _bought_today("7203") is False


class TestDateBoundary:
    def test_yesterdays_fill_does_not_count(self, isolated_db):
        """前日までの保有（＝過去の約定）は対象外＝買い増しを許す"""
        yesterday = clock.now() - timedelta(days=1)
        _add_buy("7203", st.FILLED, created_at=yesterday)
        assert _bought_today("7203") is False

    def test_todays_fill_before_now_counts(self, isolated_db):
        earlier_today = datetime.combine(clock.today(), datetime.min.time()) + timedelta(hours=9)
        _add_buy("7203", st.FILLED, created_at=earlier_today)
        assert _bought_today("7203") is True


class TestOtherSymbolsUnaffected:
    def test_other_symbol_purchase_does_not_block(self, isolated_db):
        _add_buy("6758", st.FILLED)
        assert _bought_today("7203") is False


class TestSellIsIgnored:
    def test_sell_today_does_not_count_as_bought(self, isolated_db):
        """side=SELL は対象外（BUYの二度打ち判定なので）"""
        with get_session() as session:
            intent = OrderIntent(
                symbol="7203", side="SELL", target_quantity=100,
                order_type="LIMIT", limit_price=1000.0,
                source="test", mode="live", status="SUBMITTED",
                created_at=clock.now(),
            )
            session.add(intent)
            session.flush()
            session.add(Trade(
                order_id="7203-sell", intent_id=intent.id, symbol="7203", side="SELL",
                quantity=100, price=1000.0, status=st.FILLED,
            ))
            session.commit()
        assert _bought_today("7203") is False
