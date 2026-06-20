"""
OrderManager の主要バグ修正の検証テスト

カバー範囲:
  Critical #1: パスワード参照先が kabu_station セクションであること
  Critical #2: can_place_order() タプルが正しく評価されること
  Critical #3: ライブモード約定後にポジション更新が呼ばれること
  Medium  #8: ペーパートレードが FILLED / filled_at 付きで保存されること
"""
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import src.execution.order_manager as mod


# ─── ヘルパー ─────────────────────────────────────────────────────────────


def _cfg(mode: str = "paper") -> MagicMock:
    m = MagicMock()
    m.get_section.side_effect = lambda s: {
        "trading": {
            "mode": mode,
            "order_timeout_seconds": 300,
            "daily_order_limit": 100,
        },
        "kabu_station": {"password": "secret_kabu_pw"},
    }.get(s, {})
    return m


def _session_ctx():
    """get_session() コンテキストマネージャのモックを返す"""
    session = MagicMock()
    session.scalar.return_value = None

    @contextmanager
    def ctx():
        yield session

    return ctx, session


@contextmanager
def _make_om(mode: str = "paper", limit_reached: bool = False):
    """
    パッチを有効にしたまま OrderManager を yield するコンテキストマネージャ。
    with _make_om(...) as (om, client, risk, session): で使う。
    """
    cfg_mock = _cfg(mode)
    client = MagicMock()
    client.send_order.return_value = {"Result": 0, "OrderId": "LIVE-ORD-001"}
    risk = MagicMock()
    reason = "1日の注文上限(100)に達しました" if limit_reached else ""
    risk.can_place_order.return_value = (not limit_reached, reason)
    session_ctx, session = _session_ctx()

    with patch.object(mod, "cfg", cfg_mock), \
         patch.object(mod, "get_session", session_ctx):
        om = mod.OrderManager(client, risk)
        yield om, client, risk, session


# ─── Critical #1: パスワード参照先 ──────────────────────────────────────


class TestPasswordSource:
    def test_buy_uses_kabu_station_password(self):
        """ライブ BUY 注文のパスワードが kabu_station セクションから取得されること"""
        with _make_om(mode="live") as (om, client, risk, _):
            om.buy("7203", 1000.0, 100)

        sent = client.send_order.call_args[0][0]
        assert sent["Password"] == "secret_kabu_pw", (
            f"パスワードが kabu_station セクションから取得されていない: {sent['Password']!r}"
        )

    def test_sell_uses_kabu_station_password(self):
        """ライブ SELL 注文のパスワードが kabu_station セクションから取得されること"""
        with _make_om(mode="live") as (om, client, risk, _):
            om.sell("7203", 1100.0, 100)

        sent = client.send_order.call_args[0][0]
        assert sent["Password"] == "secret_kabu_pw"

    def test_password_not_from_trading_section(self):
        """パスワードが trading セクション（不正な参照先）から取得されていないこと"""
        with _make_om(mode="live") as (om, client, risk, _):
            om.buy("7203", 1000.0, 100)

        sent = client.send_order.call_args[0][0]
        # trading セクションには "password" キーがないので空文字になるはずがない
        assert sent["Password"] != "", "trading セクションから空パスワードを取得している"


# ─── Critical #2: can_place_order タプルバグ ────────────────────────────


class TestCanPlaceOrderTuple:
    def test_limit_reached_blocks_buy(self):
        """上限到達時 (False, msg) を正しく評価して None を返す"""
        with _make_om(limit_reached=True) as (om, _, risk, _):
            result = om.buy("7203", 1000.0, 100)

        assert result is None, "注文上限時は None を返すべき"
        risk.increment_order_count.assert_not_called()

    def test_limit_reached_blocks_sell(self):
        """SELL でも上限到達時は None を返す"""
        with _make_om(limit_reached=True) as (om, _, risk, _):
            result = om.sell("7203", 1100.0, 100)

        assert result is None
        risk.increment_order_count.assert_not_called()

    def test_limit_not_reached_proceeds(self):
        """未到達時 (True, "") → increment_order_count が呼ばれる"""
        with _make_om(limit_reached=False) as (om, _, risk, _):
            result = om.buy("7203", 1000.0, 100)

        assert result is not None
        risk.increment_order_count.assert_called_once()

    def test_non_empty_tuple_was_always_truthy(self):
        """修正前バグの証明: 非空タプルは bool() で常に True になる"""
        falsy_tuple = (False, "limit reached")
        assert bool(falsy_tuple) is True
        ok, _ = falsy_tuple


# ─── increment_order_count は成功確定後にのみ呼ばれる ────────────────────


class TestIncrementOrderCountTiming:
    def test_live_rejected_order_does_not_increment(self):
        """ライブ注文がAPIに拒否された場合（Result!=0）はカウントを消費しない"""
        with _make_om(mode="live") as (om, client, risk, _):
            client.send_order.return_value = {"Result": 1, "Message": "rejected"}
            result = om.buy("7203", 1000.0, 100)

        assert result is None
        risk.increment_order_count.assert_not_called()

    def test_live_missing_order_id_does_not_increment(self):
        """OrderIdが取得できない場合もカウントを消費しない"""
        with _make_om(mode="live") as (om, client, risk, _):
            client.send_order.return_value = {"Result": 0}
            result = om.sell("7203", 1100.0, 100)

        assert result is None
        risk.increment_order_count.assert_not_called()

    def test_live_success_increments_once(self):
        """ライブ注文が成功（Result==0かつOrderId取得）した場合のみカウントする"""
        with _make_om(mode="live") as (om, client, risk, _):
            client.send_order.return_value = {"Result": 0, "OrderId": "LIVE-1"}
            result = om.buy("7203", 1000.0, 100)

        assert result == "LIVE-1"
        risk.increment_order_count.assert_called_once()


# ─── close_all_positions は成行で全決済する（A-1）───────────────────────


class TestCloseAllPositionsMarket:
    def _pos(self, symbol="7203", qty=100):
        p = MagicMock()
        p.symbol = symbol
        p.quantity = qty
        return p

    def test_calls_sell_market_for_each_position(self):
        """緊急全決済は各保有銘柄について成行 sell_market を reason='emergency' で呼ぶ"""
        with _make_om(mode="paper") as (om, client, _, session):
            scal = MagicMock()
            scal.all.return_value = [self._pos(qty=200)]
            session.scalars.return_value = scal
            om.sell_market = MagicMock()
            om.close_all_positions()
        om.sell_market.assert_called_once_with("7203", 200, reason="emergency")

    def test_does_not_use_limit_sell(self):
        """指値 sell() ではなく成行で決済する（急変時の約定漏れ防止）"""
        with _make_om(mode="paper") as (om, client, _, session):
            scal = MagicMock()
            scal.all.return_value = [self._pos()]
            session.scalars.return_value = scal
            om.sell = MagicMock()
            om.sell_market = MagicMock()
            om.close_all_positions()
        om.sell.assert_not_called()
        om.sell_market.assert_called_once()


# ─── on_order_event は補助トリガとしてAPI照合を起動するのみ（再レビュー P0-1/P0-2）──


class TestOnOrderEventTriggersReconcile:
    def test_open_trade_triggers_reconcile(self):
        """未約定(OPEN_STATUSES)のTradeなら _reconcile_trade に処理を委譲する"""
        trade_mock = MagicMock(status="PENDING")
        with _make_om(mode="live") as (om, _, _, session):
            session.scalar.return_value = trade_mock
            om._reconcile_trade = MagicMock()
            om.on_order_event({"OrderID": "LIVE-ORD-001", "OrderState": 5})
        om._reconcile_trade.assert_called_once_with(trade_mock)

    def test_already_terminal_trade_does_not_reconcile(self):
        """既にFILLED等の確定状態なら再照会しない（不要なAPI呼び出しを避ける）"""
        trade_mock = MagicMock(status="FILLED")
        with _make_om(mode="live") as (om, _, _, session):
            session.scalar.return_value = trade_mock
            om._reconcile_trade = MagicMock()
            om.on_order_event({"OrderID": "LIVE-ORD-001", "OrderState": 5})
        om._reconcile_trade.assert_not_called()

    def test_paper_trade_already_filled_does_not_reconcile(self):
        """ペーパー注文は発注時点でFILLED済みのため、イベント受信時に何もしない"""
        trade_mock = MagicMock(status="FILLED")
        with _make_om(mode="paper") as (om, _, _, session):
            session.scalar.return_value = trade_mock
            om._reconcile_trade = MagicMock()
            om.on_order_event({"OrderID": "PAPER-BUY-xxx", "OrderState": 5})
        om._reconcile_trade.assert_not_called()

    def test_unknown_order_id_ignored(self):
        """DBに無い注文IDの通知は無視する（API照会もしない）"""
        with _make_om(mode="live") as (om, client, _, session):
            session.scalar.return_value = None
            om._reconcile_trade = MagicMock()
            om.on_order_event({"OrderID": "ORD", "OrderState": 5})
        om._reconcile_trade.assert_not_called()
        client.get_orders.assert_not_called()

    def test_missing_order_id_ignored(self):
        with _make_om(mode="live") as (om, _, _, _):
            om._reconcile_trade = MagicMock()
            om.on_order_event({"OrderState": 5})
        om._reconcile_trade.assert_not_called()


# ─── Medium #8: ペーパートレードが FILLED で保存される ──────────────────


def _added_trade(session) -> object:
    """session.add() で渡された全オブジェクトから Trade のものを取り出す。

    _record_trade() は Phase5/4.2 でOrderIntentも同じセッションへ add するため、
    最初に add されたオブジェクトとは限らない（型で識別する）。
    """
    import src.data.database as db_mod
    calls = session.add.call_args_list
    assert calls, "session.add が呼ばれていない"
    for c in calls:
        obj = c[0][0]
        if isinstance(obj, db_mod.Trade):
            return obj
    raise AssertionError("Trade が session.add されていない")


class TestPaperTradeFilledStatus:
    def _first_trade_added(self, session) -> object:
        return _added_trade(session)

    def test_paper_buy_status_is_filled(self):
        """ペーパー BUY → status='FILLED'"""
        with _make_om(mode="paper") as (om, _, _, session):
            om.buy("7203", 1000.0, 100)

        trade = self._first_trade_added(session)
        assert trade.status == "FILLED", f"期待=FILLED, 実際={trade.status}"

    def test_paper_buy_has_filled_at(self):
        """ペーパー BUY → filled_at が datetime で設定されている"""
        with _make_om(mode="paper") as (om, _, _, session):
            om.buy("7203", 1000.0, 100)

        trade = self._first_trade_added(session)
        assert trade.filled_at is not None, "filled_at が None → 損益集計に含まれない"
        assert isinstance(trade.filled_at, datetime)

    def test_paper_sell_status_is_filled(self):
        """ペーパー SELL → status='FILLED'"""
        with _make_om(mode="paper") as (om, _, _, session):
            om.sell("7203", 1100.0, 100)

        trade = self._first_trade_added(session)
        assert trade.status == "FILLED"

    def test_paper_sell_has_filled_at(self):
        """ペーパー SELL → filled_at が設定されている"""
        with _make_om(mode="paper") as (om, _, _, session):
            om.sell("7203", 1100.0, 100)

        trade = self._first_trade_added(session)
        assert trade.filled_at is not None

    def test_live_buy_status_is_pending(self):
        """ライブ BUY → 約定前は PENDING（WebSocket 約定後に FILLED）"""
        with _make_om(mode="live") as (om, _, _, session):
            om.buy("7203", 1000.0, 100)

        trade = self._first_trade_added(session)
        assert trade.status == "PENDING"


class TestTradeSectorRecorded:
    """再レビュー P1-3: buy() がセクターをTradeに記録すること
    （新規銘柄はPosition未作成のため、未約定BUYのセクター引当をTrade.sectorから
    解決できるようにする）"""

    def _first_trade_added(self, session) -> object:
        return _added_trade(session)

    def test_paper_buy_records_sector(self):
        with _make_om(mode="paper") as (om, _, _, session):
            om.buy("7203", 1000.0, 100, sector="Technology")
        trade = self._first_trade_added(session)
        assert trade.sector == "Technology"

    def test_live_buy_records_sector(self):
        with _make_om(mode="live") as (om, _, _, session):
            om.buy("7203", 1000.0, 100, sector="Technology")
        trade = self._first_trade_added(session)
        assert trade.sector == "Technology"

    def test_buy_without_sector_records_none(self):
        with _make_om(mode="paper") as (om, _, _, session):
            om.buy("7203", 1000.0, 100)
        trade = self._first_trade_added(session)
        assert trade.sector is None
