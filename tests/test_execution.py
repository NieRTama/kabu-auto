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
    client.send_order.return_value = {"OrderId": "LIVE-ORD-001"}
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
        assert ok is False


# ─── Critical #3: ライブモード約定後のポジション更新 ─────────────────────


class TestOnOrderEventPositionUpdate:
    def test_live_fill_triggers_position_update(self):
        """ライブモード OrderState=5 受信時に _update_position_from_fill が呼ばれる"""
        with _make_om(mode="live") as (om, _, _, _):
            om._update_position_from_fill = MagicMock()
            om.on_order_event({"OrderID": "LIVE-ORD-001", "OrderState": 5})

        om._update_position_from_fill.assert_called_once_with("LIVE-ORD-001")

    def test_paper_fill_event_skips_position_update(self):
        """ペーパーモードでは WebSocket 約定イベントでポジション更新しない"""
        with _make_om(mode="paper") as (om, _, _, _):
            om._update_position_from_fill = MagicMock()
            om.on_order_event({"OrderID": "PAPER-BUY-xxx", "OrderState": 5})

        om._update_position_from_fill.assert_not_called()

    def test_non_fill_state_ignored(self):
        """OrderState が 5 以外は _update_position_from_fill を呼ばない"""
        with _make_om(mode="live") as (om, _, _, _):
            om._update_position_from_fill = MagicMock()
            for state in [1, 2, 3, 4]:
                om.on_order_event({"OrderID": "ORD", "OrderState": state})

        om._update_position_from_fill.assert_not_called()


# ─── Medium #8: ペーパートレードが FILLED で保存される ──────────────────


class TestPaperTradeFilledStatus:
    def _first_trade_added(self, session) -> object:
        calls = session.add.call_args_list
        assert calls, "session.add が呼ばれていない"
        return calls[0][0][0]

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
