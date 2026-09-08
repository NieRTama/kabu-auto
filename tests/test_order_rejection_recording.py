"""
発注拒否の記録・緊急全決済の成否通知のテスト（2026-09-08 のレビューで発覚）

背景: 本日拒否された3件の発注（買い2件・売り1件）がDBに一切記録されていなかった。
send_order() が HTTP 4xx/5xx を返すと kabu_client._request() が例外を投げ、
_live_buy/_live_sell/_live_sell_market/place_stop_loss の is_accepted() 判定に
到達する前に except Exception でログのみ・記録なしに落ちていた。取引履歴・
ダッシュボードから「発注を試みて失敗した」事実が消える。

もう1点、close_all_positions() は sell_market() の戻り値を見ておらず、拒否されても
「実行しました」というログだけで完了扱いになっていた。最も緊急性の高い操作で
失敗が一番見えにくい、という組み合わせだった。
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import requests

import src.execution.order_manager as mod
from src.execution import order_status as st


def _cfg(mode="live"):
    m = MagicMock()
    m.get_section.side_effect = lambda s: {
        "trading": {"mode": mode, "order_timeout_seconds": 300, "daily_order_limit": 100},
        "kabu_station": {"password": "pw"},
    }.get(s, {})
    return m


@contextmanager
def _make_om(mode="live"):
    cfg_mock = _cfg(mode)
    client = MagicMock()
    risk = MagicMock()
    risk.can_place_order.return_value = (True, "")
    session = MagicMock()
    session.scalar.return_value = None
    added = []
    session.add.side_effect = lambda obj: added.append(obj)

    @contextmanager
    def ctx():
        yield session

    with patch.object(mod, "cfg", cfg_mock), patch.object(mod, "get_session", ctx):
        om = mod.OrderManager(client, risk)
        yield om, client, added


def _http_error(code: int, message: str) -> requests.HTTPError:
    """kabu_client._request() が本文つきで投げる例外を再現する。"""
    return requests.HTTPError(f"500 Internal Server Error for POST /sendorder: Code={code} {message}")


class TestRejectedOrderIsRecordedOnException:
    """HTTPエラー（例外経路）で拒否された発注もDBへ記録されること。"""

    def test_buy_records_rejected_trade_on_http_error(self):
        with _make_om("live") as (om, client, added):
            client.send_order.side_effect = _http_error(100031, "「預り区分」をご確認ください")
            result = om._live_buy("3387", 757.0, 300)
        assert result is None
        trade = next(t for t in added if hasattr(t, "status"))
        assert trade.status == st.REJECTED
        assert trade.symbol == "3387"
        assert "100031" in trade.rationale
        assert "預り区分" in trade.rationale

    def test_sell_limit_records_rejected_trade_on_http_error(self):
        with _make_om("live") as (om, client, added):
            client.send_order.side_effect = _http_error(100378, "指定された市場でのお取引はお受けできません。")
            result = om._live_sell("9432", 170.0, 100)
        assert result is None
        trade = next(t for t in added if hasattr(t, "status"))
        assert trade.status == st.REJECTED
        assert "100378" in trade.rationale

    def test_sell_market_records_rejected_trade_on_http_error(self):
        """実際に本日9432で起きたケースそのもの"""
        with _make_om("live") as (om, client, added):
            client.send_order.side_effect = _http_error(100378, "指定された市場でのお取引はお受けできません。")
            result = om._live_sell_market("9432", 100)
        assert result is None
        trade = next(t for t in added if hasattr(t, "status"))
        assert trade.status == st.REJECTED
        assert trade.symbol == "9432"
        assert trade.side == "SELL"

    def test_stop_loss_records_rejected_trade_on_http_error(self):
        """place_stop_loss は従来、拒否時に一切記録していなかった（is_accepted=False時も）"""
        with _make_om("live") as (om, client, added):
            client.send_order.side_effect = _http_error(100378, "指定された市場でのお取引はお受けできません。")
            result = om.place_stop_loss("9432", 100, 168.0)
        assert result is None
        trade = next(t for t in added if hasattr(t, "status"))
        assert trade.status == st.REJECTED

    def test_stop_loss_records_rejected_trade_when_result_nonzero(self):
        """例外にならない拒否（Result!=0）でも同様に記録する（従来は無記録だった）"""
        with _make_om("live") as (om, client, added):
            client.send_order.return_value = {"Result": 1, "Message": "board closed"}
            result = om.place_stop_loss("9432", 100, 168.0)
        assert result is None
        trade = next(t for t in added if hasattr(t, "status"))
        assert trade.status == st.REJECTED
        assert "board closed" in trade.rationale


class TestRejectedOrderIsRecordedOnNonZeroResult:
    """従来どおりの拒否経路（HTTP200・Result!=0）でも同じ記録形式になること（退行なし）。"""

    def test_buy_still_records_on_result_nonzero(self):
        with _make_om("live") as (om, client, added):
            client.send_order.return_value = {"Result": 1, "Message": "insufficient funds"}
            result = om._live_buy("7203", 1000.0, 100)
        assert result is None
        trade = next(t for t in added if hasattr(t, "status"))
        assert trade.status == st.REJECTED
        assert "insufficient funds" in trade.rationale


class TestCloseAllPositionsReporting:
    """close_all_positions は成否を集計し、失敗があれば🔴で通知する。"""

    def _pos(self, symbol="7203", qty=100):
        p = MagicMock()
        p.symbol = symbol
        p.quantity = qty
        return p

    def test_paper_failure_triggers_alert(self):
        with _make_om("paper") as (om, client, added):
            with patch.object(mod, "get_session") as gs:
                session = MagicMock()
                scal = MagicMock()
                scal.all.return_value = [self._pos("7203"), self._pos("9432")]
                session.scalars.return_value = scal

                @contextmanager
                def ctx():
                    yield session
                gs.side_effect = ctx
                om.sell_market = MagicMock(side_effect=[None, "OID-1"])  # 1件失敗
                with patch.object(mod, "alert") as alert_mock:
                    om.close_all_positions()
        assert alert_mock.called
        title = alert_mock.call_args[0][0]
        assert "失敗" in title
        assert "9432" not in alert_mock.call_args[0][0]  # 失敗銘柄は本文側
        assert "7203" in alert_mock.call_args[0][1]

    def test_paper_all_success_reports_completion(self):
        with _make_om("paper") as (om, client, added):
            with patch.object(mod, "get_session") as gs:
                session = MagicMock()
                scal = MagicMock()
                scal.all.return_value = [self._pos("7203")]
                session.scalars.return_value = scal

                @contextmanager
                def ctx():
                    yield session
                gs.side_effect = ctx
                om.sell_market = MagicMock(return_value="OID-1")
                with patch.object(mod, "alert") as alert_mock:
                    om.close_all_positions()
        assert alert_mock.called
        assert "完了" in alert_mock.call_args[0][0]
        assert alert_mock.call_args.kwargs.get("level") == mod.LEVEL_INFO

    def test_no_positions_does_not_alert(self):
        with _make_om("paper") as (om, client, added):
            with patch.object(mod, "get_session") as gs:
                session = MagicMock()
                scal = MagicMock()
                scal.all.return_value = []
                session.scalars.return_value = scal

                @contextmanager
                def ctx():
                    yield session
                gs.side_effect = ctx
                with patch.object(mod, "alert") as alert_mock:
                    om.close_all_positions()
        alert_mock.assert_not_called()

    def test_live_failure_is_reported(self):
        """ブローカー正本(/positions)を使う経路でも同様に失敗を通知する"""
        with _make_om("live") as (om, client, added):
            client.get_positions.return_value = [
                {"Symbol": "7203", "LeavesQty": 100.0},
                {"Symbol": "9432", "LeavesQty": 100.0},
            ]
            om.sell_market = MagicMock(side_effect=[None, "OID-1"])
            with patch.object(mod, "alert") as alert_mock:
                om.close_all_positions()
        assert any("失敗" in c.args[0] for c in alert_mock.call_args_list)

    def test_live_all_success_is_reported(self):
        with _make_om("live") as (om, client, added):
            client.get_positions.return_value = [{"Symbol": "7203", "LeavesQty": 100.0}]
            om.sell_market = MagicMock(return_value="OID-1")
            with patch.object(mod, "alert") as alert_mock:
                om.close_all_positions()
        assert any("完了" in c.args[0] for c in alert_mock.call_args_list)
