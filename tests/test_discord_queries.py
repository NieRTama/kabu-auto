"""
Discordリモコンの照会コマンド（discord_queries）のテスト

従来の positions は取得単価しか出ず、「今どうなっているか」が分からなかった。
外出先から現在の取引状況を確認できるよう、含み損益・約定履歴・損益サマリ・
未約定注文を追加した。

集計は pnl_report の既存関数を使い、ここは表示の整形だけを担う
（同じ数字が日次レポートとDiscordで食い違わないようにするため）。
"""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.core import discord_queries as dq


def _pos(symbol, qty, avg):
    p = MagicMock()
    p.symbol, p.quantity, p.avg_cost = symbol, qty, avg
    return p


def _trade(symbol, side, qty, price, when, pnl=None, status="FILLED"):
    t = MagicMock()
    t.symbol, t.side, t.quantity, t.status = symbol, side, qty, status
    t.filled_quantity, t.filled_price, t.filled_at, t.pnl = qty, price, when, pnl
    t.price = price
    return t


def _session_with(rows):
    ctx = MagicMock()
    ctx.__enter__.return_value.scalars.return_value.all.return_value = rows
    return ctx


class TestPositions:
    def _run(self, positions, closes, capital=0.0):
        holdings = {
            "count": len(positions), "cost": 16000.0, "market_value": 18000.0,
            "unrealized": 2000.0, "pct": 0.004, "unpriced": [],
        }
        with patch.object(dq, "get_session", return_value=_session_with(positions)), \
             patch("src.data.market_data.latest_closes", return_value=closes), \
             patch("src.core.pnl_report.build_holdings", return_value=holdings):
            return dq.format_positions(capital)

    def test_no_positions(self):
        with patch.object(dq, "get_session", return_value=_session_with([])):
            assert dq.format_positions() == "保有建玉: なし"

    def test_shows_current_price_and_unrealized(self):
        """取得単価だけでなく現在値・含み損益を出す（このコマンドの目的）"""
        out = self._run([_pos("9432", 100, 160.0)], {"9432": 180.0})
        assert "9432" in out
        assert "取得160" in out and "現在180" in out
        assert "+2,000円" in out and "+12.5%" in out

    def test_shows_loss_without_plus_sign(self):
        out = self._run([_pos("9432", 100, 160.0)], {"9432": 150.0})
        assert "-1,000円" in out

    def test_unpriced_symbol_is_not_shown_as_zero(self):
        """価格が取れない銘柄を「損益0」に見せない（実態を隠さない）"""
        out = self._run([_pos("7203", 100, 2000.0)], {})
        assert "現在値取得不可" in out

    def test_includes_total_line(self):
        out = self._run([_pos("9432", 100, 160.0)], {"9432": 180.0})
        assert "合計:" in out and "評価額" in out


class TestTodayTrades:
    def _run(self, trades, day="2026-09-01"):
        with patch.object(dq, "get_session", return_value=_session_with(trades)), \
             patch("src.core.clock.today") as today_mock:
            today_mock.return_value.isoformat.return_value = day
            return dq.format_today_trades(day)

    def test_no_trades(self):
        assert "なし" in self._run([])

    def test_lists_buy_and_sell(self):
        rows = [
            _trade("9432", "BUY", 100, 160.0, datetime(2026, 9, 1, 9, 5)),
            _trade("6753", "SELL", 100, 700.0, datetime(2026, 9, 1, 14, 30), pnl=5000.0),
        ]
        out = self._run(rows)
        assert "2件" in out
        assert "09:05 買 9432 100株 @160" in out
        assert "14:30 売 6753 100株 @700" in out

    def test_shows_realized_pnl_total(self):
        rows = [_trade("6753", "SELL", 100, 700.0, datetime(2026, 9, 1, 14, 30), pnl=5000.0)]
        out = self._run(rows)
        assert "+5,000円" in out
        assert "本日の確定損益" in out

    def test_excludes_other_days(self):
        rows = [_trade("9432", "BUY", 100, 160.0, datetime(2026, 8, 31, 9, 5))]
        assert "なし" in self._run(rows, day="2026-09-01")


class TestPnl:
    def test_formats_all_periods(self):
        from src.core.pnl_report import PeriodPnL
        p = PeriodPnL(label="当日", realized_pnl=1000.0, pct=0.002,
                      win_count=2, loss_count=1)
        report = {"daily": p, "weekly": p, "monthly": p, "overall": p}
        with patch("src.core.pnl_report.build_report", return_value=report):
            out = dq.format_pnl(500000.0)
        assert "損益サマリ" in out
        assert out.count("当日") == 4  # 4期間ぶん（同じモックを使っているため）


class TestOpenOrders:
    def _run(self, orders):
        with patch.object(dq, "get_session", return_value=_session_with(orders)):
            return dq.format_open_orders()

    def test_no_open_orders(self):
        assert self._run([]) == "未約定注文: なし"

    def test_lists_open_orders(self):
        o = _trade("9432", "BUY", 100, 160.0, None, status="PENDING")
        o.filled_quantity = 0
        out = self._run([o])
        assert "1件" in out
        assert "買 9432 100株 @160" in out
        assert "PENDING" in out

    def test_shows_partial_fill(self):
        o = _trade("9432", "BUY", 100, 160.0, None, status="PARTIALLY_FILLED")
        o.filled_quantity = 30
        assert "約定30株" in self._run([o])


class TestWiring:
    def test_commands_registered(self):
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        # 登録は {name: 関数} と {name: (関数, 説明)} の両形式を許す
        import re
        registered = set(re.findall(r'"(\w+)":\s*\(?_cmd_\w+', src))
        for cmd in ("positions", "today", "pnl", "orders"):
            assert cmd in registered, f"{cmd} コマンドが登録されていない"

    def test_queries_are_read_only(self):
        """照会コマンドが発注・状態変更をしないこと（誤操作の余地を作らない）"""
        import inspect
        src = inspect.getsource(dq)
        for forbidden in ("order_mgr", "buy(", "sell(", "halt.engage", "session.commit"):
            assert forbidden not in src, f"照会モジュールに変更操作が入っている: {forbidden}"

    def test_all_registered_commands_are_defined(self):
        """登録したコマンド名に対応する関数が実在すること（NameError の再発防止）。

        2026-09-01、置換ミスで _cmd_today 等の定義が欠けたまま登録だけが残り、
        起動時に NameError で落ちた。構文チェックでは検出できず、
        「起動せずにコミットした」ことが原因だった。
        """
        import re
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        registered = set(re.findall(r'"\w+":\s*\(?(_cmd_\w+)', src))
        defined = set(re.findall(r'def (_cmd_\w+)\(', src))
        missing = registered - defined
        assert not missing, f"登録済みだが未定義のコマンド関数: {missing}"
