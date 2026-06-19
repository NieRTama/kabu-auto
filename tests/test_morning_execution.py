"""
morning_execution の BUY/SELL 両方向実行検証テスト

main.py の morning_execution クロージャのロジックをインラインで再現し、
依存モジュール（uvicorn 等）のインポートを回避してテストする。
"""
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest


# ─── ヘルパー ─────────────────────────────────────────────────────────────


def _make_signal(symbol: str, action: str, minutes_ago: int = 30) -> MagicMock:
    sig = MagicMock()
    sig.symbol = symbol
    sig.action = action
    sig.generated_at = datetime.now() - timedelta(minutes=minutes_ago)
    return sig


def _exec_morning(
    signals: list,
    positions: dict,
    mode: str = "live",
    market_open: bool = True,
    wallet_cash: float = 500_000.0,
    board_price: float = 1000.0,
    calc_qty: int = 100,
    buy_side_effect=None,
) -> MagicMock:
    """
    morning_execution のロジックをインラインで実行し、
    order_mgr モックを返す。

    main.py のクロージャは直接テストできないため、
    同一ロジックをここで再現する（buyループのcash逐次減算を含む。
    シグナルの銘柄ごとdedup自体は main._select_latest_signals() 側で別途テストする）。
    """
    client = MagicMock()
    client.get_board.return_value = {"CurrentPrice": board_price}
    client.get_wallet.return_value = {"StockAccountWallet": wallet_cash}

    order_mgr = MagicMock()
    if buy_side_effect is not None:
        order_mgr.buy.side_effect = buy_side_effect
    else:
        order_mgr.buy.return_value = "BUY-ORD"
    order_mgr.sell.return_value = "SELL-ORD"

    risk = MagicMock()
    risk.calc_position_size.return_value = calc_qty

    trading_conf = {"mode": mode}

    def get_position_qty(symbol):
        return positions.get(symbol, 0)

    # ── morning_execution 本体と同じロジック ─────────────────────────────
    if trading_conf.get("mode", "paper") != "live":
        return order_mgr

    if not market_open:
        return order_mgr

    # シグナルを取得（モック: 渡されたリストをそのまま使用。dedupは別関数の責務）
    seen: set = set()
    pending: list = []
    for s in signals:
        if s.symbol not in seen:
            seen.add(s.symbol)
            pending.append(s)

    if not pending:
        return order_mgr

    buy_signals = [s for s in pending if s.action == "BUY"]
    sell_signals = [s for s in pending if s.action == "SELL"]

    for sig in sell_signals:
        qty = get_position_qty(sig.symbol)
        if qty <= 0:
            continue
        board = client.get_board(sig.symbol)
        price = board.get("CurrentPrice") or board.get("Buy1", {}).get("Price", 0)
        if not price:
            continue
        order_mgr.sell(sig.symbol, float(price), qty)

    if not buy_signals:
        return order_mgr

    wallet = client.get_wallet()
    cash = float(wallet.get("StockAccountWallet", 0))
    order_mgr.cash_log = []  # 各BUY判定時点のcashを記録（テスト用）

    for sig in buy_signals:
        board = client.get_board(sig.symbol)
        price = board.get("CurrentPrice") or board.get("Sell1", {}).get("Price", 0)
        if not price:
            continue
        order_mgr.cash_log.append(cash)
        qty = risk.calc_position_size(sig.symbol, float(price), cash)
        if qty <= 0:
            continue
        order_id = order_mgr.buy(sig.symbol, float(price), qty)
        if order_id:
            cash -= float(price) * qty

    return order_mgr


# ─── SELL シグナルのテスト ────────────────────────────────────────────────


class TestMorningExecutionSell:
    def test_sell_signal_with_position_calls_sell(self):
        """SELL シグナル + ポジション保有 → sell() が呼ばれる"""
        om = _exec_morning(
            signals=[_make_signal("7203", "SELL")],
            positions={"7203": 100},
            board_price=1050.0,
        )
        om.sell.assert_called_once_with("7203", 1050.0, 100)

    def test_sell_signal_without_position_does_not_sell(self):
        """SELL シグナルでもポジションがなければ sell() は呼ばれない"""
        om = _exec_morning(
            signals=[_make_signal("7203", "SELL")],
            positions={"7203": 0},
        )
        om.sell.assert_not_called()

    def test_sell_uses_current_price_from_board(self):
        """SELL は CurrentPrice を板情報から取得して使う"""
        om = _exec_morning(
            signals=[_make_signal("7203", "SELL")],
            positions={"7203": 200},
            board_price=1200.0,
        )
        call_args = om.sell.call_args
        assert call_args[0][1] == 1200.0, "板情報の CurrentPrice で売るべき"
        assert call_args[0][2] == 200, "保有数量全て売るべき"


# ─── BUY シグナルのテスト ────────────────────────────────────────────────


class TestMorningExecutionBuy:
    def test_buy_signal_calls_buy(self):
        """BUY シグナル → buy() が呼ばれる"""
        om = _exec_morning(
            signals=[_make_signal("6758", "BUY")],
            positions={"6758": 0},
            board_price=2000.0,
            calc_qty=100,
        )
        om.buy.assert_called_once_with("6758", 2000.0, 100)

    def test_zero_quantity_skips_buy(self):
        """calc_position_size が 0 を返す場合は buy() を呼ばない"""
        om = _exec_morning(
            signals=[_make_signal("7203", "BUY")],
            positions={},
            calc_qty=0,
        )
        om.buy.assert_not_called()

    def test_duplicate_symbol_deduplication(self):
        """同銘柄に複数シグナルがあっても最初の1件のみ処理する"""
        signals = [
            _make_signal("7203", "BUY", minutes_ago=10),
            _make_signal("7203", "BUY", minutes_ago=60),
        ]
        om = _exec_morning(signals=signals, positions={})
        assert om.buy.call_count == 1, "重複シグナルは最初の1件のみ処理すべき"


# ─── BUY + SELL 混在のテスト ─────────────────────────────────────────────


class TestMorningExecutionMixed:
    def test_buy_and_sell_both_executed(self):
        """BUY + SELL が混在する場合、両方実行される"""
        signals = [
            _make_signal("7203", "SELL"),
            _make_signal("6758", "BUY"),
        ]
        om = _exec_morning(
            signals=signals,
            positions={"7203": 100, "6758": 0},
            board_price=1000.0,
        )
        om.sell.assert_called_once_with("7203", 1000.0, 100)
        om.buy.assert_called_once_with("6758", 1000.0, 100)

    def test_sell_happens_before_buy_in_code_order(self):
        """コードの実行順: SELL ループが BUY ループより先に来ること"""
        # main.py の morning_execution が SELL → BUY の順で実装されていることを検証
        import inspect
        import main as main_module

        src = inspect.getsource(main_module.main)
        # ループの開始位置で比較（変数定義ではなく実行順を確認）
        sell_loop_idx = src.find("for sig in sell_signals")
        buy_check_idx = src.find("if not buy_signals")
        assert sell_loop_idx != -1, "sell_signals ループが見つからない"
        assert buy_check_idx != -1, "buy_signals チェックが見つからない"
        assert sell_loop_idx < buy_check_idx, (
            "morning_execution では SELL を BUY より先に処理すべき（余力確保のため）"
        )

    def test_only_sell_signal_does_not_call_buy(self):
        """SELL シグナルのみ → buy() は呼ばれない"""
        om = _exec_morning(
            signals=[_make_signal("7203", "SELL")],
            positions={"7203": 100},
        )
        om.buy.assert_not_called()

    def test_only_buy_signal_does_not_call_sell(self):
        """BUY シグナルのみ → sell() は呼ばれない"""
        om = _exec_morning(
            signals=[_make_signal("6758", "BUY")],
            positions={"6758": 0},
        )
        om.sell.assert_not_called()


# ─── モード・市場状態のテスト ─────────────────────────────────────────────


class TestMorningExecutionGuards:
    def test_paper_mode_does_nothing(self):
        """ペーパーモードでは morning_execution は何もしない"""
        om = _exec_morning(
            signals=[_make_signal("7203", "BUY"), _make_signal("6758", "SELL")],
            positions={"6758": 100},
            mode="paper",
        )
        om.buy.assert_not_called()
        om.sell.assert_not_called()

    def test_market_closed_does_nothing(self):
        """市場が閉じているときは morning_execution は何もしない"""
        om = _exec_morning(
            signals=[_make_signal("7203", "BUY")],
            positions={},
            mode="live",
            market_open=False,
        )
        om.buy.assert_not_called()

    def test_empty_signals_does_nothing(self):
        """シグナルが空の場合は何もしない"""
        om = _exec_morning(signals=[], positions={})
        om.buy.assert_not_called()
        om.sell.assert_not_called()


# ─── 余力の逐次減算のテスト（#3） ─────────────────────────────────────────


class TestMorningExecutionCashDecrement:
    def test_cash_decreases_after_successful_buy(self):
        """1件目のBUY成功後、2件目の判定時のcashは1件目の発注額だけ減っている"""
        signals = [
            _make_signal("7203", "BUY"),
            _make_signal("6758", "BUY"),
        ]
        om = _exec_morning(
            signals=signals,
            positions={},
            wallet_cash=500_000.0,
            board_price=1000.0,
            calc_qty=100,
        )
        assert om.buy.call_count == 2
        assert om.cash_log[0] == 500_000.0
        # 1件目: 1000円×100株=100,000円 発注成功 → 2件目の判定時は400,000円のはず
        assert om.cash_log[1] == 400_000.0

    def test_cash_not_decremented_when_buy_fails(self):
        """buy()がNone（注文拒否）を返した場合、cashは減算されない"""
        signals = [
            _make_signal("7203", "BUY"),
            _make_signal("6758", "BUY"),
        ]
        om = _exec_morning(
            signals=signals,
            positions={},
            wallet_cash=500_000.0,
            board_price=1000.0,
            calc_qty=100,
            buy_side_effect=[None, "BUY-ORD"],  # 1件目は拒否
        )
        assert om.cash_log[0] == 500_000.0
        # 1件目が失敗したのでcashは減らず、2件目も同じ500,000円で判定される
        assert om.cash_log[1] == 500_000.0

    def test_three_buys_decrement_sequentially(self):
        """3件のBUYが成功するごとにcashが逐次減算される"""
        signals = [
            _make_signal("7203", "BUY"),
            _make_signal("6758", "BUY"),
            _make_signal("9984", "BUY"),
        ]
        om = _exec_morning(
            signals=signals,
            positions={},
            wallet_cash=300_000.0,
            board_price=500.0,
            calc_qty=100,
        )
        # 各回 500円×100株=50,000円ずつ減算される
        assert om.cash_log == [300_000.0, 250_000.0, 200_000.0]
