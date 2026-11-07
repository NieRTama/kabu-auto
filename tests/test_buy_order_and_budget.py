"""買い候補の処理順と、同じ回の2件目以降の余力のテスト。

2026-10-26 実例: 余力188,060円で 3660（93,400円）を発注した直後、次の候補の「1銘柄上限」が
630円になった。発注ループの逐次減算と position_budget の未約定引当で二重に引いていたため。
本来は (188,060 - 93,400) × 50% = 47,330円。また候補が生成時刻順だったので、
スコア 0.127 の 3660 が、より高い 0.177 の 7872 より先に枠を使った。
"""
from unittest.mock import MagicMock, patch

from src.core import scheduler as scheduler_mod
from src.data.database import Trade, get_session
from src.risk.manager import RiskManager
from src.services.trading import TradingServices, _by_score_desc


def _signal(symbol, score):
    sig = MagicMock()
    sig.symbol, sig.action, sig.combined_score = symbol, "BUY", score
    sig.rule_score, sig.ml_score = score, score
    return sig


def _risk():
    r = RiskManager()
    r._conf = {"max_position_ratio": 0.5}
    return r


def _open_buy(symbol="3660", qty=200, price=467.0):
    with get_session() as s:
        s.add(Trade(order_id=f"o-{symbol}", symbol=symbol, side="BUY", quantity=qty, price=price,
                    status="PENDING"))
        s.commit()


class TestCashForSizing:
    def test_order_of_this_run_still_open_is_not_deducted_twice(self, isolated_db):
        r = _risk()
        _open_buy()  # 93,400円が未約定
        cash = r.cash_for_sizing(188_060.0, committed=93_400.0)
        assert r.position_budget(cash) == 47_330.0

    def test_order_filled_during_the_run_stays_deducted(self, isolated_db):
        """ループ中に約定すると引当は消えるが、今回使った分は引いたままにする"""
        r = _risk()
        assert r.position_budget(r.cash_for_sizing(188_060.0, 93_400.0)) == 47_330.0

    def test_open_orders_from_earlier_run_are_still_reserved(self, isolated_db):
        """前の回の未約定は引当のまま（多重発注の守りを失わない）"""
        r = _risk()
        _open_buy()
        assert r.position_budget(r.cash_for_sizing(188_060.0, r.reserved_buy_total())) == 47_330.0

    def test_earlier_order_filled_during_the_run_stays_deducted(self, isolated_db):
        """開始時の未約定がループ中に約定しても、余力は開始時の値なので引いたままにする"""
        r = _risk()
        assert r.position_budget(r.cash_for_sizing(100_000.0, committed=30_000.0)) == 35_000.0

    def test_first_order_of_the_run_uses_full_cash(self, isolated_db):
        r = _risk()
        assert r.position_budget(r.cash_for_sizing(188_060.0, 0.0)) == 94_030.0


class _Sig:
    def __init__(self, symbol, score):
        self.symbol, self.combined_score = symbol, score


def test_buy_candidates_are_processed_by_score():
    sigs = [_Sig("3660", 0.127), _Sig("7872", 0.177), _Sig("X", None), _Sig("8058", 0.263)]
    assert [s.symbol for s in _by_score_desc(sigs)] == ["8058", "7872", "3660", "X"]


class TestPendingSignalsLoop:
    """10/26 の朝を実際の発注ループで再現する（RiskManager は本物）。"""

    def _run(self):
        client, order_mgr = MagicMock(), MagicMock()
        client.get_wallet.return_value = {"StockAccountWallet": 188_060.0}
        prices = {"3660": 467.0, "7872": 606.0, "3387": 300.0}
        client.get_board.side_effect = lambda sym: {
            "CurrentPrice": prices[sym], "Sell1": {"Price": prices[sym]}, "Buy1": {"Price": prices[sym]}}
        bought = []

        def buy(symbol, price, qty, **kw):
            bought.append((symbol, qty))
            with get_session() as s:  # 発注直後は未約定のまま残る
                s.add(Trade(order_id=f"o-{symbol}", symbol=symbol, side="BUY", quantity=qty,
                            price=price, status="PENDING"))
                s.commit()
            return f"o-{symbol}"

        order_mgr.buy.side_effect = buy
        risk = _risk()
        risk.get_current_prices = lambda syms: {}
        with patch("src.services.trading.cfg") as cfg_mock:
            cfg_mock.get_section.return_value = {"mode": "live", "no_new_buy_minutes_before_close": 0}
            svc = TradingServices(client, risk, order_mgr)
        signals = [_signal("3660", 0.127), _signal("7872", 0.177), _signal("3387", 0.118)]
        with patch.object(scheduler_mod.TradingScheduler, "is_market_open", return_value=True), \
             patch.object(scheduler_mod.TradingScheduler, "is_near_close", return_value=False), \
             patch("src.services.trading._select_latest_signals", return_value=signals), \
             patch("src.services.trading._get_position_qty", return_value=0), \
             patch("src.services.trading._bought_today", return_value=False), \
             patch("src.services.trading.watchlist_store") as watchlist_mock, \
             patch("src.services.trading.load_ohlcv"), \
             patch("src.services.trading.liquidity") as liquidity_mock, \
             patch.object(risk, "can_place_order", return_value=(True, "")):
            watchlist_mock.get_sectors.return_value = {}
            liquidity_mock.check_liquidity.return_value = (True, "")
            liquidity_mock.check_spread.return_value = (True, "")
            svc.morning_execution()
        return bought

    def test_higher_score_first_and_second_order_not_blocked(self, isolated_db):
        bought = self._run()
        assert bought[0] == ("7872", 100), "スコアの高い 7872 が先（94,030円の枠で100株）"
        # 残り 127,460円 → 枠 63,730円: 3660 は100株(46,700円)。旧実装は枠が33,430円で買えなかった
        assert bought[1] == ("3660", 100)
