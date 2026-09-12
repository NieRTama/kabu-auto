"""過去検証の執行アダプタ（src/backtest/execution.py）のテスト

退出「意図」を約定へ変換し、スリッページと手数料を一元的に控除する。
Tの引けで判断し T+1 の寄りで約定する（同じ終値で判断・約定しない）。
"""
from datetime import date

import pytest

from src.core import config as cfg
from src.backtest import execution
from src.strategy import policy


def _costs(slip=0.001, comm=0.0) -> execution.CostConfig:
    return execution.CostConfig(slippage_pct=slip, commission_pct=comm)


def _bar(session=date(2026, 9, 2), o=1000.0, h=1010.0, l=990.0, c=1005.0):
    return policy.Observation(session=session, open=o, high=h, low=l, close=c)


class TestConfigFromSettings:
    def test_reads_backtest_section(self):
        cfg.load("config.yaml")
        costs = execution.config_from_settings()
        backtest = cfg.get_section("backtest")
        assert costs.slippage_pct == backtest["slippage_pct"]
        assert costs.commission_pct == backtest["commission_pct"]


class TestFillPrices:
    def test_buy_is_unfavourable(self):
        """買いはスリッページ分だけ高く約定する"""
        assert execution.buy_fill_price(1000.0, _costs(slip=0.001)) == pytest.approx(1001.0)

    def test_sell_is_unfavourable(self):
        """売りはスリッページ分だけ安く約定する"""
        assert execution.sell_fill_price(1000.0, _costs(slip=0.001)) == pytest.approx(999.0)

    def test_matches_existing_engine_convention(self):
        """既存 engine.py の _buy_fill_price/_sell_fill_price と同じ丸め（小数2桁）"""
        assert execution.buy_fill_price(1234.567, _costs(slip=0.001)) == pytest.approx(1235.80)
        assert execution.sell_fill_price(1234.567, _costs(slip=0.001)) == pytest.approx(1233.33)


class TestEntryFill:
    def test_fills_at_next_session_open(self):
        """Tの引けで決めた買いは T+1 の寄りで約定する"""
        next_bar = _bar(session=date(2026, 9, 3), o=1020.0, c=1050.0)
        fill = execution.entry_fill(next_bar, quantity=100, costs=_costs(slip=0.001))
        assert fill.at == date(2026, 9, 3)
        assert fill.price == pytest.approx(1021.02)  # 1020 * 1.001
        assert fill.quantity == 100
        assert fill.reason == "ENTRY"

    def test_does_not_use_the_decision_session_close(self):
        """判断した日の終値では約定しない（F04の回帰防止）"""
        decision_bar = _bar(session=date(2026, 9, 2), c=1005.0)
        next_bar = _bar(session=date(2026, 9, 3), o=1020.0)
        fill = execution.entry_fill(next_bar, quantity=100, costs=_costs(slip=0.0))
        assert fill.at != decision_bar.session
        assert fill.price != pytest.approx(decision_bar.close)


def _intent(reason=policy.STOP_LINE, trigger=930.0, order_type="STOP"):
    return policy.ExitIntent(reason=reason, trigger_price=trigger, order_type=order_type)


class TestExitFillStop:
    def test_normal_intraday_touch_fills_at_trigger(self):
        """寄りが基準線より上なら、基準線で約定したとみなす"""
        bar = _bar(session=date(2026, 9, 2), o=1000.0, l=920.0)
        fill = execution.exit_fill(_intent(trigger=930.0), bar, None,
                                   quantity=100, costs=_costs(slip=0.0))
        assert fill is not None
        assert fill.at == date(2026, 9, 2)
        assert fill.price == pytest.approx(930.0)
        assert fill.reason == policy.STOP_LINE

    def test_gap_down_fills_at_open_not_trigger(self):
        """寄りが既に基準線を割っていたら min(open, trigger) で約定する。

        現行エンジンは基準線ちょうどで約定できる前提でギャップダウンに楽観的。
        """
        bar = _bar(session=date(2026, 9, 2), o=900.0, l=880.0)
        fill = execution.exit_fill(_intent(trigger=930.0), bar, None,
                                   quantity=100, costs=_costs(slip=0.0))
        assert fill.price == pytest.approx(900.0)

    def test_slippage_applied_unfavourably_on_exit(self):
        """退出はスリッページ分だけ安く約定する"""
        bar = _bar(session=date(2026, 9, 2), o=1000.0, l=920.0)
        fill = execution.exit_fill(_intent(trigger=930.0), bar, None,
                                   quantity=100, costs=_costs(slip=0.001))
        assert fill.price == pytest.approx(929.07)  # 930 * 0.999


class TestExitFillMarket:
    def test_signal_sell_fills_at_next_open(self):
        """売りシグナルは翌営業日の寄りで成行約定する"""
        bar = _bar(session=date(2026, 9, 2), c=1005.0)
        next_bar = _bar(session=date(2026, 9, 3), o=990.0)
        fill = execution.exit_fill(
            _intent(reason=policy.SIGNAL_SELL, trigger=None, order_type="MARKET"),
            bar, next_bar, quantity=100, costs=_costs(slip=0.0))
        assert fill.at == date(2026, 9, 3)
        assert fill.price == pytest.approx(990.0)

    def test_time_limit_fills_at_next_open(self):
        """満了も翌営業日の寄り。満了日の終値で判断して同じ終値で約定しない"""
        bar = _bar(session=date(2026, 9, 2), c=1005.0)
        next_bar = _bar(session=date(2026, 9, 3), o=990.0)
        fill = execution.exit_fill(
            _intent(reason=policy.TIME_LIMIT, trigger=None, order_type="MARKET"),
            bar, next_bar, quantity=100, costs=_costs(slip=0.0))
        assert fill.at == date(2026, 9, 3)
        assert fill.price != pytest.approx(bar.close)

    def test_unfilled_when_no_next_bar(self):
        """翌足が無ければ未約定（Noneを返す）。未成熟として扱えるように"""
        bar = _bar(session=date(2026, 9, 2))
        fill = execution.exit_fill(
            _intent(reason=policy.TIME_LIMIT, trigger=None, order_type="MARKET"),
            bar, None, quantity=100, costs=_costs())
        assert fill is None


class TestNetReturn:
    def test_positive_return_without_commission(self):
        entry = execution.Fill(at=date(2026, 9, 2), price=1000.0, quantity=100, reason="ENTRY")
        exit_ = execution.Fill(at=date(2026, 9, 5), price=1100.0, quantity=100, reason=policy.TRAILING)
        assert execution.net_return(entry, exit_, _costs(comm=0.0)) == pytest.approx(0.10)

    def test_commission_reduces_return(self):
        """手数料は売買それぞれの約定代金に掛かる"""
        entry = execution.Fill(at=date(2026, 9, 2), price=1000.0, quantity=100, reason="ENTRY")
        exit_ = execution.Fill(at=date(2026, 9, 5), price=1100.0, quantity=100, reason=policy.TRAILING)
        # 買い100,000 / 売り110,000 / 手数料0.1%ずつ = 100 + 110 = 210
        expected = (110000.0 - 100000.0 - 210.0) / 100000.0
        assert execution.net_return(entry, exit_, _costs(comm=0.001)) == pytest.approx(expected)

    def test_slippage_is_not_double_counted(self):
        """スリッページは約定価格に織り込み済み。net_returnで再度引かない"""
        costs = _costs(slip=0.001, comm=0.0)
        entry_price = execution.buy_fill_price(1000.0, costs)   # 1001.0
        exit_price = execution.sell_fill_price(1100.0, costs)   # 1098.9
        entry = execution.Fill(at=date(2026, 9, 2), price=entry_price, quantity=100, reason="ENTRY")
        exit_ = execution.Fill(at=date(2026, 9, 5), price=exit_price, quantity=100, reason=policy.TRAILING)
        expected = (exit_price - entry_price) / entry_price
        assert execution.net_return(entry, exit_, costs) == pytest.approx(expected)

    def test_negative_return_on_loss(self):
        entry = execution.Fill(at=date(2026, 9, 2), price=1000.0, quantity=100, reason="ENTRY")
        exit_ = execution.Fill(at=date(2026, 9, 5), price=930.0, quantity=100, reason=policy.STOP_LINE)
        assert execution.net_return(entry, exit_, _costs(comm=0.0)) == pytest.approx(-0.07)


class TestExitFillRejectsUnknownOrderType:
    def test_raises_on_unknown_order_type(self):
        """STOPでもMARKETでもないorder_typeは例外にする（黙って成行約定にしない）"""
        bar = _bar(session=date(2026, 9, 2), c=1005.0)
        next_bar = _bar(session=date(2026, 9, 3), o=990.0)
        bad_intent = policy.ExitIntent(reason="X", trigger_price=None, order_type="LIMIT")
        with pytest.raises(ValueError, match="未知の order_type"):
            execution.exit_fill(bad_intent, bar, next_bar, quantity=100, costs=_costs())


class TestNetReturnRejectsInvalidInput:
    def test_raises_on_quantity_mismatch(self):
        """部分決済（数量不一致）は按分責任が呼び出し側にあるため例外にする"""
        entry = execution.Fill(at=date(2026, 9, 2), price=1000.0, quantity=100, reason="ENTRY")
        exit_ = execution.Fill(at=date(2026, 9, 5), price=1100.0, quantity=50, reason=policy.TRAILING)
        with pytest.raises(ValueError, match="数量が一致しません"):
            execution.net_return(entry, exit_, _costs(comm=0.0))

    def test_raises_on_non_positive_buy_amount(self):
        """buy_amountが0以下は本物の0%リターンと区別するため例外にする"""
        entry = execution.Fill(at=date(2026, 9, 2), price=0.0, quantity=100, reason="ENTRY")
        exit_ = execution.Fill(at=date(2026, 9, 5), price=1100.0, quantity=100, reason=policy.TRAILING)
        with pytest.raises(ValueError, match="buy_amount"):
            execution.net_return(entry, exit_, _costs(comm=0.0))
