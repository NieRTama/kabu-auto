"""執行の制約（未約定・部分約定・出来高）のテスト

現行エンジンは「欲しい数量は必ず買える」前提だった。薄商い銘柄では
その日の出来高の何割も自分で買うことはできない（spec §8）。
"""
from datetime import date

import pytest

from src.backtest import execution
from src.strategy import policy


def _costs(slip=0.0, comm=0.0):
    return execution.CostConfig(slippage_pct=slip, commission_pct=comm)


def _bar(session=date(2026, 9, 2), o=1000.0, h=1010.0, l=990.0, c=1005.0):
    return policy.Observation(session=session, open=o, high=h, low=l, close=c)


def _liquidity(share=0.0):
    return execution.LiquidityConfig(max_volume_share=share)


class TestNoLiquidityLimit:
    def test_fills_everything_when_limit_is_disabled(self):
        """max_volume_share=0 は無制限（従来どおりの挙動）"""
        budget = execution.VolumeBudget(_liquidity(share=0.0))
        res = execution.entry_fill_limited(
            _bar(), 1000, _costs(), budget, symbol="7203", volume=100)
        assert res.filled_quantity == 1000
        assert res.fill is not None
        assert res.unfilled_reason is None


class TestVolumeShareLimit:
    def test_fills_fully_within_the_share(self):
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        res = execution.entry_fill_limited(
            _bar(), 100, _costs(), budget, symbol="7203", volume=10_000)
        assert res.filled_quantity == 100
        assert res.unfilled_reason is None

    def test_partial_fill_when_request_exceeds_the_share(self):
        """出来高の10%までしか約定できないなら、その分だけ約定する"""
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        res = execution.entry_fill_limited(
            _bar(), 5000, _costs(), budget, symbol="7203", volume=10_000)
        assert res.requested_quantity == 5000
        assert res.filled_quantity == 1000   # 10,000 × 0.1
        assert res.fill is not None
        assert res.fill.quantity == 1000
        assert res.unfilled_reason is not None
        assert "出来高" in res.unfilled_reason

    def test_partial_fill_rounds_down_to_lot_size(self):
        """部分約定も単元単位に切り捨てる"""
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        res = execution.entry_fill_limited(
            _bar(), 5000, _costs(), budget, symbol="7203", volume=1_050)
        assert res.filled_quantity == 100   # 1,050 × 0.1 = 105 → 単元切り捨てで100

    def test_unfilled_when_share_is_below_one_lot(self):
        """1単元にも満たなければ未約定"""
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        res = execution.entry_fill_limited(
            _bar(), 100, _costs(), budget, symbol="7203", volume=500)
        assert res.filled_quantity == 0
        assert res.fill is None
        assert "単元" in res.unfilled_reason

    def test_unfilled_when_volume_is_zero(self):
        """出来高ゼロ（売買停止等）では約定しない"""
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        res = execution.entry_fill_limited(
            _bar(), 100, _costs(), budget, symbol="7203", volume=0)
        assert res.filled_quantity == 0
        assert res.fill is None
        assert res.unfilled_reason is not None

    def test_repeated_calls_accumulate_consumption_against_the_same_budget(self):
        """同じ VolumeBudget に複数回エントリーすると枠が累積して減る"""
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        first = execution.entry_fill_limited(
            _bar(), 600, _costs(), budget, symbol="7203", volume=10_000)
        second = execution.entry_fill_limited(
            _bar(), 600, _costs(), budget, symbol="7203", volume=10_000)
        assert first.filled_quantity == 600          # 枠1,000株のうち600株
        assert second.filled_quantity == 400         # 残り400株のみ約定
        assert second.unfilled_reason is not None
        third = execution.entry_fill_limited(
            _bar(), 100, _costs(), budget, symbol="7203", volume=10_000)
        assert third.filled_quantity == 0             # 使い切り
        assert third.fill is None


class TestFillPriceIsUnchanged:
    def test_uses_the_same_price_as_the_base_helper(self):
        """約定価格の規約は段階B前半と同じ（寄り × (1+slip)）"""
        bar = _bar(o=1020.0)
        costs = _costs(slip=0.001)
        budget = execution.VolumeBudget(_liquidity())
        res = execution.entry_fill_limited(bar, 100, costs, budget, symbol="7203", volume=0)
        base = execution.entry_fill(bar, 100, costs)
        assert res.fill.price == pytest.approx(base.price)

    def test_fills_at_next_session_open(self):
        bar = _bar(session=date(2026, 9, 3), o=1020.0)
        budget = execution.VolumeBudget(_liquidity())
        res = execution.entry_fill_limited(bar, 100, _costs(), budget, symbol="7203", volume=0)
        assert res.fill.at == date(2026, 9, 3)


class TestBaseHelpersUnchanged:
    """段階B前半の関数は挙動を変えない（dataset.py が依存している）"""

    def test_entry_fill_still_ignores_volume(self):
        bar = _bar(o=1020.0)
        fill = execution.entry_fill(bar, 999_999, _costs())
        assert fill.quantity == 999_999

    def test_exit_fill_still_returns_a_plain_fill(self):
        intent = policy.ExitIntent(
            reason=policy.STOP_LINE, trigger_price=930.0, order_type="STOP")
        fill = execution.exit_fill(intent, _bar(o=1000.0, l=920.0), None, 100, _costs())
        assert fill is not None
        assert fill.quantity == 100


class TestExitVolumeLimit:
    """売りにも出来高の制限が掛かること（外部レビューR09）。

    買いだけに掛けて売りを無制限にすると、900株保有・当日出来高1,000株・
    参加率上限10%でも全量売れる計算になる。損切りを出しても売り切れない
    状況こそが薄商い銘柄のリスクなので、そこを消してはいけない。
    """

    def _intent(self, order_type="MARKET", trigger=None):
        return policy.ExitIntent(
            reason="test", trigger_price=trigger, order_type=order_type)

    def test_market_exit_is_capped_by_volume(self):
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        res = execution.exit_fill_limited(
            self._intent(), _bar(), _bar(date(2026, 9, 3)), 900,
            _costs(), budget, symbol="7203", volume=1_000)
        # 1,000株の10% = 100株まで
        assert res.filled_quantity == 100
        assert res.requested_quantity == 900
        assert res.unfilled_reason is not None
        assert res.fill.quantity == 100

    def test_stop_exit_is_capped_by_volume(self):
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        res = execution.exit_fill_limited(
            self._intent("STOP", trigger=950.0), _bar(), None, 900,
            _costs(), budget, symbol="7203", volume=1_000)
        assert res.filled_quantity == 100
        assert res.fill is not None

    def test_unlimited_sells_everything(self):
        budget = execution.VolumeBudget(_liquidity(share=0.0))
        res = execution.exit_fill_limited(
            self._intent(), _bar(), _bar(date(2026, 9, 3)), 900,
            _costs(), budget, symbol="7203", volume=10)
        assert res.filled_quantity == 900
        assert res.unfilled_reason is None

    def test_returns_unfilled_when_below_one_lot(self):
        budget = execution.VolumeBudget(_liquidity(share=0.01))
        res = execution.exit_fill_limited(
            self._intent(), _bar(), _bar(date(2026, 9, 3)), 900,
            _costs(), budget, symbol="7203", volume=1_000)
        # 1,000の1% = 10株 → 単元(100株)未満
        assert res.fill is None
        assert res.filled_quantity == 0
        assert "単元" in res.unfilled_reason

    def test_market_exit_without_next_bar_is_unfilled(self):
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        res = execution.exit_fill_limited(
            self._intent(), _bar(), None, 900,
            _costs(), budget, symbol="7203", volume=10_000)
        assert res.fill is None
        assert res.filled_quantity == 0

    def test_market_exit_without_next_bar_refunds_the_consumed_budget(self):
        """成行退出が翌足の欠落で未約定に終わったら、消費した枠を戻す

        exit_fill_limited() は budget.allow() で先に枠を消費してから
        exit_fill() を呼ぶ。exit_fill() が None を返す（翌営業日の足が
        無い）場合、消費したぶんを戻さないと、実際には約定していないのに
        枠だけ減った状態が残ってしまう（指摘2）。
        """
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        res = execution.exit_fill_limited(
            self._intent(), _bar(), None, 900,
            _costs(), budget, symbol="7203", volume=10_000)
        assert res.fill is None
        assert res.filled_quantity == 0
        # 出来高10,000株の10% = 1,000株の枠が用意されるはずだが、
        # 約定が成立しなかったので消費前の1,000株のまま戻っている。
        assert budget.remaining("7203") == 1_000
        # 同じ銘柄で改めて約定を試みても、枠が減ったままになっていないこと。
        again = execution.exit_fill_limited(
            self._intent(), _bar(), _bar(date(2026, 9, 3)), 900,
            _costs(), budget, symbol="7203", volume=10_000)
        assert again.filled_quantity == 900
        assert again.fill is not None


class TestVolumeBudgetIsSharedBetweenBuysAndSells:
    """同じ営業日・同じ銘柄の枠は買いと売りで共有する（規約の固定）。

    entry_fill_limited() と exit_fill_limited() の両方を実際に呼び出し、
    片方が消費した枠をもう片方が引き継いで見ることを検証する
    （budget.allow() を直接呼ぶだけでは、両関数が本当に同じ枠を
    参照しているかは確認できない＝指摘3）。
    """

    def _intent(self, order_type="MARKET", trigger=None):
        return policy.ExitIntent(
            reason="test", trigger_price=trigger, order_type=order_type)

    def test_a_sell_consumes_the_budget_a_later_sell_sees(self):
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        first = budget.allow("7203", 10_000, 600)
        second = budget.allow("7203", 10_000, 600)
        assert first == 600          # 枠1,000株のうち600株
        assert second == 400         # 残り400株
        assert budget.allow("7203", 10_000, 100) == 0   # 使い切り

    def test_exit_fill_limited_consumption_blocks_a_later_entry_fill_limited(self):
        """先に売りで900株分の枠を使い切ると、後続の買いが同じ銘柄で約定できない"""
        # 出来高9,000株の10% = 900株がこの銘柄の1日の枠。
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        sell = execution.exit_fill_limited(
            self._intent(), _bar(), _bar(date(2026, 9, 3)), 900,
            _costs(), budget, symbol="7203", volume=9_000)
        assert sell.filled_quantity == 900
        assert budget.remaining("7203") == 0   # 枠を使い切った

        # entry_fill_limited が同じ VolumeBudget を見ていなければ、独自に
        # 出来高9,000株から枠を計算し直して約定できてしまう。実際には
        # 共有の枠が0のため、単元にも満たず約定できない。
        buy = execution.entry_fill_limited(
            _bar(), 500, _costs(), budget, symbol="7203", volume=9_000)
        assert buy.filled_quantity == 0
        assert buy.fill is None

    def test_entry_fill_limited_consumption_blocks_a_later_exit_fill_limited(self):
        """逆に、先に買いで枠を使い切ると、後続の売りが同じ銘柄で約定できない"""
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        buy = execution.entry_fill_limited(
            _bar(), 1_000, _costs(), budget, symbol="7203", volume=10_000)
        assert buy.filled_quantity == 1_000   # 枠を使い切る
        assert budget.remaining("7203") == 0

        sell = execution.exit_fill_limited(
            self._intent(), _bar(), _bar(date(2026, 9, 3)), 500,
            _costs(), budget, symbol="7203", volume=10_000)
        assert sell.filled_quantity == 0
        assert sell.fill is None

    def test_budgets_are_per_symbol(self):
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        assert budget.allow("7203", 10_000, 1_000) == 1_000
        assert budget.allow("9984", 10_000, 1_000) == 1_000

    def test_allowance_is_floored_to_lots(self):
        budget = execution.VolumeBudget(_liquidity(share=0.1))
        # 1,550株の10% = 155株 → 単元切り捨てで100株
        assert budget.allow("7203", 1_550, 1_000) == 100

    def test_remaining_is_none_when_unlimited(self):
        budget = execution.VolumeBudget(_liquidity(share=0.0))
        assert budget.unlimited() is True
        assert budget.remaining("7203") is None
        assert budget.allow("7203", 1, 99_999) == 99_999
