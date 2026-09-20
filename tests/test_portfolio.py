"""ポートフォリオの状態機械（src/backtest/portfolio.py）のテスト

現金・保有・資金競合・セクター上限を実運用（src/risk/manager.py）と
同じ式で扱う。式が2つあると、バックテストで通った条件が運用で弾かれる。
"""
from datetime import date

import pytest

from src.backtest import portfolio as pf


def _holding(symbol="7203", qty=100, avg_cost=1000.0, sector="自動車",
             entry_at=date(2026, 9, 1), peak=1000.0, sessions_held=0):
    return pf.Holding(
        symbol=symbol, quantity=qty, avg_cost=avg_cost, sector=sector,
        entry_at=entry_at, peak_price=peak, sessions_held=sessions_held,
    )


class TestEmptyPortfolio:
    def test_starts_with_cash_and_nothing_held(self):
        p = pf.empty_portfolio(1_000_000.0)
        assert p.cash == pytest.approx(1_000_000.0)
        assert p.holdings == {}
        assert p.reserved == pytest.approx(0.0)

    def test_lot_size_matches_production(self):
        """単元は実運用（risk/manager.py:26）と同じ100株"""
        from src.risk.manager import LOT_SIZE as PROD_LOT_SIZE
        assert pf.LOT_SIZE == PROD_LOT_SIZE


class TestValuation:
    def test_holdings_value_uses_latest_price(self):
        p = pf.Portfolio(cash=0.0, holdings={"7203": _holding(qty=100, avg_cost=1000.0)},
                         reserved=0.0)
        assert pf.holdings_value(p, {"7203": 1200.0}) == pytest.approx(120_000.0)

    def test_falls_back_to_avg_cost_when_price_missing(self):
        """価格が取れない銘柄は取得単価で代用する（実運用と同じ規約）"""
        p = pf.Portfolio(cash=0.0, holdings={"7203": _holding(qty=100, avg_cost=1000.0)},
                         reserved=0.0)
        assert pf.holdings_value(p, {}) == pytest.approx(100_000.0)

    def test_nav_is_cash_plus_holdings(self):
        p = pf.Portfolio(cash=500_000.0,
                         holdings={"7203": _holding(qty=100, avg_cost=1000.0)},
                         reserved=0.0)
        assert pf.nav(p, {"7203": 1200.0}) == pytest.approx(620_000.0)

    def test_nav_ignores_reserved(self):
        """予約は現金の内訳であって総資産を減らさない"""
        a = pf.Portfolio(cash=500_000.0, holdings={}, reserved=0.0)
        b = pf.Portfolio(cash=500_000.0, holdings={}, reserved=100_000.0)
        assert pf.nav(a, {}) == pytest.approx(pf.nav(b, {}))

    def test_held_value_of_unheld_symbol_is_zero(self):
        p = pf.empty_portfolio(1_000_000.0)
        assert pf.held_value(p, "7203", 1000.0) == pytest.approx(0.0)

    def test_held_value_uses_current_price_not_book(self):
        """保有評価は簿価ではなく現在値（実運用 _held_value と同じ）"""
        p = pf.Portfolio(cash=0.0, holdings={"7203": _holding(qty=100, avg_cost=1000.0)},
                         reserved=0.0)
        assert pf.held_value(p, "7203", 1500.0) == pytest.approx(150_000.0)


class TestApplyBuy:
    def test_reduces_cash_and_adds_holding(self):
        p = pf.apply_buy(pf.empty_portfolio(1_000_000.0), "7203", 100, 1000.0,
                         "自動車", date(2026, 9, 2), commission_pct=0.0)
        assert p.cash == pytest.approx(900_000.0)
        assert p.holdings["7203"].quantity == 100
        assert p.holdings["7203"].avg_cost == pytest.approx(1000.0)
        assert p.holdings["7203"].sector == "自動車"
        assert p.holdings["7203"].entry_at == date(2026, 9, 2)

    def test_commission_is_charged_on_top(self):
        p = pf.apply_buy(pf.empty_portfolio(1_000_000.0), "7203", 100, 1000.0,
                         "自動車", date(2026, 9, 2), commission_pct=0.001)
        assert p.cash == pytest.approx(1_000_000.0 - 100_000.0 - 100.0)

    def test_peak_starts_at_entry_price(self):
        """ピークは取得単価から始まる（実運用 pos.peak_price or avg_cost と同じ）"""
        p = pf.apply_buy(pf.empty_portfolio(1_000_000.0), "7203", 100, 1000.0,
                         "自動車", date(2026, 9, 2), commission_pct=0.0)
        assert p.holdings["7203"].peak_price == pytest.approx(1000.0)
        assert p.holdings["7203"].sessions_held == 0

    def test_adding_to_existing_holding_averages_cost(self):
        p = pf.apply_buy(pf.empty_portfolio(1_000_000.0), "7203", 100, 1000.0,
                         "自動車", date(2026, 9, 2), commission_pct=0.0)
        p = pf.apply_buy(p, "7203", 100, 1200.0, "自動車", date(2026, 9, 3),
                         commission_pct=0.0)
        assert p.holdings["7203"].quantity == 200
        assert p.holdings["7203"].avg_cost == pytest.approx(1100.0)

    def test_adding_keeps_the_original_entry_date(self):
        """買い増しても保有開始日は動かさない（保有期間の数え方を保つ）"""
        p = pf.apply_buy(pf.empty_portfolio(1_000_000.0), "7203", 100, 1000.0,
                         "自動車", date(2026, 9, 2), commission_pct=0.0)
        p = pf.apply_buy(p, "7203", 100, 1200.0, "自動車", date(2026, 9, 5),
                         commission_pct=0.0)
        assert p.holdings["7203"].entry_at == date(2026, 9, 2)

    def test_rejects_non_positive_quantity(self):
        with pytest.raises(ValueError, match="quantity"):
            pf.apply_buy(pf.empty_portfolio(1_000_000.0), "7203", 0, 1000.0,
                         "自動車", date(2026, 9, 2), commission_pct=0.0)

    def test_rejects_a_buy_that_would_make_cash_negative(self):
        """必要額が現金を超える買いは成立させない

        数量は前日終値で決めるのに約定は翌朝の寄り。ギャップアップすると
        枠を超える。黙って現金を負にするとNAVも成績も意味を失う
        （外部レビューR08）。
        """
        # 現金10万円、前日100円で枠いっぱいの900株を決めた後、
        # 翌朝112円へギャップアップ。900 × 112 × 1.001 = 100,900.8円
        start = pf.empty_portfolio(100_000.0)
        with pytest.raises(pf.InsufficientCash):
            pf.apply_buy(start, "7203", 900, 112.0, "自動車",
                         date(2026, 9, 2), commission_pct=0.001)

    def test_cash_never_goes_negative_at_the_exact_boundary(self):
        """ちょうど買える数量は通り、1単元増やすと拒否される"""
        start = pf.empty_portfolio(100_000.0)
        qty = pf.max_affordable_quantity(100_000.0, 112.0, 0.001, lot=100)
        assert qty > 0
        after = pf.apply_buy(start, "7203", qty, 112.0, "自動車",
                             date(2026, 9, 2), commission_pct=0.001)
        assert after.cash >= 0.0
        with pytest.raises(pf.InsufficientCash):
            pf.apply_buy(start, "7203", qty + 100, 112.0, "自動車",
                         date(2026, 9, 2), commission_pct=0.001)

    def test_records_cost_basis_including_the_buy_commission(self):
        p = pf.apply_buy(pf.empty_portfolio(1_000_000.0), "7203", 100, 1000.0,
                         "自動車", date(2026, 9, 2), commission_pct=0.001)
        h = p.holdings["7203"]
        assert h.avg_cost == pytest.approx(1000.0)              # ポリシー用
        assert h.avg_cost_with_fees == pytest.approx(1001.0)    # 損益計算用

    def test_adding_to_holding_averages_the_fee_inclusive_basis_too(self):
        p = pf.apply_buy(pf.empty_portfolio(1_000_000.0), "7203", 100, 1000.0,
                         "自動車", date(2026, 9, 2), commission_pct=0.001)
        p = pf.apply_buy(p, "7203", 100, 1200.0, "自動車", date(2026, 9, 3),
                         commission_pct=0.001)
        h = p.holdings["7203"]
        assert h.avg_cost == pytest.approx(1100.0)
        # (100,100 + 120,120) / 200 = 1,101.1
        assert h.avg_cost_with_fees == pytest.approx(1101.1)


class TestMaxAffordableQuantity:
    def test_shrinks_to_what_the_cash_allows(self):
        # 900株は買えないが、800株なら 800×112×1.001 = 89,689.6円で買える
        assert pf.max_affordable_quantity(100_000.0, 112.0, 0.001, lot=100) == 800

    def test_zero_when_one_lot_is_unaffordable(self):
        assert pf.max_affordable_quantity(1_000.0, 112.0, 0.001, lot=100) == 0

    def test_result_always_fits_in_cash(self):
        for cash in (10_000.0, 100_000.0, 999_999.0):
            for price in (37.0, 112.0, 1234.5):
                qty = pf.max_affordable_quantity(cash, price, 0.001, lot=100)
                assert price * qty * 1.001 <= cash + 1e-9

    def test_rejects_bad_inputs(self):
        assert pf.max_affordable_quantity(100_000.0, 0.0, 0.001) == 0
        assert pf.max_affordable_quantity(0.0, 112.0, 0.001) == 0
        assert pf.max_affordable_quantity(-1.0, 112.0, 0.001) == 0


class TestApplySell:
    def _held(self):
        return pf.apply_buy(pf.empty_portfolio(1_000_000.0), "7203", 100, 1000.0,
                            "自動車", date(2026, 9, 2), commission_pct=0.0)

    def test_returns_cash_and_realized_profit(self):
        p, pnl = pf.apply_sell(self._held(), "7203", 100, 1200.0, commission_pct=0.0)
        assert p.cash == pytest.approx(1_020_000.0)
        assert pnl == pytest.approx(20_000.0)
        assert "7203" not in p.holdings

    def test_realizes_loss(self):
        p, pnl = pf.apply_sell(self._held(), "7203", 100, 900.0, commission_pct=0.0)
        assert pnl == pytest.approx(-10_000.0)

    def test_commission_reduces_proceeds_and_profit(self):
        p, pnl = pf.apply_sell(self._held(), "7203", 100, 1200.0, commission_pct=0.001)
        assert p.cash == pytest.approx(900_000.0 + 120_000.0 - 120.0)
        assert pnl == pytest.approx(20_000.0 - 120.0)

    def test_partial_sell_keeps_the_rest(self):
        p = pf.apply_buy(pf.empty_portfolio(1_000_000.0), "7203", 200, 1000.0,
                         "自動車", date(2026, 9, 2), commission_pct=0.0)
        p, pnl = pf.apply_sell(p, "7203", 100, 1200.0, commission_pct=0.0)
        assert p.holdings["7203"].quantity == 100
        assert p.holdings["7203"].avg_cost == pytest.approx(1000.0)
        assert pnl == pytest.approx(20_000.0)

    def test_rejects_selling_more_than_held(self):
        with pytest.raises(ValueError, match="保有"):
            pf.apply_sell(self._held(), "7203", 200, 1200.0, commission_pct=0.0)

    def test_rejects_selling_unheld_symbol(self):
        with pytest.raises(ValueError, match="保有"):
            pf.apply_sell(pf.empty_portfolio(1_000_000.0), "7203", 100, 1200.0,
                          commission_pct=0.0)


class TestRealizedPnlMatchesCash:
    """全ポジション決済後の実現損益合計と現金増減が一致すること。

    買付手数料を現金からだけ引いて原価に含めないと、同値往復で
    現金 −200円・実現損益 −100円のようにずれる（外部レビューR20）。
    """

    COMM = 0.001

    def test_round_trip_at_the_same_price(self):
        start = pf.empty_portfolio(1_000_000.0)
        p = pf.apply_buy(start, "7203", 100, 1000.0, "自動車",
                         date(2026, 9, 2), commission_pct=self.COMM)
        p, pnl = pf.apply_sell(p, "7203", 100, 1000.0, commission_pct=self.COMM)
        assert p.cash - start.cash == pytest.approx(-200.0)
        assert pnl == pytest.approx(-200.0)

    def test_partial_sales_sum_to_the_cash_change(self):
        start = pf.empty_portfolio(1_000_000.0)
        p = pf.apply_buy(start, "7203", 100, 1000.0, "自動車",
                         date(2026, 9, 2), commission_pct=self.COMM)
        p, a = pf.apply_sell(p, "7203", 50, 1000.0, commission_pct=self.COMM)
        p, b = pf.apply_sell(p, "7203", 50, 1000.0, commission_pct=self.COMM)
        assert "7203" not in p.holdings
        assert a + b == pytest.approx(p.cash - start.cash)
        assert a + b == pytest.approx(-200.0)

    def test_scaling_in_then_closing_out(self):
        start = pf.empty_portfolio(1_000_000.0)
        p = pf.apply_buy(start, "7203", 100, 1000.0, "自動車",
                         date(2026, 9, 2), commission_pct=self.COMM)
        p = pf.apply_buy(p, "7203", 100, 1200.0, "自動車",
                         date(2026, 9, 3), commission_pct=self.COMM)
        p, pnl = pf.apply_sell(p, "7203", 200, 1100.0, commission_pct=self.COMM)
        assert "7203" not in p.holdings
        # 現金: −100,100 −120,120 +219,780 = −440
        assert p.cash - start.cash == pytest.approx(-440.0)
        assert pnl == pytest.approx(-440.0)

    def test_profitable_trade_also_reconciles(self):
        start = pf.empty_portfolio(1_000_000.0)
        p = pf.apply_buy(start, "7203", 100, 1000.0, "自動車",
                         date(2026, 9, 2), commission_pct=self.COMM)
        p, pnl = pf.apply_sell(p, "7203", 100, 1200.0, commission_pct=self.COMM)
        assert p.cash - start.cash == pytest.approx(pnl)
        # 20,000 − 100（買付） − 120（売付） = 19,780
        assert pnl == pytest.approx(19_780.0)


class TestAdvanceSession:
    def _held(self):
        return pf.apply_buy(pf.empty_portfolio(1_000_000.0), "7203", 100, 1000.0,
                            "自動車", date(2026, 9, 2), commission_pct=0.0)

    def test_raises_peak_and_counts_session(self):
        p = pf.advance_session(self._held(), {"7203": 1200.0})
        assert p.holdings["7203"].peak_price == pytest.approx(1200.0)
        assert p.holdings["7203"].sessions_held == 1

    def test_peak_never_decreases(self):
        p = pf.advance_session(self._held(), {"7203": 1200.0})
        p = pf.advance_session(p, {"7203": 1100.0})
        assert p.holdings["7203"].peak_price == pytest.approx(1200.0)
        assert p.holdings["7203"].sessions_held == 2

    def test_missing_price_does_not_change_peak(self):
        p = pf.advance_session(self._held(), {})
        assert p.holdings["7203"].peak_price == pytest.approx(1000.0)
        assert p.holdings["7203"].sessions_held == 1

    def test_cash_is_untouched(self):
        before = self._held()
        after = pf.advance_session(before, {"7203": 1200.0})
        assert after.cash == pytest.approx(before.cash)


def _sizing(ratio=0.25, max_positions=5, sector_ratio=0.40):
    return pf.SizingConfig(max_position_ratio=ratio, max_positions=max_positions,
                           max_sector_ratio=sector_ratio)


class TestPositionBudget:
    def test_is_ratio_of_effective_cash(self):
        p = pf.Portfolio(cash=1_000_000.0, holdings={}, reserved=0.0)
        assert pf.position_budget(p, _sizing(ratio=0.25)) == pytest.approx(250_000.0)

    def test_reserved_reduces_the_budget(self):
        """未約定買いの引当を差し引いた実効余力で上限を計算する"""
        p = pf.Portfolio(cash=1_000_000.0, holdings={}, reserved=200_000.0)
        assert pf.position_budget(p, _sizing(ratio=0.25)) == pytest.approx(200_000.0)

    def test_never_negative(self):
        p = pf.Portfolio(cash=100_000.0, holdings={}, reserved=500_000.0)
        assert pf.position_budget(p, _sizing(ratio=0.25)) == pytest.approx(0.0)

    def test_matches_production_formula(self):
        """src/risk/manager.py:382-396 と同じ式であること"""
        cash, reserved, ratio = 1_000_000.0, 150_000.0, 0.25
        expected = max(0.0, cash - reserved) * ratio
        p = pf.Portfolio(cash=cash, holdings={}, reserved=reserved)
        assert pf.position_budget(p, _sizing(ratio=ratio)) == pytest.approx(expected)


class TestCalcQuantity:
    def test_rounds_down_to_lot_size(self):
        p = pf.Portfolio(cash=1_000_000.0, holdings={}, reserved=0.0)
        # 枠250,000円 ÷ 1,050円 = 238株 → 単元切り捨てで200株
        assert pf.calc_quantity(p, "7203", 1050.0, _sizing(ratio=0.25)) == 200

    def test_zero_when_one_lot_exceeds_budget(self):
        """単元の必要額が枠を超えたら0株（買えない）"""
        p = pf.Portfolio(cash=1_000_000.0, holdings={}, reserved=0.0)
        # 枠250,000円 < 単元必要額 300,000円
        assert pf.calc_quantity(p, "9983", 3000.0, _sizing(ratio=0.25)) == 0

    def test_existing_holding_reduces_the_remaining_budget(self):
        """既存保有の評価額を上限から差し引く（買い増しで枠を二重に使わない）"""
        p = pf.Portfolio(
            cash=1_000_000.0,
            holdings={"7203": _holding(qty=100, avg_cost=1000.0)},
            reserved=0.0)
        # 枠250,000 − 既存保有100株×1,000円=100,000 → 残り150,000 → 100株
        assert pf.calc_quantity(p, "7203", 1000.0, _sizing(ratio=0.25)) == 100

    def test_zero_when_already_at_the_cap(self):
        p = pf.Portfolio(
            cash=1_000_000.0,
            holdings={"7203": _holding(qty=300, avg_cost=1000.0)},
            reserved=0.0)
        # 枠250,000 < 既存保有300,000 → 残り0
        assert pf.calc_quantity(p, "7203", 1000.0, _sizing(ratio=0.25)) == 0

    def test_zero_for_non_positive_price(self):
        p = pf.Portfolio(cash=1_000_000.0, holdings={}, reserved=0.0)
        assert pf.calc_quantity(p, "7203", 0.0, _sizing()) == 0

    def test_matches_production_formula(self):
        """src/risk/manager.py:418-439 と同じ式であること"""
        cash, ratio, price = 1_000_000.0, 0.25, 1050.0
        held_qty, held_price = 100, 1050.0
        budget = max(0.0, cash - 0.0) * ratio
        remaining = max(0.0, budget - held_qty * held_price)
        expected = int(remaining / (price * pf.LOT_SIZE)) * pf.LOT_SIZE

        p = pf.Portfolio(
            cash=cash,
            holdings={"7203": _holding(qty=held_qty, avg_cost=held_price)},
            reserved=0.0)
        assert pf.calc_quantity(p, "7203", price, _sizing(ratio=ratio)) == expected


class TestCheckMaxPositions:
    def test_allows_when_below_the_cap(self):
        p = pf.Portfolio(cash=1_000_000.0,
                         holdings={"7203": _holding(symbol="7203")},
                         reserved=0.0)
        ok, _ = pf.check_max_positions(p, "9984", _sizing(max_positions=5))
        assert ok is True

    def test_rejects_when_candidate_would_exceed(self):
        holdings = {s: _holding(symbol=s) for s in ("A", "B", "C", "D", "E")}
        p = pf.Portfolio(cash=1_000_000.0, holdings=holdings, reserved=0.0)
        ok, reason = pf.check_max_positions(p, "F", _sizing(max_positions=5))
        assert ok is False
        assert "最大保有銘柄数" in reason

    def test_already_held_candidate_does_not_grow_the_set(self):
        """既に保有中の銘柄への買い増しは銘柄数を増やさないので通る"""
        holdings = {s: _holding(symbol=s) for s in ("A", "B", "C", "D", "E")}
        p = pf.Portfolio(cash=1_000_000.0, holdings=holdings, reserved=0.0)
        ok, _ = pf.check_max_positions(p, "C", _sizing(max_positions=5))
        assert ok is True

    def test_no_candidate_checks_current_state_only(self):
        holdings = {s: _holding(symbol=s) for s in ("A", "B", "C", "D", "E")}
        p = pf.Portfolio(cash=1_000_000.0, holdings=holdings, reserved=0.0)
        ok, _ = pf.check_max_positions(p, None, _sizing(max_positions=5))
        assert ok is True


class TestCheckSectorConcentration:
    def test_allows_the_first_purchase_from_empty(self):
        """保有ゼロから最初の1銘柄を買えること。

        分母を投資済み額にすると比率が常に100%になり、最初の1銘柄を
        永久に買えなくなる（2026-09-04の実害）。分母は総資金にする。
        """
        p = pf.empty_portfolio(1_000_000.0)
        ok, _ = pf.check_sector_concentration(
            p, "自動車", 147_500.0, {}, _sizing(sector_ratio=0.40))
        assert ok is True

    def test_rejects_when_candidate_pushes_over_the_cap(self):
        p = pf.Portfolio(
            cash=600_000.0,
            holdings={"7203": _holding(symbol="7203", qty=100, avg_cost=3000.0,
                                       sector="自動車")},
            reserved=0.0)
        # 同セクター 300,000 + 候補 200,000 = 500,000
        # 総資金 600,000 + 300,000 = 900,000 → 55.6% >= 40%
        ok, reason = pf.check_sector_concentration(
            p, "自動車", 200_000.0, {"7203": 3000.0}, _sizing(sector_ratio=0.40))
        assert ok is False
        assert "セクター集中率" in reason

    def test_other_sectors_do_not_count(self):
        p = pf.Portfolio(
            cash=600_000.0,
            holdings={"7203": _holding(symbol="7203", qty=100, avg_cost=3000.0,
                                       sector="自動車")},
            reserved=0.0)
        # 候補は別セクターなので分子は候補の200,000のみ → 22.2% < 40%
        ok, _ = pf.check_sector_concentration(
            p, "情報通信", 200_000.0, {"7203": 3000.0}, _sizing(sector_ratio=0.40))
        assert ok is True

    def test_denominator_is_total_capital_not_invested_amount(self):
        """分母は総資金（現金＋建玉）。現金が多いほど比率は下がる"""
        holdings = {"7203": _holding(symbol="7203", qty=100, avg_cost=3000.0,
                                     sector="自動車")}
        rich = pf.Portfolio(cash=5_000_000.0, holdings=holdings, reserved=0.0)
        poor = pf.Portfolio(cash=100_000.0, holdings=holdings, reserved=0.0)
        prices = {"7203": 3000.0}
        assert pf.check_sector_concentration(
            rich, "自動車", 100_000.0, prices, _sizing(sector_ratio=0.40))[0] is True
        assert pf.check_sector_concentration(
            poor, "自動車", 100_000.0, prices, _sizing(sector_ratio=0.40))[0] is False

    def test_uses_avg_cost_when_price_missing(self):
        p = pf.Portfolio(
            cash=600_000.0,
            holdings={"7203": _holding(symbol="7203", qty=100, avg_cost=3000.0,
                                       sector="自動車")},
            reserved=0.0)
        with_price = pf.check_sector_concentration(
            p, "自動車", 200_000.0, {"7203": 3000.0}, _sizing(sector_ratio=0.40))
        without_price = pf.check_sector_concentration(
            p, "自動車", 200_000.0, {}, _sizing(sector_ratio=0.40))
        assert with_price[0] == without_price[0]

    def test_matches_production_formula(self):
        """src/risk/manager.py:511-536 と同じ式であること"""
        cash, qty, price, candidate = 600_000.0, 100, 3000.0, 200_000.0
        positions_value = qty * price
        same_sector_value = positions_value + candidate
        total_value = cash + positions_value
        expected_ok = (same_sector_value / total_value) < 0.40

        p = pf.Portfolio(
            cash=cash,
            holdings={"7203": _holding(symbol="7203", qty=qty, avg_cost=price,
                                       sector="自動車")},
            reserved=0.0)
        ok, _ = pf.check_sector_concentration(
            p, "自動車", candidate, {"7203": price}, _sizing(sector_ratio=0.40))
        assert ok is expected_ok

    def test_ratio_at_exactly_the_cap_is_rejected(self):
        """上限ちょうどは却下（実運用が >= で判定しているため）"""
        p = pf.Portfolio(cash=600_000.0, holdings={}, reserved=0.0)
        # 候補240,000 / 総資金600,000 = 40.0%
        ok, _ = pf.check_sector_concentration(
            p, "自動車", 240_000.0, {}, _sizing(sector_ratio=0.40))
        assert ok is False


def _candidate(symbol="7203", sector="自動車", price=1000.0, score=0.5, tier=0):
    return pf.Candidate(symbol=symbol, sector=sector, price=price, score=score,
                        tier=tier)


class TestAllocate:
    def test_takes_candidates_in_score_order(self):
        p = pf.empty_portfolio(1_000_000.0)
        cands = [
            _candidate("A", "自動車", 1000.0, score=0.10),
            _candidate("B", "機械", 1000.0, score=0.90),
            _candidate("C", "情報通信", 1000.0, score=0.50),
        ]
        orders, _ = pf.allocate(p, cands, _sizing(ratio=0.25), {})
        assert [o.symbol for o in orders] == ["B", "C", "A"]

    def test_tier_takes_priority_over_a_higher_score_in_a_worse_tier(self):
        """tierが違う候補同士はscoreの数値を直接比較しない。

        Bはscore=0.90だがtier=1、Aはscore=0.10だがtier=0。tierを見ずscore
        だけで降順ソートする旧実装ならBが先に処理されるが、修正後は
        (tier, -score)の辞書式順序でtierが低い(=優先度が高い)Aが必ず先に
        処理される（最終ブランチレビュー・完了条件確認の残課題9の再修正）。
        """
        p = pf.empty_portfolio(1_000_000.0)
        cands = [
            _candidate("A", "自動車", 1000.0, score=0.10, tier=0),
            _candidate("B", "機械", 1000.0, score=0.90, tier=1),
        ]
        orders, _ = pf.allocate(
            p, cands, _sizing(ratio=0.10, max_positions=1, sector_ratio=1.0), {})
        assert [o.symbol for o in orders] == ["A"]

    def test_same_tier_candidates_still_sort_by_score_descending(self):
        """tierが同じ候補同士は従来通りscore降順（tier追加前の挙動を維持）。"""
        p = pf.empty_portfolio(1_000_000.0)
        cands = [
            _candidate("A", "自動車", 1000.0, score=0.10, tier=1),
            _candidate("B", "機械", 1000.0, score=0.90, tier=1),
            _candidate("C", "情報通信", 1000.0, score=0.50, tier=1),
        ]
        orders, _ = pf.allocate(p, cands, _sizing(ratio=0.25, sector_ratio=1.0), {})
        assert [o.symbol for o in orders] == ["B", "C", "A"]

    def test_cash_shrinks_for_later_candidates(self):
        """発注のたびに現金が減るので2件目以降の枠は小さくなる。

        株価1,200円で検証する。1,000円だと枠が250,000→200,000と縮んでも
        単元切り捨てでどちらも200株になり、差が観測できない。
        """
        p = pf.empty_portfolio(1_000_000.0)
        cands = [
            _candidate("A", "自動車", 1200.0, score=0.90),
            _candidate("B", "機械", 1200.0, score=0.80),
        ]
        orders, _ = pf.allocate(p, cands, _sizing(ratio=0.25), {})
        assert orders[0].quantity == 200
        assert orders[1].quantity == 100

    def test_budget_shrinks_even_when_lot_rounding_hides_it(self):
        """単元切り捨てで数量が同じでも、枠そのものは縮んでいる"""
        p = pf.empty_portfolio(1_000_000.0)
        conf = _sizing(ratio=0.25)
        before = pf.position_budget(p, conf)
        orders, _ = pf.allocate(p, [_candidate("A", "自動車", 1000.0, score=0.9)],
                                conf, {})
        after_buy = pf.apply_buy(p, "A", orders[0].quantity, 1000.0, "自動車",
                                 date(2026, 9, 2), commission_pct=0.0)
        assert pf.position_budget(after_buy, conf) < before

    def test_total_notional_never_exceeds_cash(self):
        p = pf.empty_portfolio(300_000.0)
        cands = [_candidate(s, "自動車", 1000.0, score=0.9 - i * 0.1)
                 for i, s in enumerate("ABCDEFGH")]
        orders, _ = pf.allocate(p, cands, _sizing(ratio=0.50, sector_ratio=1.0), {})
        assert sum(o.notional for o in orders) <= 300_000.0

    def test_rejects_beyond_max_positions(self):
        p = pf.empty_portfolio(10_000_000.0)
        cands = [_candidate(s, f"S{i}", 1000.0, score=0.9 - i * 0.01)
                 for i, s in enumerate("ABCDEFG")]
        orders, rejected = pf.allocate(
            p, cands, _sizing(ratio=0.10, max_positions=3, sector_ratio=1.0), {})
        assert len(orders) == 3
        assert {r.symbol for r in rejected} == {"D", "E", "F", "G"}
        assert all("最大保有銘柄数" in r.reason for r in rejected)

    def test_rejects_on_sector_concentration(self):
        p = pf.empty_portfolio(1_000_000.0)
        cands = [
            _candidate("A", "自動車", 1000.0, score=0.90),
            _candidate("B", "自動車", 1000.0, score=0.80),
            _candidate("C", "自動車", 1000.0, score=0.70),
        ]
        orders, rejected = pf.allocate(
            p, cands, _sizing(ratio=0.25, sector_ratio=0.30), {})
        assert len(orders) < 3
        assert any("セクター集中率" in r.reason for r in rejected)

    def test_rejects_when_one_lot_is_unaffordable(self):
        p = pf.empty_portfolio(100_000.0)
        cands = [_candidate("9983", "小売", 3000.0, score=0.90)]
        orders, rejected = pf.allocate(p, cands, _sizing(ratio=0.25), {})
        assert orders == []
        assert len(rejected) == 1
        assert "単元" in rejected[0].reason

    def test_every_candidate_is_either_taken_or_explained(self):
        """採用されなかった候補には必ず理由が残る"""
        p = pf.empty_portfolio(500_000.0)
        cands = [_candidate(s, f"S{i}", 1000.0, score=0.9 - i * 0.1)
                 for i, s in enumerate("ABCDE")]
        orders, rejected = pf.allocate(p, cands, _sizing(ratio=0.25), {})
        assert len(orders) + len(rejected) == len(cands)
        assert {o.symbol for o in orders} | {r.symbol for r in rejected} == set("ABCDE")

    def test_does_not_mutate_the_input_portfolio(self):
        p = pf.empty_portfolio(1_000_000.0)
        before_cash = p.cash
        pf.allocate(p, [_candidate("A", "自動車", 1000.0, score=0.9)],
                    _sizing(ratio=0.25), {})
        assert p.cash == pytest.approx(before_cash)
        assert p.holdings == {}

    def test_empty_candidates_gives_empty_result(self):
        p = pf.empty_portfolio(1_000_000.0)
        orders, rejected = pf.allocate(p, [], _sizing(), {})
        assert orders == []
        assert rejected == []

    def test_existing_holdings_constrain_new_orders(self):
        p = pf.Portfolio(
            cash=200_000.0,
            holdings={s: _holding(symbol=s, sector=f"S{i}")
                      for i, s in enumerate("ABCD")},
            reserved=0.0)
        orders, rejected = pf.allocate(
            p, [_candidate("E", "新規", 1000.0, score=0.9)],
            _sizing(ratio=0.25, max_positions=4, sector_ratio=1.0), {})
        assert orders == []
        assert "最大保有銘柄数" in rejected[0].reason
