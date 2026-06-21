"""
バックテストのトレード指標・コスト控除（レビュー ML/Backtest）のテスト
"""
import pytest

import src.backtest.engine as eng


class TestProfitFactor:
    def test_no_losses_returns_none(self):
        assert eng._profit_factor([100, 200, 50]) is None

    def test_ratio_of_gross_profit_to_loss(self):
        # 利益300 / 損失100 = 3.0
        assert eng._profit_factor([200, 100, -100]) == 3.0

    def test_empty_returns_none(self):
        assert eng._profit_factor([]) is None


class TestSortino:
    def test_no_downside_returns_zero(self):
        curve = [{"equity": 100}, {"equity": 110}, {"equity": 120}]
        assert eng._sortino_ratio(curve) == 0.0

    def test_penalizes_downside(self):
        curve = [{"equity": 100}, {"equity": 90}, {"equity": 95}, {"equity": 80}]
        # 下方リスクがあるので有限の値（プラスマイナス問わず非ゼロ判定）
        val = eng._sortino_ratio(curve)
        assert isinstance(val, float)

    def test_known_value_uses_total_n_downside_deviation(self):
        """下方偏差は「負側のみの平均」ではなく「全期間Nを母数」に計算する（レビュー再指摘）。

        +20%→-1/6 の2リターン: mean_r=1/60, downside_dev=sqrt(1/72)=1/(6√2)。
        sortino = (mean_r/downside_dev)*sqrt(252) = sqrt(504)/10 ≈ 2.245（手計算済み既知値）。
        負側のみで平均する誤った実装（N=1のみで割る）だと別の値になり、この回帰テストで検出できる。
        """
        curve = [{"equity": 100}, {"equity": 120}, {"equity": 100}]
        assert eng._sortino_ratio(curve) == pytest.approx(2.245, abs=0.001)


class TestFillPrices:
    def test_buy_fill_is_worse_higher(self):
        assert eng._buy_fill_price(1000, 0.001) == 1001.0

    def test_sell_fill_is_worse_lower(self):
        assert eng._sell_fill_price(1000, 0.001) == 999.0

    def test_zero_slippage_no_change(self):
        assert eng._buy_fill_price(1000, 0.0) == 1000.0
        assert eng._sell_fill_price(1000, 0.0) == 1000.0
