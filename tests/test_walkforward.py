"""ポートフォリオwalk-forwardエンジン（src/backtest/walkforward.py）のテスト

1営業日を5フェーズで回す。Tの終値の情報はT+1以降の注文にしか使えない
（レビューF04）。判断はpolicy、約定はexecution、現金と保有はportfolioに
委ね、本モジュールは進行と記録だけを担う。
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest import execution
from src.backtest import portfolio as pf
from src.backtest import walkforward as wf
from src.strategy import policy


def _bars(n: int, start=date(2026, 1, 5), price: float = 1000.0,
          drift: float = 0.0, high_mult: float = 1.01,
          low_mult: float = 0.99, volume: int = 1_000_000) -> pd.DataFrame:
    rows = []
    p = price
    for i in range(n):
        d = start + timedelta(days=i)
        rows.append({
            "date": d, "open": p, "high": p * high_mult,
            "low": p * low_mult, "close": p, "volume": volume,
        })
        p *= 1 + drift
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def _market(symbols=("7203", "9984"), n=40, **kwargs) -> wf.MarketData:
    return wf.MarketData(
        bars={s: _bars(n, **kwargs) for s in symbols},
        sectors={s: f"S{i}" for i, s in enumerate(symbols)},
    )


def _policy_conf(stop=-0.07, breakeven=0.02, trailing=0.04,
                 sell_thr=-0.25, max_holding=10):
    return policy.PolicyConfig(
        stop_loss_pct=stop, breakeven_trigger_pct=breakeven,
        trailing_stop_pct=trailing, sell_threshold=sell_thr,
        max_holding_sessions=max_holding)


def _costs(slip=0.0, comm=0.0):
    return execution.CostConfig(slippage_pct=slip, commission_pct=comm)


def _sizing(ratio=0.25, max_positions=5, sector_ratio=1.0):
    return pf.SizingConfig(max_position_ratio=ratio, max_positions=max_positions,
                           max_sector_ratio=sector_ratio)


def _never_buy(session, rows, model, ctx):
    """何も買わない判断規則（ループの骨格だけを見るため）"""
    return []


class TestSessions:
    def test_uses_the_union_of_symbol_sessions(self):
        md = _market(n=10)
        got = wf.sessions_between(md, date(2026, 1, 5), date(2026, 1, 14))
        assert len(got) == 10
        assert got == sorted(got)

    def test_clips_to_the_requested_range(self):
        md = _market(n=30)
        got = wf.sessions_between(md, date(2026, 1, 10), date(2026, 1, 15))
        assert got[0] >= date(2026, 1, 10)
        assert got[-1] <= date(2026, 1, 15)

    def test_closes_skips_symbols_without_a_bar(self):
        md = _market(symbols=("A", "B"), n=10)
        md.bars["B"] = md.bars["B"].iloc[:5]   # Bは途中で終わる
        closes = wf.closes_at(md, date(2026, 1, 14))
        assert "A" in closes
        assert "B" not in closes


class TestDailyLoopSkeleton:
    def test_records_one_row_per_session(self):
        md = _market(n=20)
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 24),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        assert len(res.daily) == 20
        assert list(res.daily.columns) == ["session", "nav", "cash",
                                           "n_holdings", "realized_pnl"]

    def test_nav_stays_at_initial_capital_when_nothing_traded(self):
        md = _market(n=20)
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 24),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        assert res.daily["nav"].eq(1_000_000.0).all()
        assert res.daily["n_holdings"].eq(0).all()

    def test_sessions_are_in_order(self):
        md = _market(n=20)
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 24),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        sessions = list(res.daily["session"])
        assert sessions == sorted(sessions)

    def test_starts_clean(self):
        md = _market(n=5)
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 9),
            initial_capital=500_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        assert res.trades.empty
        assert res.degraded is False
        assert res.degraded_reasons == []
