"""ポートフォリオwalk-forwardエンジン（src/backtest/walkforward.py）のテスト

1営業日を5フェーズで回す。Tの終値の情報はT+1以降の注文にしか使えない
（レビューF04）。判断はpolicy、約定はexecution、現金と保有はportfolioに
委ね、本モジュールは進行と記録だけを担う。
"""
from dataclasses import replace
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


def _buy_once(symbol: str, at_index: int = 0, score: float = 0.9):
    """指定した位置のセッションで1銘柄だけ候補にする判断規則を作る"""
    state = {"count": 0}

    def decide(session, rows, model, ctx):
        idx = state["count"]
        state["count"] += 1
        if idx != at_index or symbol not in rows:
            return []
        return [pf.Candidate(symbol=symbol, sector=ctx["sectors"][symbol],
                             price=float(rows[symbol]["close"]), score=score)]
    return decide


class TestExitDriving:
    def test_stop_loss_exits_on_the_same_session(self):
        """基準線への到達は当日中に約定する"""
        md = wf.MarketData(bars={"A": _bars(10, price=1000.0)}, sectors={"A": "S"})
        # 4本目で大きく下落させる
        idx = md.bars["A"].index[4]
        md.bars["A"].loc[idx, ["open", "high", "low", "close"]] = [1000.0, 1000.0, 850.0, 860.0]

        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 14),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())

        assert len(res.trades) == 1
        t = res.trades.iloc[0]
        assert t["reason"] == policy.STOP_LINE
        assert t["exit_at"] == date(2026, 1, 9)          # 5本目＝下落した当日
        assert t["exit_price"] == pytest.approx(930.0)   # 取得1000 × 0.93

    def test_time_limit_exits_on_the_next_session(self):
        """満了は成行なので翌営業日の寄りで約定する"""
        md = wf.MarketData(bars={"A": _bars(12, price=1000.0)}, sectors={"A": "S"})
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 16),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(max_holding=3), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())

        assert len(res.trades) == 1
        t = res.trades.iloc[0]
        assert t["reason"] == policy.TIME_LIMIT
        # 1/6に約定→1/6,1/7,1/8の3営業日で満了→翌営業日1/9に成行退出
        assert t["entry_at"] == date(2026, 1, 6)
        assert t["exit_at"] == date(2026, 1, 9)

    def test_holding_is_released_after_exit(self):
        md = wf.MarketData(bars={"A": _bars(12, price=1000.0)}, sectors={"A": "S"})
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 16),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(max_holding=3), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        assert res.daily["n_holdings"].iloc[-1] == 0

    def test_peak_comes_from_policy_not_double_counted(self):
        """保有の経過営業日数がpolicyとadvance_sessionで二重に進まない"""
        md = wf.MarketData(bars={"A": _bars(12, price=1000.0)}, sectors={"A": "S"})
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 16),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(max_holding=5), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        t = res.trades.iloc[0]
        # 1/6約定 → 5営業日(1/6〜1/10)で満了 → 翌営業日1/11に退出
        assert t["exit_at"] == date(2026, 1, 11)

    def test_realized_pnl_is_recorded_on_the_exit_session(self):
        md = wf.MarketData(bars={"A": _bars(10, price=1000.0)}, sectors={"A": "S"})
        idx = md.bars["A"].index[4]
        md.bars["A"].loc[idx, ["open", "high", "low", "close"]] = [1000.0, 1000.0, 850.0, 860.0]

        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 14),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        exit_row = res.daily[res.daily["session"] == date(2026, 1, 9)].iloc[0]
        assert exit_row["realized_pnl"] < 0
        assert res.daily[res.daily["session"] != date(2026, 1, 9)]["realized_pnl"].eq(0).all()

    def test_missing_bar_does_not_crash_and_keeps_the_holding(self):
        """その日の足が無い銘柄はポリシーを回せないので保有を持ち越す"""
        md = wf.MarketData(bars={"A": _bars(12, price=1000.0)}, sectors={"A": "S"})
        md.bars["A"] = md.bars["A"].drop(md.bars["A"].index[3])

        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 16),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(max_holding=20), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        assert res.daily["n_holdings"].max() == 1


class TestDriveExitsUnit:
    def test_no_intent_keeps_the_holding(self):
        md = wf.MarketData(bars={"A": _bars(5, price=1000.0)}, sectors={"A": "S"})
        p = pf.apply_buy(pf.empty_portfolio(1_000_000.0), "A", 100, 1000.0, "S",
                         date(2026, 1, 5), commission_pct=0.0)
        nxt, trades, pending = wf._drive_exits(
            p, md, date(2026, 1, 6), _policy_conf(max_holding=20), _costs())
        assert trades == []
        assert pending == []
        assert nxt.holdings["A"].sessions_held == 1

    def test_stop_intent_fills_immediately(self):
        md = wf.MarketData(bars={"A": _bars(5, price=1000.0)}, sectors={"A": "S"})
        idx = md.bars["A"].index[1]
        md.bars["A"].loc[idx, ["open", "high", "low", "close"]] = [1000.0, 1000.0, 850.0, 860.0]
        p = pf.apply_buy(pf.empty_portfolio(1_000_000.0), "A", 100, 1000.0, "S",
                         date(2026, 1, 5), commission_pct=0.0)
        nxt, trades, pending = wf._drive_exits(
            p, md, date(2026, 1, 6), _policy_conf(), _costs())
        assert len(trades) == 1
        assert trades[0]["reason"] == policy.STOP_LINE
        assert "A" not in nxt.holdings
        assert pending == []

    def test_time_limit_is_carried_to_the_next_session(self):
        md = wf.MarketData(bars={"A": _bars(5, price=1000.0)}, sectors={"A": "S"})
        p = pf.apply_buy(pf.empty_portfolio(1_000_000.0), "A", 100, 1000.0, "S",
                         date(2026, 1, 5), commission_pct=0.0)
        nxt, trades, pending = wf._drive_exits(
            p, md, date(2026, 1, 6), _policy_conf(max_holding=1), _costs())
        assert trades == []
        assert len(pending) == 1
        assert pending[0][1].reason == policy.TIME_LIMIT
        assert "A" in nxt.holdings   # まだ売っていない

    def test_missing_bar_only_advances_the_session_count(self):
        md = wf.MarketData(bars={"A": _bars(5, price=1000.0)}, sectors={"A": "S"})
        p = pf.apply_buy(pf.empty_portfolio(1_000_000.0), "A", 100, 1000.0, "S",
                         date(2026, 1, 5), commission_pct=0.0)
        nxt, trades, pending = wf._drive_exits(
            p, md, date(2026, 3, 1), _policy_conf(), _costs())   # 足が無い日
        assert trades == []
        assert nxt.holdings["A"].sessions_held == 1
