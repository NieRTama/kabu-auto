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
from sqlalchemy import select

from src.backtest import execution
from src.backtest import portfolio as pf
from src.backtest import walkforward as wf
from src.core import config as cfg
from src.data import database as db
from src.data.database import get_session
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


class TestFillTimeAffordability:
    """約定時点で現金・上限を引き直すこと（外部レビューR08）。"""

    def _gap_up_market(self):
        """前日終値100円、翌朝112円へギャップアップする足"""
        bars = _bars(6, price=100.0)
        idx = bars.index[1]
        bars.loc[idx, ["open", "high", "low", "close"]] = [112.0, 115.0, 111.0, 113.0]
        return wf.MarketData(bars={"A": bars}, sectors={"A": "S"})

    def test_cash_never_goes_negative_on_a_gap_up(self):
        """現金10万・前日100円で枠いっぱい→翌朝112円。全量買うと残高−900.8円

        sector_ratio を明示的に上げているのは、max_position_ratio=1.0（残余力の
        100%）と _sizing() の既定 max_sector_ratio=1.0 が同じ値だと、単一セクター・
        単一候補では候補金額が総資金にちょうど一致し、check_sector_concentration()
        の `ratio >= max_ratio` 判定（境界含む・src/risk/manager.py と同じ規約）で
        本テストが検証したい約定時点の現金制約より先にセクター集中で却下されて
        しまうため（このテストの意図はギャップアップ時の現金縮小の検証）。
        """
        md = self._gap_up_market()
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 10),
            initial_capital=100_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(max_holding=20),
            costs=_costs(comm=0.001), sizing=_sizing(ratio=1.0, sector_ratio=2.0),
            liquidity=execution.LiquidityConfig())
        assert (res.daily["cash"] >= 0).all()
        assert (res.daily["nav"] > 0).all()

    def test_shrunk_order_is_recorded_with_a_reason(self):
        """sector_ratio を上げる理由は test_cash_never_goes_negative_on_a_gap_up と同じ。"""
        md = self._gap_up_market()
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 10),
            initial_capital=100_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(max_holding=20),
            costs=_costs(comm=0.001), sizing=_sizing(ratio=1.0, sector_ratio=2.0),
            liquidity=execution.LiquidityConfig())
        reasons = " ".join(res.rejected["reason"].astype(str))
        assert "縮小" in reasons or "買付余力" in reasons

    def test_multiple_orders_share_the_same_cash(self):
        """同じ日に複数の買いが出ても、合計が現金を超えない"""
        bars_a, bars_b = _bars(6, price=1000.0), _bars(6, price=1000.0)
        md = wf.MarketData(bars={"A": bars_a, "B": bars_b},
                           sectors={"A": "S1", "B": "S2"})

        def decide(session, rows, model, ctx):
            if session != date(2026, 1, 5):
                return []
            return [pf.Candidate(symbol=s, sector=ctx["sectors"][s],
                                 price=float(rows[s]["close"]), score=0.9)
                    for s in ("A", "B")]

        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 10),
            initial_capital=150_000.0, decide=decide,
            policy_conf=_policy_conf(max_holding=20),
            costs=_costs(comm=0.001), sizing=_sizing(ratio=1.0),
            liquidity=execution.LiquidityConfig())
        assert (res.daily["cash"] >= 0).all()


class TestExitLiquidity:
    """売りにも出来高の制限が掛かること（外部レビューR09）。"""

    def _stop_market(self):
        # 出来高1,000株・参加率10% → 1日100株まで。
        # エントリー約定日（index[1]）だけは出来高を大きくして、買い自体が
        # 同じ出来高枠で縮まないようにする（このクラスが検証したいのは
        # 「売り」側の出来高制約であり、買いを縮めると保有数量が売りの
        # 1日枠ちょうどになり、損切りが1日で全量約定してしまって
        # 「売り切れずに残る」状態を再現できない）。
        bars = _bars(12, price=1000.0, volume=1_000)
        bars.loc[bars.index[1], "volume"] = 1_000_000
        idx = bars.index[4]
        bars.loc[idx, ["open", "high", "low", "close"]] = [1000.0, 1000.0, 850.0, 860.0]
        return wf.MarketData(bars={"A": bars}, sectors={"A": "S"})

    def test_stop_exit_cannot_sell_more_than_the_volume_allows(self):
        res = wf.run_walkforward(
            self._stop_market(), date(2026, 1, 5), date(2026, 1, 16),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(ratio=0.3),
            liquidity=execution.LiquidityConfig(max_volume_share=0.1))
        assert len(res.trades) >= 1
        assert (res.trades["quantity"] <= 100).all()

    def test_unsold_shares_stay_in_the_portfolio(self):
        """損切りを出しても売り切れない。残りは保有に残る

        sizing を ratio=0.3（300株）にしているのは、1日の売り枠100株の
        ちょうど2倍だと損切り発生日の翌営業日で全量を売り切ってしまい、
        「売れ残りが保有に残る」状態を1営業日も観測できないため
        （300株なら3営業日に分かれて約定するので、間の営業日で必ず残数が残る）。
        """
        res = wf.run_walkforward(
            self._stop_market(), date(2026, 1, 5), date(2026, 1, 16),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(ratio=0.3),
            liquidity=execution.LiquidityConfig(max_volume_share=0.1))
        after_stop = res.daily[res.daily["session"] > date(2026, 1, 9)]
        assert (after_stop["n_holdings"] > 0).any()

    def test_exit_and_entry_share_the_same_day_budget(self):
        """同日・同銘柄で退出が枠を使うと、買いはその残りしか約定できない"""
        budget = execution.VolumeBudget(
            execution.LiquidityConfig(max_volume_share=0.1))
        assert budget.allow("A", 1_000, 100) == 100     # 退出が使い切る
        assert budget.allow("A", 1_000, 100) == 0       # 買いは約定できない


class TestSignalSellExit:
    """売りスコアによる退出が実際に効くこと（外部レビューR10）。"""

    def _flat_market(self, n=12):
        # 値動きが無いのでストップにも満了にも掛からない
        return wf.MarketData(bars={"A": _bars(n, price=1000.0)},
                             sectors={"A": "S"})

    def test_sell_signal_exits_before_stop_or_time_limit(self):
        def exit_score_fn(symbol, row):
            # 1/8以降は強い売りシグナル
            return -0.9 if row.name >= pd.Timestamp(date(2026, 1, 8)) else 0.5

        res = wf.run_walkforward(
            self._flat_market(), date(2026, 1, 5), date(2026, 1, 16),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(sell_thr=-0.25, max_holding=30),
            costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig(),
            exit_score_fn=exit_score_fn)

        assert len(res.trades) == 1
        t = res.trades.iloc[0]
        assert t["reason"] == policy.SIGNAL_SELL
        # 成行なので翌営業日の寄りで約定する
        assert t["exit_at"] == date(2026, 1, 9)

    def test_without_a_score_function_the_position_is_held(self):
        """結線しないと売りシグナル退出は一度も起きない（回帰の見張り）"""
        res = wf.run_walkforward(
            self._flat_market(), date(2026, 1, 5), date(2026, 1, 16),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(sell_thr=-0.25, max_holding=30),
            costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        assert len(res.trades) == 0

    def test_score_reaches_the_policy_observation(self):
        """score が Observation まで届いていること（Noneで素通りしない）"""
        seen = []

        def exit_score_fn(symbol, row):
            seen.append(symbol)
            return 0.5

        wf.run_walkforward(
            self._flat_market(), date(2026, 1, 5), date(2026, 1, 16),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(max_holding=30), costs=_costs(),
            sizing=_sizing(), liquidity=execution.LiquidityConfig(),
            exit_score_fn=exit_score_fn)
        assert "A" in seen


class TestEntryTiming:
    def test_order_decided_on_t_fills_on_the_next_session(self):
        """Tの終値で判断した注文はT+1の寄りで約定する（F04の回帰防止）"""
        md = wf.MarketData(bars={"A": _bars(6, price=1000.0)}, sectors={"A": "S"})
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 10),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(max_holding=20), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        # 1/5の引けで判断 → 1/6の寄りで約定
        assert res.daily[res.daily["session"] == date(2026, 1, 5)]["n_holdings"].iloc[0] == 0
        assert res.daily[res.daily["session"] == date(2026, 1, 6)]["n_holdings"].iloc[0] == 1

    def test_no_fill_when_there_is_no_next_session(self):
        """最終営業日に決めた注文は執行されない（翌営業日が無い）"""
        md = wf.MarketData(bars={"A": _bars(3, price=1000.0)}, sectors={"A": "S"})
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 7),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=2),
            policy_conf=_policy_conf(max_holding=20), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        assert res.daily["n_holdings"].eq(0).all()

    def test_cash_decreases_on_the_fill_session(self):
        md = wf.MarketData(bars={"A": _bars(6, price=1000.0)}, sectors={"A": "S"})
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 10),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(max_holding=20), costs=_costs(), sizing=_sizing(ratio=0.25),
            liquidity=execution.LiquidityConfig())
        first = res.daily[res.daily["session"] == date(2026, 1, 5)]["cash"].iloc[0]
        second = res.daily[res.daily["session"] == date(2026, 1, 6)]["cash"].iloc[0]
        assert first == pytest.approx(1_000_000.0)
        assert second < first


class TestCapitalCompetition:
    def _buy_all(self):
        def decide(session, rows, model, ctx):
            return [pf.Candidate(symbol=s, sector=ctx["sectors"][s],
                                 price=float(r["close"]), score=1.0 / (i + 1))
                    for i, (s, r) in enumerate(sorted(rows.items()))]
        return decide

    def test_respects_max_positions(self):
        md = _market(symbols=tuple("ABCDEFG"), n=10)
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 14),
            initial_capital=10_000_000.0, decide=self._buy_all(),
            policy_conf=_policy_conf(max_holding=50), costs=_costs(),
            sizing=_sizing(ratio=0.10, max_positions=3),
            liquidity=execution.LiquidityConfig())
        assert res.daily["n_holdings"].max() <= 3

    def test_records_rejection_reasons(self):
        md = _market(symbols=tuple("ABCDEFG"), n=10)
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 14),
            initial_capital=10_000_000.0, decide=self._buy_all(),
            policy_conf=_policy_conf(max_holding=50), costs=_costs(),
            sizing=_sizing(ratio=0.10, max_positions=3),
            liquidity=execution.LiquidityConfig())
        assert len(res.rejected) > 0
        assert list(res.rejected.columns) == ["session", "symbol", "reason"]
        assert res.rejected["reason"].str.contains("最大保有銘柄数").any()

    def test_cash_never_goes_negative(self):
        md = _market(symbols=tuple("ABCDE"), n=15)
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 19),
            initial_capital=500_000.0, decide=self._buy_all(),
            policy_conf=_policy_conf(max_holding=50), costs=_costs(comm=0.001),
            sizing=_sizing(ratio=0.50), liquidity=execution.LiquidityConfig())
        assert (res.daily["cash"] >= -1e-6).all()


class TestVolumeLimit:
    def test_partial_fill_reduces_the_quantity(self):
        md = wf.MarketData(bars={"A": _bars(6, price=1000.0, volume=1000)},
                           sectors={"A": "S"})
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 10),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(max_holding=20), costs=_costs(),
            sizing=_sizing(ratio=0.25),
            liquidity=execution.LiquidityConfig(max_volume_share=0.1))
        # 出来高1,000株の10% = 100株までしか買えない
        held_cash = res.daily[res.daily["session"] == date(2026, 1, 6)]["cash"].iloc[0]
        assert held_cash == pytest.approx(1_000_000.0 - 100_000.0)

    def test_unfilled_is_recorded_as_rejected(self):
        md = wf.MarketData(bars={"A": _bars(6, price=1000.0, volume=500)},
                           sectors={"A": "S"})
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 10),
            initial_capital=1_000_000.0, decide=_buy_once("A", at_index=0),
            policy_conf=_policy_conf(max_holding=20), costs=_costs(),
            sizing=_sizing(ratio=0.25),
            liquidity=execution.LiquidityConfig(max_volume_share=0.1))
        assert res.daily["n_holdings"].eq(0).all()
        assert res.rejected["reason"].str.contains("出来高").any()


class TestWeeklyRetrain:
    def _recording_trainer(self):
        """呼ばれた as_of を記録する学習関数"""
        calls = []

        def train(as_of):
            calls.append(as_of)
            return f"model@{as_of:%Y%m%d}", 100
        return train, calls

    def test_retrains_on_the_configured_interval(self):
        md = _market(symbols=("A",), n=20)
        train, calls = self._recording_trainer()
        wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 24),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig(),
            retrain=wf.RetrainConfig(every_sessions=5, warmup_sessions=5),
            train_model=train)
        # 助走5営業日のあと、5営業日ごとに学習する
        assert len(calls) == 3

    def test_model_usage_records_each_period(self):
        md = _market(symbols=("A",), n=20)
        train, _ = self._recording_trainer()
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 24),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig(),
            retrain=wf.RetrainConfig(every_sessions=5, warmup_sessions=5),
            train_model=train)
        assert list(res.model_usage.columns) == ["model_id", "from_session",
                                                 "to_session", "n_train_events"]
        assert len(res.model_usage) == 3
        assert res.model_usage["from_session"].is_monotonic_increasing

    def test_model_is_passed_to_decide(self):
        """decide はその時点で有効なモデルを受け取る"""
        md = _market(symbols=("A",), n=20)
        train, _ = self._recording_trainer()
        seen = []

        def decide(session, rows, model, ctx):
            seen.append(model)
            return []

        wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 24),
            initial_capital=1_000_000.0, decide=decide,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig(),
            retrain=wf.RetrainConfig(every_sessions=5, warmup_sessions=5),
            train_model=train)
        assert seen[0] is None                      # 助走中はモデル無し
        # 助走5・間隔5なので index 5/10/15 で学習し、index15 は 2026-01-20
        assert seen[-1] == "model@20260120"          # 最後の学習が効いている

    def test_as_of_never_looks_ahead(self):
        """学習の基準日は、その時点のセッションを超えない"""
        md = _market(symbols=("A",), n=20)
        train, calls = self._recording_trainer()
        wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 24),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig(),
            retrain=wf.RetrainConfig(every_sessions=5, warmup_sessions=5),
            train_model=train)
        sessions = wf.sessions_between(md, date(2026, 1, 5), date(2026, 1, 24))
        for as_of in calls:
            assert as_of in sessions

    def test_disabled_when_interval_is_zero(self):
        md = _market(symbols=("A",), n=20)
        train, calls = self._recording_trainer()
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 24),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig(),
            retrain=wf.RetrainConfig(every_sessions=0, warmup_sessions=5),
            train_model=train)
        assert calls == []
        assert res.model_usage.empty

    def test_no_retrain_without_a_trainer(self):
        md = _market(symbols=("A",), n=20)
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 24),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig(),
            retrain=wf.RetrainConfig(every_sessions=5, warmup_sessions=5),
            train_model=None)
        assert res.model_usage.empty


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


class TestDegraded:
    def test_decide_exception_marks_degraded(self):
        """判断規則で例外が出たら握り潰さずdegradedを立てる"""
        md = _market(symbols=("A",), n=10)

        def broken(session, rows, model, ctx):
            raise RuntimeError("推論に失敗しました")

        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 14),
            initial_capital=1_000_000.0, decide=broken,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        assert res.degraded is True
        assert len(res.degraded_reasons) > 0
        assert "推論に失敗しました" in res.degraded_reasons[0]

    def test_run_continues_after_a_failure(self):
        """失敗しても最後まで進む（どこまで進んだかを残すため）"""
        md = _market(symbols=("A",), n=10)

        def broken(session, rows, model, ctx):
            raise RuntimeError("boom")

        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 14),
            initial_capital=1_000_000.0, decide=broken,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        assert len(res.daily) == 10

    def test_training_exception_marks_degraded(self):
        md = _market(symbols=("A",), n=20)

        def broken_train(as_of):
            raise RuntimeError("学習に失敗しました")

        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 24),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig(),
            retrain=wf.RetrainConfig(every_sessions=5, warmup_sessions=5),
            train_model=broken_train)
        assert res.degraded is True

    def test_clean_run_is_not_degraded(self):
        md = _market(symbols=("A",), n=10)
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 14),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        assert res.degraded is False


class TestSaveRun:
    def _result(self):
        md = _market(symbols=("A",), n=10)
        return wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 14),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())

    def _snapshot(self):
        return wf.RunSnapshot(
            strategy_version="rule_then_ml_v1", config_hash="abc123",
            config_json='{"buy_threshold": 0.25}', dataset_id="ds0001",
            code_version="deadbeef", execution_model_version="t1_open_v1")

    def test_records_the_full_snapshot(self, isolated_db):
        run_id = wf.save_run(
            self._result(), self._snapshot(), symbol_label="PORTFOLIO",
            start=date(2026, 1, 5), end=date(2026, 1, 14),
            initial_capital=1_000_000.0, costs=_costs(slip=0.001, comm=0.0005))

        with get_session() as session:
            row = session.scalar(select(db.BacktestRun))
        assert row.id == run_id
        assert row.strategy_version == "rule_then_ml_v1"
        assert row.config_hash == "abc123"
        assert row.dataset_id == "ds0001"
        assert row.code_version == "deadbeef"
        assert row.execution_model_version == "t1_open_v1"
        assert row.slippage_pct == pytest.approx(0.001)
        assert row.commission_pct == pytest.approx(0.0005)

    def test_stores_the_config_body_not_just_the_hash(self, isolated_db):
        """config_hash は同一性の確認には使えるが復元には使えない"""
        wf.save_run(
            self._result(), self._snapshot(), symbol_label="PORTFOLIO",
            start=date(2026, 1, 5), end=date(2026, 1, 14),
            initial_capital=1_000_000.0, costs=_costs())
        with get_session() as session:
            row = session.scalar(select(db.BacktestRun))
        assert "buy_threshold" in row.config_json

    def test_degraded_flag_is_persisted(self, isolated_db):
        md = _market(symbols=("A",), n=10)

        def broken(session, rows, model, ctx):
            raise RuntimeError("boom")

        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 14),
            initial_capital=1_000_000.0, decide=broken,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        wf.save_run(res, self._snapshot(), symbol_label="PORTFOLIO",
                    start=date(2026, 1, 5), end=date(2026, 1, 14),
                    initial_capital=1_000_000.0, costs=_costs())
        with get_session() as session:
            row = session.scalar(select(db.BacktestRun))
        assert row.degraded == 1

    def test_model_usage_rows_are_linked_to_the_run(self, isolated_db):
        md = _market(symbols=("A",), n=20)

        def train(as_of):
            return f"m@{as_of:%Y%m%d}", 50

        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 24),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig(),
            retrain=wf.RetrainConfig(every_sessions=5, warmup_sessions=5),
            train_model=train)
        run_id = wf.save_run(
            res, self._snapshot(), symbol_label="PORTFOLIO",
            start=date(2026, 1, 5), end=date(2026, 1, 24),
            initial_capital=1_000_000.0, costs=_costs())

        with get_session() as session:
            usages = list(session.scalars(select(db.RunModelUsage)).all())
        assert len(usages) == len(res.model_usage)
        assert all(u.run_id == run_id for u in usages)

    def test_daily_nav_is_stored_as_the_equity_curve(self, isolated_db):
        wf.save_run(
            self._result(), self._snapshot(), symbol_label="PORTFOLIO",
            start=date(2026, 1, 5), end=date(2026, 1, 14),
            initial_capital=1_000_000.0, costs=_costs())
        with get_session() as session:
            row = session.scalar(select(db.BacktestRun))
        assert row.equity_curve_json is not None
        assert "2026-01-05" in row.equity_curve_json


class TestPastPredictionsAreNotAffectedByFutureData:
    """過去の判断が、その後に足されたデータで変わらないこと（外部レビューR04）。"""

    def _trainer(self, events_by_session):
        """as_of までのイベント数だけをモデルIDに焼き込む学習関数"""
        def train(as_of):
            n = sum(1 for d in events_by_session if d <= as_of)
            return f"model-n{n}", n
        return train

    def test_same_past_decisions_when_future_bars_are_appended(self):
        short = _market(symbols=("A",), n=20)
        long_ = _market(symbols=("A",), n=40)
        events = sorted(set(short.bars["A"].index.date)
                        | set(long_.bars["A"].index.date))
        seen_short, seen_long = [], []

        def make_decide(sink):
            def decide(session, rows, model, ctx):
                sink.append((session, model))
                return []
            return decide

        common_end = date(2026, 1, 24)
        for md, sink in ((short, seen_short), (long_, seen_long)):
            wf.run_walkforward(
                md, date(2026, 1, 5), common_end,
                initial_capital=1_000_000.0, decide=make_decide(sink),
                policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
                liquidity=execution.LiquidityConfig(),
                retrain=wf.RetrainConfig(every_sessions=5, warmup_sessions=5),
                train_model=self._trainer(events))

        # 共通期間の判断は、後ろにデータを足しても同一
        assert seen_short == seen_long

    def test_a_run_without_retrain_is_degraded(self):
        """固定モデル実行は診断用。昇格の根拠にはできない"""
        md = _market(symbols=("A",), n=20)
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 24),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig(),
            model="fixed-model")
        assert res.degraded is True
        assert any("再学習が結線されていない" in r for r in res.degraded_reasons)

    def test_a_retrained_run_is_not_degraded_for_that_reason(self):
        md = _market(symbols=("A",), n=20)
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 24),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig(),
            retrain=wf.RetrainConfig(every_sessions=5, warmup_sessions=5),
            train_model=self._trainer([]))
        assert not any("再学習が結線されていない" in r
                       for r in res.degraded_reasons)

    def test_no_model_and_no_retrain_is_not_degraded(self):
        """ルールのみの実行は劣化ではない（意図したML無効）"""
        md = _market(symbols=("A",), n=20)
        res = wf.run_walkforward(
            md, date(2026, 1, 5), date(2026, 1, 24),
            initial_capital=1_000_000.0, decide=_never_buy,
            policy_conf=_policy_conf(), costs=_costs(), sizing=_sizing(),
            liquidity=execution.LiquidityConfig())
        assert res.degraded is False

def _strategy(buy_thr=0.25, rule_w=0.5, ml_w=0.5,
              on_failure=None):
    return wf.StrategyConfig(
        buy_threshold=buy_thr, rule_weight=rule_w, ml_weight=ml_w,
        on_model_failure=on_failure or wf.ON_FAILURE_RULE_ONLY)


def _scores(rule: float, ml):
    def score_fn(symbol, row):
        return rule, ml
    return score_fn


class TestWeightedBlend:
    def test_buys_when_blended_score_reaches_the_threshold(self):
        decide = wf.make_weighted_blend(_strategy(buy_thr=0.25), _scores(0.4, 0.6))
        ctx = {"sectors": {"A": "S"}, "portfolio": None, "closes": {}}
        rows = {"A": pd.Series({"close": 1000.0})}
        # 0.4*0.5 + (0.6-0.5)*2*0.5 = 0.2 + 0.1 = 0.30 >= 0.25
        assert [c.symbol for c in decide(date(2026, 1, 5), rows, "m", ctx)] == ["A"]

    def test_skips_below_the_threshold(self):
        decide = wf.make_weighted_blend(_strategy(buy_thr=0.25), _scores(0.2, 0.5))
        ctx = {"sectors": {"A": "S"}, "portfolio": None, "closes": {}}
        rows = {"A": pd.Series({"close": 1000.0})}
        assert decide(date(2026, 1, 5), rows, "m", ctx) == []


class TestRuleOnly:
    def test_uses_the_rule_score_directly(self):
        """縮尺を明示する。重みで割り引かない"""
        decide = wf.make_rule_only(_strategy(buy_thr=0.25), _scores(0.3, None))
        ctx = {"sectors": {"A": "S"}, "portfolio": None, "closes": {}}
        rows = {"A": pd.Series({"close": 1000.0})}
        assert [c.symbol for c in decide(date(2026, 1, 5), rows, None, ctx)] == ["A"]

    def test_ignores_the_model(self):
        decide = wf.make_rule_only(_strategy(buy_thr=0.25), _scores(0.3, 0.01))
        ctx = {"sectors": {"A": "S"}, "portfolio": None, "closes": {}}
        rows = {"A": pd.Series({"close": 1000.0})}
        assert len(decide(date(2026, 1, 5), rows, "m", ctx)) == 1


class TestRuleThenMl:
    def test_rule_gates_and_ml_ranks(self):
        """ルールが候補を決め、MLは順位だけを決める"""
        def score_fn(symbol, row):
            return ({"A": 0.30, "B": 0.30, "C": 0.10}[symbol],
                    {"A": 0.40, "B": 0.80, "C": 0.99}[symbol])

        decide = wf.make_rule_then_ml(_strategy(buy_thr=0.25), score_fn)
        ctx = {"sectors": {s: "S" for s in "ABC"}, "portfolio": None, "closes": {}}
        rows = {s: pd.Series({"close": 1000.0}) for s in "ABC"}
        got = decide(date(2026, 1, 5), rows, "m", ctx)
        # Cはルールで落ちる。A/Bは残り、ML確率の高いBが上位
        assert [c.symbol for c in got] == ["B", "A"]
        assert got[0].score > got[1].score

    def test_falls_back_to_rule_only_when_the_model_is_missing(self):
        decide = wf.make_rule_then_ml(
            _strategy(buy_thr=0.25, on_failure=wf.ON_FAILURE_RULE_ONLY),
            _scores(0.30, None))
        ctx = {"sectors": {"A": "S"}, "portfolio": None, "closes": {}}
        rows = {"A": pd.Series({"close": 1000.0})}
        assert len(decide(date(2026, 1, 5), rows, None, ctx)) == 1

    def test_halts_new_candidates_when_configured(self):
        """モデル失敗時に新規候補生成を止める設定"""
        decide = wf.make_rule_then_ml(
            _strategy(buy_thr=0.25, on_failure=wf.ON_FAILURE_HALT_NEW),
            _scores(0.30, None))
        ctx = {"sectors": {"A": "S"}, "portfolio": None, "closes": {}}
        rows = {"A": pd.Series({"close": 1000.0})}
        assert decide(date(2026, 1, 5), rows, None, ctx) == []

    def test_failure_mode_is_explicit_not_implicit(self):
        """同じ入力でも設定によって結果が変わる＝暗黙の縮尺変更ではない"""
        rows = {"A": pd.Series({"close": 1000.0})}
        ctx = {"sectors": {"A": "S"}, "portfolio": None, "closes": {}}
        keep = wf.make_rule_then_ml(
            _strategy(on_failure=wf.ON_FAILURE_RULE_ONLY), _scores(0.30, None))
        halt = wf.make_rule_then_ml(
            _strategy(on_failure=wf.ON_FAILURE_HALT_NEW), _scores(0.30, None))
        assert len(keep(date(2026, 1, 5), rows, None, ctx)) != len(
            halt(date(2026, 1, 5), rows, None, ctx))


class TestThreeStrategiesShareTheSameLoop:
    def test_all_three_run_on_the_same_market_data(self):
        md = _market(symbols=("A", "B"), n=15)
        results = {}
        for name, maker in (("blend", wf.make_weighted_blend),
                            ("rule", wf.make_rule_only),
                            ("rule_ml", wf.make_rule_then_ml)):
            decide = maker(_strategy(buy_thr=0.25), _scores(0.30, 0.70))
            results[name] = wf.run_walkforward(
                md, date(2026, 1, 5), date(2026, 1, 19),
                initial_capital=1_000_000.0, decide=decide,
                policy_conf=_policy_conf(max_holding=50), costs=_costs(),
                sizing=_sizing(ratio=0.25), liquidity=execution.LiquidityConfig())
        assert all(len(r.daily) == 15 for r in results.values())
        assert all(r.degraded is False for r in results.values())
