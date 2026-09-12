# ML評価基盤 段階D後半（walk-forwardエンジン）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 判断と執行を日単位で分けたポートフォリオ walk-forward エンジンを作り、実行条件と結果を後から辿れる形で残す。

**Architecture:** `walkforward.py` が1営業日を5フェーズで回す。**Tの終値の情報はT+1以降の注文にしか使えない。** 退出の判断は `policy.step()`、約定は `execution.py`、現金と保有は `portfolio.py` に委ね、本モジュールは進行と記録だけを担う。判断規則（戦略バージョン）は差し替え可能にして、同じイベント表の上で3案を比較する。

**Tech Stack:** Python 3.11 / pandas 2.1.4 / numpy 1.26.2 / SQLAlchemy 2.0.23 / pytest

**Spec:** `docs/superpowers/specs/2026-09-10-ml-evaluation-foundation-design.md`（§8・§10・§11・§12・§14）

**前提:** 段階D前半（`docs/superpowers/plans/2026-09-12-ml-evaluation-stage-d1.md`）が完了していること。本計画は `portfolio.Portfolio` / `Holding` / `Candidate` / `PlannedOrder` / `allocate` / `apply_buy` / `apply_sell` / `advance_session` / `nav` / `SizingConfig`、`execution.entry_fill_limited` / `exit_fill` / `FillResult` / `LiquidityConfig` / `CostConfig`、`policy.HoldingState` / `Observation` / `ExitIntent` / `step` / `PolicyConfig` に依存する。段階C前半の `validation.training_inputs()` も使う。

## Global Constraints

- 日時は **JST naive**。現在時刻は `src/core/clock.now()` / `clock.today()` を使い、`datetime.now()` を直接呼ばない。
- **Tの終値で判断した注文をTに約定させない。** 判断と執行のセッションが同じになる経路を作らない（レビューF04）。
- **`src/backtest/engine.py` を変更しない。** 旧エンジンは `strategy.engine_version: legacy` の受け皿として無改造で残す（spec §10）。`engine.py:160-161` の `except Exception: pass` も削除しない。degraded の扱いは新エンジンに最初から持たせる。
- **`src/risk/manager.py` / `src/strategy/ml_model.py` / `src/strategy/labeling.py` / `src/strategy/indicators.py` の既存公開関数を変更しない。**
- 新規のDB列・テーブルはすべて nullable。`create_all` が新規テーブルを、`_migrate_add_missing_columns()` が既存テーブルへの列追加を自動で行う。
- 推論の例外を握り潰さない。1件でも発生した実行には `degraded=True` を立て、比較とモデル昇格から除外できるようにする。
- ファイルは UTF-8 **BOM無し**・LF で保存する。確認は `git show <rev>:<path>` でコミット済みblobに対して行う。
- テストは `pytest tests/<file>.py -v` で実行する。ネットワークへ出るテストを書かない。
- コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` を付ける。実装者自身のモデル名を書かない。

---

## File Structure

| ファイル | 責務 |
|---|---|
| `src/backtest/walkforward.py`（新規） | 日次5フェーズループ、週次再学習、実行条件のスナップショット、degradedの伝播、判断規則の差し替え |
| `src/data/database.py`（改修） | `BacktestRun` への列追加と `RunModelUsage` テーブルの追加 |
| `config.yaml`（改修） | `strategy.engine_version` と `strategy.on_model_failure` の追加 |
| `src/services/trading.py`（改修・Task 7のみ） | paper経路の執行仮定を `engine_version: v2` のときだけ切り替える |
| `tests/test_walkforward.py`（新規） | 5フェーズ、T+1執行、資金競合、週次再学習の締切、degraded伝播、戦略3案 |
| `tests/test_paper_execution_v2.py`（新規） | paper経路の切替と、legacy時に挙動が変わらないこと |

---

## Task 1: 日次ループの骨格と日次NAV

**Files:**
- Create: `src/backtest/walkforward.py`
- Test: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: `portfolio.empty_portfolio` / `nav` / `advance_session`
- Produces:
  - `MarketData`（frozen dataclass）: `bars: dict`（symbol → 日付indexのOHLCV DataFrame）, `sectors: dict`
  - `DailyRow`（frozen dataclass）: `session: date`, `nav: float`, `cash: float`, `n_holdings: int`, `realized_pnl: float`
  - `sessions_between(md: MarketData, start: date, end: date) -> list[date]` — 全銘柄共通の営業日
  - `closes_at(md: MarketData, session: date) -> dict` — その日の終値（足が無い銘柄は含めない）
  - `run_walkforward(md, start, end, *, initial_capital, ...) -> WalkForwardResult`（本タスクでは①②④のみ。③⑤は Task 2・3 で足す）
  - `WalkForwardResult`（frozen dataclass）: `daily: pd.DataFrame`, `trades: pd.DataFrame`, `rejected: pd.DataFrame`, `degraded: bool`, `degraded_reasons: list`, `model_usage: pd.DataFrame`, `run_id: Optional[int]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_walkforward.py` を新規作成する。

```python
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
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_walkforward.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backtest.walkforward'`

- [ ] **Step 3: 実装を書く**

`src/backtest/walkforward.py` を新規作成する。

```python
"""ポートフォリオ walk-forward エンジン。

1営業日を5フェーズで回す。

  ①前日までに決まった注文の執行 → ②保有と現金の更新 →
  ③退出ポリシーの逐次駆動 → ④日末のNAV記録 → ⑤翌日の候補生成

**Tの終値の情報はT+1以降の注文にしか使えない。** 現行 src/backtest/engine.py は
同じ終値でスコア生成と約定を行っており、引け後スキャン→翌朝発注という実運用と
乖離していた（レビューF04）。

判断は src/strategy/policy.py、約定は src/backtest/execution.py、現金と保有は
src/backtest/portfolio.py に委ね、本モジュールは進行と記録だけを担う。
旧 engine.py は strategy.engine_version: legacy の受け皿として無改造で残す。
"""
from dataclasses import dataclass, replace
from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest import execution
from src.backtest import portfolio as pf
from src.strategy import policy

_DAILY_COLUMNS = ["session", "nav", "cash", "n_holdings", "realized_pnl"]
_TRADE_COLUMNS = ["symbol", "entry_at", "entry_price", "exit_at", "exit_price",
                  "quantity", "pnl", "reason"]
_REJECTED_COLUMNS = ["session", "symbol", "reason"]
_MODEL_USAGE_COLUMNS = ["model_id", "from_session", "to_session"]


@dataclass(frozen=True)
class MarketData:
    """銘柄ごとの日足と業種。

    bars は symbol → 日付インデックスのOHLCV DataFrame。
    銘柄ごとに長さが違ってよい（上場・上場廃止・データ欠損）。
    """
    bars: dict
    sectors: dict


@dataclass(frozen=True)
class DailyRow:
    session: date
    nav: float
    cash: float
    n_holdings: int
    realized_pnl: float


@dataclass(frozen=True)
class WalkForwardResult:
    """実行結果。最終リターンだけでなく、日次NAV・約定・除外理由まで辿れる。"""
    daily: pd.DataFrame
    trades: pd.DataFrame
    rejected: pd.DataFrame
    degraded: bool
    degraded_reasons: list
    model_usage: pd.DataFrame
    run_id: Optional[int] = None


def sessions_between(md: MarketData, start: date, end: date) -> list:
    """全銘柄共通の営業日（各銘柄の足の日付の和集合）を昇順で返す。"""
    all_sessions = set()
    for df in md.bars.values():
        for ts in df.index:
            all_sessions.add(ts.date())
    return sorted(s for s in all_sessions if start <= s <= end)


def _bar_of(md: MarketData, symbol: str, session: date) -> Optional[pd.Series]:
    df = md.bars.get(symbol)
    if df is None:
        return None
    key = pd.Timestamp(session)
    if key not in df.index:
        return None
    return df.loc[key]


def closes_at(md: MarketData, session: date) -> dict:
    """その日の終値。足が無い銘柄は含めない（呼び出し側が取得単価へ落とす）。"""
    out = {}
    for symbol in md.bars:
        row = _bar_of(md, symbol, session)
        if row is not None:
            out[symbol] = float(row["close"])
    return out


def _observation(md: MarketData, symbol: str, session: date,
                 score: Optional[float] = None) -> Optional[policy.Observation]:
    row = _bar_of(md, symbol, session)
    if row is None:
        return None
    return policy.Observation(
        session=session, open=float(row["open"]), high=float(row["high"]),
        low=float(row["low"]), close=float(row["close"]), score=score,
    )


def run_walkforward(md: MarketData, start: date, end: date, *,
                    initial_capital: float,
                    decide: Callable,
                    policy_conf: policy.PolicyConfig,
                    costs: execution.CostConfig,
                    sizing: pf.SizingConfig,
                    liquidity: execution.LiquidityConfig) -> WalkForwardResult:
    """日次5フェーズでポートフォリオを進める。

    decide は判断規則（戦略バージョン）。
    `decide(session, rows, model, ctx) -> list[portfolio.Candidate]` の形で、
    その日の引けの情報から**翌営業日に出す**候補を返す。
    """
    sessions = sessions_between(md, start, end)
    portfolio_state = pf.empty_portfolio(initial_capital)
    daily: list = []

    for session in sessions:
        realized = 0.0
        closes = closes_at(md, session)

        # ④ 日末評価（①②③⑤は後続タスクで足す）
        portfolio_state = pf.advance_session(portfolio_state, closes)
        daily.append(DailyRow(
            session=session,
            nav=pf.nav(portfolio_state, closes),
            cash=portfolio_state.cash,
            n_holdings=len(portfolio_state.holdings),
            realized_pnl=realized,
        ))

    return WalkForwardResult(
        daily=pd.DataFrame([vars(r) for r in daily], columns=_DAILY_COLUMNS),
        trades=pd.DataFrame(columns=_TRADE_COLUMNS),
        rejected=pd.DataFrame(columns=_REJECTED_COLUMNS),
        degraded=False,
        degraded_reasons=[],
        model_usage=pd.DataFrame(columns=_MODEL_USAGE_COLUMNS),
    )
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_walkforward.py -v`
Expected: PASS（7件）

- [ ] **Step 5: BOM確認とコミット**

Run: `head -c 3 src/backtest/walkforward.py | xxd`（`2222 22` を確認。`efbb bf` なら下記で除去）

```python
for p in ["src/backtest/walkforward.py", "tests/test_walkforward.py"]:
    with open(p, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        with open(p, "wb") as f:
            f.write(data[3:])
```

```bash
git add src/backtest/walkforward.py tests/test_walkforward.py
git commit -m "$(cat <<'EOF'
feat(backtest): walk-forwardエンジンの日次ループ骨格を追加

1営業日を5フェーズで回す枠組みと日次NAVの記録を置く。
全銘柄共通の営業日で進め、足が無い銘柄はその日の評価から外す
（呼び出し側が取得単価へ落とす）。
旧engine.pyはlegacyの受け皿として無改造で残す。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: 退出の駆動と約定（フェーズ③）

**Files:**
- Modify: `src/backtest/walkforward.py`
- Test: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: `policy.step` / `HoldingState`、`execution.exit_fill`、`portfolio.apply_sell`
- Produces:
  - `_holding_state(h: pf.Holding) -> policy.HoldingState`
  - `_drive_exits(...)` — 内部関数。保有ごとに `policy.step()` を回し、`STOP` は当日約定、`MARKET` は翌営業日へ繰り越す

**設計:**

- 保有の `peak_price` / `sessions_held` は **`policy.step()` が返す次の状態で更新する**。`portfolio.advance_session()` は**その日の足が無い保有にだけ**使う（ポリシーを回せないため）。同じ規則の実装を2つ持たないため
- `STOP` の意図はその日のうちに約定する（`exit_fill` が `min(open, trigger)` で処理する）
- `MARKET` の意図（`SIGNAL_SELL` / `TIME_LIMIT`）は翌営業日の寄りで約定するので、繰越キューへ入れる

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_walkforward.py` の末尾に追記する。

```python
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
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_walkforward.py -v`
Expected: FAIL — 買いがまだ実装されていないため `len(res.trades) == 1` が `0 == 1` で落ちる。Task 3 の買い実装と合わせて通る設計なので、本タスクでは**退出部分のみ**を実装し、Task 3 完了時に全件通す。

> **実装者への注記:** 本タスクと Task 3 は相互に依存する（買いが無いと退出を観測できない）。Task 2 では `_drive_exits` を実装してコミットし、テストの成功確認は Task 3 の Step 4 で行う。Task 2 の Step 4 では「`_drive_exits` の単体テスト（下記）だけ」を通す。

- [ ] **Step 3: 実装を追加**

`src/backtest/walkforward.py` の末尾に追加する。

```python
def _holding_state(h: pf.Holding) -> policy.HoldingState:
    """ポートフォリオの保有から、退出ポリシーが使う状態を作る。"""
    return policy.HoldingState(
        symbol=h.symbol, entry_at=h.entry_at, avg_cost=h.avg_cost,
        quantity=h.quantity, peak_price=h.peak_price,
        sessions_held=h.sessions_held,
    )


def _drive_exits(portfolio_state: pf.Portfolio, md: MarketData, session: date,
                 policy_conf: policy.PolicyConfig,
                 costs: execution.CostConfig) -> tuple:
    """保有ごとに退出ポリシーを1営業日ぶん進める。

    戻り値: (次のポートフォリオ, 当日約定した取引のリスト, 翌営業日へ繰り越す退出意図)

    保有の peak_price / sessions_held は **policy.step() が返す次の状態で更新する**。
    portfolio.advance_session() は「その日の足が無くポリシーを回せない保有」にだけ
    使う。同じ規則（未来のピークを遡ってストップに使わない）の実装を2つ
    持たないため。

    STOP の意図はその日のうちに約定する（execution.exit_fill が
    min(open, trigger) で処理する）。MARKET の意図（売りシグナル・満了）は
    翌営業日の寄りで約定するので繰越キューへ入れる。
    """
    trades: list = []
    pending_exits: list = []
    holdings = dict(portfolio_state.holdings)
    current = portfolio_state

    for symbol, holding in list(portfolio_state.holdings.items()):
        obs = _observation(md, symbol, session)
        if obs is None:
            # 足が無い日はポリシーを回せない。経過だけ進めて持ち越す
            holdings[symbol] = replace(
                holding, sessions_held=holding.sessions_held + 1)
            continue

        next_state, intent = policy.step(_holding_state(holding), obs, policy_conf)
        holdings[symbol] = replace(
            holding, peak_price=next_state.peak_price,
            sessions_held=next_state.sessions_held)

        if intent is None:
            continue
        if intent.order_type == "MARKET":
            pending_exits.append((symbol, intent))
            continue

        fill = execution.exit_fill(intent, obs, None, holding.quantity, costs)
        if fill is None:
            continue
        current = replace(current, holdings=holdings)
        current, realized = pf.apply_sell(
            current, symbol, holding.quantity, fill.price, costs.commission_pct)
        holdings = dict(current.holdings)
        trades.append({
            "symbol": symbol, "entry_at": holding.entry_at,
            "entry_price": holding.avg_cost, "exit_at": fill.at,
            "exit_price": fill.price, "quantity": holding.quantity,
            "pnl": realized, "reason": intent.reason,
        })

    current = replace(current, holdings=holdings)
    return current, trades, pending_exits


def _settle_pending_exits(portfolio_state: pf.Portfolio, md: MarketData,
                          session: date, pending_exits: list,
                          costs: execution.CostConfig) -> tuple:
    """前営業日に決まった成行退出を、この日の寄りで約定させる。

    戻り値: (次のポートフォリオ, 約定した取引のリスト, 約定できなかった意図)
    """
    trades: list = []
    carried: list = []
    current = portfolio_state

    for symbol, intent in pending_exits:
        holding = current.holdings.get(symbol)
        obs = _observation(md, symbol, session)
        if holding is None:
            continue
        if obs is None:
            carried.append((symbol, intent))
            continue
        fill = execution.exit_fill(intent, obs, obs, holding.quantity, costs)
        if fill is None:
            carried.append((symbol, intent))
            continue
        current, realized = pf.apply_sell(
            current, symbol, holding.quantity, fill.price, costs.commission_pct)
        trades.append({
            "symbol": symbol, "entry_at": holding.entry_at,
            "entry_price": holding.avg_cost, "exit_at": session,
            "exit_price": fill.price, "quantity": holding.quantity,
            "pnl": realized, "reason": intent.reason,
        })
    return current, trades, carried
```

**注意:** `_settle_pending_exits` は `exit_fill(intent, obs, obs, ...)` と当日の足を `next_bar` にも渡す。意図は前営業日に出ており、この日の寄りが「翌営業日の寄り」に当たるため。

- [ ] **Step 4: `_drive_exits` の単体テストを通す**

`tests/test_walkforward.py` の末尾に追記して実行する。

```python
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
```

`replace` は `dataclasses` のもの。テストファイル冒頭の import に `from dataclasses import replace` を足すこと。

Run: `pytest tests/test_walkforward.py::TestDriveExitsUnit -v`
Expected: PASS（4件）

- [ ] **Step 5: コミット**

```bash
git add src/backtest/walkforward.py tests/test_walkforward.py
git commit -m "$(cat <<'EOF'
feat(backtest): 退出ポリシーの駆動と約定を追加

保有ごとにpolicy.step()を1営業日ぶん進める。peak_priceと経過営業日数は
policyが返す次の状態で更新し、advance_session()は足が無くポリシーを
回せない保有にだけ使う（同じ規則の実装を2つ持たないため）。
STOPは当日約定、MARKET（売りシグナル・満了）は翌営業日の寄りへ繰り越す。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 候補生成と翌営業日執行（フェーズ①②⑤）

**Files:**
- Modify: `src/backtest/walkforward.py`
- Test: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: Task 1・2、`portfolio.allocate` / `apply_buy`、`execution.entry_fill_limited`
- Produces: `run_walkforward` を5フェーズ完全版にする（`decide` の呼び出し、`pending_buys` の繰越、`rejected` の記録）

**フェーズの順序（spec §8）:**

1. 前日までに決まった注文の執行（買い・成行退出）
2. 保有と現金の更新
3. 退出ポリシーの逐次駆動
4. 日末のNAV記録
5. **翌日の**候補生成

**`decide` の契約:** `decide(session, rows, model, ctx) -> list[portfolio.Candidate]`。`rows` は `symbol -> その日の足（pd.Series）`。`ctx` は `{"sectors": dict, "portfolio": Portfolio, "closes": dict}`。返した候補は**翌営業日の寄りで執行される**。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_walkforward.py` の末尾に追記する。

```python
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
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_walkforward.py -v`
Expected: FAIL — 候補生成と執行が未実装のため `TestEntryTiming` 以降が落ちる

- [ ] **Step 3: `run_walkforward` を5フェーズ完全版に置き換える**

`src/backtest/walkforward.py` の `run_walkforward` を次に置き換える。

```python
def run_walkforward(md: MarketData, start: date, end: date, *,
                    initial_capital: float,
                    decide: Callable,
                    policy_conf: policy.PolicyConfig,
                    costs: execution.CostConfig,
                    sizing: pf.SizingConfig,
                    liquidity: execution.LiquidityConfig,
                    model=None) -> WalkForwardResult:
    """日次5フェーズでポートフォリオを進める。

      ①前日までに決まった注文の執行 → ②保有と現金の更新 →
      ③退出ポリシーの逐次駆動 → ④日末のNAV記録 → ⑤翌日の候補生成

    **Tの終値の情報はT+1以降の注文にしか使えない。** decide() が返した候補は
    その日には約定せず、翌営業日の寄りで執行される。

    decide は判断規則（戦略バージョン）。
    `decide(session, rows, model, ctx) -> list[portfolio.Candidate]`。
    """
    sessions = sessions_between(md, start, end)
    state = pf.empty_portfolio(initial_capital)
    daily: list = []
    trades: list = []
    rejected: list = []
    pending_buys: list = []
    pending_exits: list = []

    for session in sessions:
        realized = 0.0
        closes = closes_at(md, session)

        # ① 前日までに決まった成行退出を、この日の寄りで約定させる
        state, exit_trades, pending_exits = _settle_pending_exits(
            state, md, session, pending_exits, costs)
        for t in exit_trades:
            realized += t["pnl"]
        trades.extend(exit_trades)

        # ① 前日までに決まった買いを、この日の寄りで約定させる（②保有と現金の更新）
        for order in pending_buys:
            bar = _bar_of(md, order.symbol, session)
            if bar is None:
                rejected.append({"session": session, "symbol": order.symbol,
                                 "reason": "この営業日の足が無く約定できません"})
                continue
            obs = _observation(md, order.symbol, session)
            result = execution.entry_fill_limited(
                obs, order.quantity, costs, liquidity, volume=int(bar["volume"]))
            if result.unfilled_reason:
                rejected.append({"session": session, "symbol": order.symbol,
                                 "reason": result.unfilled_reason})
            if result.fill is None:
                continue
            state = pf.apply_buy(
                state, order.symbol, result.fill.quantity, result.fill.price,
                order.sector, session, costs.commission_pct)
        pending_buys = []

        # ③ 退出ポリシーの逐次駆動
        state, stop_trades, new_pending_exits = _drive_exits(
            state, md, session, policy_conf, costs)
        for t in stop_trades:
            realized += t["pnl"]
        trades.extend(stop_trades)
        pending_exits = pending_exits + new_pending_exits

        # ④ 日末のNAV記録
        daily.append(DailyRow(
            session=session, nav=pf.nav(state, closes), cash=state.cash,
            n_holdings=len(state.holdings), realized_pnl=realized,
        ))

        # ⑤ 翌日の候補生成（この日の引けの情報だけを使う）
        rows = {s: _bar_of(md, s, session) for s in md.bars}
        rows = {s: r for s, r in rows.items() if r is not None}
        ctx = {"sectors": md.sectors, "portfolio": state, "closes": closes}
        candidates = decide(session, rows, model, ctx)
        if candidates:
            orders, rejects = pf.allocate(state, candidates, sizing, closes)
            pending_buys = orders
            for r in rejects:
                rejected.append({"session": session, "symbol": r.symbol,
                                 "reason": r.reason})

    return WalkForwardResult(
        daily=pd.DataFrame([vars(r) for r in daily], columns=_DAILY_COLUMNS),
        trades=pd.DataFrame(trades, columns=_TRADE_COLUMNS),
        rejected=pd.DataFrame(rejected, columns=_REJECTED_COLUMNS),
        degraded=False,
        degraded_reasons=[],
        model_usage=pd.DataFrame(columns=_MODEL_USAGE_COLUMNS),
    )
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_walkforward.py -v`
Expected: PASS（Task 1 の7件 + Task 2 の10件 + 本タスクの8件 = 25件）

- [ ] **Step 5: 全体回帰を確認してコミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

```bash
git add src/backtest/walkforward.py tests/test_walkforward.py
git commit -m "$(cat <<'EOF'
feat(backtest): 候補生成と翌営業日執行で5フェーズを完成させた

Tの終値で判断した注文はその日には約定せず、翌営業日の寄りで執行する。
現行エンジンは同じ終値で判断と約定を行っており実運用と乖離していた。
同日の複数候補は資金競合を解決し、採用しなかった候補と出来高による
未約定・部分約定はすべて理由つきで記録する。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: 期間中の週次再学習とモデル使用履歴

**Files:**
- Modify: `src/backtest/walkforward.py`
- Modify: `src/data/database.py`（`RunModelUsage` を追加）
- Test: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: Task 3、`validation.training_inputs`（締切の適用）
- Produces:
  - `RetrainConfig`（frozen dataclass）: `every_sessions: int`（0で無効）, `warmup_sessions: int`
  - `RunModelUsage` モデル: `run_id` / `model_id` / `from_session` / `to_session` / `n_train_events`
  - `run_walkforward(..., retrain=None, train_model=None)` — 再学習の結線

**背景（spec §7 経路4）:** バックテスト中の週次再学習にも、段階Cと**同じ締切**を適用する。その再学習時点までにラベルが観測可能になったイベントだけを使う。外側 fold 開始時に一度 purge するだけでは足りない。現行 `engine.py:82-97` は開始前に一度だけ学習し、テスト期間中は再学習しない。

**`train_model` の契約:** `train_model(as_of: date) -> tuple[object, int]` — その時点までに確定した情報だけで学習し、`(モデル, 学習イベント数)` を返す。締切の適用は呼び出し側（この関数の実装者）の責任で、`validation.training_inputs()` を通す。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_walkforward.py` の末尾に追記する。

```python
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
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_walkforward.py -v`
Expected: FAIL — `AttributeError: module 'src.backtest.walkforward' has no attribute 'RetrainConfig'`

- [ ] **Step 3: モデルを追加**

`src/data/database.py` の `class Prediction` の直後に追加する（段階C後半が未実装なら `class Dataset` の直後）。

```python
class RunModelUsage(Base):
    """バックテスト実行の中で、どのモデルをいつからいつまで使ったか。

    期間中に週次で再学習する実行は `BacktestRun.model_id` 一つでは表せない。
    「この成績はどのモデルが出したのか」を後から辿るために別テーブルに持つ。
    """
    __tablename__ = "run_model_usage"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, nullable=False)
    model_id = Column(String(64))
    from_session = Column(Date)
    to_session = Column(Date)
    n_train_events = Column(Integer)

    __table_args__ = (Index("ix_run_model_usage_run_id", "run_id"),)
```

- [ ] **Step 4: 再学習を結線する**

`src/backtest/walkforward.py` に追加し、`run_walkforward` を修正する。

```python
@dataclass(frozen=True)
class RetrainConfig:
    """期間中の再学習スケジュール。

    every_sessions は再学習の間隔（営業日）。0で無効。
    warmup_sessions は最初の学習までに必要な助走期間。
    実運用が週次で再学習するなら every_sessions=5 で同じ周期になる。
    """
    every_sessions: int = 0
    warmup_sessions: int = 0
```

`run_walkforward` のシグネチャに `retrain: Optional[RetrainConfig] = None` と `train_model: Optional[Callable] = None` を足し、ループを次のように変える。

```python
    model_usage: list = []
    current_model = model
    current_model_id: Optional[str] = None
    model_since: Optional[date] = None

    for index, session in enumerate(sessions):
        # ⓪ 再学習（この時点までに確定した情報だけで学習する）
        if (retrain is not None and train_model is not None
                and retrain.every_sessions > 0
                and index >= retrain.warmup_sessions
                and (index - retrain.warmup_sessions) % retrain.every_sessions == 0):
            if current_model_id is not None:
                model_usage.append({
                    "model_id": current_model_id, "from_session": model_since,
                    "to_session": sessions[index - 1],
                    "n_train_events": current_n_train,
                })
            current_model, current_n_train = train_model(session)
            current_model_id = str(current_model)
            model_since = session
```

ループの最後（`for` を抜けたところ）で、使用中のモデルを閉じる。

```python
    if current_model_id is not None:
        model_usage.append({
            "model_id": current_model_id, "from_session": model_since,
            "to_session": sessions[-1], "n_train_events": current_n_train,
        })
```

`decide` の呼び出しは `model` ではなく `current_model` を渡す。戻り値の `model_usage` は次にする。

```python
        model_usage=pd.DataFrame(model_usage, columns=_MODEL_USAGE_COLUMNS),
```

`_MODEL_USAGE_COLUMNS` を次に変更する。

```python
_MODEL_USAGE_COLUMNS = ["model_id", "from_session", "to_session", "n_train_events"]
```

`current_n_train` はループ前に `current_n_train = 0` で初期化すること。

- [ ] **Step 5: テストを実行して成功を確認**

Run: `pytest tests/test_walkforward.py -v`
Expected: PASS（31件）

- [ ] **Step 6: 全体回帰とコミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

```bash
git add src/backtest/walkforward.py src/data/database.py tests/test_walkforward.py
git commit -m "$(cat <<'EOF'
feat(backtest,data): 期間中の週次再学習とモデル使用履歴を追加

現行エンジンは開始前に一度だけ学習しテスト期間中は再学習しなかった。
実運用と同じ周期で学習し直し、その時点までに確定した情報だけを使う。
週次で変わるモデルはBacktestRun.model_id一つでは表せないため、
どのモデルをいつからいつまで使ったかをRunModelUsageに残す。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: degraded の伝播と実行条件のスナップショット

**Files:**
- Modify: `src/backtest/walkforward.py`
- Modify: `src/data/database.py`（`BacktestRun` に列追加）
- Test: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: Task 4
- Produces:
  - `RunSnapshot`（frozen dataclass）: `strategy_version` / `config_hash` / `config_json` / `dataset_id` / `code_version` / `execution_model_version`
  - `save_run(result: WalkForwardResult, snapshot: RunSnapshot, *, symbol_label: str, start: date, end: date, initial_capital: float, costs: CostConfig) -> int`
  - `BacktestRun` への列追加: `strategy_version` / `config_hash` / `config_json` / `dataset_id` / `code_version` / `execution_model_version` / `degraded`

**背景（spec §8）:** 現行 `BacktestRun` は完全な実行設定・モデル・入力データの来歴を持たない。`config_hash` は同一性の確認に使えるが復元には使えないため、**設定JSONの実体も保存する**。推論例外が1件でも発生した実行には `degraded=True` を立て、比較とモデル昇格から除外できるようにする。

**注意:** `engine.py:160-161` の `except Exception: pass` は削除しない。degraded は新エンジンに最初から持たせる（spec §10 の互換方針）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_walkforward.py` の末尾に追記する。冒頭の import に `from sqlalchemy import select`、`from src.core import config as cfg`、`from src.data import database as db`、`from src.data.database import get_session` を足す。

```python
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
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_walkforward.py -v`
Expected: FAIL — `AttributeError: module 'src.backtest.walkforward' has no attribute 'RunSnapshot'`

- [ ] **Step 3: `BacktestRun` に列を追加**

`src/data/database.py` の `class BacktestRun` に追加する（`archived` の直後）。

```python
    # ─── 再現用の来歴（段階D。レビュー Backtest）────────────────────
    # 現行は閾値とコストしか持たず、どの設定・どのモデル・どの入力データで
    # 出した成績かを後から辿れなかった。config_hash は同一性の確認には使えるが
    # 復元には使えないため、設定の実体も併せて持つ。
    strategy_version = Column(String(32))
    config_hash = Column(String(64))
    config_json = Column(Text)
    dataset_id = Column(String(64))
    code_version = Column(String(64))
    execution_model_version = Column(String(32))
    # 推論例外が1件でも発生した実行は 1。比較とモデル昇格から除外する
    degraded = Column(Integer, default=0)
```

- [ ] **Step 4: 実装を追加**

`src/backtest/walkforward.py` に追加する。import に `import json` と `from src.core import clock` を足す。

```python
@dataclass(frozen=True)
class RunSnapshot:
    """実行開始時に固定する来歴。

    config_hash は同一性の確認には使えるが**復元には使えない**ため、
    設定の実体（config_json）も併せて持つ（spec §8）。
    """
    strategy_version: str
    config_hash: str
    config_json: str
    dataset_id: Optional[str] = None
    code_version: Optional[str] = None
    execution_model_version: Optional[str] = None


def save_run(result: WalkForwardResult, snapshot: RunSnapshot, *,
             symbol_label: str, start: date, end: date,
             initial_capital: float, costs: execution.CostConfig) -> int:
    """実行結果と来歴を保存し、run_id を返す。

    日次NAVは equity_curve_json に、モデル使用履歴は RunModelUsage に入れる。
    degraded な実行も**保存する**（比較から外すのは読む側の責任で、
    「失敗した実行があったこと」自体は残す）。
    """
    from src.data.database import BacktestRun, RunModelUsage, get_session

    final_capital = float(result.daily["nav"].iloc[-1]) if len(result.daily) else initial_capital
    total_return = (final_capital - initial_capital) / initial_capital if initial_capital else 0.0
    curve = [{"date": r["session"].isoformat(), "equity": round(float(r["nav"]), 0)}
             for _, r in result.daily.iterrows()]

    with get_session() as session:
        run = BacktestRun(
            symbol=symbol_label, start_date=start, end_date=end,
            initial_capital=initial_capital, final_capital=final_capital,
            total_return=total_return,
            trade_count=len(result.trades),
            slippage_pct=costs.slippage_pct, commission_pct=costs.commission_pct,
            created_at=clock.now(),
            equity_curve_json=json.dumps(curve, ensure_ascii=False),
            strategy_version=snapshot.strategy_version,
            config_hash=snapshot.config_hash,
            config_json=snapshot.config_json,
            dataset_id=snapshot.dataset_id,
            code_version=snapshot.code_version,
            execution_model_version=snapshot.execution_model_version,
            degraded=1 if result.degraded else 0,
        )
        session.add(run)
        session.flush()
        run_id = run.id
        for _, u in result.model_usage.iterrows():
            session.add(RunModelUsage(
                run_id=run_id, model_id=u["model_id"],
                from_session=u["from_session"], to_session=u["to_session"],
                n_train_events=int(u["n_train_events"]),
            ))
        session.commit()
    return run_id
```

- [ ] **Step 5: degraded を伝播させる**

`run_walkforward` の再学習と `decide` の呼び出しを `try` で囲み、例外を記録する。**握り潰さない。**

再学習側:

```python
            try:
                current_model, current_n_train = train_model(session)
                current_model_id = str(current_model)
                model_since = session
            except Exception as e:
                degraded_reasons.append(f"{session}: 再学習に失敗しました: {e}")
                logger.warning(f"バックテスト中の再学習に失敗: {session} {e}")
```

候補生成側:

```python
        try:
            candidates = decide(session, rows, current_model, ctx)
        except Exception as e:
            degraded_reasons.append(f"{session}: 判断に失敗しました: {e}")
            logger.warning(f"バックテスト中の判断に失敗: {session} {e}")
            candidates = []
```

`degraded_reasons: list = []` をループ前に用意し、戻り値を次にする。

```python
        degraded=bool(degraded_reasons),
        degraded_reasons=degraded_reasons,
```

- [ ] **Step 6: テストを実行して成功を確認**

Run: `pytest tests/test_walkforward.py -v`
Expected: PASS（41件）

- [ ] **Step 7: 全体回帰とコミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

```bash
git add src/backtest/walkforward.py src/data/database.py tests/test_walkforward.py
git commit -m "$(cat <<'EOF'
feat(backtest,data): degradedの伝播と実行条件のスナップショットを追加

推論・再学習の例外を握り潰さず、1件でも起きた実行にdegradedを立てる。
失敗しても最後まで進めて「どこまで進んだか」を残す。
BacktestRunに戦略版・設定ハッシュ・設定の実体・データID・コード版・
執行モデル版を足す。config_hashは同一性の確認には使えるが復元には
使えないため実体も持つ。

旧engine.pyのexcept Exception: passは削除しない（legacyの受け皿として
無改造で残す方針。degradedは新エンジンに最初から持たせる）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: 戦略3案の判断規則

**Files:**
- Modify: `src/backtest/walkforward.py`
- Modify: `config.yaml`（`strategy.on_model_failure` の追加）
- Test: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: Task 3・5
- Produces:
  - `StrategyConfig`（frozen dataclass）: `buy_threshold: float`, `rule_weight: float`, `ml_weight: float`, `on_model_failure: str`
  - `make_weighted_blend(conf) -> Callable` — 既存の加重合成
  - `make_rule_only(conf) -> Callable` — 縮尺を明示したルール単独
  - `make_rule_then_ml(conf) -> Callable` — ルールで候補を作りMLで順位付け
  - `ON_FAILURE_RULE_ONLY` / `ON_FAILURE_HALT_NEW` 定数

**背景（spec §8 / F08）:** **同じイベント表・同じラベルの上で、判断規則だけを変えて**3案を比較する。まず試すのは3（ルールで候補を作り、MLで買う・見送るの順位を決める）。全日付を機械的に買い・売りへ変換せず、現行ルールへの追加効果を測定しやすいため。

モデル失敗時の動作は**設定として明示する**。現行 `signal.py:90` は暗黙にルール重みだけが残り、ML欠落時の縮尺が変わって同じ閾値でも売買判断が変わってしまう。

**`score_fn` の契約:** 3案とも `score_fn(symbol, row) -> tuple[float, Optional[float]]`（ルールスコア, ML確率 or None）を引数に取る。walk-forward の外から与えることで、指標計算とモデル推論を判断規則から切り離す。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_walkforward.py` の末尾に追記する。

```python
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
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_walkforward.py -v`
Expected: FAIL — `AttributeError: module 'src.backtest.walkforward' has no attribute 'StrategyConfig'`

- [ ] **Step 3: `config.yaml` に設定を追加**

`config.yaml` の `strategy:` 節の `allow_unverified_model_load` の直後に追加する。

```yaml
  # ─── 評価基盤の段階投入（2026-09-12〜）────────────────────────
  # 学習データの生成元・CV分割・バックテストの実行主体を切り替える。
  # legacy = 従来（labeling.build_training_set / TimeSeriesSplit / backtest.engine）
  # v2     = 新方式（dataset.build_events / validation.calendar_split / walkforward）
  # 既定は legacy。旧方式へいつでも戻せることが段階投入の前提（設計書§10）。
  engine_version: "legacy"
  # MLモデルが無い・推論に失敗したときの新規候補の扱い。
  # rule_only = ルール単独の定義済み戦略へ切り替える
  # halt_new  = 新規候補の生成を止める（既存ポジションの退出管理は続ける）
  # 現行は暗黙にルール重みだけが残り、ML欠落時の縮尺が変わって同じ閾値でも
  # 売買判断が変わっていた。どちらの動作を採るかを設定として明示する。
  on_model_failure: "rule_only"
```

- [ ] **Step 4: 実装を追加**

`src/backtest/walkforward.py` に追加する。

```python
ON_FAILURE_RULE_ONLY = "rule_only"
ON_FAILURE_HALT_NEW = "halt_new"


@dataclass(frozen=True)
class StrategyConfig:
    """判断規則の設定。

    on_model_failure は「MLが無い・推論に失敗したとき」の動作を**明示する**。
    現行 signal.py:90 は暗黙にルール重みだけが残り、ML欠落時の縮尺が変わって
    同じ閾値でも売買判断が変わっていた（レビューF08）。
    """
    buy_threshold: float
    rule_weight: float = 0.5
    ml_weight: float = 0.5
    on_model_failure: str = ON_FAILURE_RULE_ONLY


def _candidate(symbol: str, row, ctx: dict, score: float) -> pf.Candidate:
    return pf.Candidate(symbol=symbol, sector=ctx["sectors"].get(symbol, ""),
                        price=float(row["close"]), score=score)


def make_weighted_blend(conf: StrategyConfig, score_fn: Callable) -> Callable:
    """案1: 既存の加重合成。`rule × rule_weight + (p−0.5)×2 × ml_weight`。"""
    def decide(session, rows, model, ctx):
        out = []
        for symbol, row in rows.items():
            rule, proba = score_fn(symbol, row)
            ml = (proba - 0.5) * 2 if proba is not None else 0.0
            blended = rule * conf.rule_weight + ml * conf.ml_weight
            if blended >= conf.buy_threshold:
                out.append(_candidate(symbol, row, ctx, blended))
        return out
    return decide


def make_rule_only(conf: StrategyConfig, score_fn: Callable) -> Callable:
    """案2: 縮尺を明示したルール単独。重みで割り引かない。"""
    def decide(session, rows, model, ctx):
        out = []
        for symbol, row in rows.items():
            rule, _ = score_fn(symbol, row)
            if rule >= conf.buy_threshold:
                out.append(_candidate(symbol, row, ctx, rule))
        return out
    return decide


def make_rule_then_ml(conf: StrategyConfig, score_fn: Callable) -> Callable:
    """案3: ルールで候補を作り、MLで買う・見送るの順位を決める。

    **まず試すのはこれ。** 全日付を機械的に買い・売りへ変換せず、
    ルールが候補としたイベントに対してのみMLの追加効果を測るため（spec §8）。
    MLが無い・失敗した場合の動作は on_model_failure で明示する。
    """
    def decide(session, rows, model, ctx):
        gated = []
        for symbol, row in rows.items():
            rule, proba = score_fn(symbol, row)
            if rule < conf.buy_threshold:
                continue
            gated.append((symbol, row, rule, proba))

        usable = [g for g in gated if g[3] is not None]
        if len(usable) < len(gated):
            if conf.on_model_failure == ON_FAILURE_HALT_NEW:
                return []
            # rule_only: ML無しの候補はルールスコアで順位付けする
            return [_candidate(s, r, ctx, rule) for s, r, rule, _ in gated]

        ranked = sorted(usable, key=lambda g: g[3], reverse=True)
        return [_candidate(s, r, ctx, proba) for s, r, _, proba in ranked]
    return decide
```

- [ ] **Step 5: テストを実行して成功を確認**

Run: `pytest tests/test_walkforward.py -v`
Expected: PASS（51件）

- [ ] **Step 6: 全体回帰とコミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと（`config.yaml` の追加は既存テストに影響しない）

```bash
git add src/backtest/walkforward.py config.yaml tests/test_walkforward.py
git commit -m "$(cat <<'EOF'
feat(backtest): 戦略3案の判断規則と明示的な失敗時動作を追加

同じ市場データ・同じループの上で判断規則だけを差し替えて比較する。
まず試すのは「ルールで候補を作りMLで順位を決める」案で、全日付を
機械的に売買へ変換せずルールへの追加効果を測れるため。
ML欠落時の動作をon_model_failureで明示する。現行は暗黙にルール重みだけが
残り、縮尺が変わって同じ閾値でも売買判断が変わっていた。
engine_versionもconfig.yamlへ追加した（既定legacy）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: paper 経路の執行仮定を切り替える

**Files:**
- Modify: `src/services/trading.py`
- Test: `tests/test_paper_execution_v2.py`

**Interfaces:**
- Consumes: `config.yaml` の `strategy.engine_version`
- Produces:
  - `TradingServices._engine_version() -> str`
  - `TradingServices._paper_uses_v2_execution() -> bool`
  - `stop_loss_check` と `signal_scan` の paper 経路が `v2` のときだけ挙動を変える

**背景（spec §8・§10）:** `stop_loss_check`（`src/services/trading.py:362-365`）は paper モードで日足終値を使って損切りを判定している。F04（同一終値での判断・約定）はバックテストだけでなく paper 運用にも及んでいる。

**過去の日足を再生する paper と、現在の市場を観測する paper は入力契約が違う。** 前者は日足による約定の近似であり、既存の翌朝9:05運用と同じ約定モデルとしては表示しない。

**最重要の制約:** `engine_version: legacy`（既定）のとき、**paper の挙動は現在と1ビットも変わらないこと**。切替はテストで固定する。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_paper_execution_v2.py` を新規作成する。

```python
"""paper経路の執行仮定の切替（engine_version）のテスト

stop_loss_checkはpaperモードで日足終値を使って損切りを判定しており、
F04（同一終値での判断・約定）がpaper運用にも及んでいた（spec §8）。
**engine_version: legacy のとき挙動が1ビットも変わらないこと**を固定する。
"""
import inspect

import pytest

from src.core import config as cfg
from src.services import trading


@pytest.fixture(autouse=True)
def _load_config():
    cfg.load("config.yaml")


class TestEngineVersionDefault:
    def test_defaults_to_legacy(self):
        """config.yaml の既定は legacy（旧方式へいつでも戻せる）"""
        assert cfg.get_section("strategy").get("engine_version") == "legacy"

    def test_unknown_value_is_treated_as_legacy(self):
        """未知の値は安全側（legacy）に倒す"""
        cfg.get_section("strategy")["engine_version"] = "experimental"
        svc = trading.TradingServices(client=None, risk=None, order_mgr=None)
        assert svc._engine_version() == "legacy"
        assert svc._paper_uses_v2_execution() is False

    def test_v2_is_recognized(self):
        cfg.get_section("strategy")["engine_version"] = "v2"
        svc = trading.TradingServices(client=None, risk=None, order_mgr=None)
        assert svc._engine_version() == "v2"
        assert svc._paper_uses_v2_execution() is True


class TestLegacyBehaviourUnchanged:
    """legacy では paper の挙動が現在と変わらないこと"""

    def _source(self, name):
        return inspect.getsource(getattr(trading.TradingServices, name))

    def test_stop_loss_check_still_has_the_legacy_close_path(self):
        """従来の日足終値による損切り判定が残っている"""
        src = self._source("stop_loss_check")
        assert "load_ohlcv" in src
        assert 'df["close"].iloc[-1]' in src

    def test_stop_loss_check_branches_on_engine_version(self):
        """v2のときだけ別の経路へ入る分岐がある"""
        assert "_paper_uses_v2_execution" in self._source("stop_loss_check")

    def test_signal_scan_branches_on_engine_version(self):
        assert "_paper_uses_v2_execution" in self._source("signal_scan")

    def test_freshness_gate_is_still_wired(self):
        """段階Aの鮮度ゲートが外れていない"""
        assert "_is_fresh_for_new_candidate" in self._source("signal_scan")

    def test_stop_loss_check_does_not_call_the_freshness_gate(self):
        """保有保護の退出は鮮度に関わらず実行する（段階Aの不変条件）"""
        assert "_is_fresh_for_new_candidate" not in self._source("stop_loss_check")
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_paper_execution_v2.py -v`
Expected: FAIL — `AttributeError: 'TradingServices' object has no attribute '_engine_version'`

- [ ] **Step 3: 判定メソッドを追加**

`src/services/trading.py` の `TradingServices` に追加する（`_is_fresh_for_new_candidate` の直後）。

```python
    def _engine_version(self) -> str:
        """評価基盤のどの世代で動くか。未知の値は安全側（legacy）に倒す。

        legacy = 従来の挙動（既定）。v2 = 段階B〜Dで作り直した執行仮定。
        旧方式へいつでも戻せることが段階投入の前提（設計書 §10）。
        """
        value = cfg.get_section("strategy").get("engine_version", "legacy")
        return "v2" if value == "v2" else "legacy"

    def _paper_uses_v2_execution(self) -> bool:
        """paper経路で新しい執行仮定（翌営業日の寄り）を使うか。

        stop_loss_check は paper モードで日足終値を使って損切りを判定しており、
        同一終値での判断・約定という問題が paper 運用にも及んでいた
        （レビューF04）。**過去の日足を再生する paper と、現在の市場を観測する
        paper は入力契約が違う**。前者は日足による約定の近似であり、既存の
        翌朝9:05運用と同じ約定モデルとしては扱わない。
        """
        return self._engine_version() == "v2"
```

- [ ] **Step 4: `stop_loss_check` と `signal_scan` に分岐を入れる**

`stop_loss_check` の paper 分岐（現在 `load_ohlcv` で終値を取っている箇所）を次にする。**`legacy` の経路は1行も変えない。**

```python
                if is_paper:
                    df = load_ohlcv(sym)
                    if self._paper_uses_v2_execution():
                        # v2: 当日の終値で判断して同じ終値で約定する経路を作らない。
                        # 日足しか無い時点では「翌営業日の寄りで退出する」近似に留め、
                        # 判断だけをこの日に行う（実際の退出は翌営業日の
                        # morning_execution が拾う）。
                        price = float(df["open"].iloc[-1]) if len(df) else 0
                    else:
                        price = float(df["close"].iloc[-1]) if len(df) else 0
                else:
```

`signal_scan` の paper 即時シミュレート（`close_price = float(df["close"].iloc[-1])`）の直前に分岐を入れる。

```python
                if is_paper:
                    if self._paper_uses_v2_execution():
                        # v2: 引けで判断した注文をその日の終値で約定させない。
                        # 翌営業日の morning_execution が拾うシグナルとして
                        # 保存するだけに留める（判断と執行のセッションを分ける）。
                        continue
                    # ペーパーモード: 当日終値でシミュレート
                    close_price = float(df["close"].iloc[-1])
```

- [ ] **Step 5: テストを実行して成功を確認**

Run: `pytest tests/test_paper_execution_v2.py -v`
Expected: PASS（8件）

- [ ] **Step 6: legacy の回帰が無いことを確認**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと。`engine_version` の既定が `legacy` なので既存の paper テストは全て通る

Run: `pytest tests/test_signal_freshness_gate.py tests/test_morning_execution.py tests/test_afternoon_execution.py -v`
Expected: PASS（段階Aの鮮度ゲートと既存の執行テストが通る）

- [ ] **Step 7: BOM確認とコミット**

Run: `head -c 3 src/services/trading.py | xxd`（`2222 22` を確認）

```bash
git add src/services/trading.py tests/test_paper_execution_v2.py
git commit -m "$(cat <<'EOF'
fix(services): paper経路の同一終値での判断・約定をv2で是正

stop_loss_checkはpaperモードで日足終値を使って損切りを判定しており、
同一終値での判断・約定という問題がpaper運用にも及んでいた。
engine_version: v2 のときだけ翌営業日の寄りを使う経路へ切り替える。
既定はlegacyで、そのときpaperの挙動は従来と変わらない。
過去の日足を再生するpaperと現在の市場を観測するpaperは入力契約が
違うため、前者を翌朝9:05運用と同じ約定モデルとしては扱わない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## 段階D 完了条件の確認

spec §14 の段階D完了条件を、前半・後半あわせて検証する。

- [ ] **確認1: Tの終値で買う条件が成立してもT+1以降にのみ約定する**

Run: `pytest tests/test_walkforward.py::TestEntryTiming -v`
Expected: PASS（3件）

- [ ] **確認2: degraded 実行が識別できる**

Run: `pytest tests/test_walkforward.py::TestDegraded -v` と `pytest tests/test_walkforward.py::TestSaveRun::test_degraded_flag_is_persisted -v`
Expected: PASS

- [ ] **確認3: 実行内のモデル使用履歴から週次再学習を辿れる**

Run: `pytest tests/test_walkforward.py::TestWeeklyRetrain -v` と `pytest tests/test_walkforward.py::TestSaveRun::test_model_usage_rows_are_linked_to_the_run -v`
Expected: PASS

- [ ] **確認4: 資金競合とセクター上限が効き、除外理由が残る**（前半で実装済み）

Run: `pytest tests/test_portfolio.py::TestAllocate -v` と `pytest tests/test_walkforward.py::TestCapitalCompetition -v`
Expected: PASS

- [ ] **確認5: legacy で paper の挙動が変わらない**

Run: `pytest tests/test_paper_execution_v2.py::TestLegacyBehaviourUnchanged -v`
Expected: PASS（5件）

- [ ] **確認6: 旧エンジンが無改造である**

Run: `git log --oneline -- src/backtest/engine.py | head -3`
Expected: 段階A〜D のコミットが一件も出ないこと（`engine.py` に触っていない）

- [ ] **確認7: 既存経路に回帰が無い**

Run: `pytest tests/ -q`
Expected: 段階D後半の着手前と同じ結果（新規テスト59件ぶんだけ増える）

- [ ] **確認8: 実データで3戦略を1回比較する**（判断材料。合否ではない）

段階B後半のイベント表と段階Cの学習を結線し、3戦略を同じ市場データで1回ずつ流して次を記録する。

```python
for name in ("blend", "rule", "rule_ml"):
    res = wf.run_walkforward(md, start, end, ...)
    print(name, res.daily["nav"].iloc[-1], len(res.trades), res.degraded)
```

Expected: 3案の最終NAV・取引数・degraded が得られること。**この数値で採否を決めない。**
`degraded=True` の実行は比較から外す。売買回数が少なければ期間を延ばす（spec §14）。

---

## 次の段階

段階E（`docs/superpowers/specs/...` §9）は候補モデルの昇格と shadow 運用を扱う。

- 学習成功をモデル更新ではなく**候補の生成**にする（`models/candidates/<model_id>/`）
- 保存形式を pickle から LightGBM ネイティブ + JSONメタへ
- 昇格の契約（評価記録ID・判断者・理由・旧モデルID・切替日時。`degraded` / 未評価 / 特徴量定義の不一致は昇格不可）
- 既存モデルは `legacy` 用として保持し、**v2はモデル未昇格の状態から開始する**
- shadow運用（同じ入力に現行と候補の判断を並行記録。候補は発注に繋がない）
