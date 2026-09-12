# ML評価基盤 段階D前半（ポートフォリオと執行の土台）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 複数銘柄の現金・保有・資金競合・セクター上限を、実運用と同じ式で扱う純粋な状態機械を作る。あわせて未約定・部分約定・出来高制約を執行アダプタへ足す。

**Architecture:** `portfolio.py` は現金と保有だけを持つ純粋な状態機械で、DB にもネットワークにも触らない。判定式は `src/risk/manager.py` の実運用実装をそのまま写し取る（式が2つあると、バックテストで通った条件が運用で弾かれる）。`execution.py` には「約定しなかった」を表現する手段を足す。日次ループそのものは段階D後半の担当で、本計画では実装しない。

**Tech Stack:** Python 3.11 / pandas 2.1.4 / numpy 1.26.2 / pytest / dataclasses

**Spec:** `docs/superpowers/specs/2026-09-10-ml-evaluation-foundation-design.md`（§4・§8・§12・§14）

**前提:** 段階B前半（`docs/superpowers/plans/2026-09-11-ml-evaluation-stage-b1.md`）が完了していること。本計画は `execution.Fill` / `CostConfig` / `buy_fill_price` / `sell_fill_price` / `entry_fill` / `exit_fill` と `policy.Observation` / `ExitIntent` に依存する。

## Global Constraints

- 日時は **JST naive**。現在時刻は `src/core/clock.now()` / `clock.today()` を使い、`datetime.now()` を直接呼ばない。
- **`portfolio.py` は DB・ネットワーク・設定ファイルに触らない。** 純粋な状態機械にする。設定は引数で受け取る。
- **判定式は `src/risk/manager.py` から写し取る。** 1銘柄上限・既存保有の差し引き・最大保有銘柄数・セクター集中率は、実運用と同じ式であることをテストで固定する。式が分かれるとバックテストで通った条件が運用で弾かれる。
- **既存の公開関数の挙動を変えない。** `src/risk/manager.py`・`src/backtest/engine.py`・`src/services/trading.py`・`src/strategy/ml_model.py` は本計画で一切変更しない。
- 金額の比較は「単元（100株）未満は買えない」を常に守る。
- ファイルは UTF-8 **BOM無し**・LF で保存する。確認は `git show <rev>:<path>` でコミット済みblobに対して行う。
- テストは `pytest tests/<file>.py -v` で実行する。ネットワークへ出るテストを書かない。
- コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` を付ける。実装者自身のモデル名を書かない。

---

## File Structure

| ファイル | 責務 |
|---|---|
| `src/backtest/portfolio.py`（新規） | 現金・保有・予約の状態機械、1銘柄上限、最大保有銘柄数、セクター集中、資金競合の解決 |
| `src/backtest/execution.py`（改修） | 未約定・部分約定・出来高制約を追加（段階B前半の関数は挙動を変えない） |
| `tests/test_portfolio.py`（新規） | 状態遷移、実運用式との一致、資金競合 |
| `tests/test_execution_limits.py`（新規） | 未約定・部分約定・出来高制約 |

### 段階D後半（本計画のスコープ外）

日次5フェーズループ、期間中の週次再学習、実行条件のスナップショット、`degraded` の伝播、戦略3案の比較、paper経路の是正は `walkforward.py` として別計画で行う。

---

## Task 1: ポートフォリオの状態と評価額

**Files:**
- Create: `src/backtest/portfolio.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `LOT_SIZE: int`（= 100）
  - `Holding`（frozen dataclass）: `symbol: str`, `quantity: int`, `avg_cost: float`, `sector: str`, `entry_at: date`, `peak_price: float`, `sessions_held: int`
  - `Portfolio`（frozen dataclass）: `cash: float`, `holdings: dict[str, Holding]`, `reserved: float`
  - `empty_portfolio(cash: float) -> Portfolio`
  - `holdings_value(pf: Portfolio, prices: dict) -> float`
  - `nav(pf: Portfolio, prices: dict) -> float`
  - `held_value(pf: Portfolio, symbol: str, price: float) -> float`

**注意:** 価格が取れない銘柄は `avg_cost`（取得平均単価）で代用する。`src/risk/manager.py:518` の `closes.get(p.symbol) or p.avg_cost` と同じ規約。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_portfolio.py` を新規作成する。

```python
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
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_portfolio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backtest.portfolio'`

- [ ] **Step 3: 実装を書く**

`src/backtest/portfolio.py` を新規作成する。

```python
"""ポートフォリオの状態機械 — 現金と保有だけを持つ。

**DB・ネットワーク・設定ファイルに触らない純粋な状態機械にする。** 設定は
引数で受け取る。こうしておくと、資金競合や上限判定を実データなしで
決定的にテストできる。

判定式（1銘柄上限・既存保有の差し引き・最大保有銘柄数・セクター集中率）は
src/risk/manager.py の実運用実装をそのまま写し取る。式が2つあると、
バックテストで通った条件が運用で弾かれる。

現行 src/backtest/engine.py は単一銘柄の現金と数量しか持たず、複数銘柄の
資金競合を扱えない（レビュー Backtest）。
"""
from dataclasses import dataclass, replace
from datetime import date
from typing import Optional

# 単元株数。src/risk/manager.py:26 の LOT_SIZE と同じ値であること。
LOT_SIZE = 100


@dataclass(frozen=True)
class Holding:
    """1銘柄の保有。

    peak_price と sessions_held は退出ポリシー（src/strategy/policy.py の
    HoldingState）が必要とする状態で、実運用も Position.peak_price として
    永続化している。
    """
    symbol: str
    quantity: int
    avg_cost: float
    sector: str
    entry_at: date
    peak_price: float
    sessions_held: int


@dataclass(frozen=True)
class Portfolio:
    """現金・保有・未約定買いの引当。

    reserved は「発注済みだがまだ約定していない買いの金額」。現金から差し引いた
    実効余力で新規の枠を決めるために持つ（実運用 position_budget と同じ考え方）。
    """
    cash: float
    holdings: dict
    reserved: float


def empty_portfolio(cash: float) -> Portfolio:
    return Portfolio(cash=cash, holdings={}, reserved=0.0)


def _price_of(h: Holding, prices: dict) -> float:
    """評価に使う価格。取れない銘柄は取得単価で代用する。

    src/risk/manager.py:518 の `closes.get(p.symbol) or p.avg_cost` と同じ規約。
    """
    return prices.get(h.symbol) or h.avg_cost


def holdings_value(pf: Portfolio, prices: dict) -> float:
    """建玉の時価評価額の合計。"""
    return sum(h.quantity * _price_of(h, prices) for h in pf.holdings.values())


def nav(pf: Portfolio, prices: dict) -> float:
    """総資産（現金＋建玉評価額）。

    reserved は現金の内訳であって総資産を減らさないため引かない。
    """
    return pf.cash + holdings_value(pf, prices)


def held_value(pf: Portfolio, symbol: str, price: float) -> float:
    """この銘柄の現在の保有評価額（現在値ベース）。

    簿価ではなく現在値で評価する。簿価だと値上がり銘柄の実質的な集中度を
    過小評価し、値下がり銘柄では過大評価する（実運用 _held_value と同じ）。
    """
    h = pf.holdings.get(symbol)
    return h.quantity * price if h else 0.0
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_portfolio.py -v`
Expected: PASS（8件）

- [ ] **Step 5: BOM確認とコミット**

Run: `head -c 3 src/backtest/portfolio.py | xxd`（`2222 22` を確認。`efbb bf` なら下記で除去）

```python
for p in ["src/backtest/portfolio.py", "tests/test_portfolio.py"]:
    with open(p, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        with open(p, "wb") as f:
            f.write(data[3:])
```

```bash
git add src/backtest/portfolio.py tests/test_portfolio.py
git commit -m "$(cat <<'EOF'
feat(backtest): ポートフォリオの状態と評価額を追加

現行エンジンは単一銘柄の現金と数量しか持たず、複数銘柄の資金競合を
扱えなかった。現金・保有・予約を持つ純粋な状態機械を置く。
価格が取れない銘柄は取得単価で代用し、保有評価は簿価ではなく現在値で
行う（いずれも実運用 risk/manager.py と同じ規約）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: 売買による状態遷移

**Files:**
- Modify: `src/backtest/portfolio.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- Consumes: Task 1
- Produces:
  - `apply_buy(pf: Portfolio, symbol: str, quantity: int, price: float, sector: str, at: date, commission_pct: float) -> Portfolio`
  - `apply_sell(pf: Portfolio, symbol: str, quantity: int, price: float, commission_pct: float) -> tuple[Portfolio, float]` — `(次の状態, 実現損益)`
  - `advance_session(pf: Portfolio, prices: dict) -> Portfolio` — 保有のピーク更新と経過営業日数の加算

**注意:** 手数料は約定代金に対して掛かる（買いは現金をさらに減らし、売りは受取を減らす）。スリッページは `execution.py` の約定価格に織り込み済みなので、ここでは扱わない。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_portfolio.py` の末尾に追記する。

```python
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
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_portfolio.py -v`
Expected: FAIL — `AttributeError: module 'src.backtest.portfolio' has no attribute 'apply_buy'`

- [ ] **Step 3: 実装を追加**

`src/backtest/portfolio.py` の末尾に追加する。

```python
def apply_buy(pf: Portfolio, symbol: str, quantity: int, price: float,
              sector: str, at: date, commission_pct: float) -> Portfolio:
    """買い約定を反映する。

    手数料は約定代金に対して掛かる。スリッページは execution.py の約定価格に
    織り込み済みなのでここでは扱わない（二重に引かないため）。
    買い増しの場合は取得単価を加重平均し、**保有開始日は動かさない**
    （保有期間の数え方を保つため）。
    """
    if quantity <= 0:
        raise ValueError(f"quantity は正の整数: {quantity}")
    amount = price * quantity
    cash = pf.cash - amount - amount * commission_pct

    existing = pf.holdings.get(symbol)
    if existing is None:
        holding = Holding(
            symbol=symbol, quantity=quantity, avg_cost=price, sector=sector,
            entry_at=at, peak_price=price, sessions_held=0,
        )
    else:
        total_qty = existing.quantity + quantity
        avg_cost = (existing.avg_cost * existing.quantity + amount) / total_qty
        holding = replace(existing, quantity=total_qty, avg_cost=avg_cost)

    holdings = dict(pf.holdings)
    holdings[symbol] = holding
    return replace(pf, cash=cash, holdings=holdings)


def apply_sell(pf: Portfolio, symbol: str, quantity: int, price: float,
               commission_pct: float) -> tuple:
    """売り約定を反映し、(次の状態, 実現損益) を返す。

    実現損益は手数料控除後。部分決済では残りの取得単価を変えない。
    """
    existing = pf.holdings.get(symbol)
    if existing is None or existing.quantity < quantity:
        held = existing.quantity if existing else 0
        raise ValueError(f"保有数量が足りません: {symbol} 保有{held}株 < 売却{quantity}株")

    proceeds = price * quantity
    commission = proceeds * commission_pct
    cash = pf.cash + proceeds - commission
    realized = (price - existing.avg_cost) * quantity - commission

    holdings = dict(pf.holdings)
    remaining = existing.quantity - quantity
    if remaining > 0:
        holdings[symbol] = replace(existing, quantity=remaining)
    else:
        del holdings[symbol]
    return replace(pf, cash=cash, holdings=holdings), realized


def advance_session(pf: Portfolio, prices: dict) -> Portfolio:
    """1営業日ぶん保有を進める（ピーク更新と経過営業日数の加算）。

    退出ポリシーが「当日の基準線は前営業日終了時点のピークで固定する」
    規約を持つため、ピークの反映はその日の判定が終わった後に行う
    （src/strategy/policy.py の step() と同じ順序）。
    価格が取れない銘柄のピークは動かさない。
    """
    holdings = {}
    for symbol, h in pf.holdings.items():
        price = prices.get(symbol)
        peak = max(h.peak_price, price) if price else h.peak_price
        holdings[symbol] = replace(h, peak_price=peak,
                                   sessions_held=h.sessions_held + 1)
    return replace(pf, holdings=holdings)
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_portfolio.py -v`
Expected: PASS（24件）

- [ ] **Step 5: コミット**

```bash
git add src/backtest/portfolio.py tests/test_portfolio.py
git commit -m "$(cat <<'EOF'
feat(backtest): 売買によるポートフォリオの状態遷移を追加

手数料は約定代金に掛け、スリッページはexecution.pyの約定価格に
織り込み済みなのでここでは扱わない（二重に引かない）。
買い増しは取得単価を加重平均し保有開始日は動かさない。
ピークの反映は当日の判定が終わった後に行う（policy.step()と同じ順序で、
未来のピークを遡ってストップに使わないため）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 1銘柄あたりの上限と購入可能数量

**Files:**
- Modify: `src/backtest/portfolio.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- Consumes: Task 1・2
- Produces:
  - `SizingConfig`（frozen dataclass）: `max_position_ratio: float`, `max_positions: int`, `max_sector_ratio: float`
  - `position_budget(pf: Portfolio, conf: SizingConfig) -> float`
  - `calc_quantity(pf: Portfolio, symbol: str, price: float, conf: SizingConfig) -> int`

**写し取る実運用の式（`src/risk/manager.py:382-439`）:**

```
position_budget = (cash - reserved) × max_position_ratio      # 残余力に対する比率
remaining       = max(0, position_budget − held_value)         # 既存保有分を差し引く
quantity        = int(remaining / (price × LOT_SIZE)) × LOT_SIZE
```

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_portfolio.py` の末尾に追記する。

```python
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
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_portfolio.py -v`
Expected: FAIL — `AttributeError: module 'src.backtest.portfolio' has no attribute 'SizingConfig'`

- [ ] **Step 3: 実装を追加**

`src/backtest/portfolio.py` の末尾に追加する。

```python
@dataclass(frozen=True)
class SizingConfig:
    """資金配分の制限。実運用は trading 節（risk_profile で上書きされる）から来る。"""
    max_position_ratio: float   # 1銘柄あたり最大資金比率
    max_positions: int          # 最大同時保有銘柄数
    max_sector_ratio: float     # 同一セクター最大集中率


def position_budget(pf: Portfolio, conf: SizingConfig) -> float:
    """1銘柄に投じられる上限額。

    src/risk/manager.py:382-396 と同じ式。「**残余力に対する比率**」であり
    総資産に対する比率ではない。未約定買いの引当を差し引いた実効余力に
    比率を掛ける（未約定中の多重発注で余力を二重に使う事故を防ぐ）。
    """
    available = max(0.0, pf.cash - pf.reserved)
    return available * conf.max_position_ratio


def calc_quantity(pf: Portfolio, symbol: str, price: float,
                  conf: SizingConfig) -> int:
    """購入株数（単元切り捨て）。

    src/risk/manager.py:418-439 と同じ式。1銘柄あたりの上限から**既存保有の
    評価額を差し引いた残り枠**まで買える。差し引かないと、上限いっぱい
    保有した銘柄へさらに満額買い増そうとしてしまう。
    """
    if price <= 0:
        return 0
    budget = position_budget(pf, conf)
    remaining = max(0.0, budget - held_value(pf, symbol, price))
    units = int(remaining / (price * LOT_SIZE))
    return units * LOT_SIZE
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_portfolio.py -v`
Expected: PASS（34件）

- [ ] **Step 5: コミット**

```bash
git add src/backtest/portfolio.py tests/test_portfolio.py
git commit -m "$(cat <<'EOF'
feat(backtest): 1銘柄上限と購入可能数量を実運用と同じ式で追加

残余力（現金−未約定引当）に比率を掛けた枠から既存保有の評価額を
差し引いた残りまで買える。単元未満は切り捨てる。
実運用 risk/manager.py の式と一致することをテストで固定した。
式が2つあるとバックテストで通った条件が運用で弾かれるため。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: 最大保有銘柄数とセクター集中率

**Files:**
- Modify: `src/backtest/portfolio.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- Consumes: Task 1〜3
- Produces:
  - `check_max_positions(pf: Portfolio, candidate: Optional[str], conf: SizingConfig) -> tuple[bool, str]`
  - `check_sector_concentration(pf: Portfolio, sector: str, candidate_notional: float, prices: dict, conf: SizingConfig) -> tuple[bool, str]`

**写し取る実運用の式（`src/risk/manager.py:441-535`）:**

- 最大保有銘柄数は**銘柄の集合**で判定する（保有＋候補）。候補が既に保有中なら集合は増えないので通る
- セクター集中率の分子は「同一セクターの建玉評価額＋これから出す注文の金額」
- 分母は**総資金（現金＋建玉評価額）**。投資済み額を分母にすると、保有ゼロのとき比率が常に100%になり最初の1銘柄を永久に買えない
- 比率が上限**以上**なら却下（`>=`）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_portfolio.py` の末尾に追記する。

```python
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
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_portfolio.py -v`
Expected: FAIL — `AttributeError: module 'src.backtest.portfolio' has no attribute 'check_max_positions'`

- [ ] **Step 3: 実装を追加**

`src/backtest/portfolio.py` の末尾に追加する。

```python
def check_max_positions(pf: Portfolio, candidate: Optional[str],
                        conf: SizingConfig) -> tuple:
    """最大保有銘柄数チェック。(通るか, 理由) を返す。

    src/risk/manager.py:441-474 と同じ考え方で、**銘柄の集合**で判定する。
    候補が既に保有中なら集合は増えないので通る。
    """
    after = set(pf.holdings)
    if candidate:
        after = after | {candidate}
    if len(after) > conf.max_positions:
        return False, f"最大保有銘柄数({conf.max_positions})に達しています"
    return True, ""


def check_sector_concentration(pf: Portfolio, sector: str,
                               candidate_notional: float, prices: dict,
                               conf: SizingConfig) -> tuple:
    """同一セクターの集中投資チェック。(通るか, 理由) を返す。

    src/risk/manager.py:477-535 と同じ式。

    分子は「同一セクターの建玉評価額 ＋ これから出す注文の金額」。
    分母は**総資金（現金＋建玉評価額）**。投資済み額を分母にすると、保有が
    少ないほど必ず超過し、**保有ゼロでは比率が常に100%になって最初の1銘柄を
    永久に買えない**（2026-09-04に実際に起きた）。買付余力は建玉に変わるだけで
    総資金は増減しないため、分母に候補金額を足すと二重計上になる。

    比率が上限**以上**なら却下する（実運用と同じく >= で判定）。
    """
    positions_value = 0.0
    same_sector_value = candidate_notional
    for h in pf.holdings.values():
        value = h.quantity * _price_of(h, prices)
        positions_value += value
        if h.sector == sector:
            same_sector_value += value

    total_value = pf.cash + positions_value
    if total_value <= 0:
        return True, ""

    ratio = same_sector_value / total_value
    if ratio >= conf.max_sector_ratio:
        return False, (
            f"セクター集中率が上限({conf.max_sector_ratio:.0%})超: {sector} "
            f"（{same_sector_value:,.0f}円 / 総資金{total_value:,.0f}円 = {ratio:.0%}）"
        )
    return True, ""
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_portfolio.py -v`
Expected: PASS（46件）

- [ ] **Step 5: コミット**

```bash
git add src/backtest/portfolio.py tests/test_portfolio.py
git commit -m "$(cat <<'EOF'
feat(backtest): 最大保有銘柄数とセクター集中率を実運用と同じ式で追加

保有数は銘柄の集合で判定し、既保有への買い増しは集合を増やさない。
セクター集中率の分母は総資金（現金＋建玉）にする。投資済み額を分母に
すると保有ゼロで比率が常に100%になり最初の1銘柄を永久に買えない
（2026-09-04の実害）。上限ちょうどは却下（実運用と同じ>=判定）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 同日に複数候補が出たときの資金競合

**Files:**
- Modify: `src/backtest/portfolio.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- Consumes: Task 1〜4
- Produces:
  - `Candidate`（frozen dataclass）: `symbol: str`, `sector: str`, `price: float`, `score: float`
  - `PlannedOrder`（frozen dataclass）: `symbol: str`, `sector: str`, `price: float`, `quantity: int`, `notional: float`
  - `RejectedCandidate`（frozen dataclass）: `symbol: str`, `reason: str`
  - `allocate(pf: Portfolio, candidates: list, conf: SizingConfig, prices: dict) -> tuple[list, list]` — `(採用した注文, 除外した候補と理由)`

**背景（spec §8）:** 現行 `engine.py` は単一銘柄しか扱わないため、同じ日に複数の候補が出たときにどれを採るかという問題が存在しない。実運用では発注のたびに現金が減り、2件目以降の枠は自動的に小さくなる。**スコアの高い順に、現金・保有数・セクター上限を1件ずつ確認しながら確定させる。**

**除外理由は必ず記録する。** 「なぜこの候補を採らなかったか」が残らないと、成績が資金制約によるものか戦略によるものか判別できない。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_portfolio.py` の末尾に追記する。

```python
def _candidate(symbol="7203", sector="自動車", price=1000.0, score=0.5):
    return pf.Candidate(symbol=symbol, sector=sector, price=price, score=score)


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
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_portfolio.py -v`
Expected: FAIL — `AttributeError: module 'src.backtest.portfolio' has no attribute 'Candidate'`

- [ ] **Step 3: 実装を追加**

`src/backtest/portfolio.py` の末尾に追加する。

```python
@dataclass(frozen=True)
class Candidate:
    """その日の買い候補。"""
    symbol: str
    sector: str
    price: float
    score: float


@dataclass(frozen=True)
class PlannedOrder:
    """資金競合を解決したあとの、実際に出す注文。"""
    symbol: str
    sector: str
    price: float
    quantity: int
    notional: float


@dataclass(frozen=True)
class RejectedCandidate:
    """採用しなかった候補と、その理由。

    理由を残さないと、成績が資金制約によるものか戦略によるものか
    判別できない（spec §8）。
    """
    symbol: str
    reason: str


def allocate(pf: Portfolio, candidates: list, conf: SizingConfig,
             prices: dict) -> tuple:
    """同じ日の複数候補に資金を割り当てる。(採用した注文, 除外した候補) を返す。

    **スコアの高い順に1件ずつ確定させる。** 実運用では発注のたびに現金が減り、
    2件目以降の枠は自動的に小さくなる（position_budget が残余力ベースのため）。
    現行 engine.py は単一銘柄しか扱わずこの競合が存在しなかった。

    引数のポートフォリオは変更しない（割り当ての試算用に複製して進める）。
    採用されなかった候補には必ず理由を付けて返す。
    """
    working = pf
    orders: list = []
    rejected: list = []

    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        ok, reason = check_max_positions(working, cand.symbol, conf)
        if not ok:
            rejected.append(RejectedCandidate(symbol=cand.symbol, reason=reason))
            continue

        quantity = calc_quantity(working, cand.symbol, cand.price, conf)
        if quantity < LOT_SIZE:
            budget = position_budget(working, conf)
            rejected.append(RejectedCandidate(
                symbol=cand.symbol,
                reason=(f"単元({LOT_SIZE}株)の必要額 {cand.price * LOT_SIZE:,.0f}円 が"
                        f"1銘柄上限の残り枠 {budget:,.0f}円 を超過"),
            ))
            continue

        notional = cand.price * quantity
        ok, reason = check_sector_concentration(
            working, cand.sector, notional, prices, conf)
        if not ok:
            rejected.append(RejectedCandidate(symbol=cand.symbol, reason=reason))
            continue

        orders.append(PlannedOrder(
            symbol=cand.symbol, sector=cand.sector, price=cand.price,
            quantity=quantity, notional=notional,
        ))
        # 次の候補の枠を正しく縮めるため、確定したぶんを反映して進める
        working = apply_buy(working, cand.symbol, quantity, cand.price,
                            cand.sector, date.min, commission_pct=0.0)

    return orders, rejected
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_portfolio.py -v`
Expected: PASS（57件）

- [ ] **Step 5: 全体回帰を確認してコミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

```bash
git add src/backtest/portfolio.py tests/test_portfolio.py
git commit -m "$(cat <<'EOF'
feat(backtest): 同日に複数候補が出たときの資金競合を追加

スコアの高い順に1件ずつ確定させ、発注のたびに現金を減らして次の候補の
枠を縮める（実運用のposition_budgetが残余力ベースであるのと同じ挙動）。
現行エンジンは単一銘柄しか扱わずこの競合が存在しなかった。
採用しなかった候補には必ず理由を残す。理由が無いと成績が資金制約による
ものか戦略によるものか判別できない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: 未約定・部分約定・出来高制約

**Files:**
- Modify: `src/backtest/execution.py`
- Test: `tests/test_execution_limits.py`

**Interfaces:**
- Consumes: 段階B前半の `Fill` / `CostConfig` / `buy_fill_price` / `sell_fill_price`、`policy.Observation`
- Produces:
  - `FillResult`（frozen dataclass）: `fill: Optional[Fill]`, `requested_quantity: int`, `filled_quantity: int`, `unfilled_reason: Optional[str]`
  - `LiquidityConfig`（frozen dataclass）: `max_volume_share: float` — その日の出来高に対して約定できる最大の割合（0で無制限）
  - `entry_fill_limited(next_bar, quantity, costs, liquidity) -> FillResult`

**背景（spec §8）:** 現行エンジンは「欲しい数量は必ず買える」前提である。薄商い銘柄では、その日の出来高の何割も自分で買うことはできない。**未約定と部分約定を明示的に表現する**ことで、成績が執行可能性を織り込んだものになる。

**注意:** 段階B前半の `entry_fill` / `exit_fill` は**挙動を変えない**。新しい関数を追加する（段階B後半の `dataset.py` が既存の挙動に依存しているため）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_execution_limits.py` を新規作成する。

```python
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
        res = execution.entry_fill_limited(
            _bar(), 1000, _costs(), _liquidity(share=0.0), volume=100)
        assert res.filled_quantity == 1000
        assert res.fill is not None
        assert res.unfilled_reason is None


class TestVolumeShareLimit:
    def test_fills_fully_within_the_share(self):
        res = execution.entry_fill_limited(
            _bar(), 100, _costs(), _liquidity(share=0.1), volume=10_000)
        assert res.filled_quantity == 100
        assert res.unfilled_reason is None

    def test_partial_fill_when_request_exceeds_the_share(self):
        """出来高の10%までしか約定できないなら、その分だけ約定する"""
        res = execution.entry_fill_limited(
            _bar(), 5000, _costs(), _liquidity(share=0.1), volume=10_000)
        assert res.requested_quantity == 5000
        assert res.filled_quantity == 1000   # 10,000 × 0.1
        assert res.fill is not None
        assert res.fill.quantity == 1000
        assert res.unfilled_reason is not None
        assert "出来高" in res.unfilled_reason

    def test_partial_fill_rounds_down_to_lot_size(self):
        """部分約定も単元単位に切り捨てる"""
        res = execution.entry_fill_limited(
            _bar(), 5000, _costs(), _liquidity(share=0.1), volume=1_050)
        assert res.filled_quantity == 100   # 1,050 × 0.1 = 105 → 単元切り捨てで100

    def test_unfilled_when_share_is_below_one_lot(self):
        """1単元にも満たなければ未約定"""
        res = execution.entry_fill_limited(
            _bar(), 100, _costs(), _liquidity(share=0.1), volume=500)
        assert res.filled_quantity == 0
        assert res.fill is None
        assert "単元" in res.unfilled_reason

    def test_unfilled_when_volume_is_zero(self):
        """出来高ゼロ（売買停止等）では約定しない"""
        res = execution.entry_fill_limited(
            _bar(), 100, _costs(), _liquidity(share=0.1), volume=0)
        assert res.filled_quantity == 0
        assert res.fill is None
        assert res.unfilled_reason is not None


class TestFillPriceIsUnchanged:
    def test_uses_the_same_price_as_the_base_helper(self):
        """約定価格の規約は段階B前半と同じ（寄り × (1+slip)）"""
        bar = _bar(o=1020.0)
        costs = _costs(slip=0.001)
        res = execution.entry_fill_limited(bar, 100, costs, _liquidity(), volume=0)
        base = execution.entry_fill(bar, 100, costs)
        assert res.fill.price == pytest.approx(base.price)

    def test_fills_at_next_session_open(self):
        bar = _bar(session=date(2026, 9, 3), o=1020.0)
        res = execution.entry_fill_limited(bar, 100, _costs(), _liquidity(), volume=0)
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
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_execution_limits.py -v`
Expected: FAIL — `AttributeError: module 'src.backtest.execution' has no attribute 'LiquidityConfig'`

- [ ] **Step 3: 実装を追加**

`src/backtest/execution.py` の末尾に追加する。import に `from src.backtest.portfolio import LOT_SIZE` を足す。

```python
@dataclass(frozen=True)
class LiquidityConfig:
    """執行できる量の上限。

    max_volume_share は「その日の出来高に対して自分が占めてよい割合」。
    0 で無制限（従来どおり欲しい数量は必ず買える前提）。
    """
    max_volume_share: float = 0.0


@dataclass(frozen=True)
class FillResult:
    """約定の結果。**未約定と部分約定を明示的に表現する。**

    現行エンジンは「欲しい数量は必ず買える」前提で、薄商い銘柄の執行可能性を
    織り込んでいなかった（spec §8）。filled_quantity < requested_quantity なら
    部分約定、fill が None なら未約定。
    """
    fill: Optional[Fill]
    requested_quantity: int
    filled_quantity: int
    unfilled_reason: Optional[str]


def entry_fill_limited(next_bar: Observation, quantity: int, costs: CostConfig,
                       liquidity: LiquidityConfig, *, volume: int) -> FillResult:
    """出来高の制約を織り込んでエントリーを約定させる。

    約定価格の規約は entry_fill() と同じ（T+1の寄り × (1+slip)）。違うのは
    「いくつ約定できたか」だけ。段階B前半の entry_fill() は挙動を変えない
    （段階B後半の dataset.py がその挙動に依存しているため）。

    部分約定も単元単位に切り捨てる。1単元にも満たなければ未約定とする。
    """
    if liquidity.max_volume_share <= 0:
        return FillResult(
            fill=entry_fill(next_bar, quantity, costs),
            requested_quantity=quantity,
            filled_quantity=quantity,
            unfilled_reason=None,
        )

    allowed = int(volume * liquidity.max_volume_share)
    allowed_lots = (allowed // LOT_SIZE) * LOT_SIZE
    fillable = min(quantity, allowed_lots)

    if fillable < LOT_SIZE:
        return FillResult(
            fill=None,
            requested_quantity=quantity,
            filled_quantity=0,
            unfilled_reason=(
                f"出来高{volume:,}株の{liquidity.max_volume_share:.0%}では"
                f"単元({LOT_SIZE}株)に満たないため約定できません"
            ),
        )

    reason = None
    if fillable < quantity:
        reason = (
            f"出来高{volume:,}株の{liquidity.max_volume_share:.0%}まで"
            f"（{quantity:,}株のうち{fillable:,}株を約定）"
        )
    return FillResult(
        fill=entry_fill(next_bar, fillable, costs),
        requested_quantity=quantity,
        filled_quantity=fillable,
        unfilled_reason=reason,
    )
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_execution_limits.py -v`
Expected: PASS（11件）

- [ ] **Step 5: 段階B前半のテストが壊れていないことを確認**

Run: `pytest tests/test_execution.py -v`
Expected: PASS（段階B前半の14件が全て通る）

- [ ] **Step 6: 全体回帰とBOM確認、コミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

Run: `head -c 3 src/backtest/execution.py | xxd`（`2222 22` を確認）

```bash
git add src/backtest/execution.py tests/test_execution_limits.py
git commit -m "$(cat <<'EOF'
feat(backtest): 未約定・部分約定・出来高制約を追加

現行エンジンは「欲しい数量は必ず買える」前提で、薄商い銘柄の執行
可能性を織り込んでいなかった。出来高に対する自分の占有率で上限を設け、
部分約定と未約定を明示的に表現する。部分約定も単元単位に切り捨てる。
段階B前半のentry_fill/exit_fillは挙動を変えない（dataset.pyが依存）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## 段階D前半 完了条件の確認

- [ ] **確認1: 資金・上限の判定式が実運用と一致している**

Run: `pytest tests/test_portfolio.py -k matches_production -v`
Expected: PASS（3件：`position_budget` / `calc_quantity` / `check_sector_concentration`）

- [ ] **確認2: 保有ゼロから最初の1銘柄を買える**

Run: `pytest tests/test_portfolio.py::TestCheckSectorConcentration::test_allows_the_first_purchase_from_empty -v`
Expected: PASS

- [ ] **確認3: 同日の複数候補で資金が競合し、除外理由が残る**

Run: `pytest tests/test_portfolio.py::TestAllocate -v`
Expected: PASS（11件）

- [ ] **確認4: 未約定・部分約定を表現できる**

Run: `pytest tests/test_execution_limits.py -v`
Expected: PASS（11件）

- [ ] **確認5: 段階B前半の執行アダプタの挙動が変わっていない**

Run: `pytest tests/test_execution.py tests/test_policy.py -v`
Expected: PASS

- [ ] **確認6: 既存経路に回帰が無い**

Run: `pytest tests/ -q`
Expected: 段階D前半の着手前と同じ結果（新規テスト68件ぶんだけ増える）

---

## 次の段階

段階D後半は `src/backtest/walkforward.py` として日次5フェーズループを実装する。本計画が確定させた `Portfolio` / `allocate` / `FillResult` と、段階B前半の `policy.step()` / `exit_fill`、段階C の `training_inputs()` を組み合わせる。

- 日次ループ（①前日までに決まった注文の執行 → ②保有と現金の更新 → ③退出ポリシーの逐次駆動 → ④日末NAV記録 → ⑤翌日の候補生成）
- 期間中の週次再学習に段階Cと同じ締切を適用する（spec §7 経路4の残り）
- 実行条件のスナップショット（`BacktestRun` への列追加と `RunModelUsage` テーブル）
- 推論例外での `degraded` 伝播
- 戦略3案の比較（加重合成 / ルール単独 / ルール候補＋ML順位付け）
- paper経路の是正（`stop_loss_check` の終値判定、スキャン中の終値即時売買、注文の繰越）
