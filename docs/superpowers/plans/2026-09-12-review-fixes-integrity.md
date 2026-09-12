# レビュー是正 F10・F11・F15（価格の鮮度・設定の原子性・DB整合）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 価格に鮮度と由来を持たせ、設定の適用と保存を原子的にし、DBの参照整合を検査できるようにする。

**Architecture:** 価格は「値」ではなく「いつ・どこから得た値か」を伴う型で返す。設定の更新・取り込み・切替は1つの適用関数に集約し、書き込みは一時ファイル経由で原子的に行う。DBは既存データを壊さずに制約と来歴を段階的に足す。

**Tech Stack:** Python 3.11 / SQLAlchemy 2.0.23 / SQLite / pytest

**Spec:** `docs/kabu-auto-detailed-review_20260910.md`（F10・F11・F15、§7）

**現状確認（2026-09-12時点で3件とも現存）:**

| ID | 現存箇所 |
|---|---|
| F10 | `src/risk/manager.py:112` `get_current_prices` は `dict[str, float]` を返し、リアルタイム値かDB終値かを呼び出し側が区別できない |
| F11 | `src/core/risk_profile.py` の `import_profile` は `_apply()` を呼ばず保存だけする。`_persist()` は一時ファイルを介さない直接書き込み |
| F15 | `src/data/database.py` に `ForeignKey` 宣言も `PRAGMA foreign_keys` の有効化も無い |

**既に解消済みで本計画の対象外:**

- F10 の前半（含み損益がDB終値のみで計算されていた件）は `c2a0630` で修正済み。リアルタイム価格が優先される
- F15 のモデル来歴部分は段階E（`model_store.ModelMeta`）で解消済み

## Global Constraints

- **本番DBを壊さない。** 制約の追加は既存データの不整合を**検出するだけ**に留め、削除・書き換えを自動で行わない。
- **既存の公開関数の呼び出し側を壊さない。** `get_current_prices` は戻り値の型を変えず、鮮度付きの取得は別関数として足す。
- 日時は **JST naive**。現在時刻は `src/core/clock.now()` / `clock.today()` を使い、`datetime.now()` を直接呼ばない。
- **設定ファイルへの書き込みは原子的に行う。** 一時ファイルへ書いてから `os.replace`。途中で落ちても前の内容が残る。
- ファイルは UTF-8 **BOM無し**・LF で保存する。確認は `git show <rev>:<path>` でコミット済みblobに対して行う。
- テストは `pytest tests/<file>.py -v` で実行する。ネットワークへ出るテストを書かない。
- コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` を付ける。実装者自身のモデル名を書かない。

---

## File Structure

| ファイル | 責務 |
|---|---|
| `src/risk/price_quote.py`（新規） | 価格に時刻・由来・品質を持たせる型 |
| `src/risk/manager.py`（改修） | 鮮度付きの価格取得を追加（既存メソッドは戻り値の型を変えない） |
| `src/core/risk_profile.py`（改修） | 適用経路の統合と原子的な保存 |
| `src/data/database.py`（改修） | 外部キー宣言、接続ごとの `PRAGMA foreign_keys`、整合検査 |
| `tests/test_price_quote.py`（新規） | 鮮度・由来の判定 |
| `tests/test_profile_atomicity.py`（新規） | 適用経路の統合、保存の原子性 |
| `tests/test_db_integrity.py`（新規） | 外部キーの有効化、整合検査 |

---

## Task 1: 価格に鮮度と由来を持たせる

**Files:**
- Create: `src/risk/price_quote.py`
- Test: `tests/test_price_quote.py`

**Interfaces:**
- Consumes: `src/core/clock`
- Produces:
  - `SOURCE_REALTIME` / `SOURCE_DAILY_CLOSE` / `SOURCE_BOOK_VALUE` 定数
  - `QUALITY_FRESH` / `QUALITY_STALE` / `QUALITY_UNKNOWN` 定数
  - `PriceQuote`（frozen dataclass）: `symbol` / `value` / `observed_at` / `source` / `session` / `quality`
  - `classify_quality(quote, now, *, max_age_seconds) -> str`
  - `is_usable_for_new_risk(quote, now, *, max_age_seconds) -> bool`

**背景（F10）:** `get_current_prices()` は `dict[str, float]` を返すため、**リアルタイム板から取れた値なのか、DBの終値で代用した値なのかを呼び出し側が区別できない**。表示用途なら古い値でもよいが、新規リスクを取る判断（発注）では区別が要る。価格が不明な保有を「0損失」として安全扱いしない設計にする。

**方針:** 既存の `get_current_prices()` は**戻り値の型を変えない**（呼び出し側が3箇所ある）。鮮度付きの取得は別メソッドとして足し、必要な判断だけがそちらを使う。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_price_quote.py` を新規作成する。

```python
"""価格の鮮度と由来（F10）のテスト

get_current_prices は dict[str, float] を返すため、リアルタイム板の値か
DB終値で代用した値かを呼び出し側が区別できない。表示なら古くてもよいが、
新規リスクを取る判断では区別が要る。
"""
from datetime import date, datetime, timedelta

import pytest

from src.risk import price_quote as pq


def _quote(value=1000.0, observed_at=None, source=None, session=None):
    return pq.PriceQuote(
        symbol="7203", value=value,
        observed_at=observed_at or datetime(2026, 9, 12, 10, 0, 0),
        source=source or pq.SOURCE_REALTIME,
        session=session or date(2026, 9, 12),
        quality=pq.QUALITY_UNKNOWN,
    )


class TestClassifyQuality:
    def test_recent_realtime_is_fresh(self):
        now = datetime(2026, 9, 12, 10, 0, 30)
        got = pq.classify_quality(_quote(), now, max_age_seconds=60)
        assert got == pq.QUALITY_FRESH

    def test_old_realtime_is_stale(self):
        now = datetime(2026, 9, 12, 10, 5, 0)
        got = pq.classify_quality(_quote(), now, max_age_seconds=60)
        assert got == pq.QUALITY_STALE

    def test_daily_close_is_never_fresh(self):
        """DB終値は「今の価格」ではない。取得直後でもfreshにしない"""
        now = datetime(2026, 9, 12, 10, 0, 1)
        got = pq.classify_quality(
            _quote(source=pq.SOURCE_DAILY_CLOSE), now, max_age_seconds=60)
        assert got == pq.QUALITY_STALE

    def test_book_value_is_unknown(self):
        """取得単価での代用は『価格が分からない』であって古い価格ではない"""
        now = datetime(2026, 9, 12, 10, 0, 1)
        got = pq.classify_quality(
            _quote(source=pq.SOURCE_BOOK_VALUE), now, max_age_seconds=60)
        assert got == pq.QUALITY_UNKNOWN

    def test_future_timestamp_is_not_fresh(self):
        """時計のずれで未来の時刻が来ても fresh と誤認しない"""
        now = datetime(2026, 9, 12, 9, 0, 0)
        got = pq.classify_quality(_quote(), now, max_age_seconds=60)
        assert got != pq.QUALITY_FRESH

    def test_boundary_is_inclusive(self):
        now = datetime(2026, 9, 12, 10, 1, 0)   # ちょうど60秒
        assert pq.classify_quality(_quote(), now, max_age_seconds=60) == pq.QUALITY_FRESH


class TestUsableForNewRisk:
    def test_fresh_realtime_is_usable(self):
        now = datetime(2026, 9, 12, 10, 0, 30)
        assert pq.is_usable_for_new_risk(_quote(), now, max_age_seconds=60) is True

    def test_stale_is_not_usable(self):
        now = datetime(2026, 9, 12, 10, 5, 0)
        assert pq.is_usable_for_new_risk(_quote(), now, max_age_seconds=60) is False

    def test_daily_close_is_not_usable(self):
        now = datetime(2026, 9, 12, 10, 0, 1)
        assert pq.is_usable_for_new_risk(
            _quote(source=pq.SOURCE_DAILY_CLOSE), now, max_age_seconds=60) is False

    def test_unknown_is_not_usable(self):
        """価格が不明な保有を『0損失』として安全扱いしない"""
        now = datetime(2026, 9, 12, 10, 0, 1)
        assert pq.is_usable_for_new_risk(
            _quote(source=pq.SOURCE_BOOK_VALUE), now, max_age_seconds=60) is False

    def test_zero_value_is_not_usable(self):
        now = datetime(2026, 9, 12, 10, 0, 1)
        assert pq.is_usable_for_new_risk(
            _quote(value=0.0), now, max_age_seconds=60) is False


class TestQuoteCarriesProvenance:
    def test_records_where_the_value_came_from(self):
        q = _quote(source=pq.SOURCE_DAILY_CLOSE, session=date(2026, 9, 11))
        assert q.source == pq.SOURCE_DAILY_CLOSE
        assert q.session == date(2026, 9, 11)

    def test_is_immutable(self):
        q = _quote()
        with pytest.raises(Exception):
            q.value = 2000.0
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_price_quote.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.risk.price_quote'`

- [ ] **Step 3: 実装を書く**

`src/risk/price_quote.py` を新規作成する。

```python
"""価格の鮮度と由来。

get_current_prices() は dict[str, float] を返すため、リアルタイム板から
取れた値なのか、DBの終値で代用した値なのかを呼び出し側が区別できない
（レビューF10）。表示用途なら古い値でもよいが、新規リスクを取る判断では
区別が要る。

**価格が不明な保有を「0損失」として安全扱いしない。** 分からないことは
分からないままにして、判断する側が扱いを決める。
"""
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

# 値の由来
SOURCE_REALTIME = "realtime"        # ブローカーの板
SOURCE_DAILY_CLOSE = "daily_close"  # DBの日足終値
SOURCE_BOOK_VALUE = "book_value"    # 取得単価での代用（=価格が分からない）

# 値の品質
QUALITY_FRESH = "fresh"       # 新規リスクの判断に使える
QUALITY_STALE = "stale"       # 古い。表示には使えるが新規判断には使わない
QUALITY_UNKNOWN = "unknown"   # 価格が分からない


@dataclass(frozen=True)
class PriceQuote:
    """価格と、それをいつ・どこから得たか。"""
    symbol: str
    value: float
    observed_at: datetime
    source: str
    session: Optional[date] = None
    quality: str = QUALITY_UNKNOWN


def classify_quality(quote: PriceQuote, now: datetime, *,
                     max_age_seconds: int) -> str:
    """この価格がどの品質かを判定する。

    - 取得単価での代用は **unknown**（古い価格ではなく、分からない）
    - DB終値は取得直後でも **stale**（「今の価格」ではない）
    - リアルタイム板は max_age_seconds 以内なら fresh

    時計のずれで観測時刻が未来になっている場合も fresh にしない。
    """
    if quote.source == SOURCE_BOOK_VALUE:
        return QUALITY_UNKNOWN
    if quote.source != SOURCE_REALTIME:
        return QUALITY_STALE

    age = (now - quote.observed_at).total_seconds()
    if age < 0:
        return QUALITY_STALE
    return QUALITY_FRESH if age <= max_age_seconds else QUALITY_STALE


def is_usable_for_new_risk(quote: PriceQuote, now: datetime, *,
                           max_age_seconds: int) -> bool:
    """新規にリスクを取る判断（発注）に使ってよい価格か。

    既存リスクを減らす操作（退出）まで止める判定ではない。止めてよいのは
    「これから増やす」側だけである。
    """
    if not quote.value or quote.value <= 0:
        return False
    return classify_quality(quote, now, max_age_seconds=max_age_seconds) == QUALITY_FRESH
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_price_quote.py -v`
Expected: PASS（13件）

- [ ] **Step 5: BOM確認とコミット**

Run: `head -c 3 src/risk/price_quote.py | xxd`（`2222 22` を確認。`efbb bf` なら下記で除去）

```python
for p in ["src/risk/price_quote.py", "tests/test_price_quote.py"]:
    with open(p, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        with open(p, "wb") as f:
            f.write(data[3:])
```

```bash
git add src/risk/price_quote.py tests/test_price_quote.py
git commit -m "$(cat <<'EOF'
feat(risk): 価格に鮮度と由来を持たせる型を追加

get_current_pricesはdict[str,float]を返すため、リアルタイム板の値か
DB終値で代用した値かを呼び出し側が区別できなかった。
取得単価での代用は「古い価格」ではなく「分からない」として扱い、
価格が不明な保有を0損失として安全扱いしない。
DB終値は取得直後でもstaleにする（「今の価格」ではないため）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: 鮮度付きの価格取得を `RiskManager` に足す

**Files:**
- Modify: `src/risk/manager.py`
- Test: `tests/test_price_quote.py`

**Interfaces:**
- Consumes: Task 1
- Produces:
  - `RiskManager.get_price_quotes(symbols) -> dict[str, PriceQuote]`
  - `RiskManager.get_current_prices(symbols) -> dict[str, float]` — **戻り値の型も値も不変**

**背景:** 既存の `get_current_prices` の呼び出し元は `build_snapshot`・`_resolve_current_prices`（ダッシュボード）・`_held_value` 系の3経路にある。**戻り値の型を変えると全て壊れる。** 鮮度付きの取得を別メソッドとして足し、`get_current_prices` はその値だけを取り出す薄い層にする。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_price_quote.py` の末尾に追記する。

```python
class TestRiskManagerQuotes:
    def _manager(self, price_fn=None):
        from src.core import config as cfg
        from src.risk.manager import RiskManager

        cfg.load("config.yaml")
        return RiskManager(price_fn=price_fn)

    def test_realtime_values_are_marked_realtime(self, monkeypatch):
        rm = self._manager(price_fn=lambda syms: {s: 1000.0 for s in syms})
        quotes = rm.get_price_quotes(["7203"])
        assert quotes["7203"].source == pq.SOURCE_REALTIME
        assert quotes["7203"].value == pytest.approx(1000.0)

    def test_fallback_values_are_marked_daily_close(self, monkeypatch):
        from src.risk import manager as mgr

        rm = self._manager(price_fn=lambda syms: {})
        monkeypatch.setattr(mgr, "latest_closes", lambda syms: {"7203": 990.0})
        quotes = rm.get_price_quotes(["7203"])
        assert quotes["7203"].source == pq.SOURCE_DAILY_CLOSE
        assert quotes["7203"].value == pytest.approx(990.0)

    def test_missing_symbols_are_absent_not_zero(self, monkeypatch):
        """取れなかった銘柄は0ではなくキーごと無い（0損失扱いを防ぐ）"""
        from src.risk import manager as mgr

        rm = self._manager(price_fn=lambda syms: {})
        monkeypatch.setattr(mgr, "latest_closes", lambda syms: {})
        assert rm.get_price_quotes(["7203"]) == {}

    def test_observed_at_is_recorded(self):
        rm = self._manager(price_fn=lambda syms: {s: 1000.0 for s in syms})
        quotes = rm.get_price_quotes(["7203"])
        assert quotes["7203"].observed_at is not None


class TestGetCurrentPricesUnchanged:
    """既存メソッドの戻り値の型も値も変えない（呼び出し元が3経路ある）"""

    def _manager(self, price_fn=None):
        from src.core import config as cfg
        from src.risk.manager import RiskManager

        cfg.load("config.yaml")
        return RiskManager(price_fn=price_fn)

    def test_still_returns_plain_floats(self):
        rm = self._manager(price_fn=lambda syms: {s: 1000.0 for s in syms})
        got = rm.get_current_prices(["7203"])
        assert got == {"7203": 1000.0}
        assert isinstance(got["7203"], float)

    def test_still_falls_back_to_closes(self, monkeypatch):
        from src.risk import manager as mgr

        rm = self._manager(price_fn=lambda syms: {})
        monkeypatch.setattr(mgr, "latest_closes", lambda syms: {"7203": 990.0})
        assert rm.get_current_prices(["7203"]) == {"7203": 990.0}

    def test_empty_input_returns_empty(self):
        rm = self._manager()
        assert rm.get_current_prices([]) == {}
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_price_quote.py -v`
Expected: FAIL — `AttributeError: 'RiskManager' object has no attribute 'get_price_quotes'`

- [ ] **Step 3: 実装を追加**

`src/risk/manager.py` の `get_current_prices` を次の2つに分ける。import に `from src.core import clock` と `from src.risk.price_quote import PriceQuote, SOURCE_DAILY_CLOSE, SOURCE_REALTIME` を足す。

```python
    def get_price_quotes(self, symbols: list[str]) -> dict:
        """保有銘柄の「今の価格」を、**いつ・どこから得たか付きで**返す。

        ブローカーのリアルタイム板（price_fn）を優先し、取得できなかった銘柄
        （price_fn未注入のpaper運用・API障害・price_fnが対応しない銘柄）だけ
        DBの日足終値で代用する。どちらで得たかを source に残すので、
        新規リスクを取る判断では鮮度を見て弾ける（レビューF10）。

        **取れなかった銘柄はキーごと返さない。** 0を返すと「0円の保有」として
        損失0に見えてしまう。
        """
        if not symbols:
            return {}

        now = clock.now()
        quotes: dict = {}

        if self._price_fn is not None:
            cached_now = time.monotonic()
            to_fetch = []
            for s in symbols:
                cached = self._price_cache.get(s)
                if cached is not None and cached_now - cached[1] < self._PRICE_CACHE_TTL_SEC:
                    quotes[s] = PriceQuote(
                        symbol=s, value=cached[0], observed_at=now,
                        source=SOURCE_REALTIME)
                else:
                    to_fetch.append(s)
            if to_fetch:
                try:
                    fetched = dict(self._price_fn(to_fetch))
                    for s, p in fetched.items():
                        if p:
                            self._price_cache[s] = (p, cached_now)
                            quotes[s] = PriceQuote(
                                symbol=s, value=float(p), observed_at=now,
                                source=SOURCE_REALTIME)
                except Exception as e:
                    logger.warning(f"現在値のリアルタイム取得に失敗（終値で代用します）: {e}")

        missing = [s for s in symbols if s not in quotes]
        if missing:
            for s, p in latest_closes(missing).items():
                if p:
                    quotes[s] = PriceQuote(
                        symbol=s, value=float(p), observed_at=now,
                        source=SOURCE_DAILY_CLOSE)
        return quotes

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """保有銘柄の「今の価格」を返す（値だけ）。

        **戻り値の型を変えない。** 呼び出し元が build_snapshot・ダッシュボードの
        含み損益表示・保有評価の3経路にあり、型を変えると全て壊れる。
        鮮度や由来が要る判断は get_price_quotes() を使う。
        """
        return {s: q.value for s, q in self.get_price_quotes(symbols).items()}
```

**注意:** 既存の価格キャッシュ（`_price_cache`、5秒TTL）の挙動を変えないこと。429対策として `8c289c9` で入ったもので、ダッシュボードのポーリングと売買ループがAPIを二重に叩くのを防いでいる。

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_price_quote.py -v`
Expected: PASS（20件）

- [ ] **Step 5: 既存の価格まわりのテストが通ることを確認**

Run: `pytest tests/test_risk_manager_price_cache.py tests/test_unrealized_pnl_live_price.py tests/test_sector_concentration.py -v`
Expected: PASS（キャッシュの挙動と既存の呼び出し元が壊れていないこと）

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

- [ ] **Step 6: コミット**

```bash
git add src/risk/manager.py tests/test_price_quote.py
git commit -m "$(cat <<'EOF'
feat(risk): 鮮度と由来つきの価格取得を追加

リアルタイム板から取れた値かDB終値で代用した値かをsourceに残す。
取れなかった銘柄はキーごと返さない（0を返すと0円の保有として損失0に
見えるため）。
既存のget_current_pricesは戻り値の型も値も変えない。呼び出し元が
3経路あり、型を変えると全て壊れる。429対策の価格キャッシュも維持する。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: プロファイルの適用経路を統合する

**Files:**
- Modify: `src/core/risk_profile.py`
- Test: `tests/test_profile_atomicity.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `_apply_if_active(name: str, params: dict) -> bool` — アクティブなら適用する共通処理
  - `import_profile` / `update_custom` / `set_active` が同じ適用関数を通る

**背景（F11）:** `update_custom()` は更新対象がアクティブなら `_apply()` する（`src/core/risk_profile.py:271-285`）が、`import_profile(..., overwrite=True)` は同じチェックをせず保存だけする（`src/core/risk_profile.py:323-336`）。**アクティブな同名プロファイルを取り込むと、保存した内容と実行中設定が一致しなくなる。**

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_profile_atomicity.py` を新規作成する。

```python
"""リスクプロファイルの適用と保存（F11）のテスト

update_custom はアクティブなら再適用するが、import_profile は保存だけする。
アクティブな同名プロファイルを取り込むと、保存内容と実行中設定が食い違う。
"""
import json

import pytest

from src.core import config as cfg
from src.core import risk_profile as rp


@pytest.fixture(autouse=True)
def _isolated(tmp_path):
    cfg.load("config.yaml")
    rp.load(str(tmp_path / "risk_profile.json"))
    yield


def _params(stop=-0.05, ratio=0.20):
    return {
        "max_position_ratio": ratio,
        "stop_loss_pct": stop,
        "max_positions": 5,
        "max_sector_ratio": 0.30,
        "max_daily_loss": 20000,
        "buy_threshold": 0.14,
        "sell_threshold": -0.14,
    }


class TestImportAppliesWhenActive:
    def test_importing_the_active_profile_updates_running_config(self):
        """アクティブなプロファイルへの取り込みが実行中設定へ反映される（F11の核心）"""
        rp.import_profile("mine", _params(stop=-0.05))
        rp.set_active("mine")

        rp.import_profile("mine", _params(stop=-0.09), overwrite=True)

        assert cfg.get_section("trading")["stop_loss_pct"] == pytest.approx(-0.09)

    def test_importing_an_inactive_profile_does_not_change_running_config(self):
        rp.import_profile("mine", _params(stop=-0.05))
        rp.set_active("mine")
        before = cfg.get_section("trading")["stop_loss_pct"]

        rp.import_profile("other", _params(stop=-0.09))

        assert cfg.get_section("trading")["stop_loss_pct"] == pytest.approx(before)

    def test_saved_content_matches_running_config(self):
        rp.import_profile("mine", _params(stop=-0.05))
        rp.set_active("mine")
        rp.import_profile("mine", _params(stop=-0.09), overwrite=True)

        saved = rp.get_profiles()["mine"]["stop_loss_pct"]
        running = cfg.get_section("trading")["stop_loss_pct"]
        assert saved == pytest.approx(running)

    def test_update_custom_still_applies(self):
        """既存の挙動を壊さない"""
        rp.import_profile("mine", _params(stop=-0.05))
        rp.set_active("mine")
        rp.update_custom("mine", _params(stop=-0.03))
        assert cfg.get_section("trading")["stop_loss_pct"] == pytest.approx(-0.03)

    def test_set_active_still_applies(self):
        rp.import_profile("a", _params(stop=-0.05))
        rp.import_profile("b", _params(stop=-0.09))
        rp.set_active("a")
        rp.set_active("b")
        assert cfg.get_section("trading")["stop_loss_pct"] == pytest.approx(-0.09)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_profile_atomicity.py -v`
Expected: FAIL — `test_importing_the_active_profile_updates_running_config` が `-0.05` のまま

- [ ] **Step 3: 実装を修正**

`src/core/risk_profile.py` に共通処理を足す（`_apply` の直後）。

```python
def _apply_if_active(name: str, params: dict) -> bool:
    """このプロファイルがアクティブなら実行中設定へ反映する。

    update_custom だけがこの再適用を持ち、import_profile には無かったため、
    **アクティブな同名プロファイルを取り込むと保存内容と実行中設定が
    食い違っていた**（レビューF11）。適用の判断を1箇所に集める。
    """
    if _active != name:
        return False
    _apply(params)
    logger.info(f"アクティブなリスクプロファイルを再適用: {name}")
    return True
```

`update_custom` の適用部分を差し替える。

```python
    _custom[name] = validated
    _apply_if_active(name, validated)
    _record_history("update", to=name)
```

`import_profile` に適用を足す。

```python
    validated = validate_profile(params)
    _custom[name] = validated
    _apply_if_active(name, validated)
    _record_history("import", to=name)
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_profile_atomicity.py -v`
Expected: PASS（5件）

- [ ] **Step 5: 既存のプロファイルテストが通ることを確認**

Run: `pytest tests/test_risk_profile.py tests/test_custom_risk_profile.py tests/test_risk_wiring.py -v`
Expected: PASS

- [ ] **Step 6: コミット**

```bash
git add src/core/risk_profile.py tests/test_profile_atomicity.py
git commit -m "$(cat <<'EOF'
fix(config): アクティブなプロファイルへのimportが適用されなかった問題を修正

update_customは更新対象がアクティブなら再適用するが、import_profileは
同じチェックをせず保存だけしていた。アクティブな同名プロファイルを
取り込むと、保存した内容と実行中設定が一致しなくなる経路だった。
適用の判断を1箇所に集める。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: 設定の保存を原子的にする

**Files:**
- Modify: `src/core/risk_profile.py`（`_persist`）
- Test: `tests/test_profile_atomicity.py`

**Interfaces:**
- Consumes: なし
- Produces: `_persist()` が一時ファイル経由で原子的に書く

**背景（F11）:** `_persist()` は対象ファイルを直接開いて `json.dump` する。**書き込み途中でプロセスが落ちると、設定ファイルが切り詰められた不完全なJSONとして残る。** 次回起動で読めず、リスクプロファイルが既定へ戻る。段階Eの `model_store._write_atomically` と同じ方式で塞ぐ。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_profile_atomicity.py` の末尾に追記する。

```python
class TestPersistAtomicity:
    def test_writes_valid_json(self, tmp_path):
        rp.import_profile("mine", _params())
        content = rp._path.read_text(encoding="utf-8")
        assert json.loads(content)["custom"]["mine"] is not None

    def test_leaves_no_temporary_files(self, tmp_path):
        rp.import_profile("mine", _params())
        leftovers = [p.name for p in rp._path.parent.iterdir()
                     if p.name.endswith(".tmp")]
        assert leftovers == []

    def test_previous_content_survives_a_failed_write(self, tmp_path, monkeypatch):
        """書き込みが途中で落ちても、前の内容が読める状態で残る（F11）"""
        rp.import_profile("mine", _params(stop=-0.05))
        before = rp._path.read_text(encoding="utf-8")

        import json as json_mod

        def boom(*args, **kwargs):
            raise OSError("ディスクが一杯です")

        monkeypatch.setattr(json_mod, "dump", boom)
        with pytest.raises(OSError):
            rp.import_profile("other", _params(stop=-0.09))

        after = rp._path.read_text(encoding="utf-8")
        assert after == before
        assert json.loads(after)["custom"]["mine"] is not None

    def test_reload_after_a_failed_write_still_works(self, tmp_path, monkeypatch):
        rp.import_profile("mine", _params(stop=-0.05))

        import json as json_mod

        def boom(*args, **kwargs):
            raise OSError("boom")

        monkeypatch.setattr(json_mod, "dump", boom)
        with pytest.raises(OSError):
            rp.import_profile("other", _params())
        monkeypatch.undo()

        rp.load(str(rp._path))
        assert "mine" in rp.get_profiles()
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_profile_atomicity.py::TestPersistAtomicity -v`
Expected: FAIL — `test_previous_content_survives_a_failed_write` で、直接書き込みのため内容が切り詰められる

- [ ] **Step 3: 実装を修正**

`src/core/risk_profile.py` の `_persist` を次に置き換える。import に `import os` と `import tempfile` を足す。

```python
def _persist() -> None:
    """設定を**原子的に**保存する。

    対象ファイルを直接開いて書くと、書き込み途中でプロセスが落ちたときに
    切り詰められた不完全なJSONが残り、次回起動で読めずリスクプロファイルが
    既定へ戻る（レビューF11）。一時ファイルへ書いてから置換することで、
    ファイルは常に「前の完全な内容」か「新しい完全な内容」のどちらかになる。
    """
    _path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"active": _active, "custom": _custom, "history": _history}

    fd, tmp = tempfile.mkstemp(dir=str(_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, _path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_profile_atomicity.py -v`
Expected: PASS（9件）

- [ ] **Step 5: 既存テストが通ることを確認**

Run: `pytest tests/test_risk_profile.py tests/test_custom_risk_profile.py -v`
Expected: PASS

- [ ] **Step 6: コミット**

```bash
git add src/core/risk_profile.py tests/test_profile_atomicity.py
git commit -m "$(cat <<'EOF'
fix(config): リスクプロファイルの保存を原子的にした

対象ファイルを直接開いて書いていたため、書き込み途中で落ちると
切り詰められた不完全なJSONが残り、次回起動で読めずプロファイルが
既定へ戻る経路だった。一時ファイルへ書いてから置換することで、
ファイルは常に前か新しいかのどちらかの完全な内容になる。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 外部キーの有効化と参照整合の検査

**Files:**
- Modify: `src/data/database.py`
- Test: `tests/test_db_integrity.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - 接続ごとの `PRAGMA foreign_keys=ON`
  - `check_referential_integrity() -> dict` — 孤立レコードを**数えるだけ**（削除しない）
  - `check_quantity_consistency() -> dict` — 数量の不整合を数える

**背景（F15）:** `Trade.intent_id` / `OrderApproval.intent_id` / `Fill.broker_order_id` は整数列だが **`ForeignKey` 宣言が無い**。DB初期化は WAL を設定する一方、**接続ごとの `foreign_keys` 有効化が無い**（SQLiteは既定で外部キーを強制しない）。`integrity_check` は ok でも、業務上の紐付きが正しいことまでは検査しない。

**本番DBを壊さない。** 制約を既存データへ後から課すと、過去の不整合で起動できなくなる。本タスクは次の順で進める。

1. 接続ごとに `foreign_keys` を有効化する（**新しい書き込みだけ**が守られる）
2. 既存データの不整合を**検出する関数**を足す（削除・書き換えはしない）
3. `ForeignKey` 宣言自体は、検出がゼロになってから別途入れる（本計画のスコープ外）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_db_integrity.py` を新規作成する。

```python
"""DBの参照整合（F15）のテスト

Trade.intent_id / Fill.broker_order_id は整数列だが外部キー宣言が無く、
接続ごとの PRAGMA foreign_keys も設定されていない。
本番DBを壊さないため、まず検出だけを足す。
"""
import pytest
from sqlalchemy import text

from src.core import config as cfg
from src.data import database as db
from src.data.database import get_session


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


class TestForeignKeysPragma:
    def test_foreign_keys_are_enabled_per_connection(self, isolated_db):
        """SQLiteは既定で外部キーを強制しない。接続ごとに有効化する"""
        with get_session() as session:
            got = session.execute(text("PRAGMA foreign_keys")).scalar()
        assert got == 1

    def test_enabled_on_every_new_session(self, isolated_db):
        """1回目だけでなく毎回有効になる（接続プールの再利用でも）"""
        for _ in range(3):
            with get_session() as session:
                assert session.execute(text("PRAGMA foreign_keys")).scalar() == 1

    def test_wal_is_still_enabled(self, isolated_db):
        """既存のWAL設定を壊さない"""
        with get_session() as session:
            mode = session.execute(text("PRAGMA journal_mode")).scalar()
        assert str(mode).lower() == "wal"


class TestReferentialIntegrityCheck:
    def _orphan_trade(self):
        from src.data.database import Trade

        with get_session() as session:
            session.add(Trade(symbol="7203", side="BUY", quantity=100,
                              price=1000.0, status="FILLED", intent_id=999999))
            session.commit()

    def test_clean_database_reports_zero(self, isolated_db):
        got = db.check_referential_integrity()
        assert got["orphan_trades"] == 0
        assert got["orphan_fills"] == 0
        assert got["total"] == 0

    def test_detects_an_orphan_trade(self, isolated_db):
        self._orphan_trade()
        got = db.check_referential_integrity()
        assert got["orphan_trades"] == 1
        assert got["total"] == 1

    def test_detection_does_not_delete_anything(self, isolated_db):
        """検出するだけ。本番DBを勝手に壊さない"""
        from src.data.database import Trade

        self._orphan_trade()
        db.check_referential_integrity()
        with get_session() as session:
            remaining = session.query(Trade).count()
        assert remaining == 1

    def test_reports_each_category_separately(self, isolated_db):
        got = db.check_referential_integrity()
        for key in ("orphan_trades", "orphan_fills", "orphan_approvals", "total"):
            assert key in got


class TestQuantityConsistency:
    def test_clean_database_reports_zero(self, isolated_db):
        got = db.check_quantity_consistency()
        assert got["negative_positions"] == 0
        assert got["total"] == 0

    def test_detects_a_negative_position(self, isolated_db):
        from src.data.database import Position

        with get_session() as session:
            session.add(Position(symbol="7203", quantity=-100, avg_cost=1000.0))
            session.commit()
        got = db.check_quantity_consistency()
        assert got["negative_positions"] == 1

    def test_detects_a_position_without_cost(self, isolated_db):
        from src.data.database import Position

        with get_session() as session:
            session.add(Position(symbol="9984", quantity=100, avg_cost=0.0))
            session.commit()
        got = db.check_quantity_consistency()
        assert got["positions_without_cost"] == 1
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_db_integrity.py -v`
Expected: FAIL — `PRAGMA foreign_keys` が 0、および `AttributeError: module 'src.data.database' has no attribute 'check_referential_integrity'`

- [ ] **Step 3: 接続ごとの `foreign_keys` を有効化する**

`src/data/database.py` の `init()` の中、**`_engine = create_engine(...)`（`database.py:376` 付近）の直後**に接続イベントを登録する。`with _engine.connect()` で WAL を設定するより前に置くこと（イベントはそれ以降に張られる接続へ適用されるため、エンジン生成の直後が確実）。import に `from sqlalchemy import event` を足す。

```python
    # SQLiteは**接続ごと**に外部キーの強制を有効化する必要がある（既定はOFF）。
    # WALはDBファイル単位の設定なので一度で済むが、foreign_keys は接続単位。
    # 接続プールが新しい接続を作るたびに設定する（レビューF15）。
    @event.listens_for(_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
```

**注意:** `_engine` の変数名は実装に合わせること。`init()` 内でエンジンを作っている変数を使う。

- [ ] **Step 4: 検査関数を追加**

`src/data/database.py` の末尾に追加する。

```python
def check_referential_integrity() -> dict:
    """業務上の紐付きが壊れていないかを**数えるだけ**の検査。

    Trade.intent_id / Fill.broker_order_id / OrderApproval.intent_id は
    整数列だが外部キー宣言が無く、過去に孤立したレコードが残りうる
    （レビューF15）。`PRAGMA integrity_check` は ok でも、業務上の紐付きが
    正しいことまでは検査しない。

    **削除も書き換えもしない。** 既存データへ制約を後から課すと、過去の
    不整合で起動できなくなる。まず実態を数え、ゼロになってから宣言を入れる。
    """
    with get_session() as session:
        intent_ids = {row[0] for row in session.execute(select(OrderIntent.id)).all()}
        trade_ids = {row[0] for row in session.execute(select(Trade.id)).all()}

        orphan_trades = sum(
            1 for (tid,) in session.execute(
                select(Trade.intent_id).where(Trade.intent_id.isnot(None))).all()
            if tid not in intent_ids
        )
        orphan_fills = sum(
            1 for (bid,) in session.execute(
                select(Fill.broker_order_id).where(
                    Fill.broker_order_id.isnot(None))).all()
            if bid not in trade_ids
        )
        orphan_approvals = sum(
            1 for (aid,) in session.execute(
                select(OrderApproval.intent_id).where(
                    OrderApproval.intent_id.isnot(None))).all()
            if aid not in intent_ids
        )

    total = orphan_trades + orphan_fills + orphan_approvals
    if total:
        logger.warning(
            f"参照整合の不整合を検出: trades={orphan_trades} fills={orphan_fills} "
            f"approvals={orphan_approvals}（削除はしていません）"
        )
    return {
        "orphan_trades": orphan_trades,
        "orphan_fills": orphan_fills,
        "orphan_approvals": orphan_approvals,
        "total": total,
    }


def check_quantity_consistency() -> dict:
    """数量・単価の不整合を数える（削除も書き換えもしない）。"""
    with get_session() as session:
        negative = session.scalar(
            select(func.count(Position.id)).where(Position.quantity < 0)) or 0
        without_cost = session.scalar(
            select(func.count(Position.id)).where(
                Position.quantity > 0, Position.avg_cost <= 0)) or 0

    total = negative + without_cost
    if total:
        logger.warning(
            f"数量の不整合を検出: 負の建玉={negative} 単価なし={without_cost}"
            "（削除はしていません）"
        )
    return {
        "negative_positions": negative,
        "positions_without_cost": without_cost,
        "total": total,
    }
```

`func` と `select` が import 済みか確認し、無ければ `from sqlalchemy import func, select` を足す。

- [ ] **Step 5: テストを実行して成功を確認**

Run: `pytest tests/test_db_integrity.py -v`
Expected: PASS（11件）

- [ ] **Step 6: 既存のDBテストが通ることを確認**

Run: `pytest tests/test_backfill_migration.py tests/test_schema_version.py tests/test_db_backup_wal.py tests/test_fill_recording.py -v`
Expected: PASS（`foreign_keys=ON` で既存の書き込みが弾かれないこと。外部キー宣言をまだ入れていないので影響は無いはず）

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

- [ ] **Step 7: コミット**

```bash
git add src/data/database.py tests/test_db_integrity.py
git commit -m "$(cat <<'EOF'
feat(data): 外部キーの有効化と参照整合の検査を追加

SQLiteは接続ごとに外部キーの強制を有効化する必要があるが設定が無かった。
WALはDBファイル単位だがforeign_keysは接続単位なので、接続イベントで設定する。
Trade.intent_id等は整数列で外部キー宣言が無く、孤立レコードが残りうる。
integrity_checkはokでも業務上の紐付きまでは検査しない。
本番DBを壊さないため、まず数えるだけの検査を足す。削除も書き換えもしない。
ForeignKey宣言自体は検出がゼロになってから別途入れる。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: 起動時の整合検査を結線する

**Files:**
- Modify: `src/core/preflight.py`
- Test: `tests/test_db_integrity.py`

**Interfaces:**
- Consumes: Task 5 の `check_referential_integrity` / `check_quantity_consistency`
- Produces: 起動時に検査を走らせ、不整合があれば**警告する**（起動は止めない）

**背景:** 検査関数があっても呼ばれなければ意味がない。ただし**起動を止めない**。過去の不整合で本番が立ち上がらなくなるのは、直したい問題より大きな害になる。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_db_integrity.py` の末尾に追記する。

```python
class TestPreflightIntegrityCheck:
    def test_clean_database_produces_no_warning(self, isolated_db, caplog):
        from src.core import preflight

        result = preflight.check_db_integrity()
        assert result["total"] == 0

    def test_orphans_are_reported_but_do_not_block_startup(self, isolated_db):
        """不整合があっても起動は止めない（止める方が害が大きい）"""
        from src.core import preflight
        from src.data.database import Trade

        with get_session() as session:
            session.add(Trade(symbol="7203", side="BUY", quantity=100,
                              price=1000.0, status="FILLED", intent_id=999999))
            session.commit()

        result = preflight.check_db_integrity()
        assert result["total"] >= 1
        assert result["blocking"] is False

    def test_combines_both_checks(self, isolated_db):
        from src.core import preflight

        result = preflight.check_db_integrity()
        assert "referential" in result
        assert "quantity" in result
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_db_integrity.py::TestPreflightIntegrityCheck -v`
Expected: FAIL — `AttributeError: module 'src.core.preflight' has no attribute 'check_db_integrity'`

- [ ] **Step 3: 実装を追加**

`src/core/preflight.py` に追加する。

```python
def check_db_integrity() -> dict:
    """起動時にDBの参照整合と数量の不整合を検査する。

    **見つかっても起動は止めない。** 過去の不整合で本番が立ち上がらなくなるのは、
    直したい問題より大きな害になる。警告として可視化し、対処は人が決める
    （レビューF15）。
    """
    from src.data.database import check_quantity_consistency, check_referential_integrity

    referential = check_referential_integrity()
    quantity = check_quantity_consistency()
    total = referential["total"] + quantity["total"]

    if total:
        logger.warning(
            f"DB整合の検査で{total}件の不整合を検出しました。"
            "起動は継続しますが、内容を確認してください"
        )
    else:
        logger.info("DB整合の検査: 不整合なし")

    return {
        "referential": referential,
        "quantity": quantity,
        "total": total,
        "blocking": False,
    }
```

既存の preflight の実行順序に組み込む（起動チェック一覧を呼んでいる関数へ1行足す）。

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_db_integrity.py -v`
Expected: PASS（14件）

- [ ] **Step 5: 既存の preflight テストが通ることを確認**

Run: `pytest tests/test_preflight.py -v`
Expected: PASS

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

- [ ] **Step 6: BOM確認とコミット**

Run: `head -c 3 src/core/preflight.py | xxd`（`2222 22` を確認）

```bash
git add src/core/preflight.py tests/test_db_integrity.py
git commit -m "$(cat <<'EOF'
feat(core): 起動時のDB整合検査を結線

検査関数があっても呼ばれなければ意味がない。起動時に参照整合と数量の
不整合を検査して可視化する。
不整合が見つかっても起動は止めない。過去の不整合で本番が立ち上がらなく
なるのは、直したい問題より大きな害になるため。対処は人が決める。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## 完了条件の確認

- [ ] **確認1: 価格の由来が区別できる**

Run: `pytest tests/test_price_quote.py -v`
Expected: PASS（20件）

- [ ] **確認2: 既存の価格取得の呼び出し元が壊れていない**

Run: `pytest tests/test_risk_manager_price_cache.py tests/test_unrealized_pnl_live_price.py tests/test_sector_concentration.py -v`
Expected: PASS

- [ ] **確認3: アクティブなプロファイルへの取り込みが実行中設定へ反映される**

Run: `pytest tests/test_profile_atomicity.py::TestImportAppliesWhenActive -v`
Expected: PASS（5件）

- [ ] **確認4: 設定の書き込みが途中で落ちても前の内容が残る**

Run: `pytest tests/test_profile_atomicity.py::TestPersistAtomicity -v`
Expected: PASS（4件）

- [ ] **確認5: 外部キーが接続ごとに有効化される**

Run: `pytest tests/test_db_integrity.py::TestForeignKeysPragma -v`
Expected: PASS（3件）

- [ ] **確認6: 整合検査が本番DBを壊さない**

Run: `pytest tests/test_db_integrity.py::TestReferentialIntegrityCheck::test_detection_does_not_delete_anything -v`
Expected: PASS

- [ ] **確認7: 既存経路に回帰が無い**

Run: `pytest tests/ -q`
Expected: 着手前と同じ結果（新規テスト37件ぶんだけ増える）

- [ ] **確認8: 本番DBの実態を一度だけ確認する**（判断材料。合否ではない）

本番DBに対して検査を1回だけ実行し、結果を記録する。**削除も書き換えもしない。**

```python
from src.core import config as cfg
from src.data import database as db

cfg.load("config.yaml")
db.init()
print("参照整合:", db.check_referential_integrity())
print("数量:", db.check_quantity_consistency())
```

Expected: 件数が得られること。**不整合が出た場合はユーザーへ報告して指示を仰ぐ。** 自動で修復しない（実取引の履歴であるため）。

---

## 残る作業（本計画のスコープ外）

- **`ForeignKey` 宣言の追加**。上記の検査でゼロを確認してから、既存スキーマからの移行・再実行・復元をテストした上で入れる
- **F10 の結線**。`get_price_quotes` を発注判断（`validate_buy` 等）へ実際に使わせるかは、鮮度不足で発注が止まる頻度を実測してから決める。本計画は「区別できる状態」を作るところまで
- **PBKDF2 の反復回数**。現行は 200,000 回、OWASP の同方式の推奨は 600,000 回。端末上の認証時間を測った上で強化し、次回の成功ログインで更新する方式が要る（レビュー §7 F13 の付随項目）
- **設計書への反映**。`docs/詳細設計書.md` / `docs/概要設計書.md` に本計画の変更を追記する
