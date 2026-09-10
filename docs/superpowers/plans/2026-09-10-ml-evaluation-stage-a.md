# ML評価基盤 段階A（データ土台）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 日足の取得境界・確定判定・生値と調整値の分離を直し、どのシグナルがどの営業日の確定足から作られたか辿れるようにする。

**Architecture:** 純粋関数の確定足判定モジュール（`src/data/bar_status.py`）を新設し、`market_data.py` は取得境界と2系列の保存だけを担う。`trading.py` の `signal_scan` は確定していない足の銘柄を新規候補から除外するが、保有保護の退出は止めない。企業行動は価格比からの推測をやめ、yfinance の分割イベントAPIを唯一の権威とする。

**Tech Stack:** Python 3.11 / pandas 2.1.4 / yfinance>=1.4.1,<2.0.0 / SQLAlchemy 2.0.23 / pytest / jpholiday 1.0.3

**Spec:** `docs/superpowers/specs/2026-09-10-ml-evaluation-foundation-design.md`（§5・§10・§12・§14）

## Global Constraints

- 日時は **JST naive**（tzinfo を持たない日本時間の datetime）で統一する。現在時刻は必ず `src/core/clock.now()` / `clock.today()` を使い、`datetime.now()` を直接呼ばない。
- 設定は `src/core/config.py` の `cfg.get_section("<section>")` で読む。セクション名は `config.yaml` のトップレベルキー。
- 新規の DB 列は **すべて nullable**。`src/data/database.py` の `_migrate_add_missing_columns()` がモデル定義から `ALTER TABLE ADD COLUMN` を自動生成するため、**モデルに列を足す以外のマイグレーション作業は不要**。
- 既存の公開関数 `build_features()` / `build_training_set()` の挙動は変更しない（段階Aでは触らない）。
- `legacy` / `v2` の切替対象に段階Aの変更は**含めない**。取得境界・確定足判定・生値と調整値の分離は全経路へ適用する。
- ファイルは UTF-8 BOM 無し・LF で保存する。
- テストは `pytest tests/<file>.py -v` で実行する。DB を使うテストは `cfg.load("config.yaml")` → `cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")` → `db.init()` の順で隔離する（`tests/test_backtest_threshold_override.py:26-30` と同じ形）。
- ネットワークへ出るテストを書かない。yfinance は `unittest.mock.patch` で差し替える。

---

## File Structure

| ファイル | 責務 |
|---|---|
| `src/data/bar_status.py`（新規） | 「今この時刻に確定しているべきセッション」の決定と、銘柄ごとの足の鮮度分類。純粋関数のみでDB・ネットワークに触らない |
| `src/data/market_data.py`（改修） | 取得境界の一元化、生OHLCと調整済み終値の分離保存、分割イベントの取得 |
| `src/data/database.py`（改修） | `Signal.data_as_of` 列と `CorporateAction` テーブルの追加 |
| `src/services/trading.py`（改修） | `data_update` のバッチ記録、`signal_scan` の鮮度ゲート、`_save_signal` の `data_as_of` 保存 |
| `tests/test_bar_status.py`（新規） | 確定セッション決定と鮮度分類 |
| `tests/test_market_data_freshness.py`（新規） | 取得境界、2系列の保存、`update_symbol` の戻り値 |
| `tests/test_corporate_actions.py`（新規） | 分割イベントの取得と、分割で評価額が増えない不変条件 |
| `tests/test_signal_freshness_gate.py`（新規） | 新規候補の除外と、退出が止まらないこと |

---

## Task 1: 確定足判定モジュール

**Files:**
- Create: `src/data/bar_status.py`
- Test: `tests/test_bar_status.py`

**Interfaces:**
- Consumes: `src/core/market_calendar.is_business_day(d: date) -> bool`
- Produces:
  - `BarStatus` dataclass（frozen）: `symbol: str`, `last_bar_session: date | None`, `observed_at: datetime`, `is_final: bool`, `state: str`
  - `as_of_session(now: datetime, close_grace_minutes: int = 20) -> date` — その時刻に確定しているべき直近セッション
  - `classify(symbol: str, last_bar_session: date | None, now: datetime, close_grace_minutes: int = 20) -> BarStatus`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_bar_status.py` を新規作成する。

```python
"""確定足判定（src/data/bar_status.py）のテスト

背景: is_business_day() は営業日か否かを返すだけで、その足が確定済みかは
分からない。場中の未確定足・引け後の配信待ち・休場を区別する必要がある。
"""
from datetime import date, datetime

from src.data import bar_status


class TestAsOfSession:
    def test_after_close_returns_same_day(self):
        """引け後（15:00 + 猶予20分 = 15:20 以降）はその営業日が確定セッション"""
        # 2026-09-10 は木曜（営業日）
        now = datetime(2026, 9, 10, 16, 0)
        assert bar_status.as_of_session(now) == date(2026, 9, 10)

    def test_during_session_returns_previous_business_day(self):
        """場中は当日足がまだ確定していないので前営業日が基準"""
        now = datetime(2026, 9, 10, 11, 0)
        assert bar_status.as_of_session(now) == date(2026, 9, 9)

    def test_within_grace_after_close_returns_previous(self):
        """引け直後の猶予時間内は、配信待ちとみなして前営業日を基準にする"""
        now = datetime(2026, 9, 10, 15, 10)  # 15:00引け + 猶予20分未満
        assert bar_status.as_of_session(now) == date(2026, 9, 9)

    def test_on_holiday_returns_last_business_day(self):
        """休場日は直近の営業日が基準"""
        # 2026-09-12 は土曜
        now = datetime(2026, 9, 12, 16, 0)
        assert bar_status.as_of_session(now) == date(2026, 9, 11)

    def test_monday_morning_returns_friday(self):
        """月曜の場中は前営業日＝金曜"""
        # 2026-09-14 は月曜
        now = datetime(2026, 9, 14, 10, 0)
        assert bar_status.as_of_session(now) == date(2026, 9, 11)


class TestClassify:
    def test_matching_session_is_fresh(self):
        now = datetime(2026, 9, 10, 16, 0)
        st = bar_status.classify("7203", date(2026, 9, 10), now)
        assert st.state == "fresh"
        assert st.is_final is True
        assert st.symbol == "7203"
        assert st.last_bar_session == date(2026, 9, 10)
        assert st.observed_at == now

    def test_older_session_is_stale(self):
        now = datetime(2026, 9, 10, 16, 0)
        st = bar_status.classify("7203", date(2026, 9, 9), now)
        assert st.state == "stale"
        assert st.is_final is True

    def test_no_bar_is_missing(self):
        now = datetime(2026, 9, 10, 16, 0)
        st = bar_status.classify("7203", None, now)
        assert st.state == "missing"
        assert st.is_final is False

    def test_future_session_is_provisional(self):
        """基準セッションより新しい足は、まだ確定していない場中の足とみなす"""
        now = datetime(2026, 9, 10, 11, 0)  # 基準は 9/9
        st = bar_status.classify("7203", date(2026, 9, 10), now)
        assert st.state == "provisional"
        assert st.is_final is False
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_bar_status.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.bar_status'`

- [ ] **Step 3: 最小の実装を書く**

`src/data/bar_status.py` を新規作成する。

```python
"""日足が確定しているかの判定。

`market_calendar.is_business_day()` は営業日か否かを返すだけで、その足が
確定済みかは分からない。場中の未確定足・引け後の配信待ち・休場を区別しないと、
「日付は今日だがまだ確定していない足」を新規候補の根拠に使ってしまう。

本モジュールは純粋関数だけで構成し、DB・ネットワークに触らない。
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from src.core import market_calendar

# 東証の大引け（後場終了）。この時刻＋猶予を過ぎたらその日の足を確定とみなす。
_CLOSE_HOUR = 15
_CLOSE_MINUTE = 0


@dataclass(frozen=True)
class BarStatus:
    """ある銘柄の最終足が、基準セッションに対してどういう状態かを表す。"""
    symbol: str
    last_bar_session: date | None
    observed_at: datetime
    is_final: bool
    state: str  # "fresh" | "stale" | "missing" | "provisional"


def _previous_business_day(d: date) -> date:
    """d より前の直近営業日を返す。"""
    cur = d - timedelta(days=1)
    while not market_calendar.is_business_day(cur):
        cur -= timedelta(days=1)
    return cur


def as_of_session(now: datetime, close_grace_minutes: int = 20) -> date:
    """now の時点で確定しているべき直近セッションを返す。

    当日が営業日で、かつ大引け＋猶予を過ぎていれば当日。
    それ以外（場中・引け直後の配信待ち・休場）は直近の過去営業日。
    """
    today = now.date()
    if market_calendar.is_business_day(today):
        cutoff = now.replace(
            hour=_CLOSE_HOUR, minute=_CLOSE_MINUTE, second=0, microsecond=0
        ) + timedelta(minutes=close_grace_minutes)
        if now >= cutoff:
            return today
    return _previous_business_day(today)


def classify(symbol: str, last_bar_session: date | None, now: datetime,
             close_grace_minutes: int = 20) -> BarStatus:
    """最終足の営業日を基準セッションと突き合わせて鮮度を分類する。"""
    expected = as_of_session(now, close_grace_minutes)
    if last_bar_session is None:
        state, is_final = "missing", False
    elif last_bar_session == expected:
        state, is_final = "fresh", True
    elif last_bar_session < expected:
        state, is_final = "stale", True
    else:
        # 基準より新しい＝まだ確定していない場中の足
        state, is_final = "provisional", False
    return BarStatus(
        symbol=symbol,
        last_bar_session=last_bar_session,
        observed_at=now,
        is_final=is_final,
        state=state,
    )
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_bar_status.py -v`
Expected: PASS（10件）

- [ ] **Step 5: コミット**

```bash
git add src/data/bar_status.py tests/test_bar_status.py
git commit -m "feat(data): 日足の確定判定モジュールを追加"
```

---

## Task 2: 取得境界を「終了日を含む」に一元化

**Files:**
- Modify: `src/data/market_data.py:23-46`（`fetch_ohlcv`）
- Test: `tests/test_market_data_freshness.py`

**Interfaces:**
- Consumes: なし
- Produces: `fetch_ohlcv(symbol: str, start: date, end: date, retries: int = 2) -> pd.DataFrame` — `end` は**含む**。排他への変換は関数内部だけで行う

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_market_data_freshness.py` を新規作成する。

```python
"""日足取得の境界・2系列保存・更新結果のテスト

背景: yfinance の end は排他境界のため、営業日Tの引け後に実行しても
Tの足が取得できなかった（F01）。境界の解釈を1箇所に固定する。
"""
from datetime import date
from unittest.mock import patch

import pandas as pd

from src.data import market_data


def _fake_yf_frame(days: list[date]) -> pd.DataFrame:
    """yf.download が返す形（auto_adjust=False 相当）を模す。"""
    return pd.DataFrame(
        {
            "Open": [1000.0] * len(days),
            "High": [1010.0] * len(days),
            "Low": [990.0] * len(days),
            "Close": [1005.0] * len(days),
            "Adj Close": [1002.0] * len(days),
            "Volume": [100000] * len(days),
        },
        index=pd.to_datetime(days),
    )


class TestFetchBoundary:
    def test_end_is_inclusive_one_day_added_internally(self):
        """呼び出し側は end を含む日付として渡す。+1日は関数内部だけで行う。"""
        captured = {}

        def fake_download(sym, **kwargs):
            captured.update(kwargs)
            return _fake_yf_frame([date(2026, 9, 10)])

        with patch.object(market_data.yf, "download", side_effect=fake_download):
            market_data.fetch_ohlcv("7203", date(2026, 9, 1), date(2026, 9, 10))

        # yfinance へは排他境界（+1日）が渡る
        assert captured["end"] == "2026-09-11"
        assert captured["start"] == "2026-09-01"

    def test_returned_frame_includes_end_date(self):
        with patch.object(market_data.yf, "download",
                          return_value=_fake_yf_frame([date(2026, 9, 9), date(2026, 9, 10)])):
            df = market_data.fetch_ohlcv("7203", date(2026, 9, 1), date(2026, 9, 10))
        assert df.index[-1] == date(2026, 9, 10)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_market_data_freshness.py -v`
Expected: FAIL — `assert '2026-09-10' == '2026-09-11'`（現在は `end` をそのまま渡している）

- [ ] **Step 3: 実装を修正**

`src/data/market_data.py` の `fetch_ohlcv` を次のように変更する。docstring と `yf.download` の `end` 引数だけを触る。

```python
def fetch_ohlcv(symbol: str, start: date, end: date, retries: int = 2) -> pd.DataFrame:
    """yfinanceからOHLCVを取得する（一時的な通信エラーは指定回数までリトライ）。

    `end` は **その日を含む**。yfinance公式仕様の end は排他境界のため、
    ここで内部的に +1日する。呼び出し側では一切加減算しないこと
    （内部と呼び出し側の双方で1日足す事故を防ぐため、境界の解釈は
    この関数に一元化する）。
    """
    yf_sym = _to_yf_symbol(symbol)
    # yfinance の end は排他境界。引数は「含む」なので +1日して渡す。
    yf_end = end + timedelta(days=1)
    df = pd.DataFrame()
    for attempt in range(retries + 1):
        try:
            df = yf.download(yf_sym, start=start.isoformat(), end=yf_end.isoformat(),
                             auto_adjust=True, progress=False)
            break
        except Exception as e:
            if attempt < retries:
                logger.warning(f"yfinance取得失敗 (リトライ {attempt + 1}/{retries}): {symbol} {e}")
                time.sleep(2)
            else:
                logger.error(f"yfinance取得失敗（リトライ上限到達）: {symbol} {e}")
                return pd.DataFrame()
    if df.empty:
        logger.warning(f"データ取得なし: {symbol} ({start} ~ {end})")
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).date
    df.index.name = "date"
    return df
```

`timedelta` は既に `from datetime import date, timedelta` で import 済みなので追加不要。

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_market_data_freshness.py -v`
Expected: PASS（2件）

- [ ] **Step 5: コミット**

```bash
git add src/data/market_data.py tests/test_market_data_freshness.py
git commit -m "fix(data): 日足取得の終了日が排他境界で当日足を取り逃していた問題を修正"
```

---

## Task 3: 生値と調整済み終値を分離して保存

**Files:**
- Modify: `src/data/market_data.py`（`fetch_ohlcv` の `auto_adjust`、`upsert_ohlcv`、`load_ohlcv`）
- Test: `tests/test_market_data_freshness.py`

**Interfaces:**
- Consumes: Task 2 の `fetch_ohlcv`
- Produces:
  - `fetch_ohlcv` の戻り値に `adjusted_close` 列が加わる（`open/high/low/close` は生値）
  - `load_ohlcv(symbol: str, limit: int = 500, price_basis: str = "adjusted") -> pd.DataFrame` — `price_basis` は `"adjusted"`（特徴量・リターン用）または `"raw"`（株数・必要資金用）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_market_data_freshness.py` の冒頭 import ブロックを次に**置き換える**（Task 5 で使う分も含めた最終形）。

```python
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from src.core import config as cfg
from src.data import database as db
from src.data import market_data
```

そのうえで末尾に追記する。

```python
class TestRawAndAdjustedSeparation:
    def test_fetch_keeps_raw_ohlc_and_adjusted_close(self):
        """auto_adjust=False で取得し、生OHLCと調整済み終値を別々に持つ"""
        captured = {}

        def fake_download(sym, **kwargs):
            captured.update(kwargs)
            return _fake_yf_frame([date(2026, 9, 10)])

        with patch.object(market_data.yf, "download", side_effect=fake_download):
            df = market_data.fetch_ohlcv("7203", date(2026, 9, 1), date(2026, 9, 10))

        assert captured["auto_adjust"] is False
        assert df["close"].iloc[0] == 1005.0          # 生の終値
        assert df["adjusted_close"].iloc[0] == 1002.0  # 調整済み終値


class TestLoadPriceBasis:
    def test_raw_and_adjusted_return_different_close(self, tmp_path):
        cfg.load("config.yaml")
        cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
        db.init()

        df = pd.DataFrame(
            {
                "open": [1000.0], "high": [1010.0], "low": [990.0],
                "close": [1005.0], "adjusted_close": [502.5], "volume": [100000],
            },
            index=[date(2026, 9, 10)],
        )
        df.index.name = "date"
        market_data.upsert_ohlcv("7203", df)

        raw = market_data.load_ohlcv("7203", price_basis="raw")
        adj = market_data.load_ohlcv("7203", price_basis="adjusted")
        assert raw["close"].iloc[0] == 1005.0
        assert adj["close"].iloc[0] == 502.5

    def test_default_is_adjusted(self, tmp_path):
        cfg.load("config.yaml")
        cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
        db.init()

        df = pd.DataFrame(
            {
                "open": [1000.0], "high": [1010.0], "low": [990.0],
                "close": [1005.0], "adjusted_close": [502.5], "volume": [100000],
            },
            index=[date(2026, 9, 10)],
        )
        df.index.name = "date"
        market_data.upsert_ohlcv("7203", df)

        assert market_data.load_ohlcv("7203")["close"].iloc[0] == 502.5
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_market_data_freshness.py -v`
Expected: FAIL — `assert True is False`（`auto_adjust=True` のまま）および `KeyError: 'adjusted_close'`

- [ ] **Step 3: 実装を修正**

`fetch_ohlcv` の `yf.download` 呼び出しを `auto_adjust=False` に変え、列名の正規化に「Adj Close」の対応を足す。`df.columns = [c.lower() for c in df.columns]` の直後に次を挿入する。

```python
    df.columns = [c.lower() for c in df.columns]
    # auto_adjust=False では "Adj Close" が来る。列名を DB のカラム名に揃える。
    # 生OHLC（株数・必要資金の算定用）と調整済み終値（特徴量・リターン用）を分けて持つ。
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adjusted_close"})
    if "adjusted_close" not in df.columns:
        # 提供側が調整済み終値を返さない場合は生の終値で埋める（用途の分離は維持する）
        df["adjusted_close"] = df["close"]
```

`upsert_ohlcv` の2箇所（更新側と新規側）で `adjusted_close` に `close` を書いている部分を、取得した調整値を使うよう変える。

```python
                rec.adjusted_close = float(row.get("adjusted_close", row.get("close", 0)))
```

```python
                    adjusted_close=float(row.get("adjusted_close", row.get("close", 0))),
```

`load_ohlcv` に `price_basis` を足す。

```python
def load_ohlcv(symbol: str, limit: int = 500,
               price_basis: str = "adjusted") -> pd.DataFrame:
    """DBからOHLCVを読み込みDataFrameで返す（最新limit件を時系列昇順で返す）。

    price_basis:
      "adjusted" … 調整済み終値を close として返す（特徴量・リターン計算用）
      "raw"      … 生の終値を close として返す（株数・単元・必要資金・現金の算定用）

    分けないと、1対2分割で過去価格が半値に調整された銘柄について
    「当時100株買えたか」の判定が変わる。
    """
    if price_basis not in ("adjusted", "raw"):
        raise ValueError(f"price_basis は 'adjusted' か 'raw': {price_basis}")
    with get_session() as session:
        rows = list(reversed(session.scalars(
            select(OHLCV).where(OHLCV.symbol == symbol)
            .order_by(OHLCV.date.desc())
            .limit(limit)
        ).all()))
    if not rows:
        return pd.DataFrame()
    data = [
        {
            "date": r.date,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": (r.adjusted_close or r.close) if price_basis == "adjusted" else r.close,
            "volume": r.volume,
        }
        for r in rows
    ]
    df = pd.DataFrame(data).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df
```

既定を `"adjusted"` にしたので既存の呼び出し側（`trading.py` / `engine.py` / `risk/manager.py`）は変更不要である。

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_market_data_freshness.py -v`
Expected: PASS（5件）

- [ ] **Step 5: 既存テストの回帰を確認**

Run: `pytest tests/ -q`
Expected: 変更前と同じ結果（失敗が増えていないこと）

- [ ] **Step 6: コミット**

```bash
git add src/data/market_data.py tests/test_market_data_freshness.py
git commit -m "fix(data): 生OHLCと調整済み終値が同じ値で保存され用途を分けられなかった問題を修正"
```

---

## Task 4: 分割イベントの取得と不変条件

**Files:**
- Modify: `src/data/database.py`（`CorporateAction` テーブル追加）
- Modify: `src/data/market_data.py`（`fetch_splits` / `upsert_splits` / `split_factor_between` 追加）
- Test: `tests/test_corporate_actions.py`

**Interfaces:**
- Consumes: `src/data/database.get_session`
- Produces:
  - `CorporateAction` モデル: `symbol: str`, `date: date`, `action_type: str`（`"SPLIT"`）, `ratio: float`
  - `fetch_splits(symbol: str) -> pd.Series` — index=date, value=分割比率
  - `upsert_splits(symbol: str, splits: pd.Series) -> int`
  - `split_factor_between(symbol: str, start: date, end: date) -> float` — 期間内の分割比率の積

**設計上の注記:** spec §5 は分割比率を「調整済み終値と生の終値の比の変化」から導出するとしていたが、Yahoo の生の終値は分割については既に調整済みであるため、この比の変化は配当しか捉えない。分割イベントそのものを返す `yfinance.Ticker.splits` を唯一の権威とする。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_corporate_actions.py` を新規作成する。

```python
"""分割イベントの取得と、分割で評価額が増えないことの検証

背景: auto_adjust で過去価格が遡って調整されるため、調整価格のまま
「当時100株買えたか」を判定すると別の結果になる。分割比率は価格比からは
導出できない（Yahooの生終値も分割調整済みのため）ので、分割イベントの
API を唯一の権威とする。
"""
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core import config as cfg
from src.data import database as db
from src.data import market_data


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


class TestFetchSplits:
    def test_returns_split_series(self):
        fake_ticker = MagicMock()
        fake_ticker.splits = pd.Series(
            [2.0], index=pd.to_datetime([date(2026, 6, 1)])
        )
        with patch.object(market_data.yf, "Ticker", return_value=fake_ticker):
            splits = market_data.fetch_splits("7203")
        assert list(splits.values) == [2.0]
        assert splits.index[0] == date(2026, 6, 1)

    def test_empty_when_no_splits(self):
        fake_ticker = MagicMock()
        fake_ticker.splits = pd.Series(dtype=float)
        with patch.object(market_data.yf, "Ticker", return_value=fake_ticker):
            splits = market_data.fetch_splits("7203")
        assert len(splits) == 0


class TestSplitFactor:
    def test_factor_is_product_of_splits_in_range(self, isolated_db):
        market_data.upsert_splits("7203", pd.Series(
            [2.0, 3.0], index=[date(2026, 6, 1), date(2026, 7, 1)]
        ))
        assert market_data.split_factor_between(
            "7203", date(2026, 5, 1), date(2026, 8, 1)) == 6.0

    def test_factor_is_one_when_no_split_in_range(self, isolated_db):
        market_data.upsert_splits("7203", pd.Series(
            [2.0], index=[date(2026, 6, 1)]
        ))
        assert market_data.split_factor_between(
            "7203", date(2026, 7, 1), date(2026, 8, 1)) == 1.0


class TestSplitInvariant:
    def test_split_alone_does_not_change_valuation(self, isolated_db):
        """1対2分割で株数は2倍・価格は半値になり、評価額は変わらない"""
        market_data.upsert_splits("7203", pd.Series(
            [2.0], index=[date(2026, 6, 1)]
        ))
        factor = market_data.split_factor_between(
            "7203", date(2026, 5, 1), date(2026, 7, 1))

        before_qty, before_price = 100, 1000.0
        after_qty = int(before_qty * factor)
        after_price = before_price / factor

        assert after_qty == 200
        assert after_price == 500.0
        assert after_qty * after_price == before_qty * before_price
```

冒頭の import に `import pytest` を足すこと。

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_corporate_actions.py -v`
Expected: FAIL — `AttributeError: module 'src.data.market_data' has no attribute 'fetch_splits'`

- [ ] **Step 3: モデルを追加**

`src/data/database.py` の `class OHLCV` の直後（`src/data/database.py:36` の後）に追加する。

```python
class CorporateAction(Base):
    """企業行動（分割・併合）。

    調整済み価格からは分割比率を導出できない（Yahooの生終値も分割調整済みの
    ため、生値と調整値の比は配当しか表さない）。分割イベントそのものを
    権威ある情報として保存し、株数・必要資金の算定に使う。
    """
    __tablename__ = "corporate_actions"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    action_type = Column(String(10), nullable=False)  # "SPLIT"
    ratio = Column(Float)  # 1対2分割なら 2.0

    __table_args__ = (
        Index("ix_corporate_actions_symbol_date", "symbol", "date", "action_type",
              unique=True),
    )
```

`Date` / `Float` / `Index` / `String` / `Integer` / `Column` はいずれもこのファイルで import 済み。

- [ ] **Step 4: 取得・保存・集計を実装**

`src/data/market_data.py` の `upsert_ohlcv` の後に追加する。import に `CorporateAction` を足す（`from src.data.database import OHLCV, get_session` → `from src.data.database import OHLCV, CorporateAction, get_session`）。

```python
def fetch_splits(symbol: str) -> pd.Series:
    """yfinanceから分割イベントを取得する（index=date, value=比率）。

    価格系列の比からは分割を復元できないため、イベントAPIを唯一の権威とする。
    取得に失敗した場合は空のSeriesを返す（分割なしと同義に扱う）。
    """
    try:
        raw = yf.Ticker(_to_yf_symbol(symbol)).splits
    except Exception as e:
        logger.warning(f"分割イベント取得失敗: {symbol} {e}")
        return pd.Series(dtype=float)
    if raw is None or len(raw) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(raw.values, index=pd.to_datetime(raw.index).date, dtype=float)
    s.index.name = "date"
    return s


def upsert_splits(symbol: str, splits: pd.Series) -> int:
    """分割イベントをDBにupsertする。新規追加件数を返す。"""
    if splits is None or len(splits) == 0:
        return 0
    with get_session() as session:
        existing = {
            r.date: r
            for r in session.scalars(
                select(CorporateAction).where(
                    CorporateAction.symbol == symbol,
                    CorporateAction.action_type == "SPLIT",
                )
            ).all()
        }
        added = 0
        for dt, ratio in splits.items():
            if dt in existing:
                existing[dt].ratio = float(ratio)
            else:
                session.add(CorporateAction(
                    symbol=symbol, date=dt,
                    action_type="SPLIT", ratio=float(ratio),
                ))
                added += 1
        session.commit()
    return added


def split_factor_between(symbol: str, start: date, end: date) -> float:
    """start（含む）～end（含む）の間に起きた分割比率の積を返す。

    1対2分割が1回なら 2.0。分割が無ければ 1.0。
    「当時100株だった建玉が今何株か」「当時の株価が今いくらに調整されているか」を
    対応付けるのに使う。
    """
    with get_session() as session:
        rows = session.scalars(
            select(CorporateAction).where(
                CorporateAction.symbol == symbol,
                CorporateAction.action_type == "SPLIT",
                CorporateAction.date >= start,
                CorporateAction.date <= end,
            )
        ).all()
    factor = 1.0
    for r in rows:
        if r.ratio:
            factor *= float(r.ratio)
    return factor
```

- [ ] **Step 5: テストを実行して成功を確認**

Run: `pytest tests/test_corporate_actions.py -v`
Expected: PASS（5件）

- [ ] **Step 6: コミット**

```bash
git add src/data/database.py src/data/market_data.py tests/test_corporate_actions.py
git commit -m "feat(data): 分割イベントの保存と分割比率の集計を追加"
```

---

## Task 5: update_symbol が最終足の状態を返す

**Files:**
- Modify: `src/data/market_data.py:87-93`（`update_symbol`）
- Test: `tests/test_market_data_freshness.py`

**Interfaces:**
- Consumes: Task 1 の `bar_status.classify`、Task 3 の `fetch_ohlcv`、Task 4 の `fetch_splits` / `upsert_splits`
- Produces: `update_symbol(symbol: str, years: int = 3, now: datetime | None = None) -> BarStatus`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_market_data_freshness.py` の末尾に追記する（必要な import は Task 3 Step 1 で置き換えた冒頭ブロックに含まれている）。

```python
class TestUpdateSymbolReturnsStatus:
    def test_returns_fresh_when_last_bar_matches_session(self, tmp_path):
        cfg.load("config.yaml")
        cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
        db.init()

        now = datetime(2026, 9, 10, 16, 0)
        fake_ticker = MagicMock()
        fake_ticker.splits = pd.Series(dtype=float)
        with patch.object(market_data.yf, "download",
                          return_value=_fake_yf_frame([date(2026, 9, 9), date(2026, 9, 10)])), \
             patch.object(market_data.yf, "Ticker", return_value=fake_ticker):
            st = market_data.update_symbol("7203", years=1, now=now)

        assert st.state == "fresh"
        assert st.last_bar_session == date(2026, 9, 10)

    def test_returns_stale_when_update_yields_old_bar(self, tmp_path):
        cfg.load("config.yaml")
        cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
        db.init()

        now = datetime(2026, 9, 10, 16, 0)
        fake_ticker = MagicMock()
        fake_ticker.splits = pd.Series(dtype=float)
        with patch.object(market_data.yf, "download",
                          return_value=_fake_yf_frame([date(2026, 9, 8), date(2026, 9, 9)])), \
             patch.object(market_data.yf, "Ticker", return_value=fake_ticker):
            st = market_data.update_symbol("7203", years=1, now=now)

        assert st.state == "stale"

    def test_returns_missing_when_fetch_empty(self, tmp_path):
        cfg.load("config.yaml")
        cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
        db.init()

        now = datetime(2026, 9, 10, 16, 0)
        fake_ticker = MagicMock()
        fake_ticker.splits = pd.Series(dtype=float)
        with patch.object(market_data.yf, "download", return_value=pd.DataFrame()), \
             patch.object(market_data.yf, "Ticker", return_value=fake_ticker):
            st = market_data.update_symbol("7203", years=1, now=now)

        assert st.state == "missing"
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_market_data_freshness.py::TestUpdateSymbolReturnsStatus -v`
Expected: FAIL — `AttributeError: 'NoneType' object has no attribute 'state'`（現在は None を返す）

- [ ] **Step 3: 実装を修正**

`src/data/market_data.py` の `update_symbol` を置き換える。ファイル冒頭の import に次を足す。

```python
from datetime import date, datetime, timedelta

from src.core import clock
from src.data.bar_status import BarStatus, classify
```

```python
def update_symbol(symbol: str, years: int = 3,
                  now: Optional[datetime] = None) -> BarStatus:
    """銘柄の過去データと分割イベントを更新し、最終足の状態を返す。

    戻り値の BarStatus は「更新した結果、この銘柄の足は基準セッションまで
    追いついているか」を表す。呼び出し側はこれを見て、確定していない足の銘柄を
    新規候補から除外する（保有保護の退出は止めない）。
    """
    now = now or clock.now()
    end = now.date()
    start = end - timedelta(days=365 * years)
    df = fetch_ohlcv(symbol, start, end)
    added = upsert_ohlcv(symbol, df)
    split_added = upsert_splits(symbol, fetch_splits(symbol))

    last_bar = max(df.index) if not df.empty else _last_stored_session(symbol)
    status = classify(symbol, last_bar, now)
    logger.info(
        f"データ更新: {symbol} 追加={added}件 分割={split_added}件 "
        f"最終足={last_bar} 状態={status.state}"
    )
    return status


def _last_stored_session(symbol: str) -> Optional[date]:
    """DBに保存済みの最終営業日（取得が空だったときの判定に使う）。"""
    with get_session() as session:
        return session.scalar(
            select(func.max(OHLCV.date)).where(OHLCV.symbol == symbol)
        )
```

`Optional` を import に足す（`from typing import Optional`）。`func` は既に `from sqlalchemy import func, select` で import 済み。

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_market_data_freshness.py -v`
Expected: PASS（8件）

- [ ] **Step 5: コミット**

```bash
git add src/data/market_data.py tests/test_market_data_freshness.py
git commit -m "feat(data): データ更新が最終足の確定状態を返すようにした"
```

---

## Task 6: Signal に data_as_of を追加

**Files:**
- Modify: `src/data/database.py:150-158`（`Signal`）
- Modify: `src/services/trading.py:137-146`（`_save_signal`）
- Test: `tests/test_signal_freshness_gate.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `Signal.data_as_of: Date`（nullable）
  - `_save_signal(sig: TradeSignal, data_as_of: date | None = None) -> None`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_signal_freshness_gate.py` を新規作成する。

```python
"""シグナルのデータ基準日記録と、鮮度による新規候補の除外

背景: 生成日時しか持たないため、古い足から作られたシグナルを区別できなかった。
また確定していない足でも新規候補として保存されていた（F01）。
"""
from datetime import date

import pytest
from sqlalchemy import select

from src.core import clock
from src.core import config as cfg
from src.data import database as db
from src.data.database import Signal, get_session
from src.services import trading
from src.strategy.signal import Signal as TradeSignal


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


class TestSignalDataAsOf:
    def test_data_as_of_is_saved_separately_from_generated_at(self, isolated_db):
        sig = TradeSignal(symbol="7203", action="BUY", rule_score=0.3,
                          ml_score=0.1, combined_score=0.4)
        trading._save_signal(sig, data_as_of=date(2026, 9, 10))

        with get_session() as session:
            row = session.scalar(select(Signal))
        assert row.data_as_of == date(2026, 9, 10)
        # 生成日時は保存時刻（clock.now）で、データ基準日とは独立に決まる
        assert row.generated_at is not None
        assert row.generated_at.date() == clock.today()

    def test_data_as_of_is_nullable_for_legacy_rows(self, isolated_db):
        sig = TradeSignal(symbol="7203", action="HOLD", rule_score=0.0,
                          ml_score=0.0, combined_score=0.0)
        trading._save_signal(sig)

        with get_session() as session:
            row = session.scalar(select(Signal))
        assert row.data_as_of is None
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_signal_freshness_gate.py -v`
Expected: FAIL — `TypeError: _save_signal() got an unexpected keyword argument 'data_as_of'`

- [ ] **Step 3: 実装を修正**

`src/data/database.py` の `Signal` に列を足す。

```python
class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    generated_at = Column(DateTime, default=clock.now)  # JST naive（morning_executionのcutoffと統一）
    # このシグナルの根拠にした日足の最終営業日。生成日時とは別物で、
    # 「いつ作ったか」と「いつのデータで作ったか」を区別するために持つ。
    # 移行前に生成された行は不明のため NULL のままにする（推測で埋めない）。
    data_as_of = Column(Date)
    rule_score = Column(Float)
    ml_score = Column(Float)
    combined_score = Column(Float)
    action = Column(String(10))  # "BUY", "SELL", "HOLD"
```

`src/services/trading.py` の `_save_signal` を変更する。`from datetime import date` が無ければ import に足す。

```python
def _save_signal(sig: TradeSignal, data_as_of: Optional[date] = None) -> None:
    with get_session() as session:
        session.add(Signal(
            symbol=sig.symbol,
            rule_score=sig.rule_score,
            ml_score=sig.ml_score,
            combined_score=sig.combined_score,
            action=sig.action,
            data_as_of=data_as_of,
        ))
        session.commit()
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_signal_freshness_gate.py -v`
Expected: PASS（2件）

- [ ] **Step 5: 既存DBへの列追加が自動で走ることを確認**

Run: `pytest tests/test_schema_version.py -v`
Expected: PASS（`_migrate_add_missing_columns` がモデル定義から列を追加する）

- [ ] **Step 6: コミット**

```bash
git add src/data/database.py src/services/trading.py tests/test_signal_freshness_gate.py
git commit -m "feat(data): シグナルにデータ基準日を記録し生成日時と区別した"
```

---

## Task 7: signal_scan の鮮度ゲート

**Files:**
- Modify: `src/services/trading.py:162-170`（`data_update`）、`src/services/trading.py:346-370`（`signal_scan`）
- Test: `tests/test_signal_freshness_gate.py`

**Interfaces:**
- Consumes: Task 5 の `update_symbol` の戻り値、Task 6 の `_save_signal`
- Produces:
  - `TradingServices.data_update(self) -> dict[str, BarStatus]` — 銘柄ごとの最終足状態
  - `TradingServices._bar_states: dict[str, BarStatus]` — 直近の更新結果。`signal_scan` が参照する

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_signal_freshness_gate.py` の冒頭 import ブロックを次に**置き換える**（Task 8 で使う分も含めた最終形）。

```python
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

from src.core import clock
from src.core import config as cfg
from src.data import database as db
from src.data.bar_status import BarStatus
from src.data.database import Signal, get_session
from src.services import trading
from src.strategy.signal import Signal as TradeSignal
```

そのうえで末尾に追記する。

```python
def _status(symbol: str, state: str) -> BarStatus:
    return BarStatus(
        symbol=symbol,
        last_bar_session=date(2026, 9, 10) if state == "fresh" else date(2026, 9, 1),
        observed_at=datetime(2026, 9, 10, 16, 20),
        is_final=state in ("fresh", "stale"),
        state=state,
    )


class TestFreshnessGate:
    def test_stale_symbol_is_excluded_from_new_candidates(self, isolated_db):
        """確定していない/古い足の銘柄は新規候補にしない"""
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        svc._bar_states = {"7203": _status("7203", "stale")}

        assert svc._is_fresh_for_new_candidate("7203") is False

    def test_fresh_symbol_is_allowed(self, isolated_db):
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        svc._bar_states = {"7203": _status("7203", "fresh")}

        assert svc._is_fresh_for_new_candidate("7203") is True

    def test_unknown_symbol_is_not_fresh(self, isolated_db):
        """更新結果が無い銘柄は fresh と判定しない（安全側）"""
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        svc._bar_states = {}

        assert svc._is_fresh_for_new_candidate("7203") is False

    def test_data_update_records_states(self, isolated_db):
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        with patch.object(trading, "update_symbol",
                          return_value=_status("7203", "fresh")), \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]):
            states = svc.data_update()

        assert states["7203"].state == "fresh"
        assert svc._bar_states["7203"].state == "fresh"

    def test_failed_update_is_recorded_as_missing(self, isolated_db):
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        with patch.object(trading, "update_symbol",
                          side_effect=RuntimeError("network down")), \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]):
            states = svc.data_update()

        assert states["7203"].state == "missing"
        assert svc._is_fresh_for_new_candidate("7203") is False
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_signal_freshness_gate.py::TestFreshnessGate -v`
Expected: FAIL — `AttributeError: 'TradingServices' object has no attribute '_is_fresh_for_new_candidate'`

- [ ] **Step 3: 実装を修正**

`src/services/trading.py` の `TradingServices.__init__` の末尾（`src/services/trading.py:157` の後）に1行足す。

```python
        # 直近の data_update で得た銘柄ごとの最終足状態。signal_scan が新規候補の
        # 可否判定に使う。data_update より前に signal_scan が動いた場合は空のままで、
        # そのときは全銘柄が「新規候補にしない」と判定される（安全側）。
        self._bar_states: dict[str, BarStatus] = {}
```

import に `from src.data.bar_status import BarStatus` を足す。

`data_update` を置き換える。

```python
    def data_update(self) -> dict[str, BarStatus]:
        """全リストの銘柄の日足を更新し、銘柄ごとの最終足状態を返す。

        非アクティブリストもMLモデル学習データとして使うため全リストを対象にする。
        更新に失敗した銘柄は missing として記録し、その日の新規候補から外す
        （古い足のまま新しいシグナルを作らないため）。
        """
        years = self.data_conf.get("history_years", 3)
        states: dict[str, BarStatus] = {}
        for sym in watchlist_store.get_all_codes():
            try:
                states[sym] = update_symbol(sym, years=years)
            except Exception as e:
                logger.error(f"データ更新失敗: {sym} {e}")
                states[sym] = BarStatus(
                    symbol=sym, last_bar_session=None,
                    observed_at=clock.now(), is_final=False, state="missing",
                )
        self._bar_states = states
        fresh = sum(1 for s in states.values() if s.state == "fresh")
        logger.info(
            f"データ更新完了: 対象={len(states)}銘柄 確定={fresh} "
            f"未確定={len(states) - fresh}"
        )
        return states

    def _is_fresh_for_new_candidate(self, symbol: str) -> bool:
        """この銘柄を新規候補として扱ってよいか。

        更新結果が無い銘柄は fresh と判定しない（安全側）。
        なおこの判定は**新規の戦略シグナルにのみ**適用する。保有保護の退出
        （損切り・トレーリング）と既存注文の管理は鮮度に関わらず実行する。
        """
        st = self._bar_states.get(symbol)
        return st is not None and st.state == "fresh"
```

`signal_scan` のループ本体を変更する。`for sym in watchlist_store.get_codes():` の直後、`try:` の中の先頭に鮮度ゲートを入れ、`_save_signal` へ基準日を渡す。

```python
        for sym in watchlist_store.get_codes():
            try:
                if not self._is_fresh_for_new_candidate(sym):
                    st = self._bar_states.get(sym)
                    logger.info(
                        f"新規候補から除外: {sym} "
                        f"（最終足={getattr(st, 'last_bar_session', None)} "
                        f"状態={getattr(st, 'state', 'unknown')}）"
                    )
                    continue
                df = load_ohlcv(sym)
                if len(df) < 30:
                    continue
                sig = gen_signal(sym, df, self.model)
                _save_signal(sig, data_as_of=self._bar_states[sym].last_bar_session)
```

以降の行（`if sig.action not in ("BUY", "SELL"): continue` 以下）は変更しない。

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_signal_freshness_gate.py -v`
Expected: PASS（7件）

- [ ] **Step 5: 退出が止まらないことを確認**

`stop_loss_check` は `_is_fresh_for_new_candidate` を呼ばない。差分を確認する。

Run: `git diff src/services/trading.py | grep -n "stop_loss_check" || echo "stop_loss_check は未変更"`
Expected: `stop_loss_check は未変更`

- [ ] **Step 6: 既存テストの回帰を確認**

Run: `pytest tests/ -q`
Expected: 変更前と同じ結果（失敗が増えていないこと）

- [ ] **Step 7: コミット**

```bash
git add src/services/trading.py tests/test_signal_freshness_gate.py
git commit -m "fix(data): 確定していない日足から新規シグナルを作っていた問題を修正"
```

---

## Task 8: データ更新とスキャンをバッチIDで連結

**Files:**
- Modify: `src/services/trading.py`（`data_update` / `signal_scan`）
- Test: `tests/test_signal_freshness_gate.py`

**Interfaces:**
- Consumes: Task 7 の `_bar_states`
- Produces: `TradingServices._data_batch_id: str | None` — `data_update` が払い出し、`signal_scan` がログに記録する

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_signal_freshness_gate.py` の末尾に追記する。

```python
class TestDataBatchId:
    def test_data_update_issues_batch_id(self, isolated_db):
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        with patch.object(trading, "update_symbol",
                          return_value=_status("7203", "fresh")), \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]):
            svc.data_update()

        assert svc._data_batch_id is not None
        assert len(svc._data_batch_id) > 0

    def test_batch_id_changes_between_updates(self, isolated_db):
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        with patch.object(trading, "update_symbol",
                          return_value=_status("7203", "fresh")), \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]):
            svc.data_update()
            first = svc._data_batch_id
            svc.data_update()
            second = svc._data_batch_id

        assert first != second

    def test_batch_id_is_none_before_first_update(self, isolated_db):
        svc = trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                      order_mgr=MagicMock(), model=None)
        assert svc._data_batch_id is None
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_signal_freshness_gate.py::TestDataBatchId -v`
Expected: FAIL — `AttributeError: 'TradingServices' object has no attribute '_data_batch_id'`

- [ ] **Step 3: 実装を修正**

`src/services/trading.py` の `TradingServices.__init__` の `self._bar_states` の直後に足す。

```python
        # data_update が払い出す確定スナップショットの識別子。signal_scan は
        # この ID の入力集合だけを見る。16:00更新→16:20スキャンという時間差にのみ
        # 依存していると、部分更新の最中に入力が入れ替わったことに気付けない。
        self._data_batch_id: Optional[str] = None
```

ファイル冒頭の import に `import uuid` を足す（`Optional` は既に import 済みでなければ `from typing import Optional` も足す）。

`data_update` の `self._bar_states = states` の直前に払い出しを入れる。

```python
        self._data_batch_id = f"{clock.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        self._bar_states = states
```

`data_update` の完了ログにバッチIDを含める。

```python
        logger.info(
            f"データ更新完了: batch={self._data_batch_id} 対象={len(states)}銘柄 "
            f"確定={fresh} 未確定={len(states) - fresh}"
        )
```

`signal_scan` の開始ログを差し替える（`logger.info("シグナルスキャン開始...")` の行）。

```python
        logger.info(f"シグナルスキャン開始... batch={self._data_batch_id}")
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_signal_freshness_gate.py -v`
Expected: PASS（10件）

- [ ] **Step 5: 全テストを実行**

Run: `pytest tests/ -q`
Expected: 変更前と同じ結果（失敗が増えていないこと）

- [ ] **Step 6: コミット**

```bash
git add src/services/trading.py tests/test_signal_freshness_gate.py
git commit -m "feat(data): 更新とスキャンをバッチIDで連結し入力集合を追跡可能にした"
```

---

## 段階A 完了条件の確認

spec §14 の段階A完了条件に対応する検証を行う。

- [ ] **確認1: 全シグナルが data_as_of を持ち、どの営業日の足から作られたか辿れる**

Run: `pytest tests/test_signal_freshness_gate.py::TestSignalDataAsOf -v`
Expected: PASS

- [ ] **確認2: 確定していない足が新規候補として採用されない**

Run: `pytest tests/test_signal_freshness_gate.py::TestFreshnessGate -v`
Expected: PASS

- [ ] **確認3: 1対2分割で株数と価格が整合し、分割だけでは評価額が増えない**

Run: `pytest tests/test_corporate_actions.py::TestSplitInvariant -v`
Expected: PASS

- [ ] **確認4: 引け後のT日更新でT日の足が取得範囲に入る**

Run: `pytest tests/test_market_data_freshness.py::TestFetchBoundary -v`
Expected: PASS

- [ ] **確認5: 既存テストの回帰が無い**

Run: `pytest tests/ -q`
Expected: 段階A着手前と同じ結果

---

## 次の段階

段階Bは `src/strategy/policy.py`（退出ポリシー）と `src/backtest/execution.py`（過去検証アダプタ）の最小部分を同時に作る。`dataset.py` がコスト控除後のラベルを作るために執行アダプタへ依存するためである（spec §13）。段階Bの計画は本計画の完了後に作成する。
