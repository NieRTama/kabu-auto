# ML評価基盤 段階C前半（分割と漏れの遮断）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 全銘柄共通のカレンダー日付で fold を切り、学習側へ未来が入る経路を purge 以外も含めてすべて塞ぐ。

**Architecture:** `validation.py` が「どのイベントを学習に使ってよいか」だけを決める。行番号ではなく**カレンダー日付**で fold 境界を切り、`label_end_at >= 検証開始日` のイベントを学習側から除外する（purge）。purge だけでは塞がらない3経路（一意性重み・前処理・学習窓）も、すべて fold 内で完結させる。モデルの学習・評価・記録は段階C後半の担当で、本計画では一切行わない。

**Tech Stack:** Python 3.11 / pandas 2.1.4 / numpy 1.26.2 / pytest / dataclasses

**Spec:** `docs/superpowers/specs/2026-09-10-ml-evaluation-foundation-design.md`（§7・§12・§14）

**前提:** 段階B後半（`docs/superpowers/plans/2026-09-11-ml-evaluation-stage-b2.md`）が完了していること。本計画は `dataset.EVENT_COLUMNS` / `dataset.STATUS_RESOLVED` / `dataset.uniqueness_weights` / `indicators.FEATURE_COLS` に依存する。

## Global Constraints

- 日時は **JST naive**。現在時刻は `src/core/clock.now()` / `clock.today()` を使い、`datetime.now()` を直接呼ばない。
- **`validation.py` はモデルを学習しない。** 分割・除外・重み・前処理の統計量までを担当し、学習・推論・指標計算は段階C後半に置く。`lightgbm` / `sklearn` を import しない。
- **`validation.py` は設定ファイル・DB・ネットワークに直接触らない。** 必要な値は引数で受け取る。
- **既存の公開関数の挙動を変えない。** `labeling.build_training_set()` / `indicators.build_features()` / `ml_model.train()` / `train_multi()` は本計画で一切変更しない。`src/backtest/engine.py`・`src/services/trading.py` も変更しない。
- **学習側から未来を締め出す。** fold の学習締切より後に確定した情報は、ラベル・重み・前処理の統計量のいずれの経路でも学習側へ入れない。
- ファイルは UTF-8 **BOM無し**・LF で保存する。確認は `git show <rev>:<path>` でコミット済みblobに対して行う。
- テストは `pytest tests/<file>.py -v` で実行する。ネットワークへ出るテストを書かない。
- コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` を付ける。実装者自身のモデル名を書かない。

---

## 背景：purge だけでは塞がらない経路（spec §7）

現行の `ml_model._fit()` は連結した行に `TimeSeriesSplit` を当てており、銘柄Aの2026年が学習側・銘柄Bの2024年が検証側に入る分割が成立する（レビューF02）。日付で切り直すだけでは足りず、次の4経路すべてに同じ締切を適用する必要がある。本計画は 1〜3 を実装し、4 は段階C後半で扱う。

| 経路 | 漏れ方 | 本計画での扱い |
|---|---|---|
| 1. ラベル | 学習イベントのラベルが検証期間の価格で確定する | `label_end_at >= 検証開始日` を purge（Task 2） |
| 2. 一意性重み | 全期間で一度計算した重みに検証側イベントの終了時点が影響する | purge後の学習集合で再計算（Task 4） |
| 3. 前処理 | 標準化・クラス比率を全データで fit する | 学習側だけで fit（Task 5） |
| 4. 期間中の再学習／外側結果の使い方 | 再学習時点で未確定のラベルが入る／外側成績で選択する | **段階C後半** |

---

## File Structure

| ファイル | 責務 |
|---|---|
| `src/strategy/validation.py`（新規） | カレンダー分割・purge・入れ子分割・学習窓・fold内の重み再計算・前処理の統計量。モデルは扱わない |
| `tests/test_validation.py`（新規） | 日付境界分割、purge、embargo、入れ子、学習窓、重み再計算、前処理の fit 範囲、外側 fold の不可侵性 |

### 段階C後半（本計画のスコープ外）

比較対象5モデルの学習と評価、`Prediction` / `PredictionOutcome` テーブル、指標の記録、閾値選択と確率校正、学習窓の実比較は別計画で行う。

---

## Task 1: Fold 型とカレンダー分割

**Files:**
- Create: `src/strategy/validation.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `Fold`（frozen dataclass）: `index: int`, `train_start: date`, `train_end: date`, `val_start: date`, `val_end: date`
  - `sessions_of(events: pd.DataFrame) -> list[date]` — 全銘柄共通の判断セッション（`decision_at` の重複なし昇順）
  - `calendar_folds(events: pd.DataFrame, n_splits: int = 5) -> list[Fold]`

**設計:** セッションの並びを `n_splits + 1` 個の連続した区画に分け、fold *k* は区画 `0..k` を学習、区画 `k+1` を検証にする（拡大窓の walk-forward）。境界は**日付**であり行番号ではない。銘柄をまたいで同じ境界を使うため、銘柄Aの2026年と銘柄Bの2024年が同じ fold の両側に入ることがなくなる。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_validation.py` を新規作成する。

```python
"""分割と漏れの遮断（src/strategy/validation.py）のテスト

全銘柄共通のカレンダー日付でfoldを切り、学習側へ未来が入る経路を
purge以外も含めて塞ぐ（spec §7）。モデルは扱わない。
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.strategy import dataset
from src.strategy import validation


def _events(rows: list[dict]) -> pd.DataFrame:
    """テスト用の最小イベント表。

    rows の各要素は {symbol, decision_at, label_end_at, label} を持つ。
    足りない列は既定値で埋める。
    """
    out = []
    for k, r in enumerate(rows):
        out.append({
            "event_id": f"{r['symbol']}:{r['decision_at']:%Y%m%d}",
            "symbol": r["symbol"],
            "decision_at": r["decision_at"],
            "entry_at": r.get("entry_at", r["decision_at"] + timedelta(days=1)),
            "label_end_at": r.get("label_end_at"),
            "status": r.get("status", dataset.STATUS_RESOLVED
                            if r.get("label") is not None else dataset.STATUS_IMMATURE),
            "label": r.get("label"),
            "net_return": r.get("net_return", 0.01),
            "f1": r.get("f1", float(k)),
            "f2": r.get("f2", float(k) * 2),
        })
    return pd.DataFrame(out)


def _daily_events(symbols: list[str], start: date, n_sessions: int,
                  holding: int = 2) -> pd.DataFrame:
    """各銘柄が n_sessions 日ぶん連続して候補になるイベント表"""
    rows = []
    for i in range(n_sessions):
        d = start + timedelta(days=i)
        for s in symbols:
            rows.append({
                "symbol": s,
                "decision_at": d,
                "label_end_at": d + timedelta(days=holding),
                "label": i % 2,
            })
    return _events(rows)


class TestSessionsOf:
    def test_returns_sorted_unique_sessions(self):
        events = _daily_events(["7203", "9984"], date(2026, 1, 5), 4)
        got = validation.sessions_of(events)
        assert got == [date(2026, 1, 5), date(2026, 1, 6),
                       date(2026, 1, 7), date(2026, 1, 8)]

    def test_is_shared_across_symbols(self):
        """銘柄ごとではなく全銘柄共通のセッション列になる"""
        rows = [
            {"symbol": "7203", "decision_at": date(2026, 1, 5), "label": 1},
            {"symbol": "9984", "decision_at": date(2026, 1, 6), "label": 0},
        ]
        assert validation.sessions_of(_events(rows)) == [
            date(2026, 1, 5), date(2026, 1, 6)]


class TestCalendarFolds:
    def test_produces_requested_number_of_folds(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 60)
        folds = validation.calendar_folds(events, n_splits=5)
        assert len(folds) == 5
        assert [f.index for f in folds] == [0, 1, 2, 3, 4]

    def test_training_always_precedes_validation(self):
        """学習期間は検証期間より前（片方向walk-forward）"""
        events = _daily_events(["7203"], date(2026, 1, 5), 60)
        for f in validation.calendar_folds(events, n_splits=5):
            assert f.train_end < f.val_start
            assert f.val_start <= f.val_end

    def test_training_window_expands(self):
        """拡大窓: foldが進むほど学習期間の終わりが後ろへ伸びる"""
        events = _daily_events(["7203"], date(2026, 1, 5), 60)
        folds = validation.calendar_folds(events, n_splits=5)
        ends = [f.train_end for f in folds]
        assert ends == sorted(ends)
        assert ends[0] < ends[-1]

    def test_validation_periods_do_not_overlap(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 60)
        folds = validation.calendar_folds(events, n_splits=5)
        for a, b in zip(folds, folds[1:]):
            assert a.val_end < b.val_start

    def test_boundaries_are_dates_not_row_positions(self):
        """銘柄数が変わってもfold境界の日付は変わらない（行番号ではない）"""
        one = validation.calendar_folds(
            _daily_events(["7203"], date(2026, 1, 5), 60), n_splits=5)
        many = validation.calendar_folds(
            _daily_events(["7203", "9984", "6758"], date(2026, 1, 5), 60), n_splits=5)
        assert [(f.train_end, f.val_start, f.val_end) for f in one] == \
               [(f.train_end, f.val_start, f.val_end) for f in many]

    def test_raises_when_too_few_sessions(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 3)
        with pytest.raises(ValueError, match="セッション"):
            validation.calendar_folds(events, n_splits=5)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.strategy.validation'`

- [ ] **Step 3: 実装を書く**

`src/strategy/validation.py` を新規作成する。

```python
"""分割と漏れの遮断 — どのイベントを学習に使ってよいかを決める。

現行の ml_model._fit() は連結した行に TimeSeriesSplit を当てており、
銘柄Aの2026年が学習側・銘柄Bの2024年が検証側に入る分割が成立する
（レビューF02）。本モジュールは fold 境界を**全銘柄共通のカレンダー日付**で
切り、学習側へ未来が入る経路を purge 以外も含めて塞ぐ。

**モデルは扱わない。** 分割・除外・重み・前処理の統計量までを担当し、
学習・推論・指標計算は段階C後半に置く。lightgbm / sklearn を import しない。
"""
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """1つの分割。境界は日付であり行番号ではない。

    train_end は**学習締切**＝この分割で学習を実行する時点である。
    「この日までに判断されたイベント」ではなく
    **「この日までに判断され、かつこの日までにラベルが確定したイベント」**だけが
    学習候補になる（split_events の docstring を参照）。

    val_start / val_end はこのfoldの検証期間。片方向walk-forward専用のため
    常に train_start <= train_end < val_start <= val_end が成り立つ。
    この不変条件は __post_init__ で強制する。成り立たないFoldは
    purge・embargoの契約が定義できないので、黙って通さず例外にする。
    """
    index: int
    train_start: date
    train_end: date
    val_start: date
    val_end: date

    def __post_init__(self) -> None:
        if not (self.train_start <= self.train_end):
            raise ValueError(
                f"train_start <= train_end でなければならない: "
                f"{self.train_start} > {self.train_end}")
        if not (self.val_start <= self.val_end):
            raise ValueError(
                f"val_start <= val_end でなければならない: "
                f"{self.val_start} > {self.val_end}")
        if not (self.train_end < self.val_start):
            # 学習期間が検証期間を跨ぐ・後ろまで伸びる分割は片方向walk-forwardでは
            # 作れない。ここを通すと「未来で学習して過去を検証する」分割が
            # 静かに成立してしまう（外部レビューR23）。
            raise ValueError(
                f"片方向walk-forwardでは train_end < val_start が必要: "
                f"train_end={self.train_end} val_start={self.val_start}")


def sessions_of(events: pd.DataFrame) -> list:
    """全銘柄共通の判断セッション（decision_at の重複なし昇順）。

    銘柄ごとに別々の境界を使うと、同じ暦日が fold の両側に現れうる。
    分割の基準は常にこの共通セッション列にする。
    """
    values = pd.to_datetime(events["decision_at"].dropna()).dt.date.unique()
    return sorted(values)


def calendar_folds(events: pd.DataFrame, n_splits: int = 5) -> list:
    """拡大窓の walk-forward 分割を日付で作る。

    セッション列を n_splits+1 個の連続した区画に分け、fold k は区画 0..k を
    学習、区画 k+1 を検証にする。銘柄数や1日あたりの候補数が変わっても
    境界の日付は変わらない（行番号に依存しないため）。
    """
    sessions = sessions_of(events)
    n_chunks = n_splits + 1
    if len(sessions) < n_chunks:
        raise ValueError(
            f"分割に足りるセッションがありません: {len(sessions)}日 < {n_chunks}区画"
        )

    bounds = np.linspace(0, len(sessions), n_chunks + 1).astype(int)
    folds = []
    for k in range(n_splits):
        train_lo, train_hi = bounds[0], bounds[k + 1]
        val_lo, val_hi = bounds[k + 1], bounds[k + 2]
        if val_hi <= val_lo:
            continue
        folds.append(Fold(
            index=k,
            train_start=sessions[train_lo],
            train_end=sessions[train_hi - 1],
            val_start=sessions[val_lo],
            val_end=sessions[val_hi - 1],
        ))
    return folds
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_validation.py -v`
Expected: PASS（8件）

- [ ] **Step 5: BOM確認とコミット**

Run: `head -c 3 src/strategy/validation.py | xxd`（`2222 22` を確認。`efbb bf` なら下記で除去）

```python
for p in ["src/strategy/validation.py", "tests/test_validation.py"]:
    with open(p, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        with open(p, "wb") as f:
            f.write(data[3:])
```

```bash
git add src/strategy/validation.py tests/test_validation.py
git commit -m "$(cat <<'EOF'
feat(strategy): カレンダー日付によるwalk-forward分割を追加

現行は連結した行にTimeSeriesSplitを当てており、銘柄Aの2026年が学習側・
銘柄Bの2024年が検証側に入る分割が成立していた。fold境界を全銘柄共通の
判断セッション（日付）で切り、銘柄数や1日あたりの候補数が変わっても
境界が動かないようにする。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: purge を効かせて fold を適用する

**Files:**
- Modify: `src/strategy/validation.py`（`split_events` を追加）
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: Task 1 の `Fold` / `sessions_of`、`dataset.STATUS_RESOLVED`
- Produces: `split_events(events: pd.DataFrame, fold: Fold, *, embargo_sessions: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]` — `(学習イベント, 検証イベント)`

**purge の規則（spec §7）:** 学習側から `label_end_at >= fold.val_start` のイベントを除外する。学習イベントのラベルが検証期間の価格で確定していると、検証期間の情報が学習側へ入るため。

**embargo:** 既定で無効（`embargo_sessions=0`）。有効にすると `val_start` の**直前** `embargo_sessions` セッションぶんの判断日を学習側から除外する。

> **定義を変更した理由（外部レビューR23・2026-09-12）。** 当初案は「`val_end` の**後ろ**
> `embargo_sessions` セッションを除外する」としていたが、片方向walk-forwardでは
> `train_end < val_start <= val_end` が常に成り立つため、`val_end` より後の判断日は
> 学習側に**一件も入り得ない**。つまり当初の embargo は到達不能なコードであり、
> それを検証するテストも成立しなかった（後述）。
>
> 一方、検証開始の直前にある学習イベントは、purge を通した後でも系列相関で
> 検証期間の結果と相関する。片方向walk-forwardで意味を持つ embargo は
> **「検証開始前に空白セッションを置く」方向だけ**である。こちらへ揃える。

**ラベルの無いイベント:** `status != STATUS_RESOLVED` のイベントは学習にも検証にも使わない。ラベルが無く、指標も計算できないため。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_validation.py` の末尾に追記する。

```python
class TestSplitEventsBasics:
    def test_validation_is_the_fold_window(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=0)
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, val = validation.split_events(events, fold)
        assert val["decision_at"].min() >= fold.val_start
        assert val["decision_at"].max() <= fold.val_end

    def test_training_stops_at_the_cutoff(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=0)
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, val = validation.split_events(events, fold)
        assert train["decision_at"].max() <= fold.train_end

    def test_unresolved_events_are_excluded_from_both_sides(self):
        """ラベルの無いイベントは学習にも検証にも使わない"""
        rows = [
            {"symbol": "7203", "decision_at": date(2026, 1, 5),
             "label_end_at": date(2026, 1, 6), "label": 1},
            {"symbol": "7203", "decision_at": date(2026, 1, 6),
             "label_end_at": None, "label": None,
             "status": dataset.STATUS_IMMATURE},
            {"symbol": "7203", "decision_at": date(2026, 1, 7),
             "label_end_at": date(2026, 1, 8), "label": 0},
            {"symbol": "7203", "decision_at": date(2026, 1, 8),
             "label_end_at": date(2026, 1, 9), "label": 1},
        ]
        events = _events(rows)
        fold = validation.Fold(
            index=0, train_start=date(2026, 1, 5), train_end=date(2026, 1, 6),
            val_start=date(2026, 1, 7), val_end=date(2026, 1, 8))
        train, val = validation.split_events(events, fold)
        assert train["status"].eq(dataset.STATUS_RESOLVED).all()
        assert val["status"].eq(dataset.STATUS_RESOLVED).all()
        assert date(2026, 1, 6) not in set(train["decision_at"])


class TestPurge:
    def _overlapping_events(self):
        """1/6 に判断したイベントだけ、ラベルが検証期間（1/8〜）まで伸びる"""
        rows = [
            {"symbol": "7203", "decision_at": date(2026, 1, 5),
             "label_end_at": date(2026, 1, 6), "label": 1},
            {"symbol": "7203", "decision_at": date(2026, 1, 6),
             "label_end_at": date(2026, 1, 9), "label": 0},   # 検証期間へ食い込む
            {"symbol": "7203", "decision_at": date(2026, 1, 7),
             "label_end_at": date(2026, 1, 7), "label": 1},
            {"symbol": "7203", "decision_at": date(2026, 1, 8),
             "label_end_at": date(2026, 1, 9), "label": 0},
        ]
        return _events(rows), validation.Fold(
            index=0, train_start=date(2026, 1, 5), train_end=date(2026, 1, 7),
            val_start=date(2026, 1, 8), val_end=date(2026, 1, 9))

    def test_purges_events_whose_label_reaches_validation(self):
        """ラベルが検証開始日以降に確定する学習イベントを除外する"""
        events, fold = self._overlapping_events()
        train, _ = validation.split_events(events, fold)
        assert date(2026, 1, 6) not in set(train["decision_at"])

    def test_keeps_events_resolved_before_validation(self):
        events, fold = self._overlapping_events()
        train, _ = validation.split_events(events, fold)
        assert date(2026, 1, 5) in set(train["decision_at"])
        assert date(2026, 1, 7) in set(train["decision_at"])

    def test_no_training_label_resolves_after_the_cutoff(self):
        """不変条件: 学習側のどのラベルも学習締切までに確定している"""
        events, fold = self._overlapping_events()
        train, _ = validation.split_events(events, fold)
        assert (train["label_end_at"] <= fold.train_end).all()

    def test_excludes_label_resolved_after_cutoff_even_when_gap_before_validation(self):
        """判断は締切前でもラベル確定が締切後なら学習に使えない

        分割日は候補イベントのある営業日から作るため、train_end と val_start の
        間に候補の無い空白期間ができる。そこへラベル確定日が落ちると、
        「label_end_at < val_start」という条件では素通りしてしまう。
        学習締切の時点では誰も知り得ない情報なので除外されなければならない
        （外部レビューR06の反例をそのまま置く）。
        """
        rows = [
            # 1/5に判断し1/8に確定。train_end=1/5 の時点では結果が分からない
            {"symbol": "7203", "decision_at": date(2026, 1, 5),
             "label_end_at": date(2026, 1, 8), "label": 1},
            # 1/5に判断し1/5に確定。こちらは締切時点で観測できる
            {"symbol": "9984", "decision_at": date(2026, 1, 5),
             "label_end_at": date(2026, 1, 5), "label": 0},
            {"symbol": "7203", "decision_at": date(2026, 1, 12),
             "label_end_at": date(2026, 1, 12), "label": 1},
        ]
        events = _events(rows)
        fold = validation.Fold(
            index=0, train_start=date(2026, 1, 5), train_end=date(2026, 1, 5),
            val_start=date(2026, 1, 12), val_end=date(2026, 1, 12))

        train, _ = validation.split_events(events, fold)
        symbols = set(train["symbol"])
        assert "7203" not in symbols   # 締切後に確定するので除外
        assert "9984" in symbols       # 締切までに確定するので残る

    def test_purge_applies_to_every_generated_fold(self):
        events = _daily_events(["7203", "9984"], date(2026, 1, 5), 60, holding=5)
        for fold in validation.calendar_folds(events, n_splits=5):
            train, _ = validation.split_events(events, fold)
            if len(train) == 0:
                continue
            assert (train["label_end_at"] <= fold.train_end).all()


class TestFoldContract:
    def test_rejects_fold_whose_training_spans_the_validation_period(self):
        """検証期間を跨ぐ・後ろまで伸びる分割は作れない

        片方向walk-forward専用なので train_end < val_start が不変条件である。
        これを満たさないFoldを黙って受け取ると、purge・embargoの契約が
        定義できないまま「未来で学習して過去を検証する」分割が成立する
        （外部レビューR23）。
        """
        with pytest.raises(ValueError, match="train_end < val_start"):
            validation.Fold(
                index=0, train_start=date(2026, 1, 5), train_end=date(2026, 1, 12),
                val_start=date(2026, 1, 7), val_end=date(2026, 1, 8))

    def test_rejects_inverted_validation_period(self):
        with pytest.raises(ValueError, match="val_start <= val_end"):
            validation.Fold(
                index=0, train_start=date(2026, 1, 5), train_end=date(2026, 1, 6),
                val_start=date(2026, 1, 9), val_end=date(2026, 1, 8))

    def test_rejects_inverted_training_period(self):
        with pytest.raises(ValueError, match="train_start <= train_end"):
            validation.Fold(
                index=0, train_start=date(2026, 1, 7), train_end=date(2026, 1, 6),
                val_start=date(2026, 1, 9), val_end=date(2026, 1, 10))

    def test_every_generated_fold_satisfies_the_contract(self):
        """calendar_folds が作る分割は全てこの契約を満たす"""
        events = _daily_events(["7203", "9984"], date(2026, 1, 5), 60, holding=5)
        folds = validation.calendar_folds(events, n_splits=5)
        assert len(folds) > 0
        for fold in folds:
            assert fold.train_start <= fold.train_end < fold.val_start <= fold.val_end


class TestEmbargo:
    """embargo は「検証開始の直前に空白セッションを置く」方向にだけ存在する。

    `val_end` より後ろを外す向きは片方向walk-forwardでは到達不能なので
    実装しない（外部レビューR23）。
    """

    def _events_with_gap(self):
        rows = [
            {"symbol": "7203", "decision_at": date(2026, 1, 5),
             "label_end_at": date(2026, 1, 5), "label": 1},
            {"symbol": "7203", "decision_at": date(2026, 1, 6),
             "label_end_at": date(2026, 1, 6), "label": 0},
            {"symbol": "7203", "decision_at": date(2026, 1, 7),
             "label_end_at": date(2026, 1, 7), "label": 1},   # val_start直前
            {"symbol": "7203", "decision_at": date(2026, 1, 8),
             "label_end_at": date(2026, 1, 8), "label": 0},   # 検証側
        ]
        return _events(rows), validation.Fold(
            index=0, train_start=date(2026, 1, 5), train_end=date(2026, 1, 7),
            val_start=date(2026, 1, 8), val_end=date(2026, 1, 8))

    def test_default_is_disabled(self):
        events, fold = self._events_with_gap()
        train, _ = validation.split_events(events, fold)
        assert date(2026, 1, 7) in set(train["decision_at"])

    def test_excludes_the_sessions_immediately_before_validation(self):
        """embargo_sessions=1 なら val_start 直前の1営業日を学習から外す"""
        events, fold = self._events_with_gap()
        train, _ = validation.split_events(events, fold, embargo_sessions=1)
        decisions = set(train["decision_at"])
        assert date(2026, 1, 7) not in decisions   # 直前1営業日は外れる
        assert date(2026, 1, 6) in decisions       # その前は残る
        assert date(2026, 1, 5) in decisions

    def test_embargo_of_two_removes_two_sessions(self):
        events, fold = self._events_with_gap()
        train, _ = validation.split_events(events, fold, embargo_sessions=2)
        decisions = set(train["decision_at"])
        assert date(2026, 1, 7) not in decisions
        assert date(2026, 1, 6) not in decisions
        assert date(2026, 1, 5) in decisions

    def test_embargo_never_touches_the_validation_set(self):
        events, fold = self._events_with_gap()
        _, val_off = validation.split_events(events, fold)
        _, val_on = validation.split_events(events, fold, embargo_sessions=2)
        assert list(val_off["event_id"]) == list(val_on["event_id"])
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.validation' has no attribute 'split_events'`

- [ ] **Step 3: 実装を追加**

`src/strategy/validation.py` の末尾に追加する。import に `from src.strategy.dataset import STATUS_RESOLVED` を足す。

```python
def _resolved(events: pd.DataFrame) -> pd.DataFrame:
    """ラベルが確定したイベントだけを残す。

    未成熟・未約定・欠損はラベルが無く指標も計算できないため、学習にも
    検証にも使わない（損失0として扱わないのと同じ理由）。
    """
    return events[events["status"] == STATUS_RESOLVED]


def split_events(events: pd.DataFrame, fold: Fold, *,
                 embargo_sessions: int = 0) -> tuple:
    """fold を適用して (学習イベント, 検証イベント) を返す。

    **purge**: 学習側から `label_end_at > fold.train_end` のイベントを除外する。

    `train_end` は学習締切＝この分割で学習を実行する時点である。判断日だけを
    締切で切ると、「判断は締切前だがラベルの確定は締切後」というイベントが
    学習に入る。それは学習実行時点では観測できない情報であり、
    walk-forward が答えようとしている「その時点で何を知り得たか」を壊す。

    `train_end < val_start` を Fold が保証しているので、この条件は
    「ラベルが検証期間へ食い込む学習イベントを落とす」という従来の purge
    （spec §7 の `label_end_at >= val_start` を除外）を必ず含む。厳しいほうを採る。

    **embargo**: 既定で無効。有効にすると `val_start` の**直前** embargo_sessions
    セッションぶんの判断日を学習側から外す。purge を通した後でも、検証開始の
    直前にある学習イベントは系列相関で検証期間の結果と相関するため、
    境界に空白セッションを置きたい場合に使う。

    片方向walk-forwardでは `val_end` より後の判断日が学習側に入ることは
    構造的に無いので、「検証期間の後ろを外す」向きの embargo は実装しない
    （到達不能なコードになる）。両側を学習に使う分割を将来採用するなら、
    purge の契約から分けて設計し直すこと。
    """
    resolved = _resolved(events)

    val = resolved[
        (resolved["decision_at"] >= fold.val_start)
        & (resolved["decision_at"] <= fold.val_end)
    ]

    train = resolved[
        (resolved["decision_at"] >= fold.train_start)
        & (resolved["decision_at"] <= fold.train_end)
    ]
    # purge = 学習締切での実現可能性。
    # decision_at だけを train_end で切ると、判断は締切前でもラベルが締切後に
    # 確定するイベントが学習に入る。例: train_end=1/5, val_start=1/12 のとき、
    # 1/5判断・1/8確定のラベルは「1/5時点では誰も知り得ない」のに通ってしまう
    # （外部レビューR06）。学習締切とは学習を実行する時点のことなので、
    # ラベル確定日も同じ締切で切る。
    #
    # train_end < val_start が Fold で保証されているため、この条件は
    # 従来の purge（label_end_at < val_start）を必ず含む。より厳しいほうだけ残す。
    train = train[train["label_end_at"] <= fold.train_end]

    if embargo_sessions > 0:
        # 検証開始の直前セッションを学習側から落とす（前方ギャップ）。
        # 特徴量の系列相関は purge を通した後でも残るため、境界に空白を置く。
        sessions = sessions_of(resolved)
        before = [s for s in sessions if s < fold.val_start]
        embargoed = set(before[-embargo_sessions:])
        if embargoed:
            train = train[~train["decision_at"].isin(embargoed)]

    return train.reset_index(drop=True), val.reset_index(drop=True)
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_validation.py -v`
Expected: PASS（25件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/validation.py tests/test_validation.py
git commit -m "$(cat <<'EOF'
feat(strategy): 学習締切でラベル確定日まで切るfold適用を追加

train_endは学習を実行する時点である。判断日だけを締切で切ると、判断は締切前
だがラベルの確定が締切後というイベントが学習に入り、その時点では観測できない
情報で学習することになる。label_end_atも同じ締切で切る。train_end<val_startを
Foldが保証するため、この条件は従来のpurgeを必ず含む。
ラベルの無いイベントは学習にも検証にも使わない。
embargoはval_start直前の空白セッションとして定義する。val_endより後ろを外す
向きは片方向walk-forwardでは到達不能なので実装しない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 入れ子分割（外側と内側を分ける）

**Files:**
- Modify: `src/strategy/validation.py`（`inner_folds` を追加）
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: Task 1・2 の `Fold` / `calendar_folds` / `split_events`
- Produces: `inner_folds(train_events: pd.DataFrame, n_splits: int = 3) -> list[Fold]`

**背景（spec §7）:** 現行は early stopping に使った検証データでそのまま CV 指標を出しているため、報告値が楽観に寄っている（`ml_model.py:147-160`）。外側 fold は最終評価専用にして一切触らず、early stopping・閾値選択・確率校正・学習窓選択・戦略選択は**外側の学習側をさらに分割した内側 fold** で行う。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_validation.py` の末尾に追記する。

```python
class TestInnerFolds:
    def _outer(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 120, holding=1)
        fold = validation.calendar_folds(events, n_splits=5)[3]
        train, val = validation.split_events(events, fold)
        return events, fold, train, val

    def test_inner_folds_live_inside_outer_training(self):
        """内側foldは外側の学習期間の中だけで完結する"""
        _, outer, train, _ = self._outer()
        for inner in validation.inner_folds(train, n_splits=3):
            assert inner.train_start >= train["decision_at"].min()
            assert inner.val_end <= train["decision_at"].max()
            assert inner.val_end < outer.val_start

    def test_inner_folds_never_touch_outer_validation(self):
        """内側foldのどの期間も外側の検証期間に重ならない（外側は最終評価専用）"""
        _, outer, train, _ = self._outer()
        for inner in validation.inner_folds(train, n_splits=3):
            assert inner.val_start < outer.val_start
            assert inner.train_end < outer.val_start

    def test_inner_folds_are_forward_only(self):
        _, _, train, _ = self._outer()
        for inner in validation.inner_folds(train, n_splits=3):
            assert inner.train_end < inner.val_start

    def test_purge_applies_inside_inner_folds_too(self):
        """内側foldにも同じpurgeが効く"""
        _, _, train, _ = self._outer()
        for inner in validation.inner_folds(train, n_splits=3):
            inner_train, _ = validation.split_events(train, inner)
            if len(inner_train) == 0:
                continue
            assert (inner_train["label_end_at"] < inner.val_start).all()

    def test_raises_when_training_too_short(self):
        short = _daily_events(["7203"], date(2026, 1, 5), 2, holding=0)
        with pytest.raises(ValueError, match="セッション"):
            validation.inner_folds(short, n_splits=3)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.validation' has no attribute 'inner_folds'`

- [ ] **Step 3: 実装を追加**

`src/strategy/validation.py` の末尾に追加する。

```python
def inner_folds(train_events: pd.DataFrame, n_splits: int = 3) -> list:
    """外側foldの学習側をさらに分割した内側foldを作る。

    **外側foldは最終評価専用で一切触らない。** early stopping・閾値選択・
    確率校正・学習窓選択・戦略選択はすべてこの内側foldで行う。
    現行は early stopping に使った検証データでそのままCV指標を出しており、
    報告値が楽観に寄っている（ml_model.py:147-160）。

    引数は split_events() が返した学習イベントであること。外側の検証期間は
    そこに含まれていないため、内側foldがそれに触れることは構造的にない。
    """
    return calendar_folds(train_events, n_splits=n_splits)
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_validation.py -v`
Expected: PASS（22件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/validation.py tests/test_validation.py
git commit -m "$(cat <<'EOF'
feat(strategy): 外側と内側を分ける入れ子分割を追加

現行はearly stoppingに使った検証データでそのままCV指標を出しており
報告値が楽観に寄っていた。外側foldを最終評価専用にして触らず、
early stopping・閾値選択・確率校正・選択はすべて外側の学習側を
さらに分割した内側foldで行う。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: fold 内で一意性重みを再計算する

**Files:**
- Modify: `src/strategy/validation.py`（`training_weights` を追加）
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `dataset.uniqueness_weights`
- Produces: `training_weights(train_events: pd.DataFrame) -> np.ndarray`

**背景（spec §7 経路2）:** イベント表全体で一度計算した一意性重みをそのまま各 fold へ流すと、**検証側イベントの終了時点が学習側の重みへ影響する**。段階B後半で `sample_weight` を列として保存しない設計にしたのはこのためで、ここが正しい呼び出し口になる。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_validation.py` の末尾に追記する。

```python
class TestTrainingWeights:
    def test_length_matches_training_events(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=2)
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, _ = validation.split_events(events, fold)
        assert len(validation.training_weights(train)) == len(train)

    def test_weights_are_positive_for_resolved_events(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=2)
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, _ = validation.split_events(events, fold)
        w = validation.training_weights(train)
        assert (w > 0).all()

    def test_recomputed_from_the_training_subset_only(self):
        """全期間で計算した重みとは値が異なる＝fold内で計算し直している。

        差が出るのは学習期間の**末尾**のイベント。先頭付近のイベントは、
        重なり相手が全期間にも部分集合にも等しく含まれるため値が一致する。
        末尾では後続の重なり相手が切り落とされ、重みが上がる。
        """
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=5)
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, _ = validation.split_events(events, fold)

        whole = dataset.uniqueness_weights(events)
        in_fold = validation.training_weights(train)

        last_id = train["event_id"].iloc[-1]
        pos = list(events["event_id"]).index(last_id)
        assert in_fold[-1] > whole[pos]

    def test_early_training_events_keep_the_same_weight(self):
        """逆に、重なり相手が全て学習側に残る先頭付近では値が一致する。

        これが成り立たないなら、重みの計算が集合の大きさ自体に依存している
        （＝相対的な重なりを測れていない）ことになる。
        """
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=5)
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, _ = validation.split_events(events, fold)

        whole = dataset.uniqueness_weights(events)
        in_fold = validation.training_weights(train)

        first_id = train["event_id"].iloc[0]
        pos = list(events["event_id"]).index(first_id)
        assert in_fold[0] == pytest.approx(whole[pos])

    def test_validation_side_does_not_affect_training_weights(self):
        """検証側イベントの終了時点を変えても学習側の重みは変わらない（spec §7）"""
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=2)
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, _ = validation.split_events(events, fold)
        before = validation.training_weights(train)

        tampered = events.copy()
        in_val = tampered["decision_at"] >= fold.val_start
        tampered.loc[in_val, "label_end_at"] = date(2027, 1, 1)
        train_after, _ = validation.split_events(tampered, fold)
        after = validation.training_weights(train_after)

        assert list(train["event_id"]) == list(train_after["event_id"])
        assert before == pytest.approx(after)

    def test_empty_training_set_gives_empty_weights(self):
        empty = _daily_events(["7203"], date(2026, 1, 5), 10, holding=0).iloc[0:0]
        assert len(validation.training_weights(empty)) == 0
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.validation' has no attribute 'training_weights'`

- [ ] **Step 3: 実装を追加**

`src/strategy/validation.py` の末尾に追加する。import に `from src.strategy.dataset import uniqueness_weights` を足す（`STATUS_RESOLVED` は既に import 済み）。

```python
def training_weights(train_events: pd.DataFrame) -> np.ndarray:
    """学習イベント集合に対して一意性重みを計算し直す。

    **purge後の学習集合に対して呼ぶこと。** イベント表全体で一度計算した重みを
    各foldへ流すと、検証側イベントの終了時点が学習側の重みへ影響する（spec §7）。
    段階B後半で sample_weight を列として保存しなかったのはこのためで、
    ここが正しい呼び出し口になる。
    """
    return uniqueness_weights(train_events)
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_validation.py -v`
Expected: PASS（27件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/validation.py tests/test_validation.py
git commit -m "$(cat <<'EOF'
feat(strategy): 一意性重みをfold内で再計算する呼び出し口を追加

イベント表全体で一度計算した重みを各foldへ流すと、検証側イベントの
終了時点が学習側の重みへ影響する。purge後の学習集合に対して
計算し直すことで、その経路を塞ぐ。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 前処理を学習側だけで fit する

**Files:**
- Modify: `src/strategy/validation.py`（`Preprocessor` / `fit_preprocessor` / `apply_preprocessor` を追加）
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `indicators.FEATURE_COLS`
- Produces:
  - `Preprocessor`（frozen dataclass）: `means: pd.Series`, `stds: pd.Series`, `positive_rate: float`, `n_fitted: int`
  - `fit_preprocessor(train_events: pd.DataFrame, feature_cols: Optional[list] = None) -> Preprocessor`
  - `apply_preprocessor(pre: Preprocessor, events: pd.DataFrame, feature_cols: Optional[list] = None) -> pd.DataFrame`

**背景（spec §7 経路3）:** 標準化・欠損補完・特徴選択・クラス比率推定・閾値・校正は**学習側だけで fit する**。分割器だけを直しても全データ fit ならリークが残る。本計画では標準化・欠損補完・クラス比率を扱い、閾値と校正は段階C後半で同じ規約に従って実装する。

**欠損の扱い:** 学習側の平均で埋める。標準偏差が 0 の列は 1 として扱い、ゼロ除算を避ける。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_validation.py` の末尾に追記する。

```python
class TestPreprocessor:
    def _split(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=1)
        # 特徴量に学習側と検証側で異なる水準を与える
        events["f1"] = np.arange(len(events), dtype=float)
        events["f2"] = np.arange(len(events), dtype=float) * 3.0
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, val = validation.split_events(events, fold)
        return events, fold, train, val

    def test_statistics_come_from_training_only(self):
        _, _, train, _ = self._split()
        pre = validation.fit_preprocessor(train, feature_cols=["f1", "f2"])
        assert pre.means["f1"] == pytest.approx(train["f1"].mean())
        assert pre.stds["f1"] == pytest.approx(train["f1"].std(ddof=0))
        assert pre.n_fitted == len(train)

    def test_positive_rate_comes_from_training_only(self):
        _, _, train, _ = self._split()
        pre = validation.fit_preprocessor(train, feature_cols=["f1", "f2"])
        assert pre.positive_rate == pytest.approx(train["label"].mean())

    def test_changing_validation_values_does_not_change_the_fit(self):
        """検証側の特徴量を書き換えても前処理の統計量は変わらない（spec §7）"""
        events, fold, train, _ = self._split()
        pre_before = validation.fit_preprocessor(train, feature_cols=["f1", "f2"])

        tampered = events.copy()
        in_val = tampered["decision_at"] >= fold.val_start
        tampered.loc[in_val, "f1"] = 99999.0
        tampered.loc[in_val, "f2"] = -99999.0
        train_after, _ = validation.split_events(tampered, fold)
        pre_after = validation.fit_preprocessor(train_after, feature_cols=["f1", "f2"])

        assert pre_before.means["f1"] == pytest.approx(pre_after.means["f1"])
        assert pre_before.stds["f1"] == pytest.approx(pre_after.stds["f1"])
        assert pre_before.positive_rate == pytest.approx(pre_after.positive_rate)

    def test_applying_to_training_gives_zero_mean(self):
        _, _, train, _ = self._split()
        pre = validation.fit_preprocessor(train, feature_cols=["f1", "f2"])
        out = validation.apply_preprocessor(pre, train, feature_cols=["f1", "f2"])
        assert out["f1"].mean() == pytest.approx(0.0, abs=1e-9)
        assert out["f1"].std(ddof=0) == pytest.approx(1.0)

    def test_validation_is_transformed_with_training_statistics(self):
        """検証側は学習側の統計量で変換する（検証側で fit し直さない）"""
        _, _, train, val = self._split()
        pre = validation.fit_preprocessor(train, feature_cols=["f1", "f2"])
        out = validation.apply_preprocessor(pre, val, feature_cols=["f1", "f2"])
        expected = (val["f1"].iloc[0] - pre.means["f1"]) / pre.stds["f1"]
        assert out["f1"].iloc[0] == pytest.approx(expected)
        # 学習期間より後なので中心はずれる＝検証側で fit し直していない証拠
        assert out["f1"].mean() != pytest.approx(0.0, abs=1e-6)

    def test_missing_values_are_filled_with_training_mean(self):
        _, _, train, val = self._split()
        pre = validation.fit_preprocessor(train, feature_cols=["f1", "f2"])
        holed = val.copy()
        holed.loc[holed.index[0], "f1"] = np.nan
        out = validation.apply_preprocessor(pre, holed, feature_cols=["f1", "f2"])
        assert out["f1"].iloc[0] == pytest.approx(0.0)  # 平均で埋める＝標準化後は0
        assert out["f1"].notna().all()

    def test_zero_variance_column_does_not_divide_by_zero(self):
        _, _, train, _ = self._split()
        flat = train.copy()
        flat["f1"] = 5.0
        pre = validation.fit_preprocessor(flat, feature_cols=["f1", "f2"])
        out = validation.apply_preprocessor(pre, flat, feature_cols=["f1", "f2"])
        assert out["f1"].notna().all()
        assert np.isfinite(out["f1"]).all()

    def test_defaults_to_indicator_feature_columns(self):
        """feature_cols を省略すると indicators.FEATURE_COLS を使う"""
        from src.strategy import indicators

        rows = [{"symbol": "7203", "decision_at": date(2026, 1, 5),
                 "label_end_at": date(2026, 1, 5), "label": 1}]
        events = _events(rows)
        for col in indicators.FEATURE_COLS:
            events[col] = 1.0
        pre = validation.fit_preprocessor(events)
        assert list(pre.means.index) == list(indicators.FEATURE_COLS)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.validation' has no attribute 'fit_preprocessor'`

- [ ] **Step 3: 実装を追加**

`src/strategy/validation.py` の末尾に追加する。import に `from typing import Optional` と `from src.strategy.indicators import FEATURE_COLS` を足す。

```python
@dataclass(frozen=True)
class Preprocessor:
    """学習側だけで求めた前処理の統計量。

    検証側・本番側はこの統計量で変換するだけで、自分では fit し直さない。
    分割器だけを直しても全データで fit すればリークは残る（spec §7 経路3）。
    """
    means: pd.Series
    stds: pd.Series
    positive_rate: float
    n_fitted: int


def fit_preprocessor(train_events: pd.DataFrame,
                     feature_cols: Optional[list] = None) -> Preprocessor:
    """**学習イベントだけ**から前処理の統計量を求める。

    標準化の平均・標準偏差、欠損補完に使う平均、クラス比率をここで固定する。
    検証側の値は一切見ない。閾値選択と確率校正も段階C後半で同じ規約に従う。
    """
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    X = train_events.reindex(columns=cols).astype("float64")
    means = X.mean()
    stds = X.std(ddof=0)
    # 分散0の列はゼロ除算になるため1として扱う（変換後は全て0になる）
    stds = stds.mask((stds == 0) | stds.isna(), 1.0)
    labels = pd.to_numeric(train_events.get("label"), errors="coerce")
    positive_rate = float(labels.mean()) if labels is not None and len(labels) else 0.0
    return Preprocessor(
        means=means.fillna(0.0),
        stds=stds,
        positive_rate=positive_rate,
        n_fitted=len(train_events),
    )


def apply_preprocessor(pre: Preprocessor, events: pd.DataFrame,
                       feature_cols: Optional[list] = None) -> pd.DataFrame:
    """学習側の統計量で変換する（欠損は学習側の平均で埋める）。"""
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    X = events.reindex(columns=cols).astype("float64")
    X = X.fillna(pre.means)
    return (X - pre.means) / pre.stds
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_validation.py -v`
Expected: PASS（35件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/validation.py tests/test_validation.py
git commit -m "$(cat <<'EOF'
feat(strategy): 前処理を学習側だけでfitする仕組みを追加

標準化・欠損補完・クラス比率推定を学習イベントだけから求め、検証側は
その統計量で変換するだけにする。分割器だけを直しても全データでfitすれば
リークが残るため。分散0の列はゼロ除算を避けて1として扱う。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: 学習窓を明示パラメータにする

**Files:**
- Modify: `src/strategy/validation.py`（`apply_training_window` を追加）
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: Task 1・2 の `Fold` / `sessions_of`
- Produces: `apply_training_window(train_events: pd.DataFrame, window_sessions: Optional[int] = None) -> pd.DataFrame` — `None` なら拡大窓（全期間）、整数なら直近 N セッションの移動窓

**背景（spec §7 / F07）:** 現行の週次学習は `load_ohlcv(sym)` の既定値そのままで直近500行だけを読んでおり、**意図した選択ではない**。学習窓を明示パラメータにし、拡大窓と一定期間の移動窓を**内側 fold で**比較して選べるようにする。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_validation.py` の末尾に追記する。

```python
class TestTrainingWindow:
    def _train(self):
        events = _daily_events(["7203", "9984"], date(2026, 1, 5), 80, holding=1)
        fold = validation.calendar_folds(events, n_splits=5)[4]
        train, _ = validation.split_events(events, fold)
        return train

    def test_none_keeps_everything_expanding_window(self):
        train = self._train()
        out = validation.apply_training_window(train, None)
        assert list(out["event_id"]) == list(train["event_id"])

    def test_rolling_window_keeps_only_recent_sessions(self):
        train = self._train()
        out = validation.apply_training_window(train, window_sessions=10)
        assert len(validation.sessions_of(out)) == 10
        assert out["decision_at"].max() == train["decision_at"].max()

    def test_rolling_window_drops_the_oldest_sessions(self):
        train = self._train()
        out = validation.apply_training_window(train, window_sessions=10)
        assert out["decision_at"].min() > train["decision_at"].min()

    def test_keeps_all_symbols_within_the_window(self):
        """窓はセッションで切る。銘柄を落とさない"""
        train = self._train()
        out = validation.apply_training_window(train, window_sessions=10)
        assert set(out["symbol"]) == set(train["symbol"])

    def test_window_larger_than_history_keeps_everything(self):
        train = self._train()
        out = validation.apply_training_window(train, window_sessions=100000)
        assert list(out["event_id"]) == list(train["event_id"])

    def test_rejects_non_positive_window(self):
        train = self._train()
        with pytest.raises(ValueError, match="window_sessions"):
            validation.apply_training_window(train, window_sessions=0)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.validation' has no attribute 'apply_training_window'`

- [ ] **Step 3: 実装を追加**

`src/strategy/validation.py` の末尾に追加する。

```python
def apply_training_window(train_events: pd.DataFrame,
                          window_sessions: Optional[int] = None) -> pd.DataFrame:
    """学習窓を適用する。None なら拡大窓、整数なら直近 N セッションの移動窓。

    現行の週次学習は load_ohlcv(sym) の既定値そのままで直近500行だけを読んでおり、
    **意図した選択ではない**（レビューF07）。窓を明示パラメータにして、
    拡大窓と移動窓を内側foldで比較して選べるようにする。

    窓は**セッション（暦日）で切る**。行数で切ると、1日あたりの候補数が
    銘柄数や相場つきで変わったときに実期間が動いてしまう。
    """
    if window_sessions is None:
        return train_events.reset_index(drop=True)
    if window_sessions <= 0:
        raise ValueError(f"window_sessions は正の整数: {window_sessions}")

    sessions = sessions_of(train_events)
    if len(sessions) <= window_sessions:
        return train_events.reset_index(drop=True)

    cutoff = sessions[-window_sessions]
    return train_events[
        train_events["decision_at"] >= cutoff
    ].reset_index(drop=True)
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_validation.py -v`
Expected: PASS（41件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/validation.py tests/test_validation.py
git commit -m "$(cat <<'EOF'
feat(strategy): 学習窓を明示パラメータにした

現行の週次学習はload_ohlcvの既定値そのままで直近500行を読んでおり、
意図した選択ではなかった。拡大窓と移動窓を選べるようにし、
セッション（暦日）で切ることで1日あたりの候補数が変わっても
実期間が動かないようにする。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: 外側 fold の不可侵性を1本の検証にまとめる

**Files:**
- Modify: `src/strategy/validation.py`（`training_inputs` を追加）
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: Task 2・4・5・6 のすべて
- Produces:
  - `TrainingInputs`（frozen dataclass）: `events: pd.DataFrame`, `weights: np.ndarray`, `preprocessor: Preprocessor`, `fold: Fold`
  - `training_inputs(events, fold, *, window_sessions=None, embargo_sessions=0, feature_cols=None) -> TrainingInputs`

**背景（spec §14 段階C完了条件）:** 「**そのfoldの学習締切で固定されたモデルについて**、外側foldの値を変えてもモデル・前処理・重み・閾値が変わらない」。これを1つの関数の入出力として固定し、直接テストできるようにする。

`training_inputs()` は「このfoldで学習に使ってよい入力一式」を返す。学習そのものは段階C後半が行うが、**学習に入る入力がすべてこの関数を通る**ようにしておけば、漏れの検査が1箇所で済む。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_validation.py` の末尾に追記する。

```python
class TestTrainingInputs:
    def _events(self):
        events = _daily_events(["7203", "9984"], date(2026, 1, 5), 90, holding=3)
        events["f1"] = np.arange(len(events), dtype=float)
        events["f2"] = np.arange(len(events), dtype=float) * 1.5
        return events

    def test_bundles_events_weights_and_preprocessor(self):
        events = self._events()
        fold = validation.calendar_folds(events, n_splits=5)[3]
        got = validation.training_inputs(events, fold, feature_cols=["f1", "f2"])
        assert len(got.weights) == len(got.events)
        assert got.preprocessor.n_fitted == len(got.events)
        assert got.fold == fold

    def test_training_inputs_respect_purge(self):
        events = self._events()
        fold = validation.calendar_folds(events, n_splits=5)[3]
        got = validation.training_inputs(events, fold, feature_cols=["f1", "f2"])
        assert (got.events["label_end_at"] < fold.val_start).all()

    def test_training_window_is_applied_before_weights_and_fit(self):
        """窓で絞ったあとの集合で重みと前処理が決まる"""
        events = self._events()
        fold = validation.calendar_folds(events, n_splits=5)[3]
        full = validation.training_inputs(events, fold, feature_cols=["f1", "f2"])
        windowed = validation.training_inputs(
            events, fold, window_sessions=10, feature_cols=["f1", "f2"])
        assert len(windowed.events) < len(full.events)
        assert windowed.preprocessor.n_fitted == len(windowed.events)
        assert windowed.preprocessor.means["f1"] != pytest.approx(
            full.preprocessor.means["f1"])


class TestOuterFoldIsUntouchable:
    """spec §14 段階C完了条件: そのfoldの学習締切で固定された入力は、
    外側foldの値を変えても変わらない"""

    def _events(self):
        events = _daily_events(["7203", "9984"], date(2026, 1, 5), 90, holding=3)
        events["f1"] = np.arange(len(events), dtype=float)
        events["f2"] = np.arange(len(events), dtype=float) * 1.5
        return events

    def _tamper_outside_training(self, events: pd.DataFrame, fold) -> pd.DataFrame:
        """学習締切より後のイベントを、値・ラベル・終了時点すべて書き換える"""
        out = events.copy()
        after = out["decision_at"] > fold.train_end
        out.loc[after, "f1"] = -123456.0
        out.loc[after, "f2"] = 987654.0
        out.loc[after, "label"] = 1
        out.loc[after, "label_end_at"] = date(2030, 1, 1)
        out.loc[after, "net_return"] = 9.99
        return out

    def test_training_events_do_not_change(self):
        events = self._events()
        fold = validation.calendar_folds(events, n_splits=5)[3]
        before = validation.training_inputs(events, fold, feature_cols=["f1", "f2"])
        after = validation.training_inputs(
            self._tamper_outside_training(events, fold), fold, feature_cols=["f1", "f2"])
        assert list(before.events["event_id"]) == list(after.events["event_id"])

    def test_weights_do_not_change(self):
        events = self._events()
        fold = validation.calendar_folds(events, n_splits=5)[3]
        before = validation.training_inputs(events, fold, feature_cols=["f1", "f2"])
        after = validation.training_inputs(
            self._tamper_outside_training(events, fold), fold, feature_cols=["f1", "f2"])
        assert before.weights == pytest.approx(after.weights)

    def test_preprocessor_does_not_change(self):
        events = self._events()
        fold = validation.calendar_folds(events, n_splits=5)[3]
        before = validation.training_inputs(events, fold, feature_cols=["f1", "f2"])
        after = validation.training_inputs(
            self._tamper_outside_training(events, fold), fold, feature_cols=["f1", "f2"])
        assert before.preprocessor.means["f1"] == pytest.approx(after.preprocessor.means["f1"])
        assert before.preprocessor.stds["f1"] == pytest.approx(after.preprocessor.stds["f1"])
        assert before.preprocessor.positive_rate == pytest.approx(
            after.preprocessor.positive_rate)

    def test_transformed_training_features_do_not_change(self):
        events = self._events()
        fold = validation.calendar_folds(events, n_splits=5)[3]
        before = validation.training_inputs(events, fold, feature_cols=["f1", "f2"])
        after = validation.training_inputs(
            self._tamper_outside_training(events, fold), fold, feature_cols=["f1", "f2"])
        Xb = validation.apply_preprocessor(
            before.preprocessor, before.events, feature_cols=["f1", "f2"])
        Xa = validation.apply_preprocessor(
            after.preprocessor, after.events, feature_cols=["f1", "f2"])
        pd.testing.assert_frame_equal(Xb, Xa)

    def test_holds_for_every_fold(self):
        events = self._events()
        for fold in validation.calendar_folds(events, n_splits=5):
            before = validation.training_inputs(events, fold, feature_cols=["f1", "f2"])
            if len(before.events) == 0:
                continue
            after = validation.training_inputs(
                self._tamper_outside_training(events, fold), fold,
                feature_cols=["f1", "f2"])
            assert list(before.events["event_id"]) == list(after.events["event_id"])
            assert before.weights == pytest.approx(after.weights)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.validation' has no attribute 'training_inputs'`

- [ ] **Step 3: 実装を追加**

`src/strategy/validation.py` の末尾に追加する。

```python
@dataclass(frozen=True)
class TrainingInputs:
    """このfoldで学習に使ってよい入力一式。

    **学習に入る入力はすべてこの関数を通す。** そうしておけば「学習側へ
    未来が入っていないか」の検査が1箇所で済む。
    """
    events: pd.DataFrame
    weights: np.ndarray
    preprocessor: Preprocessor
    fold: Fold


def training_inputs(events: pd.DataFrame, fold: Fold, *,
                    window_sessions: Optional[int] = None,
                    embargo_sessions: int = 0,
                    feature_cols: Optional[list] = None) -> TrainingInputs:
    """foldの学習締切で固定された入力一式を返す。

    適用の順序に意味がある。purge → 学習窓 → 重み → 前処理 の順で、
    **絞り込みが終わった集合に対して重みと統計量を求める**。順序を逆にすると、
    後から落とすイベントが重みや平均に混ざる。

    spec §14 の段階C完了条件「そのfoldの学習締切で固定されたモデルについて、
    外側foldの値を変えてもモデル・前処理・重み・閾値が変わらない」は、
    この関数の出力が変わらないことで検証できる。
    """
    train, _ = split_events(events, fold, embargo_sessions=embargo_sessions)
    train = apply_training_window(train, window_sessions)
    weights = training_weights(train)
    pre = fit_preprocessor(train, feature_cols=feature_cols)
    return TrainingInputs(events=train, weights=weights, preprocessor=pre, fold=fold)
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_validation.py -v`
Expected: PASS（49件）

- [ ] **Step 5: 全体回帰を確認**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

- [ ] **Step 6: BOM確認とコミット**

Run: `head -c 3 src/strategy/validation.py | xxd`（`2222 22` を確認）

```bash
git add src/strategy/validation.py tests/test_validation.py
git commit -m "$(cat <<'EOF'
feat(strategy): fold学習入力を1つの入口にまとめた

purge→学習窓→重み→前処理の順で適用し、絞り込みが終わった集合に対して
重みと統計量を求める。学習に入る入力をすべてこの関数に通すことで、
「学習側へ未来が入っていないか」の検査が1箇所で済む。
外側foldの値を書き換えても出力が変わらないことをテストで固定する
（設計書§14の段階C完了条件）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## 段階C前半 完了条件の確認

- [ ] **確認1: fold 境界が日付であり行番号ではない**

Run: `pytest tests/test_validation.py::TestCalendarFolds::test_boundaries_are_dates_not_row_positions -v`
Expected: PASS

- [ ] **確認2: 学習側のどのラベルも検証開始日より前に確定している**

Run: `pytest tests/test_validation.py::TestPurge -v`
Expected: PASS（4件）

- [ ] **確認3: 検証側の値を変えても学習側の重み・前処理が変わらない**

Run: `pytest tests/test_validation.py::TestTrainingWeights::test_validation_side_does_not_affect_training_weights -v`
Run: `pytest tests/test_validation.py::TestPreprocessor::test_changing_validation_values_does_not_change_the_fit -v`
Expected: PASS

- [ ] **確認4: 外側 fold の値を変えても学習入力が変わらない（spec §14 段階C完了条件）**

Run: `pytest tests/test_validation.py::TestOuterFoldIsUntouchable -v`
Expected: PASS（5件）

- [ ] **確認5: 外側 fold が最終評価専用である（内側が触れない）**

Run: `pytest tests/test_validation.py::TestInnerFolds -v`
Expected: PASS（5件）

- [ ] **確認6: 既存経路に回帰が無い**

Run: `pytest tests/ -q`
Expected: 段階C前半の着手前と同じ結果（新規テスト49件ぶんだけ増える）

**残る完了条件（後半で満たす）:** 「予測明細から指標を再計算できる」は、`Prediction` / `PredictionOutcome` テーブルを作る段階C後半で検証する。

---

## 次の段階

段階C後半は、本計画が確定させた `training_inputs()` を入力として次を実装する。

- 比較対象5モデル（定数確率・多数派予測・ロジスティック回帰・小さいLightGBM・現行LightGBM）の学習と評価
- 内側 fold での early stopping・閾値選択・確率校正（`fit_preprocessor` と同じ「学習側だけで fit」の規約に従う）
- `Prediction` / `PredictionOutcome` テーブル（予測を先に保存し、満期後に実績ラベルを関連付ける）
- 記録する指標（fold別・銘柄別・期間別の件数と正例率、ROC-AUC・PR系・log loss・Brier と定数モデルとの差、校正曲線、上位群のコスト控除後成績）
- 期間中の再学習に同じ締切を適用すること（spec §7 経路4の残り）
- 拡大窓と移動窓の実比較（内側 fold で選ぶ）
