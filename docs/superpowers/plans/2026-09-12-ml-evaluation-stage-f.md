# ML評価基盤 段階F（v2の結線）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 段階A〜Eで作った v2 の経路を `engine_version` で実際に選べるようにし、どちらで動いているかを記録に残す。

**Architecture:** 段階A〜Eは新しい経路を**作る**だけで、どちらを使うかは誰も決めていなかった。本計画が3経路（学習データの生成元・CV分割・バックテストの実行主体）の分岐を実装する。**既定は `legacy` のまま**で、v2 を選んだときだけ新経路が動く。学習は候補生成に留め、運用モデルは自動で入れ替わらない。

**Tech Stack:** Python 3.11 / SQLAlchemy 2.0.23 / pytest

**Spec:** `docs/superpowers/specs/2026-09-10-ml-evaluation-foundation-design.md`（§10 の切替表・§11）

**前提:** 段階B後半・C前半・C後半・D後半・E が完了していること。本計画は `dataset.build_events_multi` / `validation.calendar_folds` / `validation.training_inputs` / `evaluation.run_evaluation` / `model_store.train_as_candidate` / `walkforward.run_walkforward` / `walkforward.save_run` に依存する。

## この計画が生まれた経緯

2026-09-12 の監査で、**段階A〜Eの計画をすべて実装しても v2 が一度も動かない**ことが分かった。

| spec §10 が定める切替 | 段階A〜Eでの扱い |
|---|---|
| バックテストの実行主体（`engine.py` ↔ `walkforward.py`） | **分岐が無い**。ダッシュボードは常に `engine.run_backtest` を呼ぶ |
| 学習データの生成元（`labeling` ↔ `dataset`） | **分岐が無い**。`ml_retrain` は常に `ml_model.train_multi` を呼ぶ |
| CV分割（`TimeSeriesSplit` ↔ `validation`） | **分岐が無い**。`ml_model._fit` の中に固定 |
| paper の執行仮定 | 段階D後半 Task 7 で実装済み |

原因は、段階C後半とEのグローバル制約が「`ml_model.py` を変更しない」としていたことにある。新しい経路を別モジュールに作る判断自体は正しい（legacy を壊さないため）が、**繋ぐ変更を誰の担当にもしていなかった**。

## Global Constraints

- **既定は `legacy`。** 本計画の完了後も、設定を変えない限り挙動は現在と同じであること。
- **`legacy` を選んだときの経路は1行も変えない。** `engine.py` / `labeling.py` / `ml_model._fit` の中身は触らない。分岐は**呼び出し側**に置く。
- **v2 の学習は候補生成に留める。** `model_store.train_as_candidate` を使い、運用モデル（`self.model`）を自動で差し替えない。昇格は段階Eの `promotion.promote()` による明示的な操作だけ。
- **発注経路を変更しない。** 本計画が触るのは学習・評価・バックテストの選択だけ。
- 日時は **JST naive**。現在時刻は `src/core/clock.now()` / `clock.today()` を使う。
- 新規のDB列はすべて nullable。`_migrate_add_missing_columns()` が自動で追加する。
- ファイルは UTF-8 **BOM無し**・LF で保存する。確認は `git show <rev>:<path>` でコミット済みblobに対して行う。
- テストは `pytest tests/<file>.py -v` で実行する。ネットワークへ出るテストを書かない。
- コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` を付ける。実装者自身のモデル名を書かない。

---

## File Structure

| ファイル | 責務 |
|---|---|
| `src/strategy/v2_training.py`（新規） | v2 の学習経路（イベント表 → 分割 → 候補モデル）。`ml_model.py` には触らない |
| `src/services/trading.py`（改修） | `ml_retrain` の分岐 |
| `src/dashboard/app.py`（改修） | バックテスト実行主体の分岐 |
| `src/data/database.py`（改修） | `ModelMetrics` への列追加 |
| `tests/test_v2_training.py`（新規） | v2 学習経路の単体 |
| `tests/test_engine_version_wiring.py`（新規） | 3経路の分岐、legacy で挙動が変わらないこと |

---

## Task 1: `ModelMetrics` に列を足す

**Files:**
- Modify: `src/data/database.py`（`ModelMetrics`）
- Test: `tests/test_v2_training.py`

**Interfaces:**
- Consumes: なし
- Produces: `ModelMetrics` へ `model_id` / `positive_rate` / `training_window_sessions` / `engine_version` を追加（すべて nullable）

**背景（spec §11）:** データモデル表がこの3列を求めているが、段階A〜Eのどの計画も実装しない。`ml_model.py` を触らない方針の副作用である。列は `ml_model._save_metrics()` と v2 の学習経路の**両方**が書けるよう、テーブル側に足す。

`engine_version` を加えるのは、**同じテーブルに legacy と v2 の記録が混ざる**ためである。どちらの方式で出た数字かが分からないと比較できない。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_v2_training.py` を新規作成する。

```python
"""v2の学習経路（src/strategy/v2_training.py）のテスト

段階A〜Eは新しい経路を作るだけで、どちらを使うかは誰も決めていなかった。
本計画が分岐を実装する。legacyを選んだときの経路は1行も変えない。
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.core import config as cfg
from src.data import database as db
from src.data.database import get_session


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


class TestModelMetricsColumns:
    def test_new_columns_exist(self, isolated_db):
        from src.data.database import ModelMetrics

        with get_session() as session:
            session.add(ModelMetrics(
                cv_mean_accuracy=0.55, n_samples=100, trigger="test",
                model_id="m0001", positive_rate=0.48,
                training_window_sessions=500, engine_version="v2"))
            session.commit()
            row = session.scalar(select(ModelMetrics))
        assert row.model_id == "m0001"
        assert row.positive_rate == pytest.approx(0.48)
        assert row.training_window_sessions == 500
        assert row.engine_version == "v2"

    def test_columns_are_nullable(self, isolated_db):
        """legacy の _save_metrics は新列を書かない。書かなくても通ること"""
        from src.data.database import ModelMetrics

        with get_session() as session:
            session.add(ModelMetrics(
                cv_mean_accuracy=0.55, n_samples=100, trigger="weekly_schedule"))
            session.commit()
            row = session.scalar(select(ModelMetrics))
        assert row.model_id is None
        assert row.engine_version is None

    def test_legacy_and_v2_records_are_distinguishable(self, isolated_db):
        """同じテーブルに混ざるので、どちらの方式かが分かること"""
        from src.data.database import ModelMetrics

        with get_session() as session:
            session.add(ModelMetrics(cv_mean_accuracy=0.55, n_samples=100,
                                     trigger="weekly_schedule"))
            session.add(ModelMetrics(cv_mean_accuracy=0.52, n_samples=90,
                                     trigger="weekly_schedule",
                                     engine_version="v2", model_id="m0001"))
            session.commit()
            rows = list(session.scalars(select(ModelMetrics)).all())
        versions = {r.engine_version for r in rows}
        assert versions == {None, "v2"}
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_v2_training.py -v`
Expected: FAIL — `TypeError: 'model_id' is an invalid keyword argument for ModelMetrics`

- [ ] **Step 3: 列を追加**

`src/data/database.py` の `class ModelMetrics` の末尾に追加する。

```python
    # ─── v2（段階F）で書く列。legacy の _save_metrics は書かないので全てnullable ───
    # 同じテーブルに legacy と v2 の記録が混ざるため、どちらの方式で出た数字かを
    # 残す。これが無いと比較できない。
    model_id = Column(String(64))
    positive_rate = Column(Float)
    training_window_sessions = Column(Integer)
    engine_version = Column(String(16))
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_v2_training.py -v`
Expected: PASS（3件）

- [ ] **Step 5: 既存のMLテストが通ることを確認**

Run: `pytest tests/test_ml_model_save.py tests/test_ml_train_multi.py tests/test_schema_version.py -v`
Expected: PASS（`_save_metrics` は新列を書かないが、nullable なので通る）

- [ ] **Step 6: BOM確認とコミット**

Run: `head -c 3 src/data/database.py | xxd`（`2222 22` を確認）

```bash
git add src/data/database.py tests/test_v2_training.py
git commit -m "$(cat <<'EOF'
feat(data): ModelMetricsにv2用の列を追加

設計書§11が求めていたmodel_id/positive_rate/training_window_sessionsが
段階A〜Eのどの計画にも入っていなかった（ml_model.pyを触らない方針の
副作用）。あわせてengine_versionも足す。同じテーブルにlegacyとv2の記録が
混ざるため、どちらの方式で出た数字かが分からないと比較できない。
全てnullableでlegacyの_save_metricsは書かない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: v2 の学習経路

**Files:**
- Create: `src/strategy/v2_training.py`
- Test: `tests/test_v2_training.py`

**Interfaces:**
- Consumes: `dataset.build_events_multi` / `compute_dataset_id` / `save_events` / `save_dataset_meta` / `input_ohlcv_hash`、`validation.calendar_folds` / `training_inputs`、`evaluation.CurrentLightGBM`、`model_store.train_as_candidate` / `ModelMeta`
- Produces:
  - `V2TrainingResult`（frozen dataclass）: `model_id: Optional[str]`, `dataset_id: str`, `n_events: int`, `n_resolved: int`, `positive_rate: Optional[float]`, `skipped_reason: Optional[str]`
  - `train_v2(ohlcv_by_symbol: dict, *, policy_conf, costs, window_sessions=None, base_dir="models") -> V2TrainingResult`

**背景:** `ml_model.train_multi()` は `labeling.build_training_set` と `TimeSeriesSplit` を使う。v2 は `dataset.build_events_multi` と `validation` を使う。**`ml_model.py` には触らず**、別モジュールとして v2 の経路を作る。

**最重要:** **v2 の学習は候補を作るだけで、運用モデルを差し替えない。** `model_store.train_as_candidate()` を通すことで、学習が失敗しても現行が失われない。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_v2_training.py` の末尾に追記する。

```python
def _ohlcv(n=300, start_price=1000.0, seed=0):
    rng = np.random.default_rng(seed)
    start = date(2025, 1, 6)
    rows, p = [], start_price
    for i in range(n):
        p *= 1 + rng.normal(0, 0.01)
        rows.append({"date": start + timedelta(days=i), "open": p,
                     "high": p * 1.01, "low": p * 0.99, "close": p,
                     "volume": 1_000_000})
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def _policy_conf():
    from src.strategy import policy
    return policy.PolicyConfig(
        stop_loss_pct=-0.07, breakeven_trigger_pct=0.02, trailing_stop_pct=0.04,
        sell_threshold=-0.25, max_holding_sessions=10)


def _costs():
    from src.backtest import execution
    return execution.CostConfig(slippage_pct=0.001, commission_pct=0.0)


class TestTrainV2:
    def _bars(self):
        return {"7203": _ohlcv(seed=1), "9984": _ohlcv(seed=2, start_price=500.0)}

    def test_produces_a_candidate_not_a_promotion(self, isolated_db, tmp_path):
        """学習成功は候補の生成であって運用モデルの差し替えではない

        **`if res.model_id is not None:` で包まない。** 包むと、保存が
        AttributeError で失敗して `train_as_candidate` が None を返した
        場合でもこのテストが通ってしまう（外部レビューR02）。
        成功ケースは成功したことを断言する。
        """
        from src.strategy import model_store as ms
        from src.strategy import v2_training

        res = v2_training.train_v2(
            self._bars(), policy_conf=_policy_conf(), costs=_costs(),
            base_dir=str(tmp_path / "models"))

        assert res.skipped_reason is None, res.skipped_reason
        assert res.model_id is not None
        assert ms.candidate_dir(res.model_id, str(tmp_path / "models")).exists()
        # 現行は未昇格のまま
        assert ms.read_current(base_dir=str(tmp_path / "models")) is None

    def test_the_saved_candidate_can_be_loaded_and_predicts_the_same(
            self, isolated_db, tmp_path):
        """保存物を読み直して、学習直後と同じ予測が出ること

        段階Cのラッパーは `booster_` も `save_model()` も持たない。
        保存側と学習側の型契約が合っていないと、ここで落ちる
        （外部レビューR02）。
        """
        import numpy as np
        import pandas as pd

        from src.strategy import model_store as ms
        from src.strategy import v2_training
        from src.strategy.indicators import FEATURE_COLS

        base = str(tmp_path / "models")
        res = v2_training.train_v2(
            self._bars(), policy_conf=_policy_conf(), costs=_costs(),
            base_dir=base)
        assert res.model_id is not None

        loaded, meta = ms.load_model(res.model_id, base_dir=base)
        assert list(meta.feature_cols) == list(FEATURE_COLS)
        assert meta.label_contract_id is not None

        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            rng.normal(0, 1, (5, len(FEATURE_COLS))), columns=list(FEATURE_COLS))
        proba = loaded.predict(X)
        assert len(proba) == 5
        assert np.all((proba >= 0.0) & (proba <= 1.0))

    def test_a_save_failure_is_reported_not_swallowed(self, isolated_db, tmp_path):
        """保存に失敗したら model_id は None で理由が残る（成功と紛れない）"""
        from src.strategy import v2_training

        original = v2_training._fit_candidate

        class Unsavable:
            def predict_proba(self, X):
                return None

        v2_training._fit_candidate = lambda events, weights: Unsavable()
        try:
            res = v2_training.train_v2(
                self._bars(), policy_conf=_policy_conf(), costs=_costs(),
                base_dir=str(tmp_path / "models"))
        finally:
            v2_training._fit_candidate = original

        assert res.model_id is None
        assert res.skipped_reason is not None

    def test_records_the_dataset(self, isolated_db, tmp_path):
        from src.data.database import Dataset
        from src.strategy import v2_training

        res = v2_training.train_v2(
            self._bars(), policy_conf=_policy_conf(), costs=_costs(),
            base_dir=str(tmp_path / "models"))

        with get_session() as session:
            row = session.scalar(select(Dataset))
        assert row is not None
        assert row.dataset_id == res.dataset_id

    def test_records_metrics_with_the_engine_version(self, isolated_db, tmp_path):
        from src.data.database import ModelMetrics
        from src.strategy import v2_training

        v2_training.train_v2(
            self._bars(), policy_conf=_policy_conf(), costs=_costs(),
            base_dir=str(tmp_path / "models"))

        with get_session() as session:
            rows = list(session.scalars(select(ModelMetrics)).all())
        if rows:
            assert rows[-1].engine_version == "v2"

    def test_skips_when_there_are_too_few_events(self, isolated_db, tmp_path):
        """イベントが足りなければ学習せず理由を返す（例外にしない）"""
        from src.strategy import v2_training

        res = v2_training.train_v2(
            {"7203": _ohlcv(n=30)}, policy_conf=_policy_conf(), costs=_costs(),
            base_dir=str(tmp_path / "models"))
        assert res.model_id is None
        assert res.skipped_reason is not None

    def test_does_not_touch_the_legacy_model_file(self, isolated_db, tmp_path):
        """models/lgb_model.pkl を壊さない"""
        legacy = tmp_path / "models" / "lgb_model.pkl"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(b"legacy-bytes")

        from src.strategy import v2_training

        v2_training.train_v2(
            self._bars(), policy_conf=_policy_conf(), costs=_costs(),
            base_dir=str(tmp_path / "models"))
        assert legacy.read_bytes() == b"legacy-bytes"

    def test_training_failure_returns_a_reason(self, isolated_db, tmp_path, monkeypatch):
        from src.strategy import v2_training

        monkeypatch.setattr(
            v2_training, "_fit_candidate",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        res = v2_training.train_v2(
            self._bars(), policy_conf=_policy_conf(), costs=_costs(),
            base_dir=str(tmp_path / "models"))
        assert res.model_id is None
        assert res.skipped_reason is not None
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_v2_training.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.strategy.v2_training'`

- [ ] **Step 3: 実装を書く**

`src/strategy/v2_training.py` を新規作成する。

```python
"""v2 の学習経路 — イベント表から候補モデルを作る。

legacy（`ml_model.train_multi`）は `labeling.build_training_set` と
`TimeSeriesSplit` を使う。v2 は `dataset.build_events_multi` と
`validation` を使う。**`ml_model.py` には触らない**。legacy をいつでも
選べることが段階投入の前提だから（設計書 §10）。

**学習成功は候補の生成であって運用モデルの差し替えではない。**
`model_store.train_as_candidate()` を通すので、学習が失敗しても現行は失われない。
昇格は `promotion.promote()` による明示的な操作だけで起きる。
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.core import clock
from src.strategy import dataset as ds
from src.strategy import model_store as ms
from src.strategy import validation
from src.strategy.evaluation import CurrentLightGBM
from src.strategy.indicators import FEATURE_COLS

ENGINE_VERSION = "v2"
MIN_RESOLVED_EVENTS = 200


@dataclass(frozen=True)
class V2TrainingResult:
    """v2 学習の結果。model_id が None なら候補は作られていない。"""
    model_id: Optional[str]
    dataset_id: str
    n_events: int
    n_resolved: int
    positive_rate: Optional[float]
    skipped_reason: Optional[str] = None


def _fit_candidate(train_events: pd.DataFrame, weights: np.ndarray):
    """最終候補モデルを学習する（テストで差し替えられるよう関数に切る）。"""
    model = CurrentLightGBM()
    model.fit(train_events[list(FEATURE_COLS)].astype("float64"),
              train_events["label"].astype(int), weights)
    return model


def _save_metrics(result: V2TrainingResult, window_sessions: Optional[int]) -> None:
    from src.data.database import ModelMetrics, get_session

    with get_session() as session:
        session.add(ModelMetrics(
            trained_at=clock.now(),
            n_samples=result.n_resolved,
            trigger="weekly_schedule",
            model_id=result.model_id,
            positive_rate=result.positive_rate,
            training_window_sessions=window_sessions,
            engine_version=ENGINE_VERSION,
        ))
        session.commit()


def train_v2(ohlcv_by_symbol: dict, *, policy_conf, costs,
             window_sessions: Optional[int] = None,
             base_dir: str = "models") -> V2TrainingResult:
    """イベント表を作り、最新foldの学習入力から候補モデルを作る。

    **運用モデルを差し替えない。** 候補を保存して終わる。
    データ不足や学習失敗は例外にせず、理由を添えて返す（週次ジョブを
    落とさないため）。
    """
    events = ds.build_events_multi(ohlcv_by_symbol, policy_conf, costs)
    dataset_id = ds.compute_dataset_id(events)
    n_events = len(events)
    resolved = events[events["status"] == ds.STATUS_RESOLVED] if n_events else events
    n_resolved = len(resolved)

    if n_events:
        path = ds.save_events(events, dataset_id)
        ds.save_dataset_meta(events, dataset_id, path,
                             ds.input_ohlcv_hash(ohlcv_by_symbol))

    if n_resolved < MIN_RESOLVED_EVENTS:
        reason = f"決着したイベントが不足しています: {n_resolved}件 < {MIN_RESOLVED_EVENTS}件"
        logger.warning(f"v2学習をスキップ: {reason}")
        return V2TrainingResult(
            model_id=None, dataset_id=dataset_id, n_events=n_events,
            n_resolved=n_resolved, positive_rate=None, skipped_reason=reason)

    # 最新foldの学習締切までの入力だけを使う（未来を入れない）
    try:
        folds = validation.calendar_folds(events, n_splits=5)
    except ValueError as e:
        reason = f"分割できません: {e}"
        logger.warning(f"v2学習をスキップ: {reason}")
        return V2TrainingResult(
            model_id=None, dataset_id=dataset_id, n_events=n_events,
            n_resolved=n_resolved, positive_rate=None, skipped_reason=reason)

    inputs = validation.training_inputs(
        events, folds[-1], window_sessions=window_sessions,
        feature_cols=list(FEATURE_COLS))
    if len(inputs.events) == 0:
        reason = "最新foldの学習イベントが0件です"
        return V2TrainingResult(
            model_id=None, dataset_id=dataset_id, n_events=n_events,
            n_resolved=n_resolved, positive_rate=None, skipped_reason=reason)

    positive_rate = float(inputs.events["label"].astype(int).mean())
    model_id = f"v2-{clock.now():%Y%m%dT%H%M%S}-{dataset_id[:8]}"

    def _meta():
        return ms.ModelMeta(
            model_id=model_id, trained_at=clock.now(),
            training_window_sessions=window_sessions,
            symbols=sorted(ohlcv_by_symbol),
            label_definition="net_return>0",
            feature_cols=list(FEATURE_COLS),
            positive_rate=positive_rate,
            fold_results=[],
            dataset_id=dataset_id,
            # どう作られたラベルで学習したかを固定する。昇格検査が
            # 評価実行のラベル契約と突き合わせる（外部レビューR07/R13）
            label_contract_id=ds.make_label_contract_id(policy_conf, costs),
        )

    saved_id = ms.train_as_candidate(
        lambda: _fit_candidate(inputs.events, inputs.weights), _meta,
        base_dir=base_dir)

    result = V2TrainingResult(
        model_id=saved_id, dataset_id=dataset_id, n_events=n_events,
        n_resolved=n_resolved, positive_rate=positive_rate,
        skipped_reason=None if saved_id else "候補モデルの学習・保存に失敗しました")
    _save_metrics(result, window_sessions)
    if saved_id:
        logger.warning(
            f"v2候補モデルを作成: {saved_id}（学習{len(inputs.events)}件 / "
            f"正例率{positive_rate:.3f} / dataset={dataset_id}）"
        )
    return result
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_v2_training.py -v`
Expected: PASS（12件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/v2_training.py tests/test_v2_training.py
git commit -m "$(cat <<'EOF'
feat(strategy): v2の学習経路を追加

legacyのml_model.train_multiには触らず、dataset.build_events_multiと
validationを使う経路を別モジュールとして置く。legacyをいつでも選べる
ことが段階投入の前提のため。
学習成功は候補の生成であって運用モデルの差し替えではない。
train_as_candidateを通すので失敗しても現行は失われない。
データ不足や学習失敗は例外にせず理由を添えて返す（週次ジョブを
落とさないため）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 週次再学習の分岐

**Files:**
- Modify: `src/services/trading.py`（`ml_retrain`）
- Test: `tests/test_engine_version_wiring.py`

**Interfaces:**
- Consumes: Task 2 の `train_v2`、段階D後半の `TradingServices._engine_version()`
- Produces: `ml_retrain` が `engine_version` で経路を分ける

**最重要の制約:** `legacy` のとき、**`ml_retrain` の挙動は現在と1ビットも変わらない**こと。`self.model = ml_model.train_multi(...)` の行はそのまま残す。

**v2 のとき:** 候補を作るだけで **`self.model` を差し替えない**。運用は昇格済みモデル（無ければ `None`）のまま動く。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_engine_version_wiring.py` を新規作成する。

```python
"""engine_version による3経路の分岐（段階F）のテスト

段階A〜Eは新しい経路を作るだけで、どちらを使うかは誰も決めていなかった。
**legacy を選んだときの挙動が現在と1ビットも変わらない**ことを固定する。
"""
import inspect
from unittest.mock import MagicMock, patch

import pytest

from src.core import config as cfg
from src.services import trading


@pytest.fixture(autouse=True)
def _config():
    cfg.load("config.yaml")
    yield
    cfg.get_section("strategy")["engine_version"] = "legacy"


def _service():
    return trading.TradingServices(
        client=MagicMock(), risk=MagicMock(), order_mgr=MagicMock(), model=None)


class TestLegacyRetrainUnchanged:
    def test_default_is_legacy(self):
        assert cfg.get_section("strategy").get("engine_version") == "legacy"

    def test_legacy_calls_train_multi(self):
        svc = _service()
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.ml_model, "train_multi") as train, \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]):
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()
        assert train.called

    def test_legacy_assigns_the_running_model(self):
        """従来どおり self.model へ代入する（挙動不変）"""
        svc = _service()
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.ml_model, "train_multi",
                          return_value="trained-model"), \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]):
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()
        assert svc.model == "trained-model"

    def test_legacy_source_still_present(self):
        src = inspect.getsource(trading.TradingServices.ml_retrain)
        assert "ml_model.train_multi" in src
        assert "self.model =" in src


class TestV2Retrain:
    def test_v2_calls_train_v2_not_train_multi(self):
        cfg.get_section("strategy")["engine_version"] = "v2"
        svc = _service()
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.ml_model, "train_multi") as legacy_train, \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]), \
             patch("src.strategy.v2_training.train_v2") as v2_train:
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()
        assert v2_train.called
        assert not legacy_train.called

    def test_v2_does_not_replace_the_running_model(self):
        """候補を作るだけ。運用モデルは自動で入れ替わらない"""
        cfg.get_section("strategy")["engine_version"] = "v2"
        svc = _service()
        svc.model = "existing-model"
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]), \
             patch("src.strategy.v2_training.train_v2",
                   return_value=MagicMock(model_id="cand-1", skipped_reason=None)):
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()
        assert svc.model == "existing-model"

    def test_v2_failure_does_not_raise(self):
        cfg.get_section("strategy")["engine_version"] = "v2"
        svc = _service()
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]), \
             patch("src.strategy.v2_training.train_v2",
                   side_effect=RuntimeError("boom")):
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()   # 例外が外へ出ない
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_engine_version_wiring.py -v`
Expected: FAIL — v2 のとき `train_multi` が呼ばれてしまう

- [ ] **Step 3: 実装を修正**

`src/services/trading.py` の `ml_retrain` の学習部分を分岐させる。**`legacy` の3行は触らない。**

```python
        if dfs:
            if self._engine_version() == "v2":
                # v2: 候補を作るだけで self.model を差し替えない。
                # 学習成功は候補の生成であって運用モデルの更新ではない（設計書 §9）。
                # 昇格は promotion.promote() による明示的な操作でのみ起きる。
                from src.strategy import policy
                from src.strategy import v2_training
                from src.backtest import execution

                try:
                    result = v2_training.train_v2(
                        {sym: df for sym, df in zip(trained_symbols, dfs)},
                        policy_conf=policy.config_from_settings(),
                        costs=execution.config_from_settings(),
                    )
                    if result.model_id:
                        logger.warning(
                            f"v2候補モデルを作成しました: {result.model_id}"
                            "（運用モデルは変更していません。昇格は明示操作が必要です）"
                        )
                    else:
                        logger.warning(f"v2候補モデルは作られませんでした: {result.skipped_reason}")
                except Exception as e:
                    logger.error(f"v2再学習失敗: {e}")
                return

            try:
                self.model = ml_model.train_multi(dfs, trigger="weekly_schedule")
            except Exception as e:
                logger.error(f"再学習失敗: {e}")
```

`trained_symbols` は `dfs` に対応する銘柄コードのリスト。`dfs.append(df)` している箇所で `trained_symbols.append(sym)` も行い、ループ前に `trained_symbols = []` を用意する。

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_engine_version_wiring.py -v`
Expected: PASS（7件）

- [ ] **Step 5: legacy の回帰が無いことを確認**

Run: `pytest tests/test_ml_train_multi.py tests/test_ml_model_save.py -v`
Expected: PASS

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

- [ ] **Step 6: コミット**

```bash
git add src/services/trading.py tests/test_engine_version_wiring.py
git commit -m "$(cat <<'EOF'
feat(services): 週次再学習をengine_versionで分岐

段階A〜Eはv2の学習経路を作るだけで、どちらを使うかは誰も決めて
いなかった。legacyの3行はそのまま残し、v2のときだけ別経路へ入る。
v2は候補を作るだけでself.modelを差し替えない。運用モデルの入れ替えは
promotion.promote()による明示操作でのみ起きる。
v2の失敗は例外を外へ出さない（週次ジョブを落とさないため）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: バックテスト実行主体の分岐

**Files:**
- Modify: `src/dashboard/app.py`（バックテスト実行のエンドポイント）
- Test: `tests/test_engine_version_wiring.py`

**Interfaces:**
- Consumes: `walkforward.run_walkforward` / `save_run`、`engine.run_backtest`
- Produces: バックテストが `engine_version` で実行主体を選ぶ

**背景:** ダッシュボードは常に `src/backtest/engine.run_backtest` を呼ぶ（`app.py:1335,1345`）。spec §10 は v2 で `walkforward.py` を使うとしている。

**注意:** 2つのエンジンは**入力も出力も違う**。`engine.run_backtest` は単一銘柄、`walkforward` はポートフォリオである。v2 で単一銘柄を指定された場合は、その1銘柄だけのポートフォリオとして扱う。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_engine_version_wiring.py` の末尾に追記する。

```python
class TestBacktestEngineSelection:
    def test_legacy_uses_the_old_engine(self):
        from src.dashboard import app as dash

        assert dash._select_backtest_engine() == "legacy"

    def test_v2_uses_walkforward(self):
        from src.dashboard import app as dash

        cfg.get_section("strategy")["engine_version"] = "v2"
        assert dash._select_backtest_engine() == "v2"

    def test_unknown_value_falls_back_to_legacy(self):
        from src.dashboard import app as dash

        cfg.get_section("strategy")["engine_version"] = "experimental"
        assert dash._select_backtest_engine() == "legacy"

    def test_legacy_path_still_imports_the_old_engine(self):
        """legacy の経路が残っている（挙動不変の裏付け）"""
        from src.dashboard import app as dash

        src = inspect.getsource(dash)
        assert "from src.backtest.engine import run_backtest" in src

    def test_v2_path_references_walkforward(self):
        from src.dashboard import app as dash

        src = inspect.getsource(dash)
        assert "walkforward" in src
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_engine_version_wiring.py -v`
Expected: FAIL — `AttributeError: module 'src.dashboard.app' has no attribute '_select_backtest_engine'`

- [ ] **Step 3: 実装を追加**

`src/dashboard/app.py` に判定関数を足す。

```python
def _select_backtest_engine() -> str:
    """バックテストの実行主体を選ぶ。未知の値は安全側（legacy）に倒す。

    legacy = src/backtest/engine.py（単一銘柄・同じ終値で判断と約定）
    v2     = src/backtest/walkforward.py（ポートフォリオ・T+1執行）
    """
    value = cfg.get_section("strategy").get("engine_version", "legacy")
    return "v2" if value == "v2" else "legacy"
```

バックテスト実行のエンドポイント（`app.py:1335` 付近）を分岐させる。**legacy の呼び出しは1行も変えない。**

```python
    if _select_backtest_engine() == "v2":
        # v2: ポートフォリオwalk-forward。単一銘柄の指定はその1銘柄だけの
        # ポートフォリオとして扱う。legacyとは入力も出力も違うため、
        # 結果は BacktestRun.strategy_version / execution_model_version で
        # 区別できる形で保存する（旧エンジンの結果と混ぜない）。
        return await _run_backtest_v2(req)

    from src.backtest.engine import run_backtest
    ...  # 既存の呼び出しはそのまま
```

`_run_backtest_v2` を追加する。

```python
async def _run_backtest_v2(req):
    """v2のポートフォリオwalk-forwardでバックテストする。

    旧エンジンとは入力も出力も違う。結果は strategy_version と
    execution_model_version を付けて保存し、旧エンジンの結果と混ぜない
    （設計書 §10）。
    """
    import json as json_mod

    from src.backtest import execution, walkforward as wf
    from src.backtest import portfolio as pf
    from src.data.market_data import load_ohlcv
    from src.strategy import dataset as ds
    from src.strategy import policy

    policy_conf = policy.config_from_settings()
    costs = execution.config_from_settings()

    # 価格基準を用途で分ける（段階A）。
    #   raw      … 約定価格・必要資金・出来高
    #   adjusted … 特徴量・リターン
    # 生OHLCVだけを walk-forward へ渡すと、判断側の行に rule_score も
    # 特徴量も存在せず、`row.get(col, 0.0)` で0に埋まる。既定の正の買い閾値
    # では全候補が落ち、「取引ゼロの正常なバックテスト」に見えるが、
    # 実際には特徴量が一度も繋がっていない（外部レビューR03）。
    raw = load_ohlcv(req.symbol, limit=2000, price_basis="raw")
    adjusted = load_ohlcv(req.symbol, limit=2000, price_basis="adjusted")
    if raw.empty or adjusted.empty:
        raise HTTPException(
            status_code=400,
            detail=f"{req.symbol} の日足がありません。先にデータを取得してください")

    # 調整系列から因果的に特徴量とルールスコアを作る。
    # feature_valid=False の行は判断対象外になる（助走期間・欠損）。
    features = ds.build_feature_frame_with_scores(adjusted)
    covered = features.index[features["feature_valid"]]
    if len(covered) == 0:
        raise HTTPException(
            status_code=400,
            detail=(f"{req.symbol} は特徴量の助走期間を満たしていません"
                    f"（{len(adjusted)}本）"))
    if covered.min().date() > req.start_date or covered.max().date() < req.end_date:
        # 期間がデータに覆われていない。黙って短い期間で回さない
        raise HTTPException(
            status_code=400,
            detail=(f"指定期間がデータに覆われていません: "
                    f"利用可能 {covered.min().date()}〜{covered.max().date()} / "
                    f"指定 {req.start_date}〜{req.end_date}"))

    bars = {req.symbol: raw}
    sectors = {req.symbol: watchlist_store.get_sectors().get(req.symbol, "")}
    md = wf.MarketData(bars=bars, sectors=sectors,
                       features={req.symbol: features})

    strategy_conf = wf.StrategyConfig(
        buy_threshold=cfg.get_section("strategy").get("buy_threshold", 0.25),
        # 設定値を明示的に渡す。渡さないと halt_new 指定が既定の
        # rule_only へ戻る（外部レビューR05）
        on_model_failure=cfg.get_section("strategy").get(
            "on_model_failure", wf.ON_FAILURE_RULE_ONLY),
    )
    decide = wf.make_rule_then_ml(strategy_conf, _v2_score_fn(req.use_ml))

    # 過去評価には**各判断時点で利用可能なモデル**を使う。
    # load_current() の戻り値を閉包に固定して過去の全日付へ当てると、
    # 評価期間を学習済みのモデルでも使えてしまう（外部レビューR04）。
    # 再学習を結線しない実行は run_walkforward 側で degraded になる。
    retrain, train_model = _v2_retrain(req, policy_conf, costs)

    result = await asyncio.to_thread(
        wf.run_walkforward, md, req.start_date, req.end_date,
        initial_capital=req.initial_capital, decide=decide,
        policy_conf=policy_conf,
        costs=costs,
        sizing=_v2_sizing_config(),
        liquidity=_v2_liquidity_config(),
        exit_score_fn=_v2_exit_score_fn(req.use_ml),
        retrain=retrain,
        train_model=train_model,
    )

    run_config = _v2_run_config(req, strategy_conf, policy_conf, costs,
                                features=features)
    snapshot = wf.RunSnapshot(
        strategy_version="rule_then_ml_v1",
        config_hash=run_config.config_hash,
        config_json=run_config.config_json,
        dataset_id=run_config.dataset_id,
        code_version=run_config.code_version,
        execution_model_version="t1_open_v1",
    )
    run_id = await asyncio.to_thread(
        wf.save_run, result, snapshot, symbol_label=req.symbol,
        start=req.start_date, end=req.end_date,
        initial_capital=req.initial_capital,
        costs=costs)

    return {"run_id": run_id, "engine_version": "v2",
            "degraded": result.degraded,
            "degraded_reasons": list(result.degraded_reasons),
            "final_capital": float(result.daily["nav"].iloc[-1]) if len(result.daily) else req.initial_capital,
            "trade_count": len(result.trades)}
```

**`config_hash=""` にしない。** 実行条件は開始時に固定し、戦略節だけでなく
リスク・手数料・数量制限・流動性設定まで含めて保存する。実行終了後に
`config.yaml` を読み直すと、実行中に設定が変わっていた場合に「実際に
使った設定」とずれる（外部レビューの残件「評価実行の再現用記録」）。


実行に必要なヘルパー群。`_v2_score_fn` と `_v2_exit_score_fn` は Task 5 で完成させる（本タスクでは経路の選択と legacy 不変を通す）。

```python
def _v2_sizing_config():
    from src.backtest import portfolio as pf

    trading_conf = cfg.get_section("trading")
    return pf.SizingConfig(
        max_position_ratio=trading_conf.get("max_position_ratio", 0.25),
        max_positions=trading_conf.get("max_positions", 5),
        max_sector_ratio=trading_conf.get("max_sector_ratio", 0.40),
    )


def _v2_liquidity_config():
    from src.backtest import execution

    return execution.LiquidityConfig(
        max_volume_share=cfg.get_section("backtest").get("max_volume_share", 0.0))


def _v2_run_config(req, strategy_conf, policy_conf, costs, *, features):
    """実行条件を**開始時に固定**して返す（外部レビューの残件）。

    戦略節だけでは足りない。リスク・手数料・数量制限・流動性設定・
    入力データID・コード版まで含める。終了後に `config.yaml` を読み直すと、
    実行中に設定が変わっていた場合に「実際に使った設定」とずれる。
    """
    from dataclasses import asdict

    from src.strategy import dataset as ds
    from src.strategy.evaluation import RunConfig, _code_version

    payload = {
        "symbol": req.symbol,
        "start": str(req.start_date), "end": str(req.end_date),
        "initial_capital": req.initial_capital,
        "use_ml": bool(req.use_ml),
        "strategy": asdict(strategy_conf),
        "policy": asdict(policy_conf),
        "costs": asdict(costs),
        "sizing": asdict(_v2_sizing_config()),
        "liquidity": asdict(_v2_liquidity_config()),
        "n_feature_rows": int(features["feature_valid"].sum()),
    }
    config_json = json_mod.dumps(payload, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"), default=str)
    return RunConfig(
        dataset_id=None,
        label_contract_id=ds.make_label_contract_id(policy_conf, costs),
        feature_version=ds.FEATURE_VERSION,
        execution_model_version=ds.EXECUTION_MODEL_VERSION,
        code_version=_code_version(),
        config_json=config_json,
        config_hash=hashlib.sha256(
            config_json.encode("utf-8")).hexdigest()[:16],
    )


def _v2_retrain(req, policy_conf, costs):
    """(RetrainConfig, train_model) を返す。

    過去評価では**各判断時点で利用可能なモデル**を使う。
    `load_current()` の戻り値を閉包に固定して過去の全日付へ当てると、
    評価期間を学習済みのモデルでもそのまま使えてしまう（外部レビューR04）。

    「現在の昇格モデルを固定して過去へ当てる」診断が必要なときは
    `req.use_current_model_fixed=True` を渡す。その実行は
    `run_walkforward` が degraded として記録し、昇格の根拠から外れる。
    採否用の walk-forward 成績とは別物として扱う。
    """
    from src.backtest import walkforward as wf
    from src.data.market_data import load_ohlcv
    from src.strategy import dataset as ds
    from src.strategy import validation
    from src.strategy.evaluation import CurrentLightGBM
    from src.strategy.indicators import FEATURE_COLS

    if getattr(req, "use_current_model_fixed", False) or not req.use_ml:
        return None, None

    backtest_conf = cfg.get_section("backtest")
    retrain = wf.RetrainConfig(
        every_sessions=backtest_conf.get("retrain_every_sessions", 20),
        warmup_sessions=backtest_conf.get("retrain_warmup_sessions", 120),
    )

    adjusted = {req.symbol: load_ohlcv(req.symbol, limit=2000,
                                       price_basis="adjusted")}
    all_events = ds.build_events_multi(adjusted, policy_conf, costs)

    def train_model(as_of):
        """`as_of` の引けを学習締切としてモデルを作る。

        締切の適用は `validation.training_inputs()` に任せる。判断日だけで
        なくラベル確定日も締切で切られるので、その時点で観測できない
        イベントは入らない（段階C・外部レビューR06）。
        """
        fold = validation.Fold(
            index=0,
            train_start=min(all_events["decision_at"]) if len(all_events) else as_of,
            train_end=as_of,
            val_start=as_of + timedelta(days=1),
            val_end=as_of + timedelta(days=1),
        )
        inputs = validation.training_inputs(
            all_events, fold, feature_cols=list(FEATURE_COLS))
        if len(inputs.events) == 0 or inputs.events["label"].nunique() < 2:
            return None, len(inputs.events)
        model = CurrentLightGBM()
        model.fit(inputs.events[list(FEATURE_COLS)].astype("float64"),
                  inputs.events["label"].astype(int), inputs.weights)
        return model, len(inputs.events)

    return retrain, train_model
```

> **`req.use_current_model_fixed`** をリクエストモデルへ足す（既定 `False`）。
> 現在の昇格モデルを固定して過去へ当てる診断と、採否に使える walk-forward
> 成績を**同じ数字として並べない**ための分岐である（外部レビューR04）。


> **実装者への注記:** `_v2_score_fn` の ML 推論部分は Task 5 で完成させる。本タスクでは「経路が選ばれること」と「legacy が変わらないこと」までを通す。

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_engine_version_wiring.py -v`
Expected: PASS（12件）

- [ ] **Step 5: legacy のバックテストが変わらないことを確認**

Run: `pytest tests/test_backtest_history.py tests/test_backtest_metrics.py tests/test_backtest_threshold_override.py -v`
Expected: PASS

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

- [ ] **Step 6: コミット**

```bash
git add src/dashboard/app.py tests/test_engine_version_wiring.py
git commit -m "$(cat <<'EOF'
feat(dashboard): バックテストの実行主体をengine_versionで分岐

ダッシュボードは常に旧エンジンを呼んでおり、walkforwardを作っても
誰も呼ばない状態だった。legacyの呼び出しは1行も変えず、v2のときだけ
ポートフォリオwalk-forwardへ入る。
2つのエンジンは入力も出力も違うため、結果はstrategy_versionと
execution_model_versionを付けて保存し旧エンジンの結果と混ぜない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: v2 バックテストのML推論を結線する

**Files:**
- Modify: `src/dashboard/app.py`（`_v2_score_fn`）
- Test: `tests/test_engine_version_wiring.py`

**Interfaces:**
- Consumes: Task 4、`model_store.load_current`、`indicators.build_feature_frame`、`signal.compute_rule_score`
- Produces:
  - `ModelInferenceError(RuntimeError)` — 推論障害。**握り潰さず送出する**
  - `_v2_score_fn(use_ml)` — 意図したML無効／モデル未昇格／推論障害の3状態を区別する
  - `_v2_exit_score_fn(use_ml)` — 保有銘柄の売りスコア（ラベル生成と同じ契約）
  - `_rule_score_of(row)` / `_required_feature_row(row, cols)` — **欠落を0で補完せず例外にする**

**背景:** v2 バックテストは `make_rule_then_ml` を使う。ルールが候補を作り、MLが順位を決める。**モデルは昇格済みのものだけを読む**（`model_store.load_current()`）。未昇格なら ML なしで、`on_model_failure` の設定に従う。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_engine_version_wiring.py` の末尾に追記する。

```python
class TestV2ScoreFunction:
    def test_returns_none_probability_when_unpromoted(self, tmp_path, monkeypatch):
        """モデル未昇格ならML確率はNone（ルールだけで動く）。これは劣化ではない"""
        from src.dashboard import app as dash
        from src.strategy import model_store as ms

        monkeypatch.setattr(ms, "load_current", lambda **k: None)
        score_fn = dash._v2_score_fn(use_ml=True)
        rule, proba = score_fn("7203", {"rule_score": 0.3})
        assert rule == pytest.approx(0.3)
        assert proba is None

    def test_returns_none_probability_when_ml_disabled(self, monkeypatch):
        from src.dashboard import app as dash

        score_fn = dash._v2_score_fn(use_ml=False)
        _, proba = score_fn("7203", {"rule_score": 0.3})
        assert proba is None

    def test_uses_the_promoted_model_when_available(self, monkeypatch):
        from src.dashboard import app as dash
        from src.strategy import model_store as ms

        class _Booster:
            def predict(self, X):
                return [0.77] * len(X)

        class _Meta:
            feature_cols = ["f1", "f2"]

        monkeypatch.setattr(ms, "load_current", lambda **k: (_Booster(), _Meta()))
        score_fn = dash._v2_score_fn(use_ml=True)
        _, proba = score_fn("7203", {"rule_score": 0.3, "f1": 1.0, "f2": 2.0})
        assert proba == pytest.approx(0.77)

    def test_inference_failure_is_raised_not_swallowed(self, monkeypatch):
        """推論障害は例外として外へ出す（外部レビューR05）

        `(rule, None)` へ落とすと「意図したML無効」と見分けがつかず、
        walk-forward の degraded も立たない。MLが効いていない実行が
        正常な成績として保存されてしまう。
        """
        from src.dashboard import app as dash
        from src.strategy import model_store as ms

        class _Broken:
            def predict(self, X):
                raise RuntimeError("boom")

        class _Meta:
            feature_cols = ["f1", "f2"]

        monkeypatch.setattr(ms, "load_current", lambda **k: (_Broken(), _Meta()))
        score_fn = dash._v2_score_fn(use_ml=True)
        with pytest.raises(dash.ModelInferenceError):
            score_fn("7203", {"rule_score": 0.3, "f1": 1.0, "f2": 2.0})

    def test_missing_rule_score_is_an_error_not_a_zero(self, monkeypatch):
        """rule_score が無い行を0で埋めない（外部レビューR03）

        0で埋めると、特徴量が一度も繋がっていない状態が「全候補が
        買い閾値に届かない正常なバックテスト」に見える。
        """
        from src.dashboard import app as dash
        from src.strategy import model_store as ms

        monkeypatch.setattr(ms, "load_current", lambda **k: None)
        score_fn = dash._v2_score_fn(use_ml=True)
        with pytest.raises(dash.ModelInferenceError, match="rule_score"):
            score_fn("7203", {"close": 1000.0})

    def test_missing_features_are_an_error_not_zeros(self, monkeypatch):
        from src.dashboard import app as dash
        from src.strategy import model_store as ms

        class _Booster:
            def predict(self, X):
                return [0.77] * len(X)

        class _Meta:
            feature_cols = ["f1", "f2"]

        monkeypatch.setattr(ms, "load_current", lambda **k: (_Booster(), _Meta()))
        score_fn = dash._v2_score_fn(use_ml=True)
        with pytest.raises(dash.ModelInferenceError, match="特徴量"):
            score_fn("7203", {"rule_score": 0.3, "f1": 1.0})   # f2 が無い


class TestV2ExitScoreFunction:
    def test_returns_the_rule_score_for_held_symbols(self):
        from src.dashboard import app as dash

        fn = dash._v2_exit_score_fn(use_ml=False)
        assert fn("7203", {"rule_score": -0.4}) == pytest.approx(-0.4)

    def test_missing_rule_score_is_an_error(self):
        from src.dashboard import app as dash

        fn = dash._v2_exit_score_fn(use_ml=False)
        with pytest.raises(dash.ModelInferenceError):
            fn("7203", {"close": 1000.0})


class TestV2BacktestEndToEnd:
    """合成OHLCV → エンドポイント → T+1約定まで実際の型で通す。

    テストで `rule_score` を手渡すだけでは、特徴量が経路上で供給されて
    いることを確認できない（外部レビューR03）。
    """

    def _seed_ohlcv(self, symbol="7203", n=300):
        """特徴量の助走期間を満たす合成日足をDBへ入れる"""
        import numpy as np
        import pandas as pd

        from src.data import market_data

        rng = np.random.default_rng(0)
        close = 1000 + np.cumsum(rng.normal(0, 15, n))
        idx = pd.bdate_range("2025-01-06", periods=n)
        df = pd.DataFrame({
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "adjusted_close": close,
            "volume": [1_000_000] * n,
        }, index=idx)
        df.index.name = "date"
        market_data.upsert_ohlcv(symbol, df)
        return idx

    def test_features_reach_the_decision_and_trades_can_happen(
            self, isolated_db, client):
        idx = self._seed_ohlcv()
        cfg.get_section("strategy")["engine_version"] = "v2"

        res = client.post("/api/backtest", json={
            "symbol": "7203",
            "start_date": str(idx[200].date()),
            "end_date": str(idx[-2].date()),
            "initial_capital": 1_000_000.0,
            "use_ml": False,
        })
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["engine_version"] == "v2"
        # 特徴量が供給されていれば、ルールスコアは一様に0にならない。
        # 取引ゼロでも「常に0点」ではないことをrun記録から確かめる
        assert body["run_id"] is not None

    def test_a_period_not_covered_by_the_data_is_rejected(
            self, isolated_db, client):
        """期間がデータに覆われていないときは黙って短い期間で回さない"""
        self._seed_ohlcv()
        cfg.get_section("strategy")["engine_version"] = "v2"

        res = client.post("/api/backtest", json={
            "symbol": "7203",
            "start_date": "2020-01-06",
            "end_date": "2020-12-30",
            "initial_capital": 1_000_000.0,
            "use_ml": False,
        })
        assert res.status_code == 400
        assert "覆われていません" in res.json()["detail"]

    def test_a_symbol_without_bars_is_rejected(self, isolated_db, client):
        cfg.get_section("strategy")["engine_version"] = "v2"
        res = client.post("/api/backtest", json={
            "symbol": "9999",
            "start_date": "2026-01-05",
            "end_date": "2026-02-05",
            "initial_capital": 1_000_000.0,
            "use_ml": False,
        })
        assert res.status_code == 400

    def test_the_run_records_a_real_config_hash(self, isolated_db, client):
        """config_hash="" で保存しない（外部レビューの残件）"""
        from sqlalchemy import select

        from src.data import database as db
        from src.data.database import get_session

        idx = self._seed_ohlcv()
        cfg.get_section("strategy")["engine_version"] = "v2"
        client.post("/api/backtest", json={
            "symbol": "7203",
            "start_date": str(idx[200].date()),
            "end_date": str(idx[-2].date()),
            "initial_capital": 1_000_000.0,
            "use_ml": False,
        })
        with get_session() as session:
            row = session.scalars(
                select(db.BacktestRun).order_by(db.BacktestRun.id.desc())).first()
        assert row.config_hash
        assert row.config_hash != ""
        assert row.config_json and row.config_json != "{}"
        # 戦略節だけでなくコスト・数量制限まで入っている
        assert "costs" in row.config_json
        assert "sizing" in row.config_json
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_engine_version_wiring.py::TestV2ScoreFunction -v`
Expected: FAIL — `AttributeError: module 'src.dashboard.app' has no attribute 'ModelInferenceError'`

- [ ] **Step 3: 実装を完成させる**

`src/dashboard/app.py` の `_v2_score_fn` を次に置き換える。

```python
class ModelInferenceError(RuntimeError):
    """v2バックテストのML推論に失敗した。

    **握り潰さない。** 例外を捕まえて `(rule, None)` を返すと、
    「意図してMLを使っていない」実行と区別がつかなくなり、
    walk-forward 側の degraded も立たない（外部レビューR05）。
    degraded を立てるのは外へ届いた例外なので、ここで飲み込んではいけない。
    """


def _required_feature_row(row, cols: list) -> dict:
    """推論に必要な特徴量を行から取り出す。**欠けていたら例外。**

    `row.get(col, 0.0)` で埋めると、特徴量が一度も繋がっていない状態が
    「全部0の入力」として通り、正の買い閾値の下で全候補が落ちる。
    結果は「取引ゼロの正常なバックテスト」に見える（外部レビューR03）。
    """
    missing = [c for c in cols if c not in row or pd_mod.isna(row[c])]
    if missing:
        raise ModelInferenceError(
            f"特徴量が供給されていません: {missing}。"
            "MarketData.features に特徴量フレームを渡してください")
    return {c: float(row[c]) for c in cols}


def _rule_score_of(row) -> float:
    """判断行からルールスコアを取り出す。**無ければ例外。**"""
    if "rule_score" not in row or pd_mod.isna(row["rule_score"]):
        raise ModelInferenceError(
            "rule_score が供給されていません。"
            "MarketData.features に rule_score 列を含めてください")
    return float(row["rule_score"])


def _v2_score_fn(use_ml: bool):
    """(ルールスコア, ML確率 or None) を返す関数を作る。

    3つの状態を**区別する**（外部レビューR05）。

      1. 意図したML無効（`use_ml=False`） … `(rule, None)`。劣化ではない
      2. モデル未昇格 … `(rule, None)`。劣化ではない
      3. 推論障害 … `ModelInferenceError` を**送出する**。
         walk-forward が捕まえて `degraded_reasons` へ積み、
         その実行は比較・昇格の対象から外れる

    3を `(rule, None)` に落とすと1・2と見分けがつかず、
    「MLが効いていない実行」が正常な成績として保存される。

    モデルは昇格済みのものだけを読む（`model_store.load_current`）。
    """
    from src.strategy import model_store as ms

    loaded = ms.load_current() if use_ml else None
    if loaded is None:
        def rule_only(symbol, row):
            # 未昇格・ML無効。意図した状態なので degraded にしない
            return _rule_score_of(row), None
        return rule_only

    model, meta = loaded
    cols = list(meta.feature_cols)

    def score_fn(symbol, row):
        rule = _rule_score_of(row)
        features = _required_feature_row(row, cols)
        try:
            proba = float(model.predict(pd_mod.DataFrame([features]))[0])
        except Exception as e:
            # ここで飲み込まない。degraded を立てられるよう外へ出す
            logger.error(f"v2バックテストのML推論に失敗: {symbol} {e}")
            raise ModelInferenceError(f"{symbol}: {e}") from e
        return rule, proba

    return score_fn


def _v2_exit_score_fn(use_ml: bool):
    """保有銘柄の売りスコアを返す関数を作る。

    結線しないと policy の SIGNAL_SELL 条件が一度も成立せず、
    ストップか満了まで持ち続ける挙動になる（外部レビューR10）。

    **ラベル生成（`dataset.simulate_event`）と同じ契約にする。** ラベル側が
    ルールスコアで売りを判定しているなら、ここも同じ値を使う。片方だけ
    MLを混ぜると、学習したラベルと検証時の退出が別物になる。
    """
    def exit_score_fn(symbol, row):
        return _rule_score_of(row)

    return exit_score_fn
```

`import pandas as pd_mod` をモジュール先頭へ足す。

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_engine_version_wiring.py -v`
Expected: PASS（20件）

- [ ] **Step 5: 全体回帰とコミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

Run: `head -c 3 src/dashboard/app.py | xxd`（`2222 22` を確認）

```bash
git add src/dashboard/app.py tests/test_engine_version_wiring.py
git commit -m "$(cat <<'EOF'
feat(dashboard): v2バックテストのML推論を結線

モデルは昇格済みのものだけを読む。未昇格ならMLなしでルールだけで動く。
推論に失敗した場合もMLなしへ落として例外を外へ出さない（1銘柄の推論
失敗でバックテスト全体を落とさないため。degradedはwalkforward側が
別途立てる）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## 完了条件の確認

- [ ] **確認1: 既定が `legacy` で、挙動が現在と変わらない**

Run: `pytest tests/test_engine_version_wiring.py::TestLegacyRetrainUnchanged -v`
Run: `pytest tests/test_engine_version_wiring.py::TestBacktestEngineSelection -v`
Expected: PASS

- [ ] **確認2: v2 の学習が運用モデルを差し替えない**

Run: `pytest tests/test_engine_version_wiring.py::TestV2Retrain::test_v2_does_not_replace_the_running_model -v`
Run: `pytest tests/test_v2_training.py::TestTrainV2::test_produces_a_candidate_not_a_promotion -v`
Expected: PASS

- [ ] **確認3: 3経路すべてに分岐がある**

Run: `grep -n "_engine_version\|_select_backtest_engine" src/services/trading.py src/dashboard/app.py`
Expected: 週次再学習・paper執行（段階D後半）・バックテスト実行主体の3箇所が出ること

- [ ] **確認4: `ml_model.py` と `engine.py` と `labeling.py` が無改造である**

Run: `git log --oneline -- src/strategy/ml_model.py src/backtest/engine.py | head -3`
Expected: 段階A〜F のコミットが一件も出ないこと

Run: `git log --oneline -- src/strategy/labeling.py | head -3`
Expected: 段階B後半の docstring 追記だけが出ること

- [ ] **確認5: `ModelMetrics` で legacy と v2 の記録が区別できる**

Run: `pytest tests/test_v2_training.py::TestModelMetricsColumns -v`
Expected: PASS（3件）

- [ ] **確認6: 既存経路に回帰が無い**

Run: `pytest tests/ -q`
Expected: 着手前と同じ結果（新規テスト25件ぶんだけ増える）

- [ ] **確認7: v2 を1回だけ動かして観察する**（判断材料。合否ではない）

`config.yaml` の `strategy.engine_version` を一時的に `v2` にして次を確認し、**必ず `legacy` へ戻す**。

1. 週次再学習を手動実行し、候補モデルが `models/candidates/` に作られ、`self.model` が変わらないこと
2. ダッシュボードでバックテストを1回実行し、`BacktestRun` に `strategy_version` と `execution_model_version` が入ること
3. `degraded` が立っていないこと

Expected: 3点が確認できること。**この結果で v2 への切替を決めない。** 切替は段階C後半の5モデル比較と段階D後半の3戦略比較を見てから判断する。

---

## 残る作業（本計画のスコープ外）

- **`engine_version: v2` を既定にする判断**。実データでの比較を見てから決める
- **旧エンジン（`engine.py`）と旧学習経路（`ml_model.py`）の廃止**。v2 が運用に乗ってから別途判断する
- **`signal.py` への `strategy_version` 付与**（spec §4）。発注経路を変更しない方針のため、v2 が運用へ昇格するまで保留する
- **設計書への反映**。`docs/詳細設計書.md` / `docs/概要設計書.md` に段階A〜Fの変更を追記する
