# ML評価基盤 段階C後半（評価の実行と記録）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 同じ入力・同じ分割で5つのモデルを比較し、予測明細を残して後から指標も売買判断も再計算できるようにする。

**Architecture:** `evaluation.py` が段階C前半の `training_inputs()` を入口としてモデルを学習・評価する。閾値選択と確率校正は**内側 fold だけ**で行い、外側 fold は最終評価専用として一切触らない。予測はイベント単位で保存し、実績ラベルは別テーブルに後から関連付ける（予測時点では未確定でありうるため）。

**Tech Stack:** Python 3.11 / pandas 2.1.4 / numpy 1.26.2 / scikit-learn 1.3.2 / lightgbm 4.1.0 / SQLAlchemy 2.0.23 / pytest

**Spec:** `docs/superpowers/specs/2026-09-10-ml-evaluation-foundation-design.md`（§7・§8・§11・§12・§14）

**前提:** 段階C前半（`docs/superpowers/plans/2026-09-11-ml-evaluation-stage-c1.md`）が完了していること。本計画は `validation.Fold` / `calendar_folds` / `split_events` / `inner_folds` / `training_inputs` / `TrainingInputs` / `Preprocessor` / `apply_preprocessor` / `apply_training_window` と `dataset.STATUS_RESOLVED` に依存する。

## Global Constraints

- 日時は **JST naive**。現在時刻は `src/core/clock.now()` / `clock.today()` を使い、`datetime.now()` を直接呼ばない。
- **`validation.py` を変更しない。** 同モジュールは「どのイベントを学習に使ってよいか」だけを担当し、`sklearn` / `lightgbm` を import しない制約を持つ（段階C前半のグローバル制約）。モデルを扱うコードは本計画で新設する `evaluation.py` に置く。
- **外側 fold は最終評価専用。** early stopping・閾値選択・確率校正・学習窓選択は内側 fold だけで行い、外側 fold の値を選択に使わない。
- **学習に入る入力は必ず `validation.training_inputs()` を通す。** 独自に events を絞り込まない（漏れの検査点を1箇所に保つため）。
- 新規のDB列・テーブルはすべて nullable。`create_all` が新規テーブルを作るためマイグレーション作業は不要。
- **既存の公開関数の挙動を変えない。** `ml_model.train()` / `train_multi()` / `_fit()` / `labeling.build_training_set()` / `indicators.build_features()` は本計画で一切変更しない。`src/backtest/engine.py`・`src/services/trading.py` も変更しない。
- ファイルは UTF-8 **BOM無し**・LF で保存する。確認は `git show <rev>:<path>` でコミット済みblobに対して行う。
- テストは `pytest tests/<file>.py -v` で実行する。ネットワークへ出るテストを書かない。乱数を使う箇所は `random_state` を固定する。
- コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` を付ける。実装者自身のモデル名を書かない。

---

## 本計画における spec からの意図的な差分（1点）

### モデルを扱う層を `evaluation.py` として分ける

spec §4 の構造表は段階Cの新規モジュールとして `src/strategy/validation.py` だけを挙げているが、**モデルの学習・評価・記録は `src/strategy/evaluation.py` に置く**。

理由: 段階C前半で `validation.py` に「モデルを学習しない・`sklearn` / `lightgbm` を import しない」という制約を置いた。分割の正しさ（どのイベントを使ってよいか）は重い依存なしに高速に検証できるべきで、実際その制約のおかげで段階C前半のテストは 42 ステップすべてが pandas/numpy だけで完結している。ここへモデルを混ぜると、分割の検査のたびに LightGBM の学習が走ることになる。

責務は次のとおり分ける。

| モジュール | 担当 |
|---|---|
| `validation.py` | どのイベントを学習に使ってよいか（分割・purge・重み・前処理の統計量） |
| `evaluation.py` | そのイベントでモデルを学習し、評価し、予測明細を残す |

---

## File Structure

| ファイル | 責務 |
|---|---|
| `src/strategy/evaluation.py`（新規） | モデルの共通インターフェースと5つの比較対象、確率校正、閾値選択、外側foldの評価ループ、指標の算出 |
| `src/data/database.py`（改修） | `Prediction` / `PredictionOutcome` テーブルの追加 |
| `tests/test_evaluation.py`（新規） | 5モデル、校正、閾値、評価ループ、指標、予測明細からの再計算、学習窓の内側比較 |

---

## Task 1: 予測明細のテーブルと保存・関連付け

**Files:**
- Modify: `src/data/database.py`（`Prediction` / `PredictionOutcome` を追加）
- Create: `src/strategy/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `dataset.STATUS_RESOLVED`
- Produces:
  - `Prediction` モデル: `event_id` / `evaluation_run_id` / `model_id` / `predicted_at` / `raw_probability` / `calibrated_probability` / `fold_index` / `purpose`
  - `PredictionOutcome` モデル: `event_id` / `actual_label` / `net_return` / `resolved_at`
  - `PURPOSE_VALIDATION` / `PURPOSE_SHADOW` 定数
  - `save_predictions(predictions: pd.DataFrame, evaluation_run_id: str, model_id: str, *, purpose: str = PURPOSE_VALIDATION) -> int`
  - `save_outcomes(events: pd.DataFrame) -> int`
  - `load_prediction_details(evaluation_run_id: str, model_id: Optional[str] = None) -> pd.DataFrame`

**背景（spec §7・§11）:** 実績ラベルは予測時点では確定していない場合がある（shadow運用が該当する）。**予測を先に保存し、満期後に実績ラベルを関連付ける**更新規約にすることで、検証用の予測と未確定の運用予測を同じテーブルで扱える。サンプル単位の明細があれば、AUCも売買判断も後から再計算できる。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_evaluation.py` を新規作成する。

```python
"""評価の実行と記録（src/strategy/evaluation.py）のテスト

同じ入力・同じ分割で5つのモデルを比較し、予測明細を残して後から
指標も売買判断も再計算できるようにする（spec §7・§14）。
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.core import config as cfg
from src.data import database as db
from src.data.database import get_session
from src.strategy import dataset
from src.strategy import evaluation


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


def _events(n_sessions: int = 60, symbols=("7203", "9984"),
            start=date(2026, 1, 5), holding: int = 2, seed: int = 0) -> pd.DataFrame:
    """テスト用のイベント表。特徴量はラベルと弱く相関させる"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sessions):
        d = start + timedelta(days=i)
        for s in symbols:
            label = int(rng.random() < 0.45)
            rows.append({
                "event_id": f"{s}:{d:%Y%m%d}",
                "symbol": s,
                "decision_at": d,
                "entry_at": d + timedelta(days=1),
                "label_end_at": d + timedelta(days=holding),
                "status": dataset.STATUS_RESOLVED,
                "label": label,
                "net_return": 0.02 if label else -0.015,
                "x1": label + rng.normal(0, 1.0),
                "x2": rng.normal(0, 1.0),
            })
    return pd.DataFrame(rows)


FEATURES = ["x1", "x2"]


class TestPredictionTables:
    def test_save_and_read_back_predictions(self, isolated_db):
        preds = pd.DataFrame({
            "event_id": ["7203:20260105", "9984:20260105"],
            "raw_probability": [0.6, 0.3],
            "calibrated_probability": [0.55, 0.35],
            "fold_index": [0, 0],
        })
        n = evaluation.save_predictions(preds, "run1", "const")
        assert n == 2

        with get_session() as session:
            rows = list(session.scalars(select(db.Prediction)).all())
        assert len(rows) == 2
        assert {r.model_id for r in rows} == {"const"}
        assert {r.evaluation_run_id for r in rows} == {"run1"}
        assert all(r.purpose == evaluation.PURPOSE_VALIDATION for r in rows)
        assert all(r.predicted_at is not None for r in rows)

    def test_shadow_purpose_is_recorded(self, isolated_db):
        preds = pd.DataFrame({
            "event_id": ["7203:20260105"],
            "raw_probability": [0.6],
            "calibrated_probability": [0.55],
            "fold_index": [-1],
        })
        evaluation.save_predictions(
            preds, "run1", "cand", purpose=evaluation.PURPOSE_SHADOW)

        with get_session() as session:
            row = session.scalar(select(db.Prediction))
        assert row.purpose == evaluation.PURPOSE_SHADOW

    def test_outcomes_are_saved_separately_from_predictions(self, isolated_db):
        """予測を先に保存し、実績は別テーブルに後から関連付ける"""
        preds = pd.DataFrame({
            "event_id": ["7203:20260105"],
            "raw_probability": [0.6],
            "calibrated_probability": [0.55],
            "fold_index": [0],
        })
        evaluation.save_predictions(preds, "run1", "const")

        # この時点では実績は無い
        details = evaluation.load_prediction_details("run1")
        assert pd.isna(details["actual_label"].iloc[0])

        events = _events(n_sessions=1, symbols=("7203",))
        events.loc[0, "event_id"] = "7203:20260105"
        events.loc[0, "label"] = 1
        evaluation.save_outcomes(events)

        details = evaluation.load_prediction_details("run1")
        assert details["actual_label"].iloc[0] == 1

    def test_only_resolved_events_become_outcomes(self, isolated_db):
        events = _events(n_sessions=2, symbols=("7203",))
        events.loc[0, "status"] = dataset.STATUS_IMMATURE
        events.loc[0, "label"] = None
        n = evaluation.save_outcomes(events)
        assert n == 1

    def test_details_can_be_filtered_by_model(self, isolated_db):
        preds = pd.DataFrame({
            "event_id": ["7203:20260105"],
            "raw_probability": [0.6],
            "calibrated_probability": [0.55],
            "fold_index": [0],
        })
        evaluation.save_predictions(preds, "run1", "const")
        evaluation.save_predictions(preds, "run1", "lgbm")

        assert len(evaluation.load_prediction_details("run1")) == 2
        assert len(evaluation.load_prediction_details("run1", model_id="lgbm")) == 1
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.strategy.evaluation'`

- [ ] **Step 3: モデルを追加**

`src/data/database.py` の `class Dataset` の直後に追加する。

```python
class Prediction(Base):
    """イベント単位の予測明細。

    サンプル単位の明細を残しておけば、指標も売買判断も後から再計算できる。
    集計済みの数値だけでは「AUC 0.5085 が何を意味するか」を後から検証できない。

    実績ラベルはここに持たない。予測時点では確定していない場合があるため
    （shadow運用が該当する）、PredictionOutcome へ後から関連付ける。
    """
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    event_id = Column(String(64), nullable=False)
    evaluation_run_id = Column(String(64), nullable=False)
    model_id = Column(String(64), nullable=False)
    predicted_at = Column(DateTime, default=clock.now)
    raw_probability = Column(Float)
    calibrated_probability = Column(Float)
    fold_index = Column(Integer)      # 出所fold。shadow等は -1
    purpose = Column(String(16))      # "validation" / "shadow"

    __table_args__ = (
        Index("ix_predictions_run_model", "evaluation_run_id", "model_id"),
        Index("ix_predictions_event_id", "event_id"),
    )


class PredictionOutcome(Base):
    """イベントの実績。予測より後に確定するため別テーブルに持つ。"""
    __tablename__ = "prediction_outcomes"
    id = Column(Integer, primary_key=True)
    event_id = Column(String(64), nullable=False)
    actual_label = Column(Integer)
    net_return = Column(Float)
    resolved_at = Column(DateTime, default=clock.now)

    __table_args__ = (Index("ix_prediction_outcomes_event_id", "event_id", unique=True),)
```

- [ ] **Step 4: 実装を書く**

`src/strategy/evaluation.py` を新規作成する。

```python
"""評価の実行と記録 — 同じ入力・同じ分割でモデルを比較する。

段階C前半の validation.training_inputs() を入口として学習し、外側foldで評価する。
閾値選択と確率校正は**内側foldだけ**で行い、外側foldは最終評価専用として触らない。
現行の ml_model._fit() は early stopping に使った検証データでそのままCV指標を
出しており、報告値が楽観に寄っている（レビュー ML）。

予測はイベント単位で保存し、実績ラベルは PredictionOutcome へ後から関連付ける。
予測時点では実績が確定していない場合があるため（shadow運用）、この分離により
検証用の予測と未確定の運用予測を同じ仕組みで扱える。

分割そのものは validation.py の担当で、本モジュールは events を独自に
絞り込まない（学習に入る入力の検査点を1箇所に保つため）。
"""
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.core import clock
from src.strategy.dataset import STATUS_RESOLVED

PURPOSE_VALIDATION = "validation"
PURPOSE_SHADOW = "shadow"


def save_predictions(predictions: pd.DataFrame, evaluation_run_id: str,
                     model_id: str, *, purpose: str = PURPOSE_VALIDATION) -> int:
    """予測明細を保存する。保存した件数を返す。

    predictions は event_id / raw_probability / calibrated_probability /
    fold_index の列を持つこと。実績ラベルはここでは書かない。
    """
    from src.data.database import Prediction, get_session

    if len(predictions) == 0:
        return 0
    now = clock.now()
    with get_session() as session:
        for _, r in predictions.iterrows():
            session.add(Prediction(
                event_id=str(r["event_id"]),
                evaluation_run_id=evaluation_run_id,
                model_id=model_id,
                predicted_at=now,
                raw_probability=float(r["raw_probability"]),
                calibrated_probability=float(r["calibrated_probability"]),
                fold_index=int(r["fold_index"]),
                purpose=purpose,
            ))
        session.commit()
    return len(predictions)


def save_outcomes(events: pd.DataFrame) -> int:
    """決着したイベントの実績を保存する（既存の event_id は上書きする）。

    予測より後に呼ぶ。未成熟・未約定にはラベルが無いので保存しない。
    """
    from src.data.database import PredictionOutcome, get_session
    from sqlalchemy import select as sa_select

    resolved = events[events["status"] == STATUS_RESOLVED]
    if len(resolved) == 0:
        return 0
    now = clock.now()
    with get_session() as session:
        existing = {
            r.event_id: r
            for r in session.scalars(sa_select(PredictionOutcome)).all()
        }
        for _, r in resolved.iterrows():
            eid = str(r["event_id"])
            label = int(r["label"])
            ret = float(r["net_return"]) if pd.notna(r.get("net_return")) else None
            if eid in existing:
                existing[eid].actual_label = label
                existing[eid].net_return = ret
                existing[eid].resolved_at = now
            else:
                session.add(PredictionOutcome(
                    event_id=eid, actual_label=label,
                    net_return=ret, resolved_at=now))
        session.commit()
    return len(resolved)


def load_prediction_details(evaluation_run_id: str,
                            model_id: Optional[str] = None) -> pd.DataFrame:
    """予測明細に実績を突き合わせて返す（実績が無い行は actual_label が NaN）。

    この明細から指標も売買判断も再計算できる（spec §14 段階C完了条件）。
    """
    from src.data.database import Prediction, PredictionOutcome, get_session
    from sqlalchemy import select as sa_select

    with get_session() as session:
        stmt = sa_select(Prediction).where(
            Prediction.evaluation_run_id == evaluation_run_id)
        if model_id is not None:
            stmt = stmt.where(Prediction.model_id == model_id)
        preds = list(session.scalars(stmt).all())
        outcomes = {
            o.event_id: o
            for o in session.scalars(sa_select(PredictionOutcome)).all()
        }

    rows = []
    for p in preds:
        o = outcomes.get(p.event_id)
        rows.append({
            "event_id": p.event_id,
            "evaluation_run_id": p.evaluation_run_id,
            "model_id": p.model_id,
            "predicted_at": p.predicted_at,
            "raw_probability": p.raw_probability,
            "calibrated_probability": p.calibrated_probability,
            "fold_index": p.fold_index,
            "purpose": p.purpose,
            "actual_label": o.actual_label if o else np.nan,
            "net_return": o.net_return if o else np.nan,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 5: テストを実行して成功を確認**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS（5件）

- [ ] **Step 6: BOM確認とコミット**

Run: `head -c 3 src/strategy/evaluation.py | xxd`（`2222 22` を確認。`efbb bf` なら下記で除去）

```python
for p in ["src/strategy/evaluation.py", "tests/test_evaluation.py", "src/data/database.py"]:
    with open(p, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        with open(p, "wb") as f:
            f.write(data[3:])
```

```bash
git add src/data/database.py src/strategy/evaluation.py tests/test_evaluation.py
git commit -m "$(cat <<'EOF'
feat(data,strategy): イベント単位の予測明細と実績の分離保存を追加

集計済みの数値だけでは「AUC 0.5085が何を意味するか」を後から検証できない。
サンプル単位の明細を残し、指標も売買判断も再計算できるようにする。
実績ラベルは予測時点では未確定でありうる（shadow運用）ため別テーブルに
持ち、満期後に関連付ける。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: モデルの共通インターフェースとベースライン3種

**Files:**
- Modify: `src/strategy/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `validation.Preprocessor` / `apply_preprocessor`
- Produces:
  - `ConstantProbability` — `fit` で学習側の正例率を覚え、全件にその確率を返す
  - `MajorityClass` — 学習側の多数派クラスを 0.0/1.0 の確率として返す
  - `LogisticRegressionModel` — 標準化した特徴量で学習する
  - 各モデルは `name: str` / `fit(X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray) -> None` / `predict_proba(X: pd.DataFrame) -> np.ndarray`（正例確率の1次元配列）を持つ

**背景（spec §7）:** 比較対象は5つに固定する。**深層モデルはこの段階では候補にしない。** 同じ入力・同じ分割で追加価値を示せるかを先に確かめる。定数確率と多数派予測は「ランダムと同じ」の床を2種類の指標で与える（定数確率は Brier / log loss の、多数派は accuracy の床）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_evaluation.py` の末尾に追記する。

```python
def _xy(events: pd.DataFrame):
    X = events[FEATURES].astype("float64")
    y = events["label"].astype(int)
    w = np.ones(len(events))
    return X, y, w


class TestConstantProbability:
    def test_predicts_the_training_positive_rate(self):
        events = _events(n_sessions=40)
        X, y, w = _xy(events)
        m = evaluation.ConstantProbability()
        m.fit(X, y, w)
        p = m.predict_proba(X)
        assert len(p) == len(X)
        assert np.allclose(p, y.mean())

    def test_does_not_look_at_features(self):
        """特徴量を変えても予測は変わらない（床としての性質）"""
        events = _events(n_sessions=40)
        X, y, w = _xy(events)
        m = evaluation.ConstantProbability()
        m.fit(X, y, w)
        shifted = X.copy()
        shifted["x1"] = shifted["x1"] + 100.0
        assert np.allclose(m.predict_proba(X), m.predict_proba(shifted))

    def test_has_a_name(self):
        assert evaluation.ConstantProbability().name == "constant_probability"


class TestMajorityClass:
    def test_predicts_one_when_positives_dominate(self):
        X = pd.DataFrame({"x1": [0.0] * 10, "x2": [0.0] * 10})
        y = pd.Series([1] * 7 + [0] * 3)
        m = evaluation.MajorityClass()
        m.fit(X, y, np.ones(10))
        assert np.allclose(m.predict_proba(X), 1.0)

    def test_predicts_zero_when_negatives_dominate(self):
        X = pd.DataFrame({"x1": [0.0] * 10, "x2": [0.0] * 10})
        y = pd.Series([1] * 3 + [0] * 7)
        m = evaluation.MajorityClass()
        m.fit(X, y, np.ones(10))
        assert np.allclose(m.predict_proba(X), 0.0)

    def test_has_a_name(self):
        assert evaluation.MajorityClass().name == "majority_class"


class TestLogisticRegressionModel:
    def test_learns_a_separable_signal(self):
        """x1がラベルと相関する合成データで、正例に高い確率を付ける"""
        rng = np.random.default_rng(3)
        y = pd.Series([0] * 100 + [1] * 100)
        X = pd.DataFrame({
            "x1": np.concatenate([rng.normal(-2, 0.5, 100), rng.normal(2, 0.5, 100)]),
            "x2": rng.normal(0, 1, 200),
        })
        m = evaluation.LogisticRegressionModel()
        m.fit(X, y, np.ones(len(X)))
        p = m.predict_proba(X)
        assert p[y == 1].mean() > p[y == 0].mean()

    def test_probabilities_are_in_range(self):
        events = _events(n_sessions=60)
        X, y, w = _xy(events)
        m = evaluation.LogisticRegressionModel()
        m.fit(X, y, w)
        p = m.predict_proba(X)
        assert ((p >= 0.0) & (p <= 1.0)).all()

    def test_standardizes_with_training_statistics_only(self):
        """学習時の統計量で変換する。推論側で fit し直さない"""
        events = _events(n_sessions=60)
        X, y, w = _xy(events)
        m = evaluation.LogisticRegressionModel()
        m.fit(X, y, w)
        p_small = m.predict_proba(X.iloc[:5])
        p_full = m.predict_proba(X)
        assert np.allclose(p_small, p_full[:5])

    def test_falls_back_to_constant_on_single_class(self):
        """学習側が片側クラスだけなら定数を返す（例外にしない）"""
        X = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 0.0, 1.0]})
        y = pd.Series([1, 1, 1])
        m = evaluation.LogisticRegressionModel()
        m.fit(X, y, np.ones(3))
        assert np.allclose(m.predict_proba(X), 1.0)

    def test_has_a_name(self):
        assert evaluation.LogisticRegressionModel().name == "logistic_regression"
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.evaluation' has no attribute 'ConstantProbability'`

- [ ] **Step 3: 実装を追加**

`src/strategy/evaluation.py` の末尾に追加する。import に次を足す。

```python
from sklearn.linear_model import LogisticRegression

from src.strategy.validation import Preprocessor, apply_preprocessor, fit_preprocessor
```

```python
class ConstantProbability:
    """学習側の正例率を全件に返す。Brier / log loss の床。

    特徴量を一切見ないため、これを上回れないモデルは「確率の質」で
    何も足していない。
    """
    name = "constant_probability"

    def __init__(self) -> None:
        self._p = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray) -> None:
        self._p = float(np.average(y.astype(float), weights=sample_weight)) \
            if len(y) and np.sum(sample_weight) > 0 else 0.5

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._p, dtype=float)


class MajorityClass:
    """学習側の多数派クラスを 0.0 / 1.0 として返す。accuracy の床。"""
    name = "majority_class"

    def __init__(self) -> None:
        self._label = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray) -> None:
        rate = float(np.average(y.astype(float), weights=sample_weight)) \
            if len(y) and np.sum(sample_weight) > 0 else 0.0
        self._label = 1.0 if rate >= 0.5 else 0.0

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._label, dtype=float)


class LogisticRegressionModel:
    """標準化した特徴量で学習するロジスティック回帰。

    標準化の統計量は**学習時に固定**し、推論側では fit し直さない
    （validation.fit_preprocessor と同じ規約）。
    """
    name = "logistic_regression"

    def __init__(self, feature_cols: Optional[list] = None) -> None:
        self._feature_cols = feature_cols
        self._pre: Optional[Preprocessor] = None
        self._model: Optional[LogisticRegression] = None
        self._constant: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray) -> None:
        cols = self._feature_cols or list(X.columns)
        self._feature_cols = cols
        frame = X.copy()
        frame["label"] = y.values
        self._pre = fit_preprocessor(frame, feature_cols=cols)

        if y.nunique() < 2:
            # 片側クラスだけでは境界を引けない。定数にフォールバックする
            self._constant = float(y.iloc[0]) if len(y) else 0.5
            self._model = None
            return

        self._constant = None
        Z = apply_preprocessor(self._pre, X, feature_cols=cols)
        self._model = LogisticRegression(max_iter=1000, random_state=42)
        self._model.fit(Z, y, sample_weight=sample_weight)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._constant is not None:
            return np.full(len(X), self._constant, dtype=float)
        Z = apply_preprocessor(self._pre, X, feature_cols=self._feature_cols)
        return self._model.predict_proba(Z)[:, 1]
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS（16件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/evaluation.py tests/test_evaluation.py
git commit -m "$(cat <<'EOF'
feat(strategy): 比較対象のベースライン3種を追加

定数確率はBrier/log lossの床、多数派予測はaccuracyの床を与える。
これを上回れないモデルは同じ入力・同じ分割で何も足していない。
ロジスティック回帰は標準化の統計量を学習時に固定し、推論側で
fitし直さない。片側クラスだけの場合は例外にせず定数へ落とす。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: LightGBM 2種

**Files:**
- Modify: `src/strategy/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: Task 2 のインターフェース
- Produces:
  - `SmallLightGBM` — `n_estimators=50` / `num_leaves=7` / `learning_rate=0.05`
  - `CurrentLightGBM` — 現行 `ml_model._fit()` と同じ `n_estimators=200` / `num_leaves=31` / `learning_rate=0.05`
  - `default_model_factories() -> dict[str, Callable[[], object]]` — 5モデルの生成関数

**注意:** 現行 `ml_model._fit()` は fold ごとに early stopping を行い、その `best_iteration_` の平均を最終モデルの木数にしている。本モジュールでは early stopping を**内側 fold**で行う（Task 5）。ここでは素の分類器として実装し、`fit` は追加の検証データを取らない。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_evaluation.py` の末尾に追記する。

```python
class TestLightGbmModels:
    def _separable(self):
        rng = np.random.default_rng(11)
        y = pd.Series([0] * 150 + [1] * 150)
        X = pd.DataFrame({
            "x1": np.concatenate([rng.normal(-1.5, 1.0, 150), rng.normal(1.5, 1.0, 150)]),
            "x2": rng.normal(0, 1, 300),
        })
        return X, y

    def test_small_lightgbm_learns_a_signal(self):
        X, y = self._separable()
        m = evaluation.SmallLightGBM()
        m.fit(X, y, np.ones(len(X)))
        p = m.predict_proba(X)
        assert p[y == 1].mean() > p[y == 0].mean()

    def test_current_lightgbm_learns_a_signal(self):
        X, y = self._separable()
        m = evaluation.CurrentLightGBM()
        m.fit(X, y, np.ones(len(X)))
        p = m.predict_proba(X)
        assert p[y == 1].mean() > p[y == 0].mean()

    def test_probabilities_are_in_range(self):
        X, y = self._separable()
        for model in (evaluation.SmallLightGBM(), evaluation.CurrentLightGBM()):
            model.fit(X, y, np.ones(len(X)))
            p = model.predict_proba(X)
            assert ((p >= 0.0) & (p <= 1.0)).all()

    def test_current_matches_existing_hyperparameters(self):
        """現行 ml_model._fit() と同じハイパーパラメータであること"""
        m = evaluation.CurrentLightGBM()
        assert m.params["n_estimators"] == 200
        assert m.params["num_leaves"] == 31
        assert m.params["learning_rate"] == pytest.approx(0.05)

    def test_small_is_smaller_than_current(self):
        small = evaluation.SmallLightGBM()
        current = evaluation.CurrentLightGBM()
        assert small.params["num_leaves"] < current.params["num_leaves"]
        assert small.params["n_estimators"] < current.params["n_estimators"]

    def test_falls_back_to_constant_on_single_class(self):
        X = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 0.0, 1.0]})
        y = pd.Series([0, 0, 0])
        m = evaluation.SmallLightGBM()
        m.fit(X, y, np.ones(3))
        assert np.allclose(m.predict_proba(X), 0.0)

    def test_names_are_distinct(self):
        assert evaluation.SmallLightGBM().name == "small_lightgbm"
        assert evaluation.CurrentLightGBM().name == "current_lightgbm"


class TestDefaultModelFactories:
    def test_provides_exactly_five_models(self):
        """比較対象は5つに固定する（深層モデルはこの段階では候補にしない）"""
        factories = evaluation.default_model_factories()
        assert set(factories) == {
            "constant_probability", "majority_class", "logistic_regression",
            "small_lightgbm", "current_lightgbm",
        }

    def test_each_factory_builds_a_fresh_model(self):
        factories = evaluation.default_model_factories()
        for name, make in factories.items():
            a, b = make(), make()
            assert a is not b
            assert a.name == name
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.evaluation' has no attribute 'SmallLightGBM'`

- [ ] **Step 3: 実装を追加**

`src/strategy/evaluation.py` の末尾に追加する。import に `import lightgbm as lgb` を足す。

```python
class _LightGbmBase:
    """LightGBM分類器の共通部。

    early stopping はここでは行わない。内側foldで決めるため（spec §7）。
    現行 ml_model._fit() は fold ごとの best_iteration_ の平均を最終モデルの
    木数にしているが、その fold は early stopping と指標報告を兼ねており
    報告値が楽観に寄る。
    """
    name = "lightgbm"
    params: dict = {}

    def __init__(self) -> None:
        self._model = None
        self._constant: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray) -> None:
        if y.nunique() < 2:
            self._constant = float(y.iloc[0]) if len(y) else 0.5
            self._model = None
            return
        self._constant = None
        self._model = lgb.LGBMClassifier(**self.params)
        self._model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._constant is not None:
            return np.full(len(X), self._constant, dtype=float)
        return self._model.predict_proba(X)[:, 1]


class SmallLightGBM(_LightGbmBase):
    """小さいLightGBM。現行より表現力を抑えた比較対象。"""
    name = "small_lightgbm"
    params = {
        "n_estimators": 50, "num_leaves": 7, "learning_rate": 0.05,
        "random_state": 42, "verbose": -1,
    }


class CurrentLightGBM(_LightGbmBase):
    """現行 ml_model._fit() と同じハイパーパラメータのLightGBM。"""
    name = "current_lightgbm"
    params = {
        "n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05,
        "random_state": 42, "verbose": -1,
    }


def default_model_factories() -> dict:
    """比較対象5つの生成関数。

    **深層モデルはこの段階では候補にしない。** 同じ入力・同じ分割で
    追加価値を示せるかを先に確かめる（spec §7）。
    """
    return {
        "constant_probability": ConstantProbability,
        "majority_class": MajorityClass,
        "logistic_regression": LogisticRegressionModel,
        "small_lightgbm": SmallLightGBM,
        "current_lightgbm": CurrentLightGBM,
    }
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS（25件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/evaluation.py tests/test_evaluation.py
git commit -m "$(cat <<'EOF'
feat(strategy): 比較対象のLightGBM2種と5モデルの生成関数を追加

現行と同じハイパーパラメータのものと、表現力を抑えた小さいものを置く。
early stoppingはここでは行わない（内側foldで決める）。現行実装は
early stoppingと指標報告を同じfoldで兼ねており報告値が楽観に寄るため。
深層モデルはこの段階では候補にしない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: 指標の算出（定数モデルとの差を含む）

**Files:**
- Modify: `src/strategy/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: なし
- Produces: `compute_metrics(y_true, p, *, baseline_rate: float) -> dict` — `n` / `positive_rate` / `roc_auc` / `average_precision` / `log_loss` / `brier` / `brier_vs_constant` / `log_loss_vs_constant`

**背景（spec §7・レビュー ML）:** 「Brier 0.2476 は 0.25 未満だから採用可能」とは判断しない。二値ラベルに常に 0.5 を予測すれば二乗誤差は 0.25 である。**クラス比率を予測する定数モデルとの差**で見る。AUC も検証 fold が片側クラスのみだと未定義になるため `None` を返す。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_evaluation.py` の末尾に追記する。

```python
class TestComputeMetrics:
    def test_perfect_prediction_has_auc_one(self):
        y = pd.Series([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["roc_auc"] == pytest.approx(1.0)

    def test_reversed_prediction_has_auc_zero(self):
        y = pd.Series([0, 0, 1, 1])
        p = np.array([0.9, 0.8, 0.2, 0.1])
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["roc_auc"] == pytest.approx(0.0)

    def test_auc_is_none_when_single_class(self):
        """検証foldが片側クラスのみだとAUCは未定義。例外にせずNoneにする"""
        y = pd.Series([1, 1, 1])
        p = np.array([0.2, 0.5, 0.9])
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["roc_auc"] is None

    def test_brier_matches_mean_squared_error(self):
        y = pd.Series([0, 1])
        p = np.array([0.25, 0.75])
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["brier"] == pytest.approx(0.0625)

    def test_constant_half_gives_brier_quarter(self):
        """常に0.5を予測すれば二乗誤差は0.25。単独では採用根拠にならない"""
        y = pd.Series([0, 1, 0, 1])
        p = np.full(4, 0.5)
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["brier"] == pytest.approx(0.25)
        assert m["brier_vs_constant"] == pytest.approx(0.0)

    def test_better_than_constant_is_positive(self):
        """定数モデルより良ければ差は正になる"""
        y = pd.Series([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["brier_vs_constant"] > 0
        assert m["log_loss_vs_constant"] > 0

    def test_worse_than_constant_is_negative(self):
        y = pd.Series([0, 0, 1, 1])
        p = np.array([0.9, 0.8, 0.2, 0.1])
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["brier_vs_constant"] < 0

    def test_baseline_uses_training_rate_not_validation_rate(self):
        """定数モデルの確率は学習側の正例率。検証側の正例率を使わない"""
        y = pd.Series([1, 1, 1, 0])
        p = np.full(4, 0.3)
        with_train_rate = evaluation.compute_metrics(y, p, baseline_rate=0.3)
        with_val_rate = evaluation.compute_metrics(y, p, baseline_rate=0.75)
        assert with_train_rate["brier_vs_constant"] != pytest.approx(
            with_val_rate["brier_vs_constant"])
        assert with_train_rate["brier_vs_constant"] == pytest.approx(0.0)

    def test_records_counts_and_positive_rate(self):
        y = pd.Series([0, 1, 1, 1])
        p = np.full(4, 0.5)
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["n"] == 4
        assert m["positive_rate"] == pytest.approx(0.75)

    def test_empty_input_returns_zero_count(self):
        m = evaluation.compute_metrics(pd.Series([], dtype=int), np.array([]),
                                       baseline_rate=0.5)
        assert m["n"] == 0
        assert m["roc_auc"] is None
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.evaluation' has no attribute 'compute_metrics'`

- [ ] **Step 3: 実装を追加**

`src/strategy/evaluation.py` の末尾に追加する。import に次を足す。

```python
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)
```

```python
_EPS = 1e-15


def _log_loss_safe(y_true: np.ndarray, p: np.ndarray) -> float:
    """確率を刈り込んでから log loss を計算する（0/1予測で無限大にしない）。"""
    clipped = np.clip(p, _EPS, 1.0 - _EPS)
    return float(log_loss(y_true, clipped, labels=[0, 1]))


def compute_metrics(y_true, p, *, baseline_rate: float) -> dict:
    """確率予測の質を測る。**定数モデルとの差**を併せて返す。

    「Brier 0.2476 は 0.25 未満だから採用可能」とは判断しない。二値ラベルに
    常に 0.5 を予測すれば二乗誤差は 0.25 である（レビュー ML）。
    クラス比率を予測する定数モデルを基準に置き、その差で見る。
    基準の確率は**学習側の正例率**を使う（検証側の正例率を使うと、
    検証側の情報が基準へ入る）。

    検証データが片側クラスのみだと ROC-AUC と平均適合率は未定義になるため
    None を返す（例外にしない）。
    """
    y = np.asarray(pd.Series(y_true).astype(float))
    prob = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        return {
            "n": 0, "positive_rate": None, "roc_auc": None,
            "average_precision": None, "log_loss": None, "brier": None,
            "brier_vs_constant": None, "log_loss_vs_constant": None,
        }

    both_classes = len(np.unique(y)) > 1
    const = np.full(n, float(baseline_rate))

    # pos_label を明示する。片側クラスのみの検証データでも版によっては
    # 「どちらが正例か決められない」と拒否されうるため。
    brier = float(brier_score_loss(y, prob, pos_label=1))
    brier_const = float(brier_score_loss(y, const, pos_label=1))
    ll = _log_loss_safe(y, prob)
    ll_const = _log_loss_safe(y, const)

    return {
        "n": n,
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, prob)) if both_classes else None,
        "average_precision": float(average_precision_score(y, prob)) if both_classes else None,
        "log_loss": ll,
        "brier": brier,
        # 正なら定数モデルより良い（誤差が小さい）
        "brier_vs_constant": brier_const - brier,
        "log_loss_vs_constant": ll_const - ll,
    }
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS（35件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/evaluation.py tests/test_evaluation.py
git commit -m "$(cat <<'EOF'
feat(strategy): 定数モデルとの差を併記する指標の算出を追加

二値ラベルに常に0.5を予測すれば二乗誤差は0.25になるため、Brierの
絶対値だけでは採用根拠にならない。クラス比率を予測する定数モデルを
基準に置き、その差で見る。基準の確率は学習側の正例率を使う
（検証側を使うと検証側の情報が基準へ入る）。
片側クラスのみでAUCが未定義になる場合は例外にせずNoneを返す。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 内側 fold での確率校正と閾値選択

**Files:**
- Modify: `src/strategy/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `validation.inner_folds` / `split_events` / `training_inputs`、Task 2〜4
- Produces:
  - `Calibrator`（frozen dataclass）: `kind: str`（`"identity"` / `"isotonic"`）, `model: Optional[object]`
  - `fit_calibrator(raw_p: np.ndarray, y: pd.Series, *, min_samples: int = 50) -> Calibrator`
  - `apply_calibrator(cal: Calibrator, raw_p: np.ndarray) -> np.ndarray`
  - `select_threshold(p: np.ndarray, net_return: np.ndarray, *, candidates: Optional[np.ndarray] = None) -> float`
  - `fit_inner(events, fold, make_model, *, window_sessions=None, feature_cols=None, inner_splits=3) -> tuple[Calibrator, float]`

**背景（spec §7・§8）:** 確率校正と閾値選択は**内側 fold の予測**から決める。外側 fold の値は一切使わない。閾値は期待値 `p × 平均利益 − (1−p) × 平均損失` を最大化する点で、コストは `net_return` に織り込み済みなので**式の末尾で再度引かない**。利益側と損失側が非対称なら採用確率の境界は 0.5 にならない。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_evaluation.py` の末尾に追記する。冒頭の import に `from src.strategy import validation` を足す。

```python
class TestCalibrator:
    def test_identity_when_too_few_samples(self):
        raw = np.array([0.1, 0.9])
        cal = evaluation.fit_calibrator(raw, pd.Series([0, 1]), min_samples=50)
        assert cal.kind == "identity"
        assert np.allclose(evaluation.apply_calibrator(cal, raw), raw)

    def test_identity_when_single_class(self):
        raw = np.linspace(0.1, 0.9, 100)
        cal = evaluation.fit_calibrator(raw, pd.Series([1] * 100), min_samples=50)
        assert cal.kind == "identity"

    def test_isotonic_pulls_overconfident_probabilities_toward_truth(self):
        """実際の正例率が0.5なのに0.9を出していたら、校正後は下がる"""
        rng = np.random.default_rng(5)
        y = pd.Series(rng.integers(0, 2, 400))
        raw = np.where(y == 1, 0.95, 0.9)  # 常に自信過剰
        cal = evaluation.fit_calibrator(raw, y, min_samples=50)
        assert cal.kind == "isotonic"
        out = evaluation.apply_calibrator(cal, raw)
        assert out.mean() < raw.mean()

    def test_calibrated_values_stay_in_range(self):
        rng = np.random.default_rng(6)
        y = pd.Series(rng.integers(0, 2, 300))
        raw = rng.random(300)
        cal = evaluation.fit_calibrator(raw, y, min_samples=50)
        out = evaluation.apply_calibrator(cal, raw)
        assert ((out >= 0.0) & (out <= 1.0)).all()

    def test_is_monotonic(self):
        """校正は順序を壊さない（AUCを変えない）"""
        rng = np.random.default_rng(7)
        y = pd.Series(rng.integers(0, 2, 300))
        raw = rng.random(300)
        cal = evaluation.fit_calibrator(raw, y, min_samples=50)
        grid = np.linspace(0.0, 1.0, 50)
        out = evaluation.apply_calibrator(cal, grid)
        assert np.all(np.diff(out) >= -1e-12)


class TestSelectThreshold:
    def test_picks_threshold_that_excludes_losers(self):
        """低い確率のイベントが損失なら、それを外す閾値を選ぶ"""
        p = np.array([0.1, 0.2, 0.8, 0.9])
        ret = np.array([-0.05, -0.04, 0.03, 0.06])
        t = evaluation.select_threshold(p, ret)
        selected = p >= t
        assert selected.tolist() == [False, False, True, True]

    def test_takes_everything_when_all_profitable(self):
        p = np.array([0.1, 0.5, 0.9])
        ret = np.array([0.01, 0.02, 0.03])
        t = evaluation.select_threshold(p, ret)
        assert (p >= t).all()

    def test_boundary_is_not_always_half(self):
        """利益と損失が非対称なら境界は0.5にならない（spec §8）"""
        p = np.array([0.3, 0.45, 0.55, 0.7])
        # 0.45 の取引まで採った方が総和が大きい非対称な収益
        ret = np.array([-0.01, 0.05, 0.01, 0.02])
        t = evaluation.select_threshold(p, ret)
        assert t != pytest.approx(0.5)
        assert (p >= t).sum() == 3

    def test_returns_a_value_even_when_nothing_is_profitable(self):
        p = np.array([0.1, 0.5, 0.9])
        ret = np.array([-0.01, -0.02, -0.03])
        t = evaluation.select_threshold(p, ret)
        assert 0.0 <= t <= 1.0

    def test_empty_input_returns_half(self):
        assert evaluation.select_threshold(np.array([]), np.array([])) == pytest.approx(0.5)


class TestFitInner:
    def test_uses_only_inner_folds(self):
        """外側の検証期間を書き換えても、内側で決めた校正と閾値は変わらない"""
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]

        cal_a, thr_a = evaluation.fit_inner(
            events, fold, evaluation.LogisticRegressionModel, feature_cols=FEATURES)

        tampered = events.copy()
        in_val = tampered["decision_at"] >= fold.val_start
        tampered.loc[in_val, "label"] = 1
        tampered.loc[in_val, "net_return"] = 5.0
        tampered.loc[in_val, "x1"] = -99.0
        cal_b, thr_b = evaluation.fit_inner(
            tampered, fold, evaluation.LogisticRegressionModel, feature_cols=FEATURES)

        assert thr_a == pytest.approx(thr_b)
        assert cal_a.kind == cal_b.kind

    def test_returns_usable_threshold(self):
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]
        _, thr = evaluation.fit_inner(
            events, fold, evaluation.ConstantProbability, feature_cols=FEATURES)
        assert 0.0 <= thr <= 1.0
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.evaluation' has no attribute 'fit_calibrator'`

- [ ] **Step 3: 実装を追加**

`src/strategy/evaluation.py` の末尾に追加する。import に次を足す。

```python
from dataclasses import dataclass

from sklearn.isotonic import IsotonicRegression

from src.strategy import validation
from src.strategy.indicators import FEATURE_COLS
```

```python
@dataclass(frozen=True)
class Calibrator:
    """確率校正。**内側foldの予測から作る。** 外側foldの値は使わない。"""
    kind: str                 # "identity" / "isotonic"
    model: Optional[object]


def fit_calibrator(raw_p: np.ndarray, y: pd.Series, *,
                   min_samples: int = 50) -> Calibrator:
    """等調回帰で確率を校正する。データが足りない／片側クラスなら恒等にする。

    校正は順序を変えないため AUC は動かない。動くのは Brier と log loss。
    """
    y_arr = np.asarray(pd.Series(y).astype(float))
    if len(y_arr) < min_samples or len(np.unique(y_arr)) < 2:
        return Calibrator(kind="identity", model=None)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(raw_p, dtype=float), y_arr)
    return Calibrator(kind="isotonic", model=iso)


def apply_calibrator(cal: Calibrator, raw_p: np.ndarray) -> np.ndarray:
    if cal.kind == "identity" or cal.model is None:
        return np.asarray(raw_p, dtype=float)
    return np.clip(cal.model.predict(np.asarray(raw_p, dtype=float)), 0.0, 1.0)


def select_threshold(p: np.ndarray, net_return: np.ndarray, *,
                     candidates: Optional[np.ndarray] = None) -> float:
    """コスト控除後の総収益を最大にする採用閾値を返す。

    期待値は `p × 平均利益 − (1−p) × 平均損失` で扱う。**コストは net_return に
    織り込み済みなので、ここで再度引かない**（spec §8）。利益側と損失側が
    非対称なら、採用確率の境界は 0.5 にならない。

    どの閾値でも総収益が正にならない場合も最良の点を返す。「そもそも取引するか」
    の判断は呼び出し側が行う。
    """
    prob = np.asarray(p, dtype=float)
    ret = np.asarray(net_return, dtype=float)
    if len(prob) == 0:
        return 0.5
    grid = np.unique(prob) if candidates is None else np.asarray(candidates, dtype=float)

    best_t, best_total = float(grid[0]), -np.inf
    for t in grid:
        total = float(ret[prob >= t].sum())
        if total > best_total:
            best_total, best_t = total, float(t)
    return best_t


def fit_inner(events: pd.DataFrame, fold, make_model, *,
              window_sessions: Optional[int] = None,
              feature_cols: Optional[list] = None,
              inner_splits: int = 3) -> tuple:
    """内側foldの予測から校正と閾値を決める。

    **外側foldの値は一切使わない。** 外側の学習側をさらに分割し、その
    内側検証で得た予測だけを材料にする（spec §7）。
    戻り値: (Calibrator, 採用閾値)
    """
    outer = validation.training_inputs(
        events, fold, window_sessions=window_sessions, feature_cols=feature_cols)
    outer_train = outer.events
    if len(outer_train) == 0:
        return Calibrator(kind="identity", model=None), 0.5

    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    raw_parts, y_parts, ret_parts = [], [], []
    for inner in validation.inner_folds(outer_train, n_splits=inner_splits):
        inner_inputs = validation.training_inputs(
            outer_train, inner, window_sessions=window_sessions, feature_cols=feature_cols)
        _, inner_val = validation.split_events(outer_train, inner)
        if len(inner_inputs.events) == 0 or len(inner_val) == 0:
            continue
        model = make_model()
        model.fit(inner_inputs.events[cols].astype("float64"),
                  inner_inputs.events["label"].astype(int),
                  inner_inputs.weights)
        raw_parts.append(model.predict_proba(inner_val[cols].astype("float64")))
        y_parts.append(inner_val["label"].astype(int))
        ret_parts.append(inner_val["net_return"].astype(float).values)

    if not raw_parts:
        return Calibrator(kind="identity", model=None), 0.5

    raw = np.concatenate(raw_parts)
    y = pd.concat(y_parts, ignore_index=True)
    ret = np.concatenate(ret_parts)

    cal = fit_calibrator(raw, y)
    threshold = select_threshold(apply_calibrator(cal, raw), ret)
    return cal, threshold
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS（47件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/evaluation.py tests/test_evaluation.py
git commit -m "$(cat <<'EOF'
feat(strategy): 内側foldでの確率校正と閾値選択を追加

校正と閾値は外側foldの学習側をさらに分割した内側foldの予測だけから
決める。外側foldの値は選択に使わない。
閾値はコスト控除後の総収益を最大にする点を選ぶ。コストはnet_returnに
織り込み済みなので式の末尾で再度引かない。利益と損失が非対称なら
境界は0.5にならない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: 外側 fold の評価ループと予測明細の書き出し

**Files:**
- Modify: `src/strategy/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: Task 1〜5 のすべて
- Produces:
  - `FoldResult`（frozen dataclass）: `fold_index` / `model_id` / `n_train` / `n_val` / `train_positive_rate` / `threshold` / `metrics: dict` / `predictions: pd.DataFrame`
  - `evaluate_fold(events, fold, model_id, make_model, *, window_sessions=None, feature_cols=None, inner_splits=3) -> Optional[FoldResult]`
  - `run_evaluation(events, *, model_factories=None, n_splits=5, window_sessions=None, feature_cols=None, evaluation_run_id=None, persist=True) -> dict`

**注意:** `evaluate_fold` は学習入力を必ず `validation.training_inputs()` から取る。独自に `events` を絞り込まない。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_evaluation.py` の末尾に追記する。

```python
class TestEvaluateFold:
    def test_returns_predictions_for_every_validation_event(self):
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]
        _, val = validation.split_events(events, fold)

        res = evaluation.evaluate_fold(
            events, fold, "logistic_regression",
            evaluation.LogisticRegressionModel, feature_cols=FEATURES)
        assert res is not None
        assert len(res.predictions) == len(val)
        assert set(res.predictions["event_id"]) == set(val["event_id"])

    def test_prediction_columns_are_complete(self):
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]
        res = evaluation.evaluate_fold(
            events, fold, "const", evaluation.ConstantProbability,
            feature_cols=FEATURES)
        for col in ("event_id", "raw_probability", "calibrated_probability", "fold_index"):
            assert col in res.predictions.columns
        assert (res.predictions["fold_index"] == fold.index).all()

    def test_baseline_rate_comes_from_training_side(self):
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]
        inputs = validation.training_inputs(events, fold, feature_cols=FEATURES)
        res = evaluation.evaluate_fold(
            events, fold, "const", evaluation.ConstantProbability,
            feature_cols=FEATURES)
        assert res.train_positive_rate == pytest.approx(
            inputs.events["label"].astype(int).mean())

    def test_outer_validation_values_do_not_change_training(self):
        """外側の検証側を書き換えても学習件数と学習側正例率は変わらない"""
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]
        a = evaluation.evaluate_fold(
            events, fold, "const", evaluation.ConstantProbability, feature_cols=FEATURES)

        tampered = events.copy()
        in_val = tampered["decision_at"] >= fold.val_start
        tampered.loc[in_val, "label"] = 1
        b = evaluation.evaluate_fold(
            tampered, fold, "const", evaluation.ConstantProbability, feature_cols=FEATURES)

        assert a.n_train == b.n_train
        assert a.train_positive_rate == pytest.approx(b.train_positive_rate)

    def test_returns_none_when_fold_has_no_data(self):
        events = _events(n_sessions=120)
        empty_fold = validation.Fold(
            index=9, train_start=date(2030, 1, 1), train_end=date(2030, 1, 2),
            val_start=date(2030, 1, 3), val_end=date(2030, 1, 4))
        assert evaluation.evaluate_fold(
            events, empty_fold, "const", evaluation.ConstantProbability,
            feature_cols=FEATURES) is None


class TestRunEvaluation:
    def test_evaluates_every_model_on_every_fold(self, isolated_db):
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events, model_factories={
                "constant_probability": evaluation.ConstantProbability,
                "logistic_regression": evaluation.LogisticRegressionModel,
            },
            n_splits=3, feature_cols=FEATURES, persist=False)
        models = {r.model_id for r in out["fold_results"]}
        assert models == {"constant_probability", "logistic_regression"}

    def test_all_models_see_the_same_folds(self, isolated_db):
        """同じ分割で比較する（モデルごとに分割を変えない）"""
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events, model_factories={
                "constant_probability": evaluation.ConstantProbability,
                "logistic_regression": evaluation.LogisticRegressionModel,
            },
            n_splits=3, feature_cols=FEATURES, persist=False)
        by_model = {}
        for r in out["fold_results"]:
            by_model.setdefault(r.model_id, []).append((r.fold_index, r.n_train, r.n_val))
        values = list(by_model.values())
        assert all(v == values[0] for v in values)

    def test_persists_predictions_and_outcomes(self, isolated_db):
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events, model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=3, feature_cols=FEATURES, persist=True)

        details = evaluation.load_prediction_details(out["evaluation_run_id"])
        assert len(details) > 0
        assert details["actual_label"].notna().all()

    def test_evaluation_run_id_is_recorded_on_every_row(self, isolated_db):
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events, model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=3, feature_cols=FEATURES, persist=True)
        details = evaluation.load_prediction_details(out["evaluation_run_id"])
        assert (details["evaluation_run_id"] == out["evaluation_run_id"]).all()

    def test_summary_frame_has_one_row_per_model_and_fold(self, isolated_db):
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events, model_factories={
                "constant_probability": evaluation.ConstantProbability,
                "small_lightgbm": evaluation.SmallLightGBM,
            },
            n_splits=3, feature_cols=FEATURES, persist=False)
        summary = out["summary"]
        assert len(summary) == len(out["fold_results"])
        for col in ("fold_index", "model_id", "n_train", "n_val",
                    "train_positive_rate", "threshold", "roc_auc", "brier",
                    "brier_vs_constant"):
            assert col in summary.columns

    def test_defaults_to_the_five_comparison_models(self, isolated_db):
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events, n_splits=3, feature_cols=FEATURES, persist=False)
        assert {r.model_id for r in out["fold_results"]} == set(
            evaluation.default_model_factories())
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.evaluation' has no attribute 'evaluate_fold'`

- [ ] **Step 3: 実装を追加**

`src/strategy/evaluation.py` の末尾に追加する。

```python
@dataclass(frozen=True)
class FoldResult:
    """1つの外側foldで1モデルを評価した結果。"""
    fold_index: int
    model_id: str
    n_train: int
    n_val: int
    train_positive_rate: float
    threshold: float
    metrics: dict
    predictions: pd.DataFrame


def evaluate_fold(events: pd.DataFrame, fold, model_id: str, make_model, *,
                  window_sessions: Optional[int] = None,
                  feature_cols: Optional[list] = None,
                  inner_splits: int = 3) -> Optional[FoldResult]:
    """1つの外側foldでモデルを学習し、検証側で評価する。

    学習入力は必ず validation.training_inputs() から取る（独自に events を
    絞り込まない）。校正と閾値は内側foldで決めてから外側検証へ適用する。
    """
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    inputs = validation.training_inputs(
        events, fold, window_sessions=window_sessions, feature_cols=cols)
    _, val = validation.split_events(events, fold)
    if len(inputs.events) == 0 or len(val) == 0:
        return None

    calibrator, threshold = fit_inner(
        events, fold, make_model, window_sessions=window_sessions,
        feature_cols=cols, inner_splits=inner_splits)

    model = make_model()
    y_train = inputs.events["label"].astype(int)
    model.fit(inputs.events[cols].astype("float64"), y_train, inputs.weights)

    raw = model.predict_proba(val[cols].astype("float64"))
    calibrated = apply_calibrator(calibrator, raw)
    train_rate = float(y_train.mean())

    predictions = pd.DataFrame({
        "event_id": val["event_id"].values,
        "raw_probability": raw,
        "calibrated_probability": calibrated,
        "fold_index": fold.index,
    })
    metrics = compute_metrics(
        val["label"].astype(int), calibrated, baseline_rate=train_rate)

    return FoldResult(
        fold_index=fold.index, model_id=model_id,
        n_train=len(inputs.events), n_val=len(val),
        train_positive_rate=train_rate, threshold=threshold,
        metrics=metrics, predictions=predictions,
    )


def run_evaluation(events: pd.DataFrame, *, model_factories: Optional[dict] = None,
                   n_splits: int = 5, window_sessions: Optional[int] = None,
                   feature_cols: Optional[list] = None,
                   evaluation_run_id: Optional[str] = None,
                   persist: bool = True) -> dict:
    """全モデルを**同じ分割**で評価する。

    戻り値: {"evaluation_run_id", "fold_results", "summary"}
    persist=True なら予測明細と実績をDBへ保存する。
    """
    factories = model_factories or default_model_factories()
    if evaluation_run_id is None:
        evaluation_run_id = f"{clock.now():%Y%m%dT%H%M%S}"

    folds = validation.calendar_folds(events, n_splits=n_splits)
    results = []
    for model_id, make_model in factories.items():
        for fold in folds:
            res = evaluate_fold(
                events, fold, model_id, make_model,
                window_sessions=window_sessions, feature_cols=feature_cols)
            if res is None:
                continue
            results.append(res)
            if persist:
                save_predictions(res.predictions, evaluation_run_id, model_id)

    if persist:
        save_outcomes(events)

    rows = []
    for r in results:
        row = {
            "fold_index": r.fold_index, "model_id": r.model_id,
            "n_train": r.n_train, "n_val": r.n_val,
            "train_positive_rate": r.train_positive_rate,
            "threshold": r.threshold,
        }
        row.update(r.metrics)
        rows.append(row)
    summary = pd.DataFrame(rows)

    logger.info(
        f"評価実行 {evaluation_run_id}: モデル{len(factories)}件 × fold{len(folds)}件 "
        f"→ 結果{len(results)}件"
    )
    return {
        "evaluation_run_id": evaluation_run_id,
        "fold_results": results,
        "summary": summary,
    }
```

（`FEATURE_COLS` は Task 5 で import 済み。追加の import は不要。）

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS（59件）

- [ ] **Step 5: 全体回帰を確認してコミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

```bash
git add src/strategy/evaluation.py tests/test_evaluation.py
git commit -m "$(cat <<'EOF'
feat(strategy): 外側foldの評価ループと予測明細の書き出しを追加

全モデルを同じ分割で評価する。学習入力は必ずtraining_inputs()から取り、
独自にeventsを絞り込まない。校正と閾値は内側foldで決めてから
外側検証へ適用する。
予測明細と実績をDBへ保存し、後から指標を再計算できるようにする。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: 予測明細からの再計算と学習窓の内側比較

**Files:**
- Modify: `src/strategy/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: Task 1・4・5・6
- Produces:
  - `recompute_metrics(details: pd.DataFrame, *, baseline_rate: float, model_id: Optional[str] = None) -> dict`
  - `select_training_window(events, fold, make_model, candidates, *, feature_cols=None, inner_splits=3) -> Optional[int]`

**背景（spec §14 段階C完了条件）:** 「予測明細から指標を再計算できる」。保存した明細だけを入力に、実行時と同じ指標が出ることをテストで固定する。

**学習窓（spec §7 / F07）:** 拡大窓と移動窓の選択は**内側 fold で**行う。外側成績を見て窓を選び、同じ成績を最終証拠として使わない。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_evaluation.py` の末尾に追記する。

```python
class TestRecomputeFromDetails:
    def test_metrics_match_the_original_run(self, isolated_db):
        """保存した明細だけから、実行時と同じ指標が出る（spec §14 完了条件）"""
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events,
            model_factories={"logistic_regression": evaluation.LogisticRegressionModel},
            n_splits=3, feature_cols=FEATURES, persist=True)

        details = evaluation.load_prediction_details(
            out["evaluation_run_id"], model_id="logistic_regression")

        for res in out["fold_results"]:
            sub = details[details["fold_index"] == res.fold_index]
            again = evaluation.recompute_metrics(
                sub, baseline_rate=res.train_positive_rate)
            assert again["n"] == res.metrics["n"]
            assert again["brier"] == pytest.approx(res.metrics["brier"])
            if res.metrics["roc_auc"] is not None:
                assert again["roc_auc"] == pytest.approx(res.metrics["roc_auc"])

    def test_trading_decisions_can_be_recomputed(self, isolated_db):
        """明細から採用群も引き直せる（閾値を変えた検討ができる）"""
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events,
            model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=3, feature_cols=FEATURES, persist=True)
        details = evaluation.load_prediction_details(out["evaluation_run_id"])
        taken = details[details["calibrated_probability"] >= 0.0]
        assert taken["net_return"].notna().all()

    def test_skips_rows_without_outcomes(self, isolated_db):
        preds = pd.DataFrame({
            "event_id": ["a", "b"],
            "raw_probability": [0.6, 0.4],
            "calibrated_probability": [0.6, 0.4],
            "fold_index": [0, 0],
        })
        evaluation.save_predictions(preds, "run1", "m")
        details = evaluation.load_prediction_details("run1")
        got = evaluation.recompute_metrics(details, baseline_rate=0.5)
        assert got["n"] == 0


class TestSelectTrainingWindow:
    def test_returns_one_of_the_candidates(self):
        events = _events(n_sessions=150)
        fold = validation.calendar_folds(events, n_splits=3)[1]
        got = evaluation.select_training_window(
            events, fold, evaluation.LogisticRegressionModel,
            candidates=[None, 20, 40], feature_cols=FEATURES)
        assert got in (None, 20, 40)

    def test_choice_does_not_depend_on_outer_validation(self):
        """外側検証の値を書き換えても選ばれる窓は変わらない（選択は内側で）"""
        events = _events(n_sessions=150)
        fold = validation.calendar_folds(events, n_splits=3)[1]
        a = evaluation.select_training_window(
            events, fold, evaluation.LogisticRegressionModel,
            candidates=[None, 20, 40], feature_cols=FEATURES)

        tampered = events.copy()
        in_val = tampered["decision_at"] >= fold.val_start
        tampered.loc[in_val, "label"] = 1
        tampered.loc[in_val, "net_return"] = 9.9
        b = evaluation.select_training_window(
            tampered, fold, evaluation.LogisticRegressionModel,
            candidates=[None, 20, 40], feature_cols=FEATURES)
        assert a == b

    def test_empty_candidates_returns_none(self):
        events = _events(n_sessions=150)
        fold = validation.calendar_folds(events, n_splits=3)[1]
        assert evaluation.select_training_window(
            events, fold, evaluation.ConstantProbability,
            candidates=[], feature_cols=FEATURES) is None
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.evaluation' has no attribute 'recompute_metrics'`

- [ ] **Step 3: 実装を追加**

`src/strategy/evaluation.py` の末尾に追加する。

```python
def recompute_metrics(details: pd.DataFrame, *, baseline_rate: float,
                      model_id: Optional[str] = None) -> dict:
    """保存した予測明細だけから指標を計算し直す。

    実績が未確定の行（shadow等）は除外する。集計済みの数値しか無い状態では
    「その数字が何を意味するか」を後から検証できないため、明細から同じ指標を
    再現できることを保証する（spec §14 段階C完了条件）。
    """
    sub = details if model_id is None else details[details["model_id"] == model_id]
    sub = sub[sub["actual_label"].notna()]
    return compute_metrics(
        sub["actual_label"].astype(int),
        sub["calibrated_probability"].astype(float).values,
        baseline_rate=baseline_rate,
    )


def select_training_window(events: pd.DataFrame, fold, make_model,
                           candidates: list, *,
                           feature_cols: Optional[list] = None,
                           inner_splits: int = 3) -> Optional[int]:
    """学習窓を**内側foldだけ**で選ぶ。

    外側成績を見て窓を選び、同じ成績を最終証拠として使わない（spec §7）。
    各候補について内側foldの予測からコスト控除後の総収益を求め、最大の窓を返す。
    候補が空なら None（拡大窓）を返す。
    """
    if not candidates:
        return None

    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    best_window, best_total = None, -np.inf
    for window in candidates:
        outer = validation.training_inputs(
            events, fold, window_sessions=window, feature_cols=cols)
        if len(outer.events) == 0:
            continue

        total = 0.0
        scored = False
        for inner in validation.inner_folds(outer.events, n_splits=inner_splits):
            inner_inputs = validation.training_inputs(
                outer.events, inner, window_sessions=window, feature_cols=cols)
            _, inner_val = validation.split_events(outer.events, inner)
            if len(inner_inputs.events) == 0 or len(inner_val) == 0:
                continue
            model = make_model()
            model.fit(inner_inputs.events[cols].astype("float64"),
                      inner_inputs.events["label"].astype(int),
                      inner_inputs.weights)
            p = model.predict_proba(inner_val[cols].astype("float64"))
            ret = inner_val["net_return"].astype(float).values
            threshold = select_threshold(p, ret)
            total += float(ret[p >= threshold].sum())
            scored = True

        if scored and total > best_total:
            best_total, best_window = total, window
    return best_window
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS（65件）

- [ ] **Step 5: 全体回帰を確認**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

- [ ] **Step 6: BOM確認とコミット**

Run: `head -c 3 src/strategy/evaluation.py | xxd`（`2222 22` を確認）

```bash
git add src/strategy/evaluation.py tests/test_evaluation.py
git commit -m "$(cat <<'EOF'
feat(strategy): 予測明細からの指標再計算と学習窓の内側選択を追加

保存した明細だけから実行時と同じ指標が出ることをテストで固定する
（設計書§14の段階C完了条件）。閾値を変えた検討も明細から引き直せる。
学習窓は内側foldだけで選ぶ。外側成績を見て窓を選び同じ成績を最終証拠に
使うことをしない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## 段階C 完了条件の確認

spec §14 の段階C完了条件を、前半・後半あわせて検証する。

- [ ] **確認1: 外側 fold の値を変えても学習入力が変わらない**（前半で実装済み）

Run: `pytest tests/test_validation.py::TestOuterFoldIsUntouchable -v`
Expected: PASS（5件）

- [ ] **確認2: 外側 fold の値を変えても閾値と校正が変わらない**

Run: `pytest tests/test_evaluation.py::TestFitInner::test_uses_only_inner_folds -v`
Expected: PASS

- [ ] **確認3: 外側 fold の値を変えても学習窓の選択が変わらない**

Run: `pytest tests/test_evaluation.py::TestSelectTrainingWindow::test_choice_does_not_depend_on_outer_validation -v`
Expected: PASS

- [ ] **確認4: 予測明細から指標を再計算できる**

Run: `pytest tests/test_evaluation.py::TestRecomputeFromDetails -v`
Expected: PASS（3件）

- [ ] **確認5: 全モデルが同じ分割で比較されている**

Run: `pytest tests/test_evaluation.py::TestRunEvaluation::test_all_models_see_the_same_folds -v`
Expected: PASS

- [ ] **確認6: 既存経路に回帰が無い**

Run: `pytest tests/ -q`
Expected: 段階C後半の着手前と同じ結果（新規テスト65件ぶんだけ増える）

- [ ] **確認7: 実データで5モデルを1回比較する**（判断材料。合否ではない）

段階B後半で生成したイベント表を入力に `run_evaluation()` を1回実行し、次を記録する。

```python
out = evaluation.run_evaluation(events, n_splits=5)
print(out["summary"].groupby("model_id")[
    ["n_val", "roc_auc", "brier", "brier_vs_constant", "log_loss_vs_constant"]].mean())
```

Expected: モデルごとの平均が得られること。**この数値で採否を決めない。**
`brier_vs_constant` が 0 近傍なら、そのモデルは確率の質で定数モデルに何も足していない。
AUC が 0.5 近傍なら順位付けにも寄与していない。spec §14 のとおり合格条件は固定せず、
売買回数が少なければ期間を延ばし、比較の不確実性が大きければ保留する。

---

## 段階C で残した項目（段階D以降）

spec §7 の「purge だけでは塞がらない経路」のうち **経路4の前半（期間中の再学習に同じ締切を適用する）** は、バックテスト中の週次再学習を実装する段階Dで扱う。本計画の `run_evaluation()` は fold ごとに1回学習する形であり、期間中の再学習は含まない。

あわせて次も段階D以降に残る。

- 取引に採用した上位群のコスト控除後成績を、ポートフォリオとして（資金競合・セクター上限込みで）評価すること
- 重要特徴の安定性の記録
- 予測確率のヒストグラムと校正曲線の可視化（明細は保存済みなので後から描ける）
- shadow運用での予測保存（`PURPOSE_SHADOW` は用意済み。運用への結線は段階E）
