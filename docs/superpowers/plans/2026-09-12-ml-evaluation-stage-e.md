# ML評価基盤 段階E（候補モデルの昇格とshadow）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 学習成功を「モデル更新」ではなく「候補の生成」にし、評価を通った候補だけが現行になる経路を作る。候補は発注に繋がない形で並行記録する。

**Architecture:** モデルの実体はバージョン別の**不変ディレクトリ**へ保存し、現行を指す**小さな参照だけを原子的に更新する**。保存形式は pickle をやめて LightGBM ネイティブ + JSON メタにする。昇格は自動では起きず、明示的な判断と記録を伴う。shadow は同じ入力に現行と候補の判断を並行記録するだけで、発注経路には一切繋がない。

**Tech Stack:** Python 3.11 / lightgbm 4.1.0 / SQLAlchemy 2.0.23 / pytest / json / hashlib

**Spec:** `docs/superpowers/specs/2026-09-10-ml-evaluation-foundation-design.md`（§9・§10・§11・§12・§14）

**前提:** 段階C後半（`docs/superpowers/plans/2026-09-11-ml-evaluation-stage-c2.md`）が完了していること。本計画は `evaluation.PURPOSE_SHADOW` / `save_predictions` / `load_prediction_details` と、`indicators.FEATURE_COLS` に依存する。

## Global Constraints

- 日時は **JST naive**。現在時刻は `src/core/clock.now()` / `clock.today()` を使い、`datetime.now()` を直接呼ばない。
- **`src/strategy/ml_model.py` を変更しない。** 旧 pickle 経路（`MODEL_PATH` / `load()` / SHA256検証）は `engine_version: legacy` の受け皿として無改造で残す（spec §10）。
- **`models/lgb_model.pkl` を削除・移動・上書きしない。** legacy 運用がそのまま動く必要がある。
- **v2はモデル未昇格の状態から開始する。** 旧モデルを起動時に自動で現行へ複製しない。v2では特徴量とラベルの契約が変わるため、旧モデルをv2の現行として扱えない（spec §9）。
- **自動昇格を実装しない。** 本計画の完了範囲は候補保存とshadowまで。昇格は明示的な呼び出しでのみ起きる。
- **shadow の候補を発注経路に繋がない。** 記録のみ。
- 新規のDB列・テーブルはすべて nullable。`create_all` が新規テーブルを作る。
- ファイルは UTF-8 **BOM無し**・LF で保存する。確認は `git show <rev>:<path>` でコミット済みblobに対して行う。
- テストは `pytest tests/<file>.py -v` で実行する。ネットワークへ出るテストを書かない。乱数は `random_state` を固定する。
- コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` を付ける。実装者自身のモデル名を書かない。

---

## File Structure

| ファイル | 責務 |
|---|---|
| `src/strategy/model_store.py`（新規） | バージョン別の不変ディレクトリへの保存、現行を指す参照の原子的な更新、ロールバック |
| `src/strategy/promotion.py`（新規） | 昇格の契約（不可条件の検査と記録） |
| `src/strategy/shadow.py`（新規） | 現行と候補の並行記録（発注には繋がない） |
| `src/data/database.py`（改修） | `ModelPromotion` テーブルの追加 |
| `tests/test_model_store.py`（新規） | 保存形式、候補と現行の分離、原子的な切替、ロールバック |
| `tests/test_model_promotion.py`（新規） | 昇格不可条件、記録、途中失敗時の一貫性 |
| `tests/test_shadow.py`（新規） | 並行記録、発注に繋がらないこと |

---

## Task 1: バージョン別ディレクトリへの保存

**Files:**
- Create: `src/strategy/model_store.py`
- Test: `tests/test_model_store.py`

**Interfaces:**
- Consumes: `indicators.FEATURE_COLS`
- Produces:
  - `ModelMeta`（frozen dataclass）: `model_id` / `trained_at` / `training_window_sessions` / `symbols` / `label_definition` / `feature_cols` / `positive_rate` / `fold_results` / `code_version` / `lightgbm_version` / `dataset_id` / `model_kind` / `label_contract_id`
  - `KIND_BOOSTER = "lightgbm_booster"` / `KIND_CONSTANT = "constant"` — 保存形式
  - `CandidateExists(FileExistsError)` — 既存 model_id への再保存
  - `ConstantModel(probability)` — 単一クラス学習の結果。`predict(X)` を持つ
  - `save_candidate(model, meta: ModelMeta, *, base_dir: str = "models") -> Path` — 既存IDは `CandidateExists`。一時ディレクトリへ全成果物を書き、読み直して検証してから `os.replace()` で公開する
  - `load_model(model_id: str, *, base_dir: str = "models") -> tuple[object, ModelMeta]` — 戻り値のモデルは `lgb.Booster` か `ConstantModel`。どちらも `predict(X) -> 1次元配列`

**保存できるモデル型（`_extract_artifact`）:** ①段階Cのラッパー（`is_constant` / `constant_probability` / `booster` を公開するもの）②`booster_` を持つ scikit-learn API のモデル ③`lgb.Booster`。それ以外は `TypeError`。**握り潰さない**（外部レビューR02）。
  - `read_meta(model_id: str, *, base_dir: str = "models") -> ModelMeta`
  - `candidate_dir(model_id, base_dir) -> Path` / `current_ref_path(base_dir) -> Path`

**保存形式（spec §9）:** pickle を廃し、**LightGBM ネイティブ形式 + JSON メタ**にする。pickle は信頼できないデータのロードで任意コードを実行しうる。ネイティブ形式ならその経路が無い。

**レイアウト:**

```
models/
  lgb_model.pkl          ← legacy。触らない
  lgb_model.meta.json    ← legacy。触らない
  candidates/
    <model_id>/
      model.txt          ← LightGBM Booster.save_model()
      meta.json
  current.json           ← 現行を指す参照（Task 3）
```

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_model_store.py` を新規作成する。

```python
"""モデルの保存と現行の管理（src/strategy/model_store.py）のテスト

学習成功はモデル更新ではなく候補の生成である（spec §9）。実体は
バージョン別の不変ディレクトリへ置き、現行を指す小さな参照だけを
原子的に更新する。pickleは任意コード実行の経路なので使わない。
"""
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.strategy import model_store as ms


def _trained_model():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 200), "f2": rng.normal(0, 1, 200)})
    y = pd.Series((X["f1"] > 0).astype(int))
    m = lgb.LGBMClassifier(n_estimators=10, num_leaves=4,
                           random_state=42, verbose=-1)
    m.fit(X, y)
    return m, X


def _meta(model_id="m0001"):
    return ms.ModelMeta(
        model_id=model_id,
        trained_at=datetime(2026, 9, 12, 10, 0, 0),
        training_window_sessions=500,
        symbols=["7203", "9984"],
        label_definition="net_return>0",
        feature_cols=["f1", "f2"],
        positive_rate=0.48,
        fold_results=[{"fold": 0, "roc_auc": 0.52}],
        code_version="abc1234",
        lightgbm_version=lgb.__version__,
        dataset_id="ds0001",
    )


class TestSaveCandidate:
    def test_writes_native_format_not_pickle(self, tmp_path):
        model, _ = _trained_model()
        path = ms.save_candidate(model, _meta(), base_dir=str(tmp_path))
        assert (path / "model.txt").exists()
        assert (path / "meta.json").exists()
        assert not list(path.glob("*.pkl"))

    def test_model_file_is_text_not_a_pickle_stream(self, tmp_path):
        """LightGBMネイティブ形式はテキスト。pickleのマジックバイトが無い"""
        model, _ = _trained_model()
        path = ms.save_candidate(model, _meta(), base_dir=str(tmp_path))
        head = (path / "model.txt").read_bytes()[:16]
        assert b"\x80" not in head[:2]      # pickle protocol marker
        assert b"tree" in (path / "model.txt").read_bytes()[:200].lower()

    def test_goes_under_candidates_not_current(self, tmp_path):
        model, _ = _trained_model()
        path = ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        assert path == Path(tmp_path) / "candidates" / "m0001"
        assert not (Path(tmp_path) / "current.json").exists()

    def test_meta_records_the_full_contract(self, tmp_path):
        model, _ = _trained_model()
        path = ms.save_candidate(model, _meta(), base_dir=str(tmp_path))
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        for key in ("model_id", "trained_at", "training_window_sessions",
                    "symbols", "label_definition", "feature_cols",
                    "positive_rate", "fold_results", "code_version",
                    "lightgbm_version", "dataset_id"):
            assert key in meta

    def test_each_model_id_gets_its_own_directory(self, tmp_path):
        model, _ = _trained_model()
        a = ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        b = ms.save_candidate(model, _meta("m0002"), base_dir=str(tmp_path))
        assert a != b
        assert a.exists() and b.exists()

    def test_does_not_touch_the_legacy_pickle(self, tmp_path):
        """legacy の models/lgb_model.pkl を壊さない"""
        legacy = Path(tmp_path) / "lgb_model.pkl"
        legacy.write_bytes(b"legacy-model-bytes")
        model, _ = _trained_model()
        ms.save_candidate(model, _meta(), base_dir=str(tmp_path))
        assert legacy.read_bytes() == b"legacy-model-bytes"


class TestLoadModel:
    def test_round_trip_preserves_predictions(self, tmp_path):
        model, X = _trained_model()
        before = model.predict_proba(X)[:, 1]
        ms.save_candidate(model, _meta(), base_dir=str(tmp_path))
        loaded, _ = ms.load_model("m0001", base_dir=str(tmp_path))
        after = loaded.predict(X)
        assert np.allclose(before, after, atol=1e-9)

    def test_reads_back_the_meta(self, tmp_path):
        model, _ = _trained_model()
        ms.save_candidate(model, _meta(), base_dir=str(tmp_path))
        meta = ms.read_meta("m0001", base_dir=str(tmp_path))
        assert meta.model_id == "m0001"
        assert meta.feature_cols == ["f1", "f2"]
        assert meta.dataset_id == "ds0001"

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ms.load_model("nope", base_dir=str(tmp_path))


class TestSaveWrapperModels:
    """段階Cのラッパー型がそのまま保存できること（外部レビューR02）。

    `CurrentLightGBM` は `_model` と `predict_proba()` しか持たない。
    `getattr(model, "booster_", model).save_model(...)` という書き方だと
    AttributeError になり、それを握り潰した呼び出し側が
    「保存できていないのに学習成功」と扱ってしまう。
    """

    def _wrapper(self, single_class=False):
        from src.strategy.evaluation import CurrentLightGBM
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"f1": rng.normal(0, 1, 200), "f2": rng.normal(0, 1, 200)})
        y = pd.Series(np.ones(200, dtype=int) if single_class
                      else (X["f1"] > 0).astype(int))
        m = CurrentLightGBM()
        m.fit(X, y, np.ones(len(X)))
        return m, X

    def test_saves_a_two_class_wrapper(self, tmp_path):
        model, X = self._wrapper()
        path = ms.save_candidate(model, _meta("w0001"), base_dir=str(tmp_path))
        assert (path / "model.txt").exists()
        meta = ms.read_meta("w0001", base_dir=str(tmp_path))
        assert meta.model_kind == ms.KIND_BOOSTER

    def test_wrapper_round_trip_preserves_predictions(self, tmp_path):
        model, X = self._wrapper()
        before = model.predict_proba(X)
        ms.save_candidate(model, _meta("w0001"), base_dir=str(tmp_path))
        loaded, _ = ms.load_model("w0001", base_dir=str(tmp_path))
        assert np.allclose(before, loaded.predict(X), atol=1e-9)

    def test_saves_a_constant_model_without_a_booster(self, tmp_path):
        """単一クラスの学習結果には Booster が無い。定数として保存する"""
        model, X = self._wrapper(single_class=True)
        assert model.is_constant is True
        path = ms.save_candidate(model, _meta("c0001"), base_dir=str(tmp_path))
        assert (path / "constant.json").exists()
        assert not (path / "model.txt").exists()
        meta = ms.read_meta("c0001", base_dir=str(tmp_path))
        assert meta.model_kind == ms.KIND_CONSTANT

    def test_constant_model_round_trip_preserves_predictions(self, tmp_path):
        model, X = self._wrapper(single_class=True)
        before = model.predict_proba(X)
        ms.save_candidate(model, _meta("c0001"), base_dir=str(tmp_path))
        loaded, _ = ms.load_model("c0001", base_dir=str(tmp_path))
        after = loaded.predict(X)
        assert np.allclose(before, after, atol=1e-12)
        assert len(after) == len(X)

    def test_unsupported_type_raises_instead_of_silently_failing(self, tmp_path):
        class NotAModel:
            def predict_proba(self, X):
                return None

        with pytest.raises(TypeError, match="保存できないモデル型"):
            ms.save_candidate(NotAModel(), _meta("x0001"), base_dir=str(tmp_path))
        # 失敗したときにディレクトリを残さない
        assert not (Path(tmp_path) / "candidates" / "x0001").exists()


class TestCandidateDirectoriesAreImmutable:
    """候補ディレクトリは不変であること（外部レビューR12）。

    `exist_ok=True` で既存を受け入れて中身を上書きすると、そのIDを
    current や rollback 先が指していた場合に**昇格操作なしで実体が
    入れ替わる**。
    """

    def test_rejects_a_second_save_with_the_same_id(self, tmp_path):
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        with pytest.raises(ms.CandidateExists):
            ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))

    def test_the_original_files_are_untouched_after_a_rejected_save(self, tmp_path):
        model, X = _trained_model()
        path = ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        original = (path / "model.txt").read_bytes()

        other, _ = _trained_model()
        with pytest.raises(ms.CandidateExists):
            ms.save_candidate(other, _meta("m0001"), base_dir=str(tmp_path))
        assert (path / "model.txt").read_bytes() == original

    def test_the_model_current_points_at_cannot_be_replaced(self, tmp_path):
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        before = ms.load_model("m0001", base_dir=str(tmp_path))[0].model_to_string()

        with pytest.raises(ms.CandidateExists):
            ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        after = ms.load_model("m0001", base_dir=str(tmp_path))[0].model_to_string()
        assert before == after

    def test_a_failure_after_writing_the_model_leaves_nothing_behind(self, tmp_path):
        """モデル保存後にメタの書き込みが落ちても、公開されない"""
        model, _ = _trained_model()
        import src.strategy.model_store as mod

        original = mod._meta_to_json

        def boom(meta):
            raise OSError("ディスクが一杯です")

        mod._meta_to_json = boom
        try:
            with pytest.raises(OSError):
                ms.save_candidate(model, _meta("m0009"), base_dir=str(tmp_path))
        finally:
            mod._meta_to_json = original

        assert not (Path(tmp_path) / "candidates" / "m0009").exists()
        # 一時ディレクトリも残さない
        leftovers = list((Path(tmp_path) / "candidates").glob(".m0009.*"))
        assert leftovers == []

    def test_publishing_is_all_or_nothing(self, tmp_path):
        """公開されたディレクトリには必ずメタと本体が揃っている"""
        model, _ = _trained_model()
        path = ms.save_candidate(model, _meta("m0010"), base_dir=str(tmp_path))
        assert (path / "meta.json").exists()
        assert (path / "model.txt").exists()
        # 読み直せる（保存時に検証済み）
        loaded, meta = ms.load_model("m0010", base_dir=str(tmp_path))
        assert meta.model_id == "m0010"
        assert loaded is not None
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_model_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.strategy.model_store'`

- [ ] **Step 3: 実装を書く**

`src/strategy/model_store.py` を新規作成する。

```python
"""モデルの保存と現行の管理。

学習成功は**モデル更新ではなく候補の生成**である（spec §9）。現行の
ml_model.py は週次再学習が成功するとその戻り値をそのまま self.model へ
代入しており、成績に応じた合格判断がその経路に無かった（レビューF09）。

実体はバージョン別の**不変ディレクトリ**へ保存し、現行を指す**小さな参照
だけを原子的に更新する**。途中書き込みや学習失敗で現行版が失われず、
前のモデルへ戻せる。

保存形式は pickle を廃し **LightGBMネイティブ形式 + JSONメタ** にする。
pickle は信頼できないデータのロードで任意コードを実行しうる経路であり、
ハッシュ照合を足しても「モデルとメタを同時に置換できる相手」には効かない。

**旧 pickle 経路（ml_model.MODEL_PATH）は触らない。** engine_version: legacy
の受け皿として無改造で残す（spec §10）。
"""
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
from loguru import logger

CURRENT_REF = "current.json"
CANDIDATES_DIR = "candidates"
MODEL_FILE = "model.txt"
META_FILE = "meta.json"
CONSTANT_FILE = "constant.json"

# 保存形式。二値がそろったモデルと、単一クラス時の定数モデルを区別する
# （定数モデルには Booster が存在しない・外部レビューR02）。
KIND_BOOSTER = "lightgbm_booster"
KIND_CONSTANT = "constant"


@dataclass(frozen=True)
class ModelMeta:
    """モデルを再現・検証するための来歴。

    特徴量定義（feature_cols）を持つのは、昇格時に現行と食い違っていないかを
    検査するため。ラベル定義が変わったモデルを黙って現行にすると、
    同じ数字が別の意味になる。
    """
    model_id: str
    trained_at: datetime
    training_window_sessions: Optional[int] = None
    symbols: list = field(default_factory=list)
    label_definition: str = ""
    feature_cols: list = field(default_factory=list)
    positive_rate: Optional[float] = None
    fold_results: list = field(default_factory=list)
    code_version: Optional[str] = None
    lightgbm_version: Optional[str] = None
    dataset_id: Optional[str] = None
    # 保存形式。save_candidate() が書き込み時に確定させる
    model_kind: str = KIND_BOOSTER
    # ラベル契約ID（段階B後半 dataset.make_label_contract_id）。
    # このモデルがどう作られたラベルで学習されたかを固定する
    label_contract_id: Optional[str] = None


def candidate_dir(model_id: str, base_dir: str = "models") -> Path:
    return Path(base_dir) / CANDIDATES_DIR / model_id


def current_ref_path(base_dir: str = "models") -> Path:
    return Path(base_dir) / CURRENT_REF


def _meta_to_json(meta: ModelMeta) -> str:
    payload = asdict(meta)
    payload["trained_at"] = meta.trained_at.isoformat()
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _meta_from_json(text: str) -> ModelMeta:
    payload = json.loads(text)
    payload["trained_at"] = datetime.fromisoformat(payload["trained_at"])
    return ModelMeta(**payload)


class CandidateExists(FileExistsError):
    """同じ model_id の候補が既に存在する。"""


def _extract_artifact(model) -> tuple:
    """モデルから保存形式を取り出す。`(kind, payload)` を返す。

    段階Cの `_LightGbmBase` 系ラッパーは `_model` と `predict_proba()` しか
    持たず、`booster_` も `save_model()` も無い。`getattr(model, "booster_",
    model).save_model(...)` のような書き方は AttributeError になり、
    それを握り潰すと**保存できていないのに学習成功として扱われる**
    （外部レビューR02）。保存できる形は次の3つだけと決め、
    それ以外は**その場で例外にする**。

      1. 段階Cのラッパー … `is_constant` / `constant_probability` / `booster`
      2. scikit-learn API の LGBMClassifier … `booster_`
      3. `lgb.Booster` そのもの

    単一クラスしか見なかった定数モデルには Booster が存在しない。
    「二値がそろったモデル」と「定数モデル」で保存方式を分ける。
    """
    if hasattr(model, "is_constant"):
        if model.is_constant:
            return KIND_CONSTANT, float(model.constant_probability)
        booster = model.booster
        if booster is None:
            raise TypeError(
                "is_constant=False なのに booster が None です: "
                f"{type(model).__name__}")
        return KIND_BOOSTER, booster
    if hasattr(model, "booster_"):
        return KIND_BOOSTER, model.booster_
    if isinstance(model, lgb.Booster):
        return KIND_BOOSTER, model
    raise TypeError(
        f"保存できないモデル型です: {type(model).__name__}。"
        "is_constant/constant_probability/booster を公開するか、"
        "booster_ を持つか、lgb.Booster であること")


def save_candidate(model, meta: ModelMeta, *, base_dir: str = "models") -> Path:
    """学習結果を**候補として**保存する。現行は一切触らない。

    **既存の model_id へは書かない。** 候補ディレクトリは不変である。
    `exist_ok=True` で受け入れて中身を上書きすると、その ID を current や
    rollback 先が指していた場合に**昇格操作なしで実体が入れ替わる**
    （外部レビューR12）。同じIDでの再保存は `CandidateExists` にする。

    **全成果物を一時ディレクトリへ書き、読み直して検証してから公開する。**
    モデルを書いた後にメタの保存が失敗すると、中途半端なディレクトリが
    残って「保存済みだが読めない候補」になる。公開は `os.replace()` に
    よるディレクトリの原子的な差し替えで行う。
    """
    final = candidate_dir(meta.model_id, base_dir)
    if final.exists():
        raise CandidateExists(
            f"この model_id の候補は既に存在します: {meta.model_id}。"
            "候補は不変です。学習し直したなら新しいIDを付けてください")

    final.parent.mkdir(parents=True, exist_ok=True)
    kind, payload = _extract_artifact(model)

    staging = Path(tempfile.mkdtemp(prefix=f".{meta.model_id}.", dir=str(final.parent)))
    try:
        if kind == KIND_BOOSTER:
            payload.save_model(str(staging / MODEL_FILE))
        else:
            (staging / CONSTANT_FILE).write_text(
                json.dumps({"probability": payload}), encoding="utf-8")
        stored = replace(meta, model_kind=kind)
        (staging / META_FILE).write_text(_meta_to_json(stored), encoding="utf-8")

        # 公開前に読み直して、実際に復元できることを確かめる
        _load_from_dir(staging)

        os.replace(str(staging), str(final))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    logger.info(f"候補モデルを保存: {meta.model_id} → {final}（{kind}）")
    return final


def read_meta(model_id: str, *, base_dir: str = "models") -> ModelMeta:
    path = candidate_dir(model_id, base_dir) / META_FILE
    if not path.exists():
        raise FileNotFoundError(f"メタが見つかりません: {path}")
    return _meta_from_json(path.read_text(encoding="utf-8"))


class ConstantModel:
    """単一クラスしか見なかった学習の結果。常に同じ確率を返す。

    Booster が存在しないので、読み出し側が `predict()` を一様に呼べるよう
    最小の互換型を置く。`lgb.Booster.predict()` と同じく、正例確率の
    1次元配列を返す（外部レビューR02）。
    """

    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict(self, X, **kwargs):
        return np.full(len(X), self.probability, dtype=float)


def _load_from_dir(path: Path) -> tuple:
    """ディレクトリから (モデル, ModelMeta) を復元する。

    保存直後の検証にも使うので、`candidate_dir()` ではなく実パスを取る。
    """
    meta = _meta_from_json((path / META_FILE).read_text(encoding="utf-8"))
    if meta.model_kind == KIND_CONSTANT:
        payload = json.loads((path / CONSTANT_FILE).read_text(encoding="utf-8"))
        return ConstantModel(payload["probability"]), meta
    model_path = path / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
    return lgb.Booster(model_file=str(model_path)), meta


def load_model(model_id: str, *, base_dir: str = "models") -> tuple:
    """候補モデルを読み込む。(モデル, ModelMeta) を返す。

    モデルは `lgb.Booster` か `ConstantModel`。どちらも
    `predict(X) -> 正例確率の1次元配列` を持つ（scikit-learn APIの
    `predict_proba()[:, 1]` と同じ値）。読み出し側は型で分岐しない。
    """
    path = candidate_dir(model_id, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"候補が見つかりません: {path}")
    return _load_from_dir(path)
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_model_store.py -v`
Expected: PASS（22件）

- [ ] **Step 5: BOM確認とコミット**

Run: `head -c 3 src/strategy/model_store.py | xxd`（`2222 22` を確認。`efbb bf` なら下記で除去）

```python
for p in ["src/strategy/model_store.py", "tests/test_model_store.py"]:
    with open(p, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        with open(p, "wb") as f:
            f.write(data[3:])
```

```bash
git add src/strategy/model_store.py tests/test_model_store.py
git commit -m "$(cat <<'EOF'
feat(strategy): 候補モデルをバージョン別ディレクトリへ保存する仕組みを追加

学習成功をモデル更新ではなく候補の生成にする。実体はバージョン別の
不変ディレクトリへ置き、現行には一切触らない。
保存形式はpickleを廃しLightGBMネイティブ形式+JSONメタにする。pickleは
任意コード実行の経路で、ハッシュ照合を足してもモデルとメタを同時に
置換できる相手には効かない。
旧pickle経路(ml_model.MODEL_PATH)はlegacyの受け皿として触らない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: 現行を指す参照の原子的な更新

**Files:**
- Modify: `src/strategy/model_store.py`
- Test: `tests/test_model_store.py`

**Interfaces:**
- Consumes: Task 1
- Produces:
  - `CurrentRef`（frozen dataclass）: `model_id: str`, `switched_at: datetime`, `previous_model_id: Optional[str]`
  - `read_current(*, base_dir) -> Optional[CurrentRef]`
  - `set_current(model_id: str, *, base_dir, previous_model_id=None) -> CurrentRef`
  - `load_current(*, base_dir) -> Optional[tuple]` — `(Booster, ModelMeta)` または未昇格なら `None`
  - `rollback(*, base_dir) -> Optional[CurrentRef]` — 1つ前のモデルへ戻す

**背景（spec §9・§14）:** 「途中書き込みや学習失敗で現行版が失われないようにし、前のモデルへ戻せるようにする」。**参照ファイルの書き込みは原子的に行う**（一時ファイルへ書いてから `os.replace`）。書き込み途中で落ちても、参照は常に「前の完全な内容」か「新しい完全な内容」のどちらかになる。

**v2はモデル未昇格の状態から開始する。** `current.json` が無い状態を正常として扱い、`load_current()` は `None` を返す。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_model_store.py` の末尾に追記する。

```python
class TestCurrentRef:
    def test_starts_unpromoted(self, tmp_path):
        """v2はモデル未昇格の状態から開始する（spec §9）"""
        assert ms.read_current(base_dir=str(tmp_path)) is None
        assert ms.load_current(base_dir=str(tmp_path)) is None

    def test_set_current_points_at_the_model(self, tmp_path):
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ref = ms.set_current("m0001", base_dir=str(tmp_path))
        assert ref.model_id == "m0001"
        assert ref.previous_model_id is None
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"

    def test_load_current_returns_the_model_and_meta(self, tmp_path):
        model, X = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        booster, meta = ms.load_current(base_dir=str(tmp_path))
        assert meta.model_id == "m0001"
        assert len(booster.predict(X)) == len(X)

    def test_switching_records_the_previous(self, tmp_path):
        model, _ = _trained_model()
        for mid in ("m0001", "m0002"):
            ms.save_candidate(model, _meta(mid), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        ref = ms.set_current("m0002", base_dir=str(tmp_path))
        assert ref.model_id == "m0002"
        assert ref.previous_model_id == "m0001"

    def test_rejects_a_model_that_was_never_saved(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ms.set_current("ghost", base_dir=str(tmp_path))

    def test_the_reference_is_small_and_the_model_stays_put(self, tmp_path):
        """切替は参照だけを書き換える。モデルの実体はコピーも移動もしない"""
        model, _ = _trained_model()
        path = ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        before = (path / "model.txt").read_bytes()
        ms.set_current("m0001", base_dir=str(tmp_path))
        assert (path / "model.txt").read_bytes() == before
        assert ms.current_ref_path(str(tmp_path)).stat().st_size < 1000


class TestAtomicSwitch:
    def test_no_partial_reference_is_left_behind(self, tmp_path):
        """一時ファイルが残らない（原子的な置換の副作用が無い）"""
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        leftovers = [p for p in Path(tmp_path).iterdir()
                     if p.name != ms.CURRENT_REF and p.is_file()]
        assert leftovers == []

    def test_previous_reference_survives_a_failed_switch(self, tmp_path):
        """存在しないモデルへの切替が失敗しても、現行は前のまま残る"""
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            ms.set_current("ghost", base_dir=str(tmp_path))
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"

    def test_reference_survives_a_reread(self, tmp_path):
        """再起動後に同じモデルへ復帰する（参照を読み直しても同じ）"""
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        first = ms.read_current(base_dir=str(tmp_path))
        second = ms.read_current(base_dir=str(tmp_path))
        assert first.model_id == second.model_id
        assert first.switched_at == second.switched_at


class TestRollback:
    def test_returns_to_the_previous_model(self, tmp_path):
        model, _ = _trained_model()
        for mid in ("m0001", "m0002"):
            ms.save_candidate(model, _meta(mid), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        ms.set_current("m0002", base_dir=str(tmp_path))

        ref = ms.rollback(base_dir=str(tmp_path))
        assert ref.model_id == "m0001"
        assert ms.load_current(base_dir=str(tmp_path))[1].model_id == "m0001"

    def test_records_the_rolled_back_model_as_previous(self, tmp_path):
        model, _ = _trained_model()
        for mid in ("m0001", "m0002"):
            ms.save_candidate(model, _meta(mid), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        ms.set_current("m0002", base_dir=str(tmp_path))
        ref = ms.rollback(base_dir=str(tmp_path))
        assert ref.previous_model_id == "m0002"

    def test_returns_none_when_there_is_nothing_to_roll_back_to(self, tmp_path):
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        assert ms.rollback(base_dir=str(tmp_path)) is None
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"

    def test_returns_none_when_unpromoted(self, tmp_path):
        assert ms.rollback(base_dir=str(tmp_path)) is None
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_model_store.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.model_store' has no attribute 'read_current'`

- [ ] **Step 3: 実装を追加**

`src/strategy/model_store.py` の末尾に追加する。import に `from src.core import clock` を足す。

```python
@dataclass(frozen=True)
class CurrentRef:
    """現行モデルを指す参照。**実体ではなくIDだけを持つ小さなファイル。**

    切替でモデルの実体をコピー・移動しないため、途中で落ちても実体は壊れない。
    previous_model_id はロールバック先。
    """
    model_id: str
    switched_at: datetime
    previous_model_id: Optional[str] = None


def _write_atomically(path: Path, text: str) -> None:
    """一時ファイルへ書いてから置換する。

    書き込み途中で落ちても、参照は常に「前の完全な内容」か
    「新しい完全な内容」のどちらかになる（spec §9）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_current(*, base_dir: str = "models") -> Optional[CurrentRef]:
    """現行の参照を読む。**未昇格（参照が無い）を正常として None を返す。**

    v2はモデル未昇格の状態から開始する（spec §9）。
    """
    path = current_ref_path(base_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CurrentRef(
        model_id=payload["model_id"],
        switched_at=datetime.fromisoformat(payload["switched_at"]),
        previous_model_id=payload.get("previous_model_id"),
    )


def set_current(model_id: str, *, base_dir: str = "models",
                previous_model_id: Optional[str] = None) -> CurrentRef:
    """現行の参照を差し替える。

    **実体が存在することを先に確かめる。** 存在しないモデルを指す参照を
    書くと、次回の起動でモデルを読めなくなる。検査で落ちた場合、
    現行の参照は前のまま残る。
    """
    meta_path = candidate_dir(model_id, base_dir) / META_FILE
    model_path = candidate_dir(model_id, base_dir) / MODEL_FILE
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"保存されていないモデルは現行にできません: {model_id}")

    if previous_model_id is None:
        existing = read_current(base_dir=base_dir)
        previous_model_id = existing.model_id if existing else None

    ref = CurrentRef(model_id=model_id, switched_at=clock.now(),
                     previous_model_id=previous_model_id)
    _write_atomically(current_ref_path(base_dir), json.dumps({
        "model_id": ref.model_id,
        "switched_at": ref.switched_at.isoformat(),
        "previous_model_id": ref.previous_model_id,
    }, ensure_ascii=False, indent=2))
    logger.warning(f"現行モデルを切替: {previous_model_id} → {model_id}")
    return ref


def load_current(*, base_dir: str = "models") -> Optional[tuple]:
    """現行モデルを読み込む。未昇格なら None。"""
    ref = read_current(base_dir=base_dir)
    if ref is None:
        return None
    return load_model(ref.model_id, base_dir=base_dir)


def rollback(*, base_dir: str = "models") -> Optional[CurrentRef]:
    """1つ前のモデルへ戻す。戻り先が無ければ None（現行は変えない）。"""
    ref = read_current(base_dir=base_dir)
    if ref is None or ref.previous_model_id is None:
        return None
    return set_current(ref.previous_model_id, base_dir=base_dir,
                       previous_model_id=ref.model_id)
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_model_store.py -v`
Expected: PASS（22件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/model_store.py tests/test_model_store.py
git commit -m "$(cat <<'EOF'
feat(strategy): 現行モデルの参照を原子的に切り替える仕組みを追加

切替は実体のコピー・移動ではなく小さな参照ファイルの置換で行う。
一時ファイルへ書いてからos.replaceするので、途中で落ちても参照は
常に前か新しいかのどちらかの完全な内容になる。
存在しないモデルを指す参照を書かないよう実体の存在を先に確かめ、
検査で落ちた場合は現行が前のまま残る。1つ前へのロールバックも持つ。
v2はモデル未昇格（参照なし）を正常な開始状態として扱う。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 昇格の契約

**Files:**
- Create: `src/strategy/promotion.py`
- Modify: `src/data/database.py`（`ModelPromotion` を追加）
- Test: `tests/test_model_promotion.py`

**Interfaces:**
- Consumes: Task 1・2、`evaluation.load_prediction_details`
- Produces:
  - `ModelPromotion` モデル: `model_id` / `evaluation_run_id` / `decided_by` / `reason` / `previous_model_id` / `state` / `switched_at`
  - `PROMOTION_PENDING` / `PROMOTION_COMMITTED` / `PROMOTION_FAILED`
  - `recover_promotions(*, base_dir="models") -> list` — 起動時に未決着の昇格を参照の実体と突き合わせて閉じる。**自動で参照は書き換えない**
  - `promotion_history(*, limit=50) -> list`
  - `PromotionCheck`（frozen dataclass）: `ok: bool`, `blockers: list`
  - `check_promotable(model_id, *, evaluation_run_id, degraded, base_dir, expected_feature_cols) -> PromotionCheck`
  - `promote(model_id, *, evaluation_run_id, decided_by, reason, degraded, base_dir, expected_feature_cols) -> int` — **2段階で切り替える**（pending記録 → 参照切替 → committed確定）。切替に失敗したら failed で閉じて送出する。`ModelPromotion.id` を返す

**昇格不可の条件（spec §9）:**

| 条件 | 理由 |
|---|---|
| `degraded` な実行 | 推論例外が起きた実行の成績は比較に使えない |
| 未評価（`evaluation_run_id` が無い・予測明細が無い） | 何を根拠に昇格するのか残らない |
| 特徴量定義の不一致 | ラベルや特徴量が変わったモデルを黙って現行にすると、同じ数字が別の意味になる |

**自動昇格は実装しない。** `promote()` は明示的な呼び出しでのみ動き、`decided_by` と `reason` を必須にする。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_model_promotion.py` を新規作成する。

```python
"""昇格の契約（src/strategy/promotion.py）のテスト

自動昇格は実装しない。昇格は明示的な呼び出しでのみ起き、
判断者と理由の記録を必須にする（spec §9）。
"""
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.core import config as cfg
from src.data import database as db
from src.data.database import get_session
from src.strategy import evaluation
from src.strategy import model_store as ms
from src.strategy import promotion


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


def _saved_model(tmp_path, model_id="m0001", feature_cols=("f1", "f2")):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({c: rng.normal(0, 1, 200) for c in feature_cols})
    y = pd.Series((X[feature_cols[0]] > 0).astype(int))
    m = lgb.LGBMClassifier(n_estimators=10, num_leaves=4, random_state=42, verbose=-1)
    m.fit(X, y)
    meta = ms.ModelMeta(
        model_id=model_id, trained_at=datetime(2026, 9, 12, 10, 0, 0),
        feature_cols=list(feature_cols), dataset_id="ds0001",
        lightgbm_version=lgb.__version__)
    ms.save_candidate(m, meta, base_dir=str(tmp_path))
    return model_id


_LC = "testcontract"


def _recorded_evaluation(run_id="run1", model_id="m0001", *,
                         resolved=True, purpose=None, degraded=False):
    """評価実行を1件ぶん保存する。

    **予測・実績・実行記録の3つを揃える。** 予測だけを保存した状態を
    「評価済み」と呼ばないため（外部レビューR13）。
    `resolved=False` で実績を保存しない状態を作れる。
    """
    from src.strategy.evaluation import (
        PURPOSE_SHADOW, PURPOSE_VALIDATION, RunConfig, save_evaluation_run,
        save_outcomes, save_predictions)

    purpose = purpose or PURPOSE_VALIDATION
    preds = pd.DataFrame({
        "event_id": ["7203:20260105"],
        "label_contract_id": [_LC],
        "raw_probability": [0.6],
        "calibrated_probability": [0.55],
        "fold_index": [0 if purpose == PURPOSE_VALIDATION else -1],
    })
    save_predictions(preds, run_id, model_id, purpose=purpose)

    if resolved:
        events = pd.DataFrame({
            "event_id": ["7203:20260105"],
            "label_contract_id": [_LC],
            "status": ["resolved"],
            "label": [1],
            "net_return": [0.03],
        })
        save_outcomes(events)

    save_evaluation_run(
        run_id,
        RunConfig(dataset_id="ds0001", label_contract_id=_LC,
                  feature_version="f1", execution_model_version="t1_open_v1",
                  code_version="abc1234", config_json="{}", config_hash="cfg1"),
        purpose=purpose, model_id=model_id, n_folds=1, n_predictions=1,
        degraded_reasons=(["推論に失敗しました"] if degraded else []))


class TestPromotionBlockers:
    def test_degraded_run_blocks_promotion(self, isolated_db, tmp_path):
        _saved_model(tmp_path)
        _recorded_evaluation()
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=True,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("degraded" in b for b in got.blockers)

    def test_missing_evaluation_blocks_promotion(self, isolated_db, tmp_path):
        _saved_model(tmp_path)
        got = promotion.check_promotable(
            "m0001", evaluation_run_id=None, degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("未評価" in b for b in got.blockers)

    def test_evaluation_without_predictions_blocks_promotion(self, isolated_db, tmp_path):
        """評価IDがあっても予測明細が無ければ根拠にならない"""
        _saved_model(tmp_path)
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run-with-no-rows", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("予測明細" in b for b in got.blockers)

    def test_feature_mismatch_blocks_promotion(self, isolated_db, tmp_path):
        _saved_model(tmp_path, feature_cols=("f1", "f2"))
        _recorded_evaluation()
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2", "f3"])
        assert got.ok is False
        assert any("特徴量" in b for b in got.blockers)

    def test_unsaved_model_blocks_promotion(self, isolated_db, tmp_path):
        got = promotion.check_promotable(
            "ghost", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1"])
        assert got.ok is False

    def test_all_blockers_are_reported_together(self, isolated_db, tmp_path):
        """1つ直せば通る、を繰り返さずに済むよう全件返す"""
        _saved_model(tmp_path, feature_cols=("f1",))
        got = promotion.check_promotable(
            "m0001", evaluation_run_id=None, degraded=True,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert len(got.blockers) >= 3

    def test_clean_candidate_passes(self, isolated_db, tmp_path):
        _saved_model(tmp_path)
        _recorded_evaluation()
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is True
        assert got.blockers == []


class TestPromote:
    def _clean(self, tmp_path, model_id="m0001"):
        _saved_model(tmp_path, model_id=model_id)
        _recorded_evaluation(model_id=model_id)
        return model_id

    def test_records_who_decided_and_why(self, isolated_db, tmp_path):
        self._clean(tmp_path)
        pid = promotion.promote(
            "m0001", evaluation_run_id="run1", decided_by="garnet",
            reason="内側foldで定数モデルを上回ったため", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])

        with get_session() as session:
            row = session.scalar(select(db.ModelPromotion))
        assert row.id == pid
        assert row.model_id == "m0001"
        assert row.evaluation_run_id == "run1"
        assert row.decided_by == "garnet"
        assert "定数モデル" in row.reason
        assert row.switched_at is not None

    def test_switches_the_current_reference(self, isolated_db, tmp_path):
        self._clean(tmp_path)
        promotion.promote(
            "m0001", evaluation_run_id="run1", decided_by="garnet",
            reason="ok", degraded=False, base_dir=str(tmp_path),
            expected_feature_cols=["f1", "f2"])
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"

    def test_records_the_previous_model(self, isolated_db, tmp_path):
        self._clean(tmp_path, "m0001")
        self._clean(tmp_path, "m0002")
        promotion.promote("m0001", evaluation_run_id="run1", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])
        promotion.promote("m0002", evaluation_run_id="run1", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])
        with get_session() as session:
            rows = list(session.scalars(select(db.ModelPromotion)).all())
        assert rows[-1].previous_model_id == "m0001"

    def test_refuses_a_blocked_candidate(self, isolated_db, tmp_path):
        _saved_model(tmp_path)
        with pytest.raises(ValueError, match="昇格できません"):
            promotion.promote(
                "m0001", evaluation_run_id="run1", decided_by="garnet",
                reason="ok", degraded=True, base_dir=str(tmp_path),
                expected_feature_cols=["f1", "f2"])

    def test_refused_promotion_leaves_the_current_untouched(self, isolated_db, tmp_path):
        """昇格が拒否されても現行は動かない（途中状態を残さない）"""
        self._clean(tmp_path, "m0001")
        promotion.promote("m0001", evaluation_run_id="run1", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])
        _saved_model(tmp_path, model_id="m0002")

        with pytest.raises(ValueError):
            promotion.promote("m0002", evaluation_run_id=None, decided_by="g",
                              reason="ng", degraded=False, base_dir=str(tmp_path),
                              expected_feature_cols=["f1", "f2"])
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"

    def test_requires_a_decider_and_a_reason(self, isolated_db, tmp_path):
        """自動昇格を実装しない。判断者と理由は必須"""
        self._clean(tmp_path)
        with pytest.raises(ValueError, match="判断者"):
            promotion.promote("m0001", evaluation_run_id="run1", decided_by="",
                              reason="ok", degraded=False, base_dir=str(tmp_path),
                              expected_feature_cols=["f1", "f2"])
        with pytest.raises(ValueError, match="理由"):
            promotion.promote("m0001", evaluation_run_id="run1", decided_by="g",
                              reason="", degraded=False, base_dir=str(tmp_path),
                              expected_feature_cols=["f1", "f2"])


class TestUnresolvedEvaluationBlocksPromotion:
    """予測だけでは「評価済み」にしない（外部レビューR13）。

    `load_prediction_details()` は実績が無くても行を返す。行数だけを
    見る条件では、ラベルが全て未確定の shadow 予測でも昇格できてしまう。
    """

    def test_predictions_without_outcomes_block_promotion(self, isolated_db, tmp_path):
        _saved_model(tmp_path)
        _recorded_evaluation(resolved=False)
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("実績が1件も確定していません" in b for b in got.blockers)

    def test_shadow_only_blocks_promotion(self, isolated_db, tmp_path):
        from src.strategy.evaluation import PURPOSE_SHADOW
        _saved_model(tmp_path)
        _recorded_evaluation(purpose=PURPOSE_SHADOW)
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("shadow" in b for b in got.blockers)

    def test_stored_degraded_blocks_even_when_the_caller_says_otherwise(
            self, isolated_db, tmp_path):
        """呼び出し側の bool ではなく保存済みの実行記録を根拠にする"""
        _saved_model(tmp_path)
        _recorded_evaluation(degraded=True)
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,   # 嘘の申告
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("保存済みの実行記録が degraded" in b for b in got.blockers)

    def test_missing_run_record_blocks_promotion(self, isolated_db, tmp_path):
        """予測明細はあるが実行記録が無い → 何を測ったのか復元できない"""
        from src.strategy.evaluation import save_predictions
        _saved_model(tmp_path)
        save_predictions(pd.DataFrame({
            "event_id": ["7203:20260105"], "label_contract_id": [_LC],
            "raw_probability": [0.6], "calibrated_probability": [0.55],
            "fold_index": [0]}), "run1", "m0001")
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("評価実行の記録がありません" in b for b in got.blockers)

    def test_run_for_a_different_model_blocks_promotion(self, isolated_db, tmp_path):
        _saved_model(tmp_path, model_id="m0002")
        _recorded_evaluation(model_id="m0001")
        got = promotion.check_promotable(
            "m0002", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("別のモデル" in b for b in got.blockers)

    def test_a_fully_resolved_validation_run_is_promotable(self, isolated_db, tmp_path):
        _saved_model(tmp_path)
        _recorded_evaluation()
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is True, got.blockers


class TestPromotionIsRecoverable:
    """切替の途中で落ちても決着できること（外部レビューR11）。

    参照ファイルの原子的置換は、DBを含む取引の原子性ではない。
    「参照は新モデルなのに昇格記録が無い」状態を作らないため、
    昇格の意図を先に永続化し、切替後に確定させる。
    """

    def _ready(self, tmp_path, model_id="m0001"):
        _saved_model(tmp_path, model_id=model_id)
        _recorded_evaluation(model_id=model_id)

    def test_successful_promotion_is_committed(self, isolated_db, tmp_path):
        self._ready(tmp_path)
        pid = promotion.promote(
            "m0001", evaluation_run_id="run1", decided_by="g", reason="ok",
            degraded=False, base_dir=str(tmp_path),
            expected_feature_cols=["f1", "f2"])
        with get_session() as session:
            row = session.get(db.ModelPromotion, pid)
            assert row.state == promotion.PROMOTION_COMMITTED
            assert row.switched_at is not None

    def test_a_failed_switch_is_recorded_as_failed(self, isolated_db, tmp_path):
        """参照の切替に失敗しても、記録だけが committed で残らない"""
        self._ready(tmp_path)

        def boom(*args, **kwargs):
            raise OSError("参照ファイルを書けません")

        original = ms.set_current
        ms.set_current = boom
        try:
            with pytest.raises(OSError):
                promotion.promote(
                    "m0001", evaluation_run_id="run1", decided_by="g",
                    reason="ok", degraded=False, base_dir=str(tmp_path),
                    expected_feature_cols=["f1", "f2"])
        finally:
            ms.set_current = original

        with get_session() as session:
            row = session.scalar(select(db.ModelPromotion))
        assert row.state == promotion.PROMOTION_FAILED
        assert row.switched_at is None
        # 現行は切り替わっていない
        assert ms.read_current(base_dir=str(tmp_path)) is None

    def test_recovery_commits_a_pending_row_whose_switch_actually_happened(
            self, isolated_db, tmp_path):
        """切替後・確定前に落ちた場合 → 実体に合わせて committed にする"""
        self._ready(tmp_path)
        with get_session() as session:
            row = db.ModelPromotion(
                model_id="m0001", evaluation_run_id="run1", decided_by="g",
                reason="ok", previous_model_id=None,
                state=promotion.PROMOTION_PENDING, switched_at=None)
            session.add(row)
            session.commit()
            pid = row.id
        ms.set_current("m0001", base_dir=str(tmp_path))   # 切替は完了していた

        resolved = promotion.recover_promotions(base_dir=str(tmp_path))
        assert len(resolved) == 1
        assert resolved[0]["resolved_to"] == promotion.PROMOTION_COMMITTED
        with get_session() as session:
            assert session.get(db.ModelPromotion, pid).state == \
                promotion.PROMOTION_COMMITTED

    def test_recovery_fails_a_pending_row_whose_switch_never_happened(
            self, isolated_db, tmp_path):
        """切替前に落ちた場合 → failed にする。参照は触らない"""
        self._ready(tmp_path)
        with get_session() as session:
            row = db.ModelPromotion(
                model_id="m0001", evaluation_run_id="run1", decided_by="g",
                reason="ok", previous_model_id=None,
                state=promotion.PROMOTION_PENDING, switched_at=None)
            session.add(row)
            session.commit()
            pid = row.id

        resolved = promotion.recover_promotions(base_dir=str(tmp_path))
        assert resolved[0]["resolved_to"] == promotion.PROMOTION_FAILED
        with get_session() as session:
            assert session.get(db.ModelPromotion, pid).state == \
                promotion.PROMOTION_FAILED
        # 参照は書き換えない（自動でやり直さない）
        assert ms.read_current(base_dir=str(tmp_path)) is None

    def test_recovery_is_idempotent(self, isolated_db, tmp_path):
        self._ready(tmp_path)
        promotion.promote("m0001", evaluation_run_id="run1", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])
        assert promotion.recover_promotions(base_dir=str(tmp_path)) == []
        assert promotion.recover_promotions(base_dir=str(tmp_path)) == []

    def test_no_committed_row_without_a_switched_at(self, isolated_db, tmp_path):
        """不変条件: committed なら切替時刻が必ずある"""
        self._ready(tmp_path)
        promotion.promote("m0001", evaluation_run_id="run1", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])
        for row in promotion.promotion_history():
            if row.state == promotion.PROMOTION_COMMITTED:
                assert row.switched_at is not None
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_model_promotion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.strategy.promotion'`

- [ ] **Step 3: モデルを追加**

`src/data/database.py` の `class RunModelUsage` の直後に追加する（段階D後半が未実装なら `class Prediction` の直後）。

```python
class ModelPromotion(Base):
    """モデルを現行へ昇格した記録。

    自動昇格は行わない。**誰が・何を根拠に・なぜ切り替えたか**を残す
    （spec §9）。学習が成功しただけで運用モデルが入れ替わる経路を無くす。
    """
    __tablename__ = "model_promotions"
    id = Column(Integer, primary_key=True)
    model_id = Column(String(64), nullable=False)
    evaluation_run_id = Column(String(64))
    decided_by = Column(String(64))
    reason = Column(Text)
    previous_model_id = Column(String(64))
    # 参照の切替が完了したか。pending / committed / failed。
    # 参照ファイルとDBは別の永続化先なので、片方だけが進んだ状態が
    # 起こりうる。それを検出して決着できるように持つ（外部レビューR11）。
    state = Column(String(16), default="committed")
    # 実際に切り替わった時刻。pending / failed では None
    switched_at = Column(DateTime)

    __table_args__ = (
        Index("ix_model_promotions_model_id", "model_id"),
        Index("ix_model_promotions_state", "state"),
    )
```

- [ ] **Step 4: 実装を書く**

`src/strategy/promotion.py` を新規作成する。

```python
"""昇格の契約 — 評価を通った候補だけが現行になる。

現行の ml_model は週次再学習が成功するとその戻り値をそのまま運用モデルへ
代入しており、成績に応じた合格判断がその経路に無かった（レビューF09）。

**自動昇格は実装しない。** 昇格は明示的な呼び出しでのみ起き、判断者と理由の
記録を必須にする。AUC等の採用数値をここで決める必要はない（spec §9）。
"""
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.strategy import model_store as ms


@dataclass(frozen=True)
class PromotionCheck:
    """昇格できるかの判定。できない理由は**全件**返す。

    1つ直せば通る、を繰り返さずに済むようにするため。
    """
    ok: bool
    blockers: list = field(default_factory=list)


def check_promotable(model_id: str, *, evaluation_run_id: Optional[str],
                     degraded: bool, base_dir: str,
                     expected_feature_cols: list) -> PromotionCheck:
    """昇格不可の条件を検査する（spec §9）。

    - degraded な実行: 推論例外が起きた実行・再学習が結線されていない
      診断実行の成績は比較に使えない。**呼び出し側の bool ではなく
      保存済みの実行記録から読む**（外部レビューR13）
    - 未評価: 何を根拠に昇格するのかが残らない
    - **実績が未確定**: 予測明細の行数だけでは「評価済み」と言えない。
      `load_prediction_details()` は実績が無くても行を返すので、
      ラベルが全て未確定の shadow 予測でも行数条件は通ってしまう
    - **shadow だけ**: 並行記録は現行と候補の比較であって評価ではない
    - 評価実行と候補モデルの食い違い、ラベル契約の食い違い
    - 特徴量定義の不一致: ラベルや特徴量が変わったモデルを黙って現行にすると、
      同じ数字が別の意味になる

    引数の `degraded` は**補助的な早期拒否**としてのみ使う。保存状態と
    食い違う場合は保存状態を優先する。
    """
    from src.strategy.evaluation import (
        PURPOSE_SHADOW, load_evaluation_run, load_prediction_details)

    _load_run = load_evaluation_run
    blockers: list = []

    try:
        meta = ms.read_meta(model_id, base_dir=base_dir)
    except FileNotFoundError:
        return PromotionCheck(ok=False,
                              blockers=[f"候補として保存されていません: {model_id}"])

    if degraded:
        blockers.append("degraded な実行の成績は昇格の根拠にできません")

    if not evaluation_run_id:
        blockers.append("未評価です（evaluation_run_id がありません）")
    else:
        # 保存済みの実行記録を根拠にする。呼び出し側が渡した degraded を
        # そのまま信じない（外部レビューR13）
        run = _load_run(evaluation_run_id)
        if run is None:
            blockers.append(
                f"評価実行の記録がありません（evaluation_run_id={evaluation_run_id}）")
        else:
            if run.degraded:
                blockers.append(
                    "保存済みの実行記録が degraded です。その成績は昇格の根拠にできません")
            if run.model_id and run.model_id != model_id:
                blockers.append(
                    f"評価実行が別のモデルのものです: 実行={run.model_id} 候補={model_id}")
            if run.label_contract_id and meta.label_contract_id                     and run.label_contract_id != meta.label_contract_id:
                blockers.append(
                    "ラベル契約が一致しません: "
                    f"実行={run.label_contract_id} モデル={meta.label_contract_id}")

        details = load_prediction_details(evaluation_run_id, model_id=model_id)
        if len(details) == 0:
            blockers.append(
                f"予測明細がありません（evaluation_run_id={evaluation_run_id}）")
        else:
            # **行数だけでは「評価済み」と言えない。**
            # load_prediction_details は実績が無くても行を返すので、
            # ラベルが全て未確定の shadow 予測でもこの条件を通せてしまう
            # （外部レビューR13）。実績の確定を根拠にする。
            resolved = details["actual_label"].notna()
            if not resolved.any():
                blockers.append(
                    "実績が1件も確定していません（予測だけでは成績を測れません）")

            purposes = set(details["purpose"].dropna().astype(str))
            if purposes and purposes <= {PURPOSE_SHADOW}:
                blockers.append(
                    "shadow記録だけでは昇格できません"
                    "（並行記録は現行と候補の比較であって評価ではありません）")

            unresolved = int((~resolved).sum())
            if unresolved:
                logger.info(
                    f"昇格検査: 未確定の予測が{unresolved}件あります"
                    f"（確定{int(resolved.sum())}件で判断します）")

    if list(meta.feature_cols) != list(expected_feature_cols):
        blockers.append(
            f"特徴量定義が一致しません: モデル={list(meta.feature_cols)} "
            f"期待={list(expected_feature_cols)}"
        )

    return PromotionCheck(ok=not blockers, blockers=blockers)


# 昇格の状態。参照の切替とDB記録は別の永続化先なので、片方だけが進んだ
# 状態が起こりうる。それを**検出して決着できる**ようにする（外部レビューR11）。
PROMOTION_PENDING = "pending"       # 記録済み。参照の切替はまだ
PROMOTION_COMMITTED = "committed"   # 参照も切り替わった
PROMOTION_FAILED = "failed"         # 切替に失敗。現行は前のまま


def _close_promotion(promotion_id: int, state: str, *,
                     switched_at: Optional[datetime]) -> None:
    """昇格記録を確定させる。"""
    from src.data.database import ModelPromotion, get_session

    with get_session() as session:
        row = session.get(ModelPromotion, promotion_id)
        if row is None:
            logger.error(f"昇格記録が見つかりません: id={promotion_id}")
            return
        row.state = state
        row.switched_at = switched_at
        session.commit()


def recover_promotions(*, base_dir: str = "models") -> list:
    """起動時に、決着していない昇格を実体と突き合わせて閉じる。

    `promote()` は「記録(pending) → 参照切替 → 記録(committed)」の順で進む。
    2と3の間でプロセスが落ちると pending が残る。このとき参照ファイルが
    既に新モデルを指しているかどうかで、実際に切り替わったかが分かる。

      - 参照が pending の model_id を指している → 切替は完了していた。committed
      - 指していない → 切替前に落ちた。failed

    **自動で参照を書き換えない。** 実体に合わせて記録のほうを直すだけである。
    取引の履歴に関わる状態なので、勝手に「やり直す」ことはしない。
    決着した件数と内容を返し、呼び出し側がユーザーへ報告する。

    戻り値: `[{"promotion_id", "model_id", "resolved_to"}, ...]`
    """
    from src.data.database import ModelPromotion, get_session
    from sqlalchemy import select as sa_select

    ref = ms.read_current(base_dir=base_dir)
    current_id = ref.model_id if ref else None

    resolved = []
    with get_session() as session:
        rows = list(session.scalars(sa_select(ModelPromotion).where(
            ModelPromotion.state == PROMOTION_PENDING)).all())
        for row in rows:
            if current_id == row.model_id:
                row.state = PROMOTION_COMMITTED
                row.switched_at = ref.switched_at
                outcome = PROMOTION_COMMITTED
            else:
                row.state = PROMOTION_FAILED
                row.switched_at = None
                outcome = PROMOTION_FAILED
            resolved.append({"promotion_id": row.id, "model_id": row.model_id,
                             "resolved_to": outcome})
        if rows:
            session.commit()

    for item in resolved:
        logger.warning(
            f"未決着の昇格を決着させました: {item['model_id']} "
            f"→ {item['resolved_to']}（参照の実体に合わせました）")
    return resolved


def promotion_history(*, limit: int = 50) -> list:
    """昇格履歴（新しい順）。pending が残っていれば混じる。"""
    from src.data.database import ModelPromotion, get_session
    from sqlalchemy import select as sa_select

    with get_session() as session:
        rows = list(session.scalars(
            sa_select(ModelPromotion)
            .order_by(ModelPromotion.id.desc()).limit(limit)).all())
        for r in rows:
            session.expunge(r)
        return rows


def promote(model_id: str, *, evaluation_run_id: Optional[str],
            decided_by: str, reason: str, degraded: bool,
            base_dir: str, expected_feature_cols: list) -> int:
    """候補を現行へ昇格する。ModelPromotion.id を返す。

    **検査に通らなければ現行を一切変更せずに例外を投げる。** 途中状態を残さない。
    判断者と理由は必須（自動昇格を作らないため）。
    """
    from src.data.database import ModelPromotion, get_session

    if not decided_by:
        raise ValueError("判断者（decided_by）は必須です")
    if not reason:
        raise ValueError("理由（reason）は必須です")

    check = check_promotable(
        model_id, evaluation_run_id=evaluation_run_id, degraded=degraded,
        base_dir=base_dir, expected_feature_cols=expected_feature_cols)
    if not check.ok:
        raise ValueError("昇格できません: " + " / ".join(check.blockers))

    previous = ms.read_current(base_dir=base_dir)
    previous_id = previous.model_id if previous else None

    # ── 2段階で切り替える（外部レビューR11）──────────────────────────
    #
    # 参照の切替を先に行い、その後で履歴をcommitすると、DB書込みの失敗や
    # 両処理の間でのプロセス停止によって「参照は新モデルなのに昇格記録が
    # 無い」状態が残る。順序を逆にしても「記録はあるのに参照が古い」という
    # 逆向きの不整合が残るだけで、解決しない。
    #
    # 参照ファイル単体の原子的置換（os.replace）は、DBを含む取引の
    # 原子性ではない。**片方だけが進んだ状態を検出して決着できる**ように、
    # 昇格の意図を先に永続化し、切替後に確定させる。
    #
    #   1. pending として記録（switched_at は None）
    #   2. 参照を切り替える（os.replace）
    #   3. committed へ更新
    #
    # 2と3の間で落ちた場合、起動時に pending が残る。
    # `recover_promotions()` が参照の実体と突き合わせて決着させる。

    with get_session() as session:
        row = ModelPromotion(
            model_id=model_id, evaluation_run_id=evaluation_run_id,
            decided_by=decided_by, reason=reason,
            previous_model_id=previous_id,
            state=PROMOTION_PENDING, switched_at=None,
        )
        session.add(row)
        session.commit()
        promotion_id = row.id

    try:
        ref = ms.set_current(model_id, base_dir=base_dir)
    except BaseException:
        # 切替に失敗した。現行は前のまま。意図を失敗として閉じる
        _close_promotion(promotion_id, PROMOTION_FAILED, switched_at=None)
        raise

    _close_promotion(promotion_id, PROMOTION_COMMITTED,
                     switched_at=ref.switched_at)

    logger.warning(
        f"モデルを昇格: {previous_id} → {model_id}（判断者={decided_by}）")
    return promotion_id
```

- [ ] **Step 5: テストを実行して成功を確認**

Run: `pytest tests/test_model_promotion.py -v`
Expected: PASS（13件）

- [ ] **Step 6: 全体回帰とコミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

```bash
git add src/strategy/promotion.py src/data/database.py tests/test_model_promotion.py
git commit -m "$(cat <<'EOF'
feat(strategy,data): 昇格の契約を追加

現行は週次再学習が成功するとその戻り値をそのまま運用モデルへ代入して
おり、成績に応じた合格判断がその経路に無かった。
自動昇格は実装せず、明示的な呼び出しと判断者・理由の記録を必須にする。
degradedな実行・未評価・特徴量定義の不一致は昇格不可とし、理由は全件
返す（1つ直せば通る、を繰り返さずに済むように）。
検査に通らなければ現行を一切変更せず例外にする。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: shadow 運用（並行記録）

**Files:**
- Create: `src/strategy/shadow.py`
- Modify: `src/data/database.py`（`ShadowComparisonRow` を追加）
- Test: `tests/test_shadow.py`

**Interfaces:**
- Consumes: Task 1・2、`evaluation.save_predictions` / `PURPOSE_SHADOW`
- Produces:
  - `ShadowComparison`（frozen dataclass）: `event_id` / `current_probability` / `candidate_probability` / `current_takes` / `candidate_takes` / `agreement: str`
  - `compare(event_ids, features, *, current, candidate, threshold) -> list[ShadowComparison]`
  - `record_shadow(comparisons, *, evaluation_run_id, candidate_model_id, current_model_id, threshold, label_contract_id) -> int` — **比較そのもの**（両モデルID・両確率・閾値・採否・一致区分・ラベル契約）を保存する
  - `load_shadow_comparisons(evaluation_run_id) -> pd.DataFrame` — 保存済みの比較を読み直す。ここから `disagreement_summary()` を再計算できることが並行記録の合格条件
  - `ShadowComparisonRow` モデル: `evaluation_run_id` / `event_id` / `label_contract_id` / `current_model_id` / `candidate_model_id` / `current_probability` / `candidate_probability` / `threshold` / `current_takes` / `candidate_takes` / `agreement` / `recorded_at`
  - `disagreement_summary(comparisons) -> dict`

**背景（spec §9）:** 同じ入力に対し現行と候補の判断を**並行記録する**。**候補は発注に繋がない。** 差が何によって生じたかを、見送った候補の結果も含めて記録する。

**発注に繋がらないことの担保:** `shadow.py` は発注系モジュール（`src/execution/`）を import しない。これをテストで固定する。

**保存先のテーブル。** `src/data/database.py` の `ModelPromotion` の直後に追加する。

```python
class ShadowComparisonRow(Base):
    """shadow運用での「現行 vs 候補」の比較1件。

    `Prediction` には候補の確率しか入らない。それだけでは再起動後に
    「その時どちらを採り、なぜ見送ったか」を復元できない（外部レビューR21）。
    **両モデルID・両確率・判断閾値・採否・一致区分**を同じ行に置く。

    現行が未昇格のときも行は作る。`current_model_id` と
    `current_probability` を NULL にして「現行が無かった」という状態を残す。
    行ごと作らないと「比較しなかった」のか「現行が無かった」のかを
    後から区別できない。
    """
    __tablename__ = "shadow_comparisons"
    id = Column(Integer, primary_key=True)
    evaluation_run_id = Column(String(64), nullable=False)
    event_id = Column(String(64), nullable=False)
    label_contract_id = Column(String(32), nullable=False)
    current_model_id = Column(String(64))          # 未昇格なら NULL
    candidate_model_id = Column(String(64), nullable=False)
    current_probability = Column(Float)            # 未昇格なら NULL
    candidate_probability = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    current_takes = Column(Integer, default=0)
    candidate_takes = Column(Integer, default=0)
    agreement = Column(String(24))
    recorded_at = Column(DateTime, default=clock.now)

    __table_args__ = (
        Index("ix_shadow_comparisons_run", "evaluation_run_id"),
        Index("ix_shadow_comparisons_event",
              "evaluation_run_id", "label_contract_id", "event_id",
              unique=True),
    )
```

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_shadow.py` を新規作成する。

```python
"""shadow運用（src/strategy/shadow.py）のテスト

同じ入力に現行と候補の判断を並行記録する。**候補は発注に繋がない**。
差が何によって生じたかを、見送った候補の結果も含めて記録する（spec §9）。
"""
import inspect

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.core import config as cfg
from src.data import database as db
from src.data.database import get_session
from src.strategy import evaluation
from src.strategy import shadow


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


class _FixedModel:
    """与えた確率をそのまま返すテスト用モデル"""

    def __init__(self, probabilities):
        self._p = np.asarray(probabilities, dtype=float)

    def predict(self, X):
        return self._p[:len(X)]


def _features(n=4):
    return pd.DataFrame({"f1": np.arange(n, dtype=float),
                         "f2": np.arange(n, dtype=float)})


class TestCompare:
    def test_records_both_probabilities(self):
        got = shadow.compare(
            ["a", "b"], _features(2),
            current=_FixedModel([0.2, 0.8]),
            candidate=_FixedModel([0.7, 0.3]),
            threshold=0.5)
        assert [c.current_probability for c in got] == pytest.approx([0.2, 0.8])
        assert [c.candidate_probability for c in got] == pytest.approx([0.7, 0.3])

    def test_classifies_agreement(self):
        got = shadow.compare(
            ["both", "neither", "only_current", "only_candidate"], _features(4),
            current=_FixedModel([0.9, 0.1, 0.9, 0.1]),
            candidate=_FixedModel([0.9, 0.1, 0.1, 0.9]),
            threshold=0.5)
        assert [c.agreement for c in got] == [
            "both_take", "both_skip", "only_current", "only_candidate"]

    def test_includes_events_the_candidate_skipped(self):
        """見送った候補の結果も記録に残す（採った分だけでは差が測れない）"""
        got = shadow.compare(
            ["x"], _features(1),
            current=_FixedModel([0.9]), candidate=_FixedModel([0.1]),
            threshold=0.5)
        assert len(got) == 1
        assert got[0].candidate_takes is False

    def test_works_without_a_current_model(self):
        """v2はモデル未昇格から始まる。現行が無くても候補は記録できる"""
        got = shadow.compare(
            ["x"], _features(1),
            current=None, candidate=_FixedModel([0.8]), threshold=0.5)
        assert got[0].current_probability is None
        assert got[0].current_takes is False
        assert got[0].candidate_takes is True


class TestRecordShadow:
    def test_saves_with_the_shadow_purpose(self, isolated_db):
        comparisons = shadow.compare(
            ["a"], _features(1),
            current=_FixedModel([0.2]), candidate=_FixedModel([0.7]),
            threshold=0.5)
        n = shadow.record_shadow(comparisons, evaluation_run_id="shadow1",
                                 candidate_model_id="m0002",
                                 current_model_id="m0001", threshold=0.5,
                                 label_contract_id=_LC)
        assert n == 1

        with get_session() as session:
            row = session.scalar(select(db.Prediction))
        assert row.purpose == evaluation.PURPOSE_SHADOW
        assert row.model_id == "m0002"
        assert row.evaluation_run_id == "shadow1"

    def test_uses_a_sentinel_fold_index(self, isolated_db):
        """shadowはfoldに属さない"""
        comparisons = shadow.compare(
            ["a"], _features(1),
            current=_FixedModel([0.2]), candidate=_FixedModel([0.7]),
            threshold=0.5)
        shadow.record_shadow(comparisons, evaluation_run_id="shadow1",
                             candidate_model_id="m0002",
                             current_model_id="m0001", threshold=0.5,
                             label_contract_id=_LC)
        with get_session() as session:
            row = session.scalar(select(db.Prediction))
        assert row.fold_index == -1

    def test_outcomes_are_not_written(self, isolated_db):
        """予測時点で実績は未確定。PredictionOutcomeは書かない"""
        comparisons = shadow.compare(
            ["a"], _features(1),
            current=_FixedModel([0.2]), candidate=_FixedModel([0.7]),
            threshold=0.5)
        shadow.record_shadow(comparisons, evaluation_run_id="shadow1",
                             candidate_model_id="m0002",
                             current_model_id="m0001", threshold=0.5,
                             label_contract_id=_LC)
        with get_session() as session:
            outcomes = list(session.scalars(select(db.PredictionOutcome)).all())
        assert outcomes == []


class TestShadowRecordsBothSides:
    """比較そのものが保存されること（外部レビューR21）。

    候補の確率だけを書くと、再起動後に「その時どちらを採り、なぜ見送ったか」
    を復元できない。関数名や一時的な戻り値では並行記録にならない。
    """

    def _record(self, run_id="shadow1", current=_FixedModel([0.2]),
                current_id="m0001"):
        comparisons = shadow.compare(
            ["a", "b"], _features(2),
            current=current, candidate=_FixedModel([0.7, 0.3]),
            threshold=0.5)
        shadow.record_shadow(
            comparisons, evaluation_run_id=run_id,
            candidate_model_id="m0002", current_model_id=current_id,
            threshold=0.5, label_contract_id=_LC)
        return comparisons

    def test_stores_both_model_ids_and_both_probabilities(self, isolated_db):
        self._record(current=_FixedModel([0.2, 0.9]))
        got = shadow.load_shadow_comparisons("shadow1")
        assert len(got) == 2
        assert set(got["current_model_id"]) == {"m0001"}
        assert set(got["candidate_model_id"]) == {"m0002"}
        assert got["current_probability"].notna().all()
        assert got["candidate_probability"].notna().all()

    def test_stores_the_threshold_and_the_decisions(self, isolated_db):
        self._record(current=_FixedModel([0.2, 0.9]))
        got = shadow.load_shadow_comparisons("shadow1").set_index("event_id")
        assert set(got["threshold"]) == {0.5}
        assert got.loc["a", "current_takes"] is False or             got.loc["a", "current_takes"] == False    # noqa: E712
        assert got.loc["a", "candidate_takes"] == True   # noqa: E712
        assert got.loc["b", "current_takes"] == True     # noqa: E712
        assert got.loc["b", "candidate_takes"] == False  # noqa: E712

    def test_the_summary_can_be_recomputed_from_the_database(self, isolated_db):
        """再起動後に当時の比較内訳を復元できること（合格条件）"""
        comparisons = self._record(current=_FixedModel([0.2, 0.9]))
        expected = shadow.disagreement_summary(comparisons)

        stored = shadow.load_shadow_comparisons("shadow1")
        rebuilt = shadow.disagreement_summary([
            shadow.ShadowComparison(
                event_id=r["event_id"],
                current_probability=r["current_probability"],
                candidate_probability=r["candidate_probability"],
                current_takes=bool(r["current_takes"]),
                candidate_takes=bool(r["candidate_takes"]),
                agreement=r["agreement"])
            for _, r in stored.iterrows()])
        assert rebuilt == expected

    def test_records_the_unpromoted_current_as_its_own_state(self, isolated_db):
        """現行が未昇格でも行は残す

        行を作らないと「比較しなかった」のか「現行が無かった」のかを
        後から区別できない。
        """
        self._record(current=None, current_id=None)
        got = shadow.load_shadow_comparisons("shadow1")
        assert len(got) == 2
        assert got["current_model_id"].isna().all()
        assert got["current_probability"].isna().all()
        assert (~got["current_takes"]).all()

    def test_stores_the_label_contract(self, isolated_db):
        """どのラベル契約に対する比較かを残す（実績と結合するため）"""
        self._record()
        got = shadow.load_shadow_comparisons("shadow1")
        assert set(got["label_contract_id"]) == {_LC}

    def test_runs_are_isolated(self, isolated_db):
        self._record(run_id="shadow1")
        self._record(run_id="shadow2")
        assert len(shadow.load_shadow_comparisons("shadow1")) == 2
        assert len(shadow.load_shadow_comparisons("shadow2")) == 2


class TestDisagreementSummary:
    def test_counts_each_category(self):
        comparisons = shadow.compare(
            ["a", "b", "c", "d"], _features(4),
            current=_FixedModel([0.9, 0.1, 0.9, 0.1]),
            candidate=_FixedModel([0.9, 0.1, 0.1, 0.9]),
            threshold=0.5)
        got = shadow.disagreement_summary(comparisons)
        assert got["both_take"] == 1
        assert got["both_skip"] == 1
        assert got["only_current"] == 1
        assert got["only_candidate"] == 1
        assert got["n"] == 4

    def test_agreement_rate_is_reported(self):
        comparisons = shadow.compare(
            ["a", "b", "c", "d"], _features(4),
            current=_FixedModel([0.9, 0.1, 0.9, 0.1]),
            candidate=_FixedModel([0.9, 0.1, 0.1, 0.9]),
            threshold=0.5)
        assert shadow.disagreement_summary(comparisons)["agreement_rate"] == pytest.approx(0.5)

    def test_empty_input_is_safe(self):
        got = shadow.disagreement_summary([])
        assert got["n"] == 0
        assert got["agreement_rate"] is None


class TestNotWiredToOrdering:
    """候補は発注に繋がらない（spec §9）"""

    def _import_lines(self):
        """import文だけを取り出す。

        ソース全体を対象にすると、docstringが発注系に言及しただけで落ちる
        脆いテストになる。依存の有無だけを見る。
        """
        src = inspect.getsource(shadow)
        return "\n".join(
            line for line in src.splitlines()
            if line.startswith("import ") or line.startswith("from ")
        )

    def test_module_does_not_import_execution_modules(self):
        joined = self._import_lines()
        assert "execution" not in joined
        assert "order" not in joined
        assert "kabu_client" not in joined

    def test_module_does_not_import_the_trading_service(self):
        assert "trading" not in self._import_lines()

    def test_public_functions_return_records_not_orders(self):
        """公開関数はどれも記録用の値だけを返す"""
        for name in ("compare", "record_shadow", "disagreement_summary"):
            assert callable(getattr(shadow, name))
        assert not hasattr(shadow, "place_order")
        assert not hasattr(shadow, "execute")
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_shadow.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.strategy.shadow'`

- [ ] **Step 3: 実装を書く**

`src/strategy/shadow.py` を新規作成する。

```python
"""shadow運用 — 現行と候補の判断を並行記録する。

同じ入力に対し現行と候補の判断を記録し、差が何によって生じたかを
**見送った候補の結果も含めて**残す。採った分だけでは差が測れない。

**候補は発注に繋がらない。** 本モジュールは発注系（src/execution）も
取引サービス（src/services/trading）も import しない。記録だけを行う。
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.evaluation import PURPOSE_SHADOW, save_predictions

# shadowの予測はどのfoldにも属さない
SHADOW_FOLD_INDEX = -1

AGREEMENT_BOTH_TAKE = "both_take"
AGREEMENT_BOTH_SKIP = "both_skip"
AGREEMENT_ONLY_CURRENT = "only_current"
AGREEMENT_ONLY_CANDIDATE = "only_candidate"


@dataclass(frozen=True)
class ShadowComparison:
    """1イベントについての、現行と候補の判断の突き合わせ。"""
    event_id: str
    current_probability: Optional[float]
    candidate_probability: float
    current_takes: bool
    candidate_takes: bool
    agreement: str


def _predict(model, features: pd.DataFrame) -> Optional[np.ndarray]:
    if model is None:
        return None
    return np.asarray(model.predict(features), dtype=float)


def compare(event_ids: list, features: pd.DataFrame, *,
            current, candidate, threshold: float) -> list:
    """同じ入力に現行と候補を当て、判断の一致・不一致を記録用に並べる。

    現行が無い（v2はモデル未昇格から始まる）場合は、現行側を「採らない」
    として扱い確率は None にする。
    """
    current_p = _predict(current, features)
    candidate_p = _predict(candidate, features)
    if candidate_p is None:
        raise ValueError("候補モデルは必須です")

    out = []
    for i, event_id in enumerate(event_ids):
        cur = float(current_p[i]) if current_p is not None else None
        cand = float(candidate_p[i])
        cur_takes = cur is not None and cur >= threshold
        cand_takes = cand >= threshold
        if cur_takes and cand_takes:
            agreement = AGREEMENT_BOTH_TAKE
        elif not cur_takes and not cand_takes:
            agreement = AGREEMENT_BOTH_SKIP
        elif cur_takes:
            agreement = AGREEMENT_ONLY_CURRENT
        else:
            agreement = AGREEMENT_ONLY_CANDIDATE
        out.append(ShadowComparison(
            event_id=str(event_id), current_probability=cur,
            candidate_probability=cand, current_takes=cur_takes,
            candidate_takes=cand_takes, agreement=agreement,
        ))
    return out


def record_shadow(comparisons: list, *, evaluation_run_id: str,
                  candidate_model_id: str, current_model_id: Optional[str],
                  threshold: float, label_contract_id: str) -> int:
    """**比較そのもの**を保存する。保存件数を返す。

    候補の確率だけを書くと、再起動後に「その時どちらを採り、なぜ見送ったか」
    を復元できない（外部レビューR21）。並行記録と呼べるのは、
    **両モデルID・両確率・判断閾値・採否・一致区分・入力契約**が
    同じ行に揃っているときだけである。関数名や一時的な戻り値では満たさない。

    現行が未昇格のときは `current_model_id=None` / `current_probability=None`
    で保存し、「現行が無かった」という状態として残す。行を作らないと
    「比較しなかった」のか「現行が無かった」のか後から区別できない。

    実績（PredictionOutcome）はここでは書かない。予測時点では確定して
    いないため、満期後に別途関連付ける（spec §7）。結合キーは
    `(label_contract_id, event_id)`。
    """
    from src.data.database import ShadowComparisonRow, get_session

    if not comparisons:
        return 0

    # 候補の予測は従来どおり Prediction へ（指標の再計算に使う）
    preds = pd.DataFrame({
        "event_id": [c.event_id for c in comparisons],
        "label_contract_id": [label_contract_id] * len(comparisons),
        "raw_probability": [c.candidate_probability for c in comparisons],
        "calibrated_probability": [c.candidate_probability for c in comparisons],
        "fold_index": [SHADOW_FOLD_INDEX] * len(comparisons),
    })
    n = save_predictions(preds, evaluation_run_id, candidate_model_id,
                         purpose=PURPOSE_SHADOW)

    # 比較の内訳はこちらへ。これが「並行記録」の実体
    now = clock.now()
    with get_session() as session:
        for c in comparisons:
            session.add(ShadowComparisonRow(
                evaluation_run_id=evaluation_run_id,
                event_id=c.event_id,
                label_contract_id=label_contract_id,
                current_model_id=current_model_id,
                candidate_model_id=candidate_model_id,
                current_probability=c.current_probability,
                candidate_probability=c.candidate_probability,
                threshold=float(threshold),
                current_takes=1 if c.current_takes else 0,
                candidate_takes=1 if c.candidate_takes else 0,
                agreement=c.agreement,
                recorded_at=now,
            ))
        session.commit()

    logger.info(
        f"shadow記録: run={evaluation_run_id} 現行={current_model_id} "
        f"候補={candidate_model_id} 閾値={threshold} {n}件")
    return n


def load_shadow_comparisons(evaluation_run_id: str) -> pd.DataFrame:
    """保存済みの比較を読み直す。

    この出力から `disagreement_summary()` を再計算できること
    （＝再起動後に当時の比較を復元できること）が並行記録の合格条件である。
    """
    from src.data.database import ShadowComparisonRow, get_session
    from sqlalchemy import select as sa_select

    with get_session() as session:
        rows = list(session.scalars(sa_select(ShadowComparisonRow).where(
            ShadowComparisonRow.evaluation_run_id == evaluation_run_id)).all())
    return pd.DataFrame([{
        "event_id": r.event_id,
        "label_contract_id": r.label_contract_id,
        "current_model_id": r.current_model_id,
        "candidate_model_id": r.candidate_model_id,
        "current_probability": r.current_probability,
        "candidate_probability": r.candidate_probability,
        "threshold": r.threshold,
        "current_takes": bool(r.current_takes),
        "candidate_takes": bool(r.candidate_takes),
        "agreement": r.agreement,
    } for r in rows])


def disagreement_summary(comparisons: list) -> dict:
    """判断の一致・不一致の内訳。

    「候補が現行と何件違ったか」だけでなく、どちらの向きに違ったかを分けて
    数える。片側にだけ寄っているなら、それは閾値の差であって能力の差では
    ないかもしれない。
    """
    counts = {
        AGREEMENT_BOTH_TAKE: 0, AGREEMENT_BOTH_SKIP: 0,
        AGREEMENT_ONLY_CURRENT: 0, AGREEMENT_ONLY_CANDIDATE: 0,
    }
    for c in comparisons:
        counts[c.agreement] += 1
    n = len(comparisons)
    agreed = counts[AGREEMENT_BOTH_TAKE] + counts[AGREEMENT_BOTH_SKIP]
    return {
        **counts, "n": n,
        "agreement_rate": (agreed / n) if n else None,
    }
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_shadow.py -v`
Expected: PASS（14件）

- [ ] **Step 5: 全体回帰とBOM確認、コミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

Run: `head -c 3 src/strategy/shadow.py | xxd`（`2222 22` を確認）

```bash
git add src/strategy/shadow.py tests/test_shadow.py
git commit -m "$(cat <<'EOF'
feat(strategy): shadow運用の並行記録を追加

同じ入力に現行と候補の判断を当て、見送った候補の結果も含めて記録する。
採った分だけでは差が測れないため。一致・不一致はどちらの向きに違ったかを
分けて数える（片側に寄っているなら閾値の差であって能力の差ではない
かもしれない）。
候補は発注に繋がない。発注系も取引サービスもimportしないことを
テストで固定する。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 学習失敗で現行が失われないことの検証

**Files:**
- Modify: `src/strategy/model_store.py`
- Test: `tests/test_model_store.py`

**Interfaces:**
- Consumes: Task 1・2
- Produces: `train_as_candidate(train_fn, meta_fn, *, base_dir) -> Optional[str]` — 学習を候補生成として実行し、失敗しても現行に影響を与えない

**背景（spec §14 段階E完了条件）:** 「学習失敗で現行モデルが失われない」「昇格処理が途中で失敗しても現行が一貫して復元できる」。現行 `ml_model` は週次再学習の戻り値を運用モデルへ直接代入するため、学習の途中経過が運用へ漏れる。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_model_store.py` の末尾に追記する。

```python
class TestTrainAsCandidate:
    def _promoted(self, tmp_path):
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))

    def test_successful_training_creates_a_candidate_not_a_switch(self, tmp_path):
        """学習成功はモデル更新ではなく候補の生成（spec §9）"""
        self._promoted(tmp_path)
        model, _ = _trained_model()

        model_id = ms.train_as_candidate(
            lambda: model, lambda: _meta("m0002"), base_dir=str(tmp_path))

        assert model_id == "m0002"
        assert ms.candidate_dir("m0002", str(tmp_path)).exists()
        # 現行は変わらない
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"

    def test_training_failure_leaves_the_current_intact(self, tmp_path):
        """学習失敗で現行モデルが失われない（spec §14 段階E完了条件）"""
        self._promoted(tmp_path)

        def broken():
            raise RuntimeError("学習に失敗しました")

        assert ms.train_as_candidate(
            broken, lambda: _meta("m0002"), base_dir=str(tmp_path)) is None
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"
        assert ms.load_current(base_dir=str(tmp_path))[1].model_id == "m0001"

    def test_metadata_failure_leaves_no_half_written_candidate(self, tmp_path):
        """メタの生成で落ちたら候補ディレクトリを残さない"""
        self._promoted(tmp_path)
        model, _ = _trained_model()

        def broken_meta():
            raise RuntimeError("メタの生成に失敗しました")

        assert ms.train_as_candidate(
            lambda: model, broken_meta, base_dir=str(tmp_path)) is None
        assert not ms.candidate_dir("m0002", str(tmp_path)).exists()

    def test_failure_when_unpromoted_stays_unpromoted(self, tmp_path):
        def broken():
            raise RuntimeError("boom")

        assert ms.train_as_candidate(
            broken, lambda: _meta("m0002"), base_dir=str(tmp_path)) is None
        assert ms.read_current(base_dir=str(tmp_path)) is None

    def test_previous_candidate_survives_a_new_failure(self, tmp_path):
        """過去の候補も壊さない"""
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        before = (ms.candidate_dir("m0001", str(tmp_path)) / "model.txt").read_bytes()

        def broken():
            raise RuntimeError("boom")

        ms.train_as_candidate(broken, lambda: _meta("m0002"), base_dir=str(tmp_path))
        after = (ms.candidate_dir("m0001", str(tmp_path)) / "model.txt").read_bytes()
        assert after == before
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_model_store.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.model_store' has no attribute 'train_as_candidate'`

- [ ] **Step 3: 実装を追加**

`src/strategy/model_store.py` の末尾に追加する。import に `import shutil` を足す。

```python
def train_as_candidate(train_fn, meta_fn, *,
                       base_dir: str = "models") -> Optional[str]:
    """学習を**候補の生成として**実行する。成功したら model_id を返す。

    **失敗しても現行には一切触れない。** 現行の ml_model は週次再学習の
    戻り値を運用モデルへ直接代入するため、学習の途中経過が運用へ漏れる
    （レビューF09）。ここでは学習・メタ生成・保存のいずれで落ちても、
    現行の参照も過去の候補も変わらない。

    書きかけの候補ディレクトリは残さない（次回の読み込みで壊れたモデルを
    掴まないため）。
    """
    try:
        model = train_fn()
        meta = meta_fn()
    except Exception as e:
        logger.error(f"候補モデルの学習に失敗しました（現行は変更していません）: {e}")
        return None

    path = candidate_dir(meta.model_id, base_dir)
    existed = path.exists()
    try:
        save_candidate(model, meta, base_dir=base_dir)
    except Exception as e:
        logger.error(f"候補モデルの保存に失敗しました（現行は変更していません）: {e}")
        if not existed and path.exists():
            shutil.rmtree(path, ignore_errors=True)
        return None
    return meta.model_id
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_model_store.py -v`
Expected: PASS（27件）

- [ ] **Step 5: 全体回帰とコミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

```bash
git add src/strategy/model_store.py tests/test_model_store.py
git commit -m "$(cat <<'EOF'
feat(strategy): 学習を候補生成として実行し失敗時も現行を守る

現行のml_modelは週次再学習の戻り値を運用モデルへ直接代入するため、
学習の途中経過が運用へ漏れる。学習・メタ生成・保存のいずれで落ちても
現行の参照も過去の候補も変わらないようにする。
書きかけの候補ディレクトリは残さない（次回の読み込みで壊れたモデルを
掴まないため）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## 段階E 完了条件の確認

spec §14 の段階E完了条件を検証する。

- [ ] **確認1: 学習失敗で現行モデルが失われない**

Run: `pytest tests/test_model_store.py::TestTrainAsCandidate -v`
Expected: PASS（5件）

- [ ] **確認2: shadow の候補が発注に繋がらない**

Run: `pytest tests/test_shadow.py::TestNotWiredToOrdering -v`
Expected: PASS（3件）

- [ ] **確認3: 昇格処理が途中で失敗しても現行が一貫して復元できる**

Run: `pytest tests/test_model_promotion.py::TestPromote::test_refused_promotion_leaves_the_current_untouched -v`
Run: `pytest tests/test_model_store.py::TestAtomicSwitch -v`
Expected: PASS

- [ ] **確認4: 自動昇格が無い**

Run: `pytest tests/test_model_promotion.py::TestPromote::test_requires_a_decider_and_a_reason -v`
Expected: PASS

Run: `grep -rn "promote(" src/ --include=*.py | grep -v "def promote" | grep -v "tests/"`
Expected: 出力が無いこと（本番コードのどこからも自動で呼ばれていない）

- [ ] **確認5: legacy の pickle 経路が無傷である**

Run: `git log --oneline -- src/strategy/ml_model.py | head -3`
Expected: 段階A〜E のコミットが一件も出ないこと

Run: `pytest tests/test_ml_model_save.py tests/test_ml_train_multi.py -v`
Expected: PASS

Run: `ls models/lgb_model.pkl`
Expected: 存在すること（削除・移動していない）

- [ ] **確認6: v2がモデル未昇格から始まる**

Run: `pytest tests/test_model_store.py::TestCurrentRef::test_starts_unpromoted -v`
Expected: PASS

- [ ] **確認7: 既存経路に回帰が無い**

Run: `pytest tests/ -q`
Expected: 段階Eの着手前と同じ結果（新規テスト54件ぶんだけ増える）

---

## 段階A〜E 全体の完了状態

本計画の完了時点で、spec §14 の完了条件は次のとおり満たされる。

| 段階 | 完了条件 | 検証 |
|---|---|---|
| A | シグナルが `data_as_of` を持つ。確定していない足を新規候補にしない。分割で評価額が増えない | 実装済み（`e9b1999`） |
| B | 将来データで過去の特徴量が変わらない。未成熟ラベルを学習しない。ラベルの終了が実行と一致 | `test_indicators.py` / `test_policy.py` / `test_dataset.py` |
| C | 外側foldの値を変えても学習入力・閾値・校正・学習窓が変わらない。予測明細から指標を再計算できる | `test_validation.py` / `test_evaluation.py` |
| D | Tの終値で判断してもT+1以降にのみ約定。degradedが識別できる。モデル使用履歴を辿れる | `test_walkforward.py` / `test_portfolio.py` |
| E | 学習失敗で現行が失われない。shadowが発注に繋がらない。昇格の途中失敗から復元できる | `test_model_store.py` / `test_model_promotion.py` / `test_shadow.py` |

**全体の合格条件は固定していない。** spec §14 のとおり「AUCが0.55を超えたら採用」「paperで1ヶ月成功したら合格」といった一律基準は置かない。売買回数が少なければ期間を延ばし、比較の不確実性が大きければ保留する。利益目標・許容ドローダウン・学習窓の最終値は、ユーザーの資金と運用目的に依存する未確定事項である。

## 残る作業（本計画のスコープ外）

- **F10〜F15**（リスク価格の鮮度、プロファイル適用の原子性、初期設定の保護、ログアウトとAPIトークン、ログのマスキング、DB制約と監査メタデータ）は別プラン
- **`engine_version: v2` への実際の切替判断**。段階A〜Eは「測れる状態」を作るところまでで、v2を運用へ昇格させるかは実データでの比較を見てから決める
- **旧エンジン（`engine.py`）と旧 pickle 経路（`ml_model.py`）の廃止**。v2が運用に乗ってから別途判断する
- **`docs/詳細設計書.md` / `docs/概要設計書.md` への反映**。新テーブル（`CorporateAction` / `Dataset` / `Prediction` / `PredictionOutcome` / `RunModelUsage` / `ModelPromotion`）と新モジュール群を追記する。main へマージして GitHub へ push する際に Obsidian vault への同期も必要
