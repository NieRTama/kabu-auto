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

from src.core import clock

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

      1. 段階Cのラッパー（LightGBM系） … `is_constant` / `constant_probability` / `booster`
      2. scikit-learn API の LGBMClassifier … `booster_`
      3. `lgb.Booster` そのもの

    単一クラスしか見なかった定数モデルには Booster が存在しない。
    「二値がそろったモデル」と「定数モデル」で保存方式を分ける。

    **`is_constant` だけでは「段階Cのラッパーである」判定として不十分。**
    段階C2の `LogisticRegressionModel` も `_LightGbmBase` と全く同じ
    `is_constant` / `constant_probability` を意図的に公開しているが、
    `booster` はLightGBM固有の概念なので持たない（evaluation.py の
    コメント参照）。`is_constant` の有無だけで分岐すると、二値が揃った
    `LogisticRegressionModel` を保存しようとしたときに契約どおりの
    `TypeError` ではなく `booster` 属性アクセス時の `AttributeError` に
    なってしまう（外部レビュー最終ブランチレビュー M2）。
    `hasattr(model, "booster")` を先に確認し、無ければその場で
    明示的な `TypeError` にする（黙って握り潰さない・外部レビューR02）。
    """
    if hasattr(model, "is_constant"):
        if model.is_constant:
            return KIND_CONSTANT, float(model.constant_probability)
        if not hasattr(model, "booster"):
            raise TypeError(
                f"保存できないモデル型です: {type(model).__name__}。"
                "is_constant/constant_probability はありますが booster を"
                "公開していません。LightGBM系モデル以外は現状保存できません")
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
    cand_dir = candidate_dir(model_id, base_dir)
    meta_path = cand_dir / META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"保存されていないモデルは現行にできません: {model_id}")

    # メタを読んでモデル種別を確認
    meta = read_meta(model_id, base_dir=base_dir)
    if meta.model_kind == KIND_BOOSTER:
        model_path = cand_dir / MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError(f"保存されていないモデルは現行にできません: {model_id}")
    elif meta.model_kind == KIND_CONSTANT:
        const_path = cand_dir / CONSTANT_FILE
        if not const_path.exists():
            raise FileNotFoundError(f"保存されていないモデルは現行にできません: {model_id}")
    else:
        raise FileNotFoundError(f"不明なモデル種別: {meta.model_kind}")

    if previous_model_id is None:
        existing = read_current(base_dir=base_dir)
        if existing is not None and existing.model_id == model_id:
            # 既に現行と同じモデルへの再設定。ここで previous_model_id を
            # 自分自身（＝existing.model_id）に書き換えると、以後
            # rollback() が過去のモデルへ永久に戻れなくなる
            # （外部レビュー最終ブランチレビュー M4）。既存の previous を
            # そのまま引き継ぎ、ロールバック可能な状態を壊さない
            previous_model_id = existing.previous_model_id
        else:
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
