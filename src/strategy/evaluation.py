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
import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)

from src.core import clock
from src.strategy import validation
from src.strategy.dataset import STATUS_RESOLVED
from src.strategy.indicators import FEATURE_COLS
from src.strategy.validation import Preprocessor, apply_preprocessor, fit_preprocessor

PURPOSE_VALIDATION = "validation"
PURPOSE_SHADOW = "shadow"


def save_predictions(predictions: pd.DataFrame, evaluation_run_id: str,
                     model_id: str, *, purpose: str = PURPOSE_VALIDATION) -> int:
    """予測明細を保存する。保存した件数を返す。

    predictions は event_id / label_contract_id / raw_probability /
    calibrated_probability / fold_index の列を持つこと。
    実績ラベルはここでは書かない。

    `train_positive_rate` / `threshold` 列は任意（外部レビューI-3）。
    無い呼び出し元との後方互換のため、無ければ既定値で埋めずに
    NULLのまま保存する。

    `label_contract_id` を必須にするのは、後で実績と結合するときの
    キーがこの組だからである（外部レビューR07）。欠けたまま保存すると
    結合先が定まらないので、既定値で埋めずに例外にする。

    同じ `(evaluation_run_id, model_id)` の既存明細を保存前に削除してから
    挿入する（run単位の置換）。`EvaluationRun` / `PredictionOutcome` は
    同じIDでの再実行を上書きとして扱うのに、ここだけ素の追記だと
    同一runを2回実行しただけで明細が静かに二重化する（外部レビューI-2）。
    """
    from sqlalchemy import delete as sa_delete

    from src.data.database import Prediction, get_session

    if len(predictions) == 0:
        return 0
    if "label_contract_id" not in predictions.columns:
        raise ValueError(
            "predictions に label_contract_id 列がありません。"
            "実績との結合キーなので省略できません")
    now = clock.now()
    has_train_rate = "train_positive_rate" in predictions.columns
    has_threshold = "threshold" in predictions.columns
    with get_session() as session:
        session.execute(sa_delete(Prediction).where(
            Prediction.evaluation_run_id == evaluation_run_id,
            Prediction.model_id == model_id))
        for _, r in predictions.iterrows():
            session.add(Prediction(
                event_id=str(r["event_id"]),
                label_contract_id=str(r["label_contract_id"]),
                evaluation_run_id=evaluation_run_id,
                model_id=model_id,
                predicted_at=now,
                raw_probability=float(r["raw_probability"]),
                calibrated_probability=float(r["calibrated_probability"]),
                fold_index=int(r["fold_index"]),
                purpose=purpose,
                train_positive_rate=(
                    float(r["train_positive_rate"])
                    if has_train_rate and pd.notna(r["train_positive_rate"]) else None),
                threshold=(
                    float(r["threshold"])
                    if has_threshold and pd.notna(r["threshold"]) else None),
            ))
        session.commit()
    return len(predictions)


def save_outcomes(events: pd.DataFrame) -> int:
    """決着したイベントの実績を保存する。

    上書きの単位は **`(label_contract_id, event_id)`** である。同じ銘柄・
    同じ判断日でも、別の退出ポリシーやコストで作ったラベルは別の行になる。
    `event_id` だけを鍵にすると、コストを変えて評価をやり直した瞬間に
    過去runの実績が置き換わり、保存済みの予測と結合し直したときの指標が
    後から変わってしまう（外部レビューR07）。

    予測より後に呼ぶ。未成熟・未約定にはラベルが無いので保存しない。
    """
    from src.data.database import PredictionOutcome, get_session
    from sqlalchemy import select as sa_select

    resolved = events[events["status"] == STATUS_RESOLVED]
    if len(resolved) == 0:
        return 0
    if "label_contract_id" not in resolved.columns:
        raise ValueError(
            "events に label_contract_id 列がありません。"
            "実績の同一性を決める鍵なので省略できません")
    now = clock.now()
    with get_session() as session:
        existing = {
            (r.label_contract_id, r.event_id): r
            for r in session.scalars(sa_select(PredictionOutcome)).all()
        }
        for _, r in resolved.iterrows():
            key = (str(r["label_contract_id"]), str(r["event_id"]))
            label = int(r["label"])
            ret = float(r["net_return"]) if pd.notna(r.get("net_return")) else None
            if key in existing:
                existing[key].actual_label = label
                existing[key].net_return = ret
                existing[key].resolved_at = now
            else:
                session.add(PredictionOutcome(
                    label_contract_id=key[0], event_id=key[1],
                    actual_label=label, net_return=ret, resolved_at=now))
        session.commit()
    return len(resolved)


def load_prediction_details(evaluation_run_id: str,
                            model_id: Optional[str] = None) -> pd.DataFrame:
    """予測明細に実績を突き合わせて返す（実績が無い行は actual_label が NaN）。

    この明細から指標も売買判断も再計算できる（spec §14 段階C完了条件）。
    `train_positive_rate`（学習側正例率）と `threshold`（採用閾値）を列に
    含めるため、`brier_vs_constant` / `log_loss_vs_constant` だけでなく
    **その run 自身が採用した売買判断**も、保存済みの明細だけから
    復元できる（外部レビューI-3。以前はこの2値がどこにも永続化されておらず、
    任意の閾値で引き直すことはできても「その run 自身の判断」は
    再現できなかった）。
    """
    from src.data.database import Prediction, PredictionOutcome, get_session
    from sqlalchemy import select as sa_select

    with get_session() as session:
        stmt = sa_select(Prediction).where(
            Prediction.evaluation_run_id == evaluation_run_id)
        if model_id is not None:
            stmt = stmt.where(Prediction.model_id == model_id)
        preds = list(session.scalars(stmt).all())
        # 結合キーは (label_contract_id, event_id)。event_id 単独で引くと、
        # 別コストで作られた実績を拾って過去runの指標が変わる（外部レビューR07）
        outcomes = {
            (o.label_contract_id, o.event_id): o
            for o in session.scalars(sa_select(PredictionOutcome)).all()
        }

    rows = []
    for p in preds:
        o = outcomes.get((p.label_contract_id, p.event_id))
        rows.append({
            "event_id": p.event_id,
            "label_contract_id": p.label_contract_id,
            "evaluation_run_id": p.evaluation_run_id,
            "model_id": p.model_id,
            "predicted_at": p.predicted_at,
            "raw_probability": p.raw_probability,
            "calibrated_probability": p.calibrated_probability,
            "fold_index": p.fold_index,
            "purpose": p.purpose,
            "train_positive_rate": p.train_positive_rate,
            "threshold": p.threshold,
            "actual_label": o.actual_label if o else np.nan,
            "net_return": o.net_return if o else np.nan,
        })
    return pd.DataFrame(rows)


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

    # ─── 保存のための契約（段階Eの model_store が使う）─────────────────
    # _LightGbmBase と全く同じ `self._constant` という状態を持つため、
    # 単一クラス縮退の検知手段も同じ形で公開する（外部レビューI-5）。
    # `booster` はLightGBM固有の概念なのでこちらには無い。

    @property
    def is_constant(self) -> bool:
        """単一クラスしか見なかったため定数を返すモデルか。"""
        return self._constant is not None

    @property
    def constant_probability(self) -> Optional[float]:
        """定数モデルのときの確率。二値モデルなら None。"""
        return self._constant


class _LightGbmBase:
    """LightGBM分類器の共通部。

    現行 ml_model._fit() は fold ごとの best_iteration_ の平均を最終モデルの
    木数にしているが、その fold は early stopping と指標報告を兼ねており
    報告値が楽観に寄る。本計画はその二重利用をやめる。

    **木数は候補ごとに固定し、early stopping は本段階では実装しない
    （意図的な設計差分・外部レビュー残件2026-09-12）。** spec §7 は
    「early stopping は内側foldで決める」としているが、本段階の `fit_inner()`
    は通常の `fit()` を呼ぶだけで、木数は 50（Small）/ 200（Current）に固定する。
    現時点で内側foldが担うのは**閾値選択と確率校正**であり、木数選択は
    含まれない。固定木数どうしの比較は公平（同じ内側集合・同じ検証集合）なので
    段階Cの目的である「正しく測る」は満たすが、
    **「木数を内側で選んだ」とは書かないこと**。

    木数選択を入れる場合は、`_LightGbmBase.fit()` に `eval_set` を渡す
    `fit_with_early_stopping()` を別メソッドとして足し、`fit_inner()` から
    内側検証で `best_iteration_` を決め、外側学習ではその木数を固定値として
    使う形にする。外側検証を eval_set に使うと元の問題に戻るので、
    その経路だけは絶対に作らないこと。
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

    # ─── 保存のための契約（段階Eの model_store が使う）─────────────────
    # このラッパーは `_model` と `predict_proba()` しか持たないため、
    # `getattr(model, "booster_", model).save_model(...)` のような
    # 書き方では保存できず AttributeError になる（外部レビューR02）。
    # 「二値がそろったモデル」と「単一クラス時の定数モデル」は保存形式が
    # 違うので、どちらであるかを型として公開する。

    @property
    def is_constant(self) -> bool:
        """単一クラスしか見なかったため定数を返すモデルか。"""
        return self._constant is not None

    @property
    def constant_probability(self) -> Optional[float]:
        """定数モデルのときの確率。二値モデルなら None。"""
        return self._constant

    @property
    def booster(self):
        """LightGBM の Booster。定数モデルなら None。

        `lgb.LGBMClassifier.booster_` を取り出したもの。model_store は
        これを `save_model()` でネイティブ形式へ書く。
        """
        return None if self._model is None else self._model.booster_


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
    try:
        inner_fold_list = validation.inner_folds(outer_train, n_splits=inner_splits)
    except ValueError:
        # 内側に分割するだけのセッション数が無い。evaluate_fold が退化foldに
        # 対して return None（skip）する規約を既に持っているのと同じ扱いで、
        # ここでは例外を外へ漏らさず「校正なし・閾値0.5」の既存の退化パスへ
        # 合流する（外側fold自体は続行させる。外部レビューI-1）。
        return Calibrator(kind="identity", model=None), 0.5

    raw_parts, y_parts, ret_parts = [], [], []
    for inner in inner_fold_list:
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
    calibrator_kind: str
    model_is_constant: bool


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
    # ConstantProbability.fit() と同じ重み付き平均に揃える。単純平均のままだと
    # 一意性重み（inputs.weights）を無視した値になり、ConstantProbability
    # 自身を評価しても vs定数指標が厳密に0にならない（レビュー指摘）。
    train_rate = float(np.average(y_train.astype(float), weights=inputs.weights)) \
        if len(y_train) and np.sum(inputs.weights) > 0 else 0.5

    predictions = pd.DataFrame({
        "event_id": val["event_id"].values,
        # 実績との結合キー。イベント表から持ち回り、ここで作り直さない
        "label_contract_id": val["label_contract_id"].values,
        "raw_probability": raw,
        "calibrated_probability": calibrated,
        "fold_index": fold.index,
        # 保存済み明細だけからbrier_vs_constant等・その run自身の売買判断を
        # 復元できるようにする（外部レビューI-3）
        "train_positive_rate": train_rate,
        "threshold": threshold,
    })
    metrics = compute_metrics(
        val["label"].astype(int), calibrated, baseline_rate=train_rate)

    return FoldResult(
        fold_index=fold.index, model_id=model_id,
        n_train=len(inputs.events), n_val=len(val),
        train_positive_rate=train_rate, threshold=threshold,
        metrics=metrics, predictions=predictions,
        calibrator_kind=calibrator.kind,
        # ConstantProbability/MajorityClass は元々定数を返す設計なので対象外。
        # is_constant を公開しないモデルは getattr の既定値で False 扱いにする
        model_is_constant=bool(getattr(model, "is_constant", False)),
    )


@dataclass(frozen=True)
class RunConfig:
    """評価実行の条件一式。**開始時に固定する。**

    終了後に `config.yaml` を読み直すと、実行中に設定が変わっていた場合に
    「実際に使った設定」とずれる。戦略節だけでなく、リスク・手数料・
    数量制限・分割設定まで含める（外部レビューの残件「評価実行の再現用記録」）。
    """
    dataset_id: Optional[str]
    label_contract_id: Optional[str]
    feature_version: str
    execution_model_version: str
    code_version: Optional[str]
    config_json: str
    config_hash: str


def _code_version() -> Optional[str]:
    """現在のコード版（git の短縮SHA）。取れなければ None。"""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except Exception:
        return None


def capture_run_config(events: pd.DataFrame, *, n_splits: int,
                       window_sessions: Optional[int],
                       feature_cols: Optional[list]) -> RunConfig:
    """実行条件を今この瞬間の値で固めて返す。"""
    from src.core import config as cfg

    payload = {
        "n_splits": n_splits,
        "window_sessions": window_sessions,
        "feature_cols": list(feature_cols) if feature_cols else list(FEATURE_COLS),
        "n_events": int(len(events)),
        # 実行条件は戦略節だけでは足りない。約定コスト・リスク・数量制限まで含める
        "strategy": cfg.get_section("strategy"),
        "trading": cfg.get_section("trading"),
        "backtest": cfg.get_section("backtest"),
    }
    config_json = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), default=str)
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:16]

    def _one(col):
        if col not in events.columns or len(events) == 0:
            return None
        values = set(events[col].dropna().astype(str))
        return values.pop() if len(values) == 1 else None

    return RunConfig(
        dataset_id=_one("dataset_id"),
        label_contract_id=_one("label_contract_id"),
        feature_version=_one("feature_version") or "",
        execution_model_version=_one("execution_model_version") or "",
        code_version=_code_version(),
        config_json=config_json,
        config_hash=config_hash,
    )


def save_evaluation_run(evaluation_run_id: str, run_config: RunConfig, *,
                        purpose: str, model_id: Optional[str],
                        n_folds: int, n_predictions: int,
                        degraded_reasons: Optional[list] = None) -> None:
    """評価実行そのものを保存する（同じIDなら上書き）。

    段階Eの `check_promotable()` はこの行を昇格可否の根拠にする。
    """
    from src.data.database import EvaluationRun, get_session
    from sqlalchemy import select as sa_select

    reasons = list(degraded_reasons or [])
    now = clock.now()
    with get_session() as session:
        row = session.scalar(sa_select(EvaluationRun).where(
            EvaluationRun.evaluation_run_id == evaluation_run_id))
        if row is None:
            row = EvaluationRun(evaluation_run_id=evaluation_run_id,
                                started_at=now)
            session.add(row)
        row.finished_at = now
        row.purpose = purpose
        row.model_id = model_id
        row.dataset_id = run_config.dataset_id
        row.label_contract_id = run_config.label_contract_id
        row.feature_version = run_config.feature_version
        row.execution_model_version = run_config.execution_model_version
        row.code_version = run_config.code_version
        row.config_hash = run_config.config_hash
        row.config_json = run_config.config_json
        row.n_folds = int(n_folds)
        row.n_predictions = int(n_predictions)
        row.degraded = 1 if reasons else 0
        row.degraded_reasons = json.dumps(reasons, ensure_ascii=False)
        session.commit()


def load_evaluation_run(evaluation_run_id: str):
    """保存済みの評価実行を返す（無ければ None）。"""
    from src.data.database import EvaluationRun, get_session
    from sqlalchemy import select as sa_select

    with get_session() as session:
        row = session.scalar(sa_select(EvaluationRun).where(
            EvaluationRun.evaluation_run_id == evaluation_run_id))
        if row is None:
            return None
        session.expunge(row)
        return row


def _new_evaluation_run_id() -> str:
    """既定の evaluation_run_id を生成する。

    秒精度のタイムスタンプだけでは、同一秒内の連続呼び出しで衝突しうる
    （Windowsの時計分解能は約15.6ms）。段階B2の Dataset.collection_id と
    全く同じ種類のバグで、そのときは100%再現した実績がある。
    uuid4 の先頭8桁を付与して構造的に一意性を確保する。
    """
    return f"{clock.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"


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
        evaluation_run_id = _new_evaluation_run_id()

    # 実行条件は**開始時に固定する**。終了後に読み直すと、実行中に設定が
    # 変わっていた場合に「実際に使った設定」とずれる。
    run_config = capture_run_config(
        events, n_splits=n_splits, window_sessions=window_sessions,
        feature_cols=feature_cols)
    degraded_reasons: list = []

    folds = validation.calendar_folds(events, n_splits=n_splits)
    results = []
    for model_id, make_model in factories.items():
        model_predictions = []
        for fold in folds:
            res = evaluate_fold(
                events, fold, model_id, make_model,
                window_sessions=window_sessions, feature_cols=feature_cols)
            if res is None:
                # (a) 退化fold（学習/検証データが空）はskipする。evaluate_fold
                # 自身の規約（return None）に合わせ、理由をここで記録する
                # （外部レビューI-4）。
                degraded_reasons.append(
                    f"fold {fold.index} model={model_id}: "
                    "学習/検証データが空のためskip")
                continue
            if res.calibrator_kind == "identity":
                # (b) 内側foldの校正がidentity（未校正）に落ちた。データ不足や
                # 片側クラスなど、fit_calibrator / fit_inner の退化パスに
                # 合流した結果である（外部レビューI-4）。
                degraded_reasons.append(
                    f"fold {res.fold_index} model={model_id}: "
                    "内側foldの校正がidentity(未校正)に縮退")
            if res.model_is_constant:
                # (c) 学習側が単一クラスでモデルが定数に縮退した
                # （外部レビューI-4・I-5）。
                degraded_reasons.append(
                    f"fold {res.fold_index} model={model_id}: モデルが定数に縮退")
            if res.metrics.get("roc_auc") is None:
                # (d) 検証側が片側クラスで roc_auc が未定義になった
                # （外部レビューI-4）。
                degraded_reasons.append(
                    f"fold {res.fold_index} model={model_id}: "
                    "roc_aucが未定義(検証側が片側クラス)")
            results.append(res)
            if persist:
                model_predictions.append(res.predictions)
        if persist and model_predictions:
            # モデル単位で一括保存する。fold単位で都度 save_predictions() を
            # 呼ぶと、その関数の「同一 (run_id, model_id) は置換する」という
            # 冪等化（外部レビューI-2）が直前foldの分まで消してしまう。
            save_predictions(
                pd.concat(model_predictions, ignore_index=True),
                evaluation_run_id, model_id)

    if persist:
        save_outcomes(events)
        save_evaluation_run(
            evaluation_run_id, run_config,
            purpose=PURPOSE_VALIDATION,
            model_id=(list(factories)[0] if len(factories) == 1 else None),
            n_folds=len(folds),
            n_predictions=sum(len(r.predictions) for r in results),
            degraded_reasons=degraded_reasons)

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
        "degraded_reasons": degraded_reasons,
    }


def _single_column_value(df: pd.DataFrame, col: str) -> float:
    """列の値がこの範囲内で単一値であることを確認してから返す。

    `capture_run_config._one()` と同じ考え方（外部レビューI-3）。単一で
    なければ、その範囲を跨いだ baseline を暗黙に選ぶことになり誤りうる
    ため ValueError にする。
    """
    if col not in df.columns or len(df) == 0:
        raise ValueError(
            f"baseline_rate を省略するには、明細に{col}列が必要です")
    values = set(df[col].dropna().tolist())
    if len(values) != 1:
        raise ValueError(
            f"baseline_rate を省略するには{col}がこの範囲内で単一値である"
            f"必要があります（fold等で絞り込んでください）: {sorted(values)}")
    return float(values.pop())


def recompute_metrics(details: pd.DataFrame, *, baseline_rate: Optional[float] = None,
                      model_id: Optional[str] = None) -> dict:
    """保存した予測明細だけから指標を計算し直す。

    実績が未確定の行（shadow等）は除外する。集計済みの数値しか無い状態では
    「その数字が何を意味するか」を後から検証できないため、明細から同じ指標を
    再現できることを保証する（spec §14 段階C完了条件）。

    `baseline_rate` を省略すると、`details` の `train_positive_rate` 列
    （外部レビューI-3で永続化）から導出する。その列がこの呼び出しの対象
    範囲内（`model_id` で絞り込んだ後）で単一値であることを要求し、
    単一でなければ ValueError にする（複数foldをまたいだ明細をそのまま
    渡すと、どのfoldのbaselineを使うべきか一意に決まらないため）。
    """
    sub = details if model_id is None else details[details["model_id"] == model_id]
    sub = sub[sub["actual_label"].notna()]
    if baseline_rate is None:
        baseline_rate = _single_column_value(sub, "train_positive_rate")
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

    **内側の検証期間は全候補で共通にする。** 候補窓は各内側foldの**学習側にだけ**
    適用し、検証側には触らない。窓で外側集合を先に切ってから内側foldを作り直すと、
    短い窓と拡大窓で検証日も件数も変わる。比較値がコスト控除後の収益の**総和**で
    ある以上、件数が多い候補がそれだけで有利になり、「学習窓の効果」と
    「評価期間・件数の差」が混ざる（外部レビューR19）。

    比較値は総和ではなく**1イベントあたりの平均**にする。共通の検証集合を使えば
    件数は揃うが、採用閾値を超えた件数は候補ごとに変わるため、総和のままだと
    「多く拾う窓」が有利に出る。

    候補が空なら None（拡大窓）を返す。
    """
    if not candidates:
        return None

    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)

    # 内側foldは「窓を適用する前の外側学習集合」から一度だけ作る。
    # これで検証側の event_id が全候補で同一になる。
    base = validation.training_inputs(events, fold, feature_cols=cols)
    if len(base.events) == 0:
        return None
    inner_folds = list(validation.inner_folds(base.events, n_splits=inner_splits))
    if not inner_folds:
        return None

    best_window, best_score = None, -np.inf
    for window in candidates:
        total, n_taken, scored = 0.0, 0, False
        for inner in inner_folds:
            # 窓は学習側にだけ掛ける
            inner_inputs = validation.training_inputs(
                base.events, inner, window_sessions=window, feature_cols=cols)
            # 検証側は窓に依存しない（window_sessions を渡さない）
            _, inner_val = validation.split_events(base.events, inner)
            if len(inner_inputs.events) == 0 or len(inner_val) == 0:
                continue
            model = make_model()
            model.fit(inner_inputs.events[cols].astype("float64"),
                      inner_inputs.events["label"].astype(int),
                      inner_inputs.weights)
            p = model.predict_proba(inner_val[cols].astype("float64"))
            ret = inner_val["net_return"].astype(float).values
            threshold = select_threshold(p, ret)
            taken = ret[p >= threshold]
            total += float(taken.sum())
            n_taken += int(len(taken))
            scored = True

        if not scored:
            continue
        # 1件も採らない窓は「収益0」ではなく比較対象外にする。
        # 総和0が負の窓に勝ってしまうのを避ける。
        if n_taken == 0:
            continue
        score = total / n_taken
        if score > best_score:
            best_score, best_window = score, window
    return best_window


def inner_validation_event_ids(events: pd.DataFrame, fold, *,
                               feature_cols: Optional[list] = None,
                               inner_splits: int = 3) -> list:
    """select_training_window が使う内側検証集合の event_id を返す（検証用）。

    「全候補窓で検証集合が同一である」ことをテストから確かめるために切り出す。
    """
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    base = validation.training_inputs(events, fold, feature_cols=cols)
    out = []
    for inner in validation.inner_folds(base.events, n_splits=inner_splits):
        _, inner_val = validation.split_events(base.events, inner)
        out.extend(list(inner_val["event_id"]))
    return out
