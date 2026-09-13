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
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)

from src.core import clock
from src.strategy.dataset import STATUS_RESOLVED
from src.strategy.validation import Preprocessor, apply_preprocessor, fit_preprocessor

PURPOSE_VALIDATION = "validation"
PURPOSE_SHADOW = "shadow"


def save_predictions(predictions: pd.DataFrame, evaluation_run_id: str,
                     model_id: str, *, purpose: str = PURPOSE_VALIDATION) -> int:
    """予測明細を保存する。保存した件数を返す。

    predictions は event_id / label_contract_id / raw_probability /
    calibrated_probability / fold_index の列を持つこと。
    実績ラベルはここでは書かない。

    `label_contract_id` を必須にするのは、後で実績と結合するときの
    キーがこの組だからである（外部レビューR07）。欠けたまま保存すると
    結合先が定まらないので、既定値で埋めずに例外にする。
    """
    from src.data.database import Prediction, get_session

    if len(predictions) == 0:
        return 0
    if "label_contract_id" not in predictions.columns:
        raise ValueError(
            "predictions に label_contract_id 列がありません。"
            "実績との結合キーなので省略できません")
    now = clock.now()
    with get_session() as session:
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
