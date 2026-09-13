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

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression

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
