"""
機械学習モデル（LightGBM）の学習・推論

モデルは表形式の金融データで実績のあるLightGBMを採用（2025-2026の各種比較研究でも
日足スイングのような小規模サンプルでは深層学習より優位とされる）。
ラベリングはトリプルバリア法（labeling.py）を用い、ラベル期間の重なりに対する
サンプル一意性重みを付与してルックアヘッド・過学習を抑制する。
"""
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from src.strategy.indicators import FEATURE_COLS, build_features
from src.strategy.labeling import build_training_set

MODEL_PATH = Path("models/lgb_model.pkl")


def train(df: pd.DataFrame, trigger: Optional[str] = None,
          save: bool = True) -> lgb.LGBMClassifier:
    """単一銘柄のOHLCVからトリプルバリアラベル＋サンプル重みでLightGBMを学習し保存する。

    trigger が指定された場合（"weekly_schedule" / "manual"）、
    CV精度・特徴量重要度をDBに保存する。
    バックテスト内での学習時は trigger=None で呼び出してDBに記録しない。

    注意: df は単一銘柄の時系列であること。複数銘柄を学習する場合は
    train_multi() を使う（build_features の rolling/ewm や トリプルバリア法の
    「N日後」判定は単一時系列前提のため、複数銘柄を結合したdfを渡すと
    銘柄境界をまたいで指標・ラベルが破壊される）。
    """
    X, y, weights = build_training_set(df)
    return _fit(X, y, weights, trigger=trigger, save=save)


def train_multi(dfs: list[pd.DataFrame], trigger: Optional[str] = None,
                 save: bool = True) -> lgb.LGBMClassifier:
    """複数銘柄のOHLCVから学習する。

    各dfごとに個別に build_training_set() で特徴量・ラベル・サンプル重みを
    作成してから連結する（生のOHLCVをconcatしてから処理すると、移動平均/RSI/
    トリプルバリア法が銘柄境界をまたいで壊れるため、これは行わない）。
    """
    X_parts, y_parts, w_parts = [], [], []
    for df in dfs:
        try:
            X, y, w = build_training_set(df)
        except ValueError as e:
            logger.warning(f"学習データ生成をスキップ: {e}")
            continue
        if len(X) == 0:
            continue
        X_parts.append(X)
        y_parts.append(y)
        w_parts.append(w)

    if not X_parts:
        raise ValueError("学習データが不足しています: 0件")

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    weights = np.concatenate(w_parts)
    return _fit(X, y, weights, trigger=trigger, save=save)


def _fit(X: pd.DataFrame, y: pd.Series, weights: np.ndarray,
         trigger: Optional[str] = None, save: bool = True) -> lgb.LGBMClassifier:
    """特徴量・ラベル・サンプル重みからLightGBMを学習し保存する（train/train_multi共通部）。

    TimeSeriesSplitによるCVは、train_multi() 経由（複数銘柄連結）の場合は
    「銘柄ごとの時系列順序」は保たれるが「全体としての時系列順序」は厳密ではない
    （銘柄Aの全期間→銘柄Bの全期間の順に連結されるため）。CV精度はモデル選択の
    近似指標として扱い、厳密な銘柄別時系列CVは将来の改善課題とする。
    """
    if len(X) < 100:
        raise ValueError(f"学習データが不足しています: {len(X)}件")

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    best_n_list = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_tr = weights[train_idx]
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                num_leaves=31, random_state=42, verbose=-1)
        m.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(20, verbose=False),
                         lgb.log_evaluation(period=-1)])
        preds = m.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
        best_n_list.append(m.best_iteration_ or 200)

    best_n_estimators = int(np.mean(best_n_list))
    cv_mean = float(np.mean(scores))
    cv_std = float(np.std(scores))

    logger.info(f"MLモデルCV精度: {cv_mean:.3f} (+/-{cv_std:.3f})")

    # CV後に全データで最終モデルを学習（サンプル重み込み）
    model = lgb.LGBMClassifier(n_estimators=best_n_estimators, learning_rate=0.05,
                                num_leaves=31, random_state=42, verbose=-1)
    model.fit(X, y, sample_weight=weights)
    logger.info("MLモデル学習完了（全データ・トリプルバリア法）")

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

    if trigger is not None:
        _save_metrics(
            cv_mean=cv_mean,
            cv_std=cv_std,
            n_samples=len(X),
            n_estimators=best_n_estimators,
            feature_importances=dict(zip(FEATURE_COLS, model.feature_importances_.tolist())),
            trigger=trigger,
        )

    return model


def _save_metrics(
    cv_mean: float, cv_std: float, n_samples: int,
    n_estimators: int, feature_importances: dict, trigger: str,
) -> None:
    from src.data.database import ModelMetrics, get_session
    with get_session() as session:
        session.add(ModelMetrics(
            trained_at=datetime.now(),
            cv_mean_accuracy=round(cv_mean, 4),
            cv_std_accuracy=round(cv_std, 4),
            n_samples=n_samples,
            n_estimators=n_estimators,
            feature_importances_json=json.dumps(feature_importances),
            trigger=trigger,
        ))
        session.commit()
    logger.info(f"MLモデルメトリクス保存完了 (trigger={trigger})")


def load() -> Optional[lgb.LGBMClassifier]:
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_proba(model: lgb.LGBMClassifier, df: pd.DataFrame) -> float:
    """最新の特徴量で上昇確率を返す（0.0〜1.0）。
    df に既に特徴量列がある場合は再計算しない。
    """
    if FEATURE_COLS[0] not in df.columns:
        df = build_features(df)
    if df.empty or len(df) < 2:
        return 0.5
    latest = df[FEATURE_COLS].iloc[[-1]]
    proba = model.predict_proba(latest)[0][1]
    return float(proba)
