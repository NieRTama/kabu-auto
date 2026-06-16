"""
機械学習モデル（LightGBM）の学習・推論

モデルは表形式の金融データで実績のあるLightGBMを採用（2025-2026の各種比較研究でも
日足スイングのような小規模サンプルでは深層学習より優位とされる）。
ラベリングはトリプルバリア法（labeling.py）を用い、ラベル期間の重なりに対する
サンプル一意性重みを付与してルックアヘッド・過学習を抑制する。
"""
import pickle
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


def train(df: pd.DataFrame) -> lgb.LGBMClassifier:
    """トリプルバリアラベル＋サンプル重みでLightGBMを学習し保存する"""
    X, y, weights = build_training_set(df)
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

    logger.info(f"MLモデルCV精度: {np.mean(scores):.3f} (+/-{np.std(scores):.3f})")

    # CV後に全データで最終モデルを学習（サンプル重み込み）
    model = lgb.LGBMClassifier(n_estimators=best_n_estimators, learning_rate=0.05,
                                num_leaves=31, random_state=42, verbose=-1)
    model.fit(X, y, sample_weight=weights)
    logger.info("MLモデル学習完了（全データ・トリプルバリア法）")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model


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
