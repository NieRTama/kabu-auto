"""
機械学習モデル（LightGBM）の学習・推論
ルックアヘッドバイアスを防ぐため時系列分割を使用する。
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

MODEL_PATH = Path("models/lgb_model.pkl")


def train(df: pd.DataFrame) -> lgb.LGBMClassifier:
    """時系列分割でLightGBMを学習し保存する"""
    df = build_features(df)
    if len(df) < 100:
        raise ValueError(f"学習データが不足しています: {len(df)}件")

    X = df[FEATURE_COLS]
    y = df["label"]

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    model = None
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                num_leaves=31, random_state=42, verbose=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(20, verbose=False),
                         lgb.log_evaluation(period=-1)])
        preds = m.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
        model = m

    logger.info(f"MLモデル学習完了 CV精度: {np.mean(scores):.3f} (+/-{np.std(scores):.3f})")
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
    """最新の特徴量で上昇確率を返す（0.0〜1.0）"""
    df = build_features(df)
    if df.empty or len(df) < 2:
        return 0.5
    latest = df[FEATURE_COLS].iloc[[-1]]
    proba = model.predict_proba(latest)[0][1]
    return float(proba)
