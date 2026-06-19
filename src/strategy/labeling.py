"""
トリプルバリア法によるラベリング（López de Prado, Advances in Financial ML）

素朴な「N日後に+X%」ラベルの問題点を解消する、プロのクオンツ標準手法:
  1. 利益確定バリア（上） … +pt_mult × ボラティリティ
  2. 損切りバリア（下）   … -sl_mult × ボラティリティ
  3. 時間バリア（垂直）   … 最大保有日数

「今買ったら、損切りに当たる前に利益目標に到達するか？」という
実際の売買判断そのものを学習対象にする。さらに、ラベル期間の重なり
（label concurrency）がIID仮定を破る問題に対し、サンプルの一意性に
基づく重み（sample uniqueness weight）を付与して過学習を抑える。
"""
from typing import Tuple

import numpy as np
import pandas as pd

from src.core import config as cfg
from src.strategy.indicators import FEATURE_COLS, build_features


def get_daily_volatility(close: pd.Series, span: int = 20) -> pd.Series:
    """日次リターンの指数加重標準偏差（動的ボラティリティ推定）"""
    returns = close.pct_change()
    return returns.ewm(span=span).std()


def triple_barrier_labels(
    df: pd.DataFrame,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    vol_span: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """各行にトリプルバリアラベルを付与する。

    戻り値:
        labels: 各行のラベル（1=利益目標に先に到達, 0=損切り/時間切れ, NaN=評価不能）
        t_ends: 各イベントが終了した行の位置インデックス（-1=評価不能）
    """
    close = df["close"].reset_index(drop=True)
    vol = get_daily_volatility(close, vol_span)
    n = len(close)
    labels = np.full(n, np.nan)
    t_ends = np.full(n, -1, dtype=int)

    for i in range(n):
        v = vol.iloc[i]
        if np.isnan(v) or v <= 0:
            continue
        entry = close.iloc[i]
        upper = entry * (1 + pt_mult * v)
        lower = entry * (1 - sl_mult * v)
        end = min(i + max_holding, n - 1)
        if end <= i:
            continue

        label = None
        t_end = end
        for j in range(i + 1, end + 1):
            price = close.iloc[j]
            if price >= upper:
                label, t_end = 1, j
                break
            if price <= lower:
                label, t_end = 0, j
                break
        if label is None:
            # 時間バリア到達: 最終リターンの符号でラベル付け
            label = 1 if close.iloc[end] > entry else 0
            t_end = end

        labels[i] = label
        t_ends[i] = t_end

    return labels, t_ends


def get_sample_weights(t_ends: np.ndarray) -> np.ndarray:
    """ラベル期間の重なり（concurrency）に基づくサンプル一意性重みを計算する。

    同時に多数のイベントが進行している期間のサンプルは情報の重複が大きいため
    重みを下げる。López de Prado の average uniqueness に相当。
    """
    n = len(t_ends)
    concurrency = np.zeros(n)
    for i in range(n):
        if t_ends[i] < 0:
            continue
        concurrency[i:t_ends[i] + 1] += 1

    weights = np.zeros(n)
    for i in range(n):
        if t_ends[i] < 0:
            continue
        span = concurrency[i:t_ends[i] + 1]
        span = span[span > 0]
        if len(span) > 0:
            weights[i] = float(np.mean(1.0 / span))
    return weights


def _assert_single_symbol_timeseries(df: pd.DataFrame) -> None:
    """df が単一銘柄の時系列（日付インデックスが重複なく昇順）であることを検査する。

    本モジュールの triple_barrier_labels()/get_daily_volatility() は行番号を
    「N日後」として扱うため、複数銘柄のOHLCVを単純に連結したDataFrame（日付の
    重複・逆順が生じる）を渡すと、移動平均やラベルが銘柄境界をまたいで破壊される。
    複数銘柄を学習する場合は、銘柄ごとに build_training_set() を呼んでから
    結果（X, y, weights）を連結すること（ml_model.train_multi() を参照）。
    """
    if df.index.has_duplicates:
        raise ValueError(
            "build_training_set: 日付インデックスに重複があります。"
            "複数銘柄のOHLCVを連結してから渡していませんか？"
            "銘柄ごとに個別に呼び出してください。"
        )
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "build_training_set: 日付インデックスが昇順ではありません。"
            "複数銘柄のOHLCVを連結してから渡していませんか？"
            "銘柄ごとに個別に呼び出してください。"
        )


def build_training_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """特徴量・トリプルバリアラベル・サンプル重みを生成する。

    df は単一銘柄の時系列（日付インデックス・重複なし・昇順）であること。
    複数銘柄を学習する場合は ml_model.train_multi() を使う。

    戻り値: (X, y, sample_weights)
    """
    _assert_single_symbol_timeseries(df)
    conf = cfg.get_section("strategy")
    pt_mult = conf.get("tb_profit_mult", 2.0)
    sl_mult = conf.get("tb_stop_mult", 2.0)
    max_holding = conf.get("tb_max_holding", 10)
    vol_span = conf.get("tb_vol_span", 20)

    feat = build_features(df).reset_index(drop=True)
    labels, t_ends = triple_barrier_labels(feat, pt_mult, sl_mult, max_holding, vol_span)
    weights = get_sample_weights(t_ends)

    mask = ~np.isnan(labels)
    X = feat.loc[mask, FEATURE_COLS].reset_index(drop=True)
    y = pd.Series(labels[mask], name="label").astype(int).reset_index(drop=True)
    w = weights[mask]
    return X, y, w
