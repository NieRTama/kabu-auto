"""v2 の学習経路 — イベント表から候補モデルを作る。

legacy（`ml_model.train_multi`）は `labeling.build_training_set` と
`TimeSeriesSplit` を使う。v2 は `dataset.build_events_multi` と
`validation` を使う。**`ml_model.py` には触らない**。legacy をいつでも
選べることが段階投入の前提だから（設計書 §10）。

**学習成功は候補の生成であって運用モデルの差し替えではない。**
`model_store.train_as_candidate()` を通すので、学習が失敗しても現行は失われない。
昇格は `promotion.promote()` による明示的な操作だけで起きる。
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.core import clock
from src.strategy import dataset as ds
from src.strategy import model_store as ms
from src.strategy import validation
from src.strategy.evaluation import CurrentLightGBM
from src.strategy.indicators import FEATURE_COLS

ENGINE_VERSION = "v2"
MIN_RESOLVED_EVENTS = 200


@dataclass(frozen=True)
class V2TrainingResult:
    """v2 学習の結果。model_id が None なら候補は作られていない。"""
    model_id: Optional[str]
    dataset_id: str
    n_events: int
    n_resolved: int
    positive_rate: Optional[float]
    skipped_reason: Optional[str] = None


def _fit_candidate(train_events: pd.DataFrame, weights: np.ndarray):
    """最終候補モデルを学習する（テストで差し替えられるよう関数に切る）。"""
    model = CurrentLightGBM()
    model.fit(train_events[list(FEATURE_COLS)].astype("float64"),
              train_events["label"].astype(int), weights)
    return model


def _save_metrics(result: V2TrainingResult, window_sessions: Optional[int],
                   trigger: str) -> None:
    from src.data.database import ModelMetrics, get_session

    with get_session() as session:
        session.add(ModelMetrics(
            trained_at=clock.now(),
            n_samples=result.n_resolved,
            trigger=trigger,
            model_id=result.model_id,
            positive_rate=result.positive_rate,
            training_window_sessions=window_sessions,
            engine_version=ENGINE_VERSION,
        ))
        session.commit()


def train_v2(ohlcv_by_symbol: dict, *, policy_conf, costs,
             window_sessions: Optional[int] = None,
             trigger: str = "weekly_schedule",
             base_dir: str = "models") -> V2TrainingResult:
    """イベント表を作り、最新foldの学習入力から候補モデルを作る。

    **運用モデルを差し替えない。** 候補を保存して終わる。
    データ不足や学習失敗は例外にせず、理由を添えて返す（週次ジョブを
    落とさないため）。

    決着イベント数が `MIN_RESOLVED_EVENTS` に届かずスキップする場合も
    `ModelMetrics` に1行だけ記録する（段階F残課題6）。「今週の再学習が
    動いたが見送られた」という履歴を残すため。この行は
    `model_id=None`・`positive_rate=None`（学習していないため）・
    `n_samples=n_resolved`・`engine_version="v2"` になる。
    """
    events = ds.build_events_multi(ohlcv_by_symbol, policy_conf, costs)
    dataset_id = ds.compute_dataset_id(events)
    n_events = len(events)
    resolved = events[events["status"] == ds.STATUS_RESOLVED] if n_events else events
    n_resolved = len(resolved)

    if n_events:
        path = ds.save_events(events, dataset_id)
        ds.save_dataset_meta(events, dataset_id, path,
                             ds.input_ohlcv_hash(ohlcv_by_symbol))

    if n_resolved < MIN_RESOLVED_EVENTS:
        reason = f"決着したイベントが不足しています: {n_resolved}件 < {MIN_RESOLVED_EVENTS}件"
        logger.warning(f"v2学習をスキップ: {reason}")
        result = V2TrainingResult(
            model_id=None, dataset_id=dataset_id, n_events=n_events,
            n_resolved=n_resolved, positive_rate=None, skipped_reason=reason)
        _save_metrics(result, window_sessions, trigger)
        return result

    # 最新foldの学習締切までの入力だけを使う（未来を入れない）
    try:
        folds = validation.calendar_folds(events, n_splits=5)
    except ValueError as e:
        reason = f"分割できません: {e}"
        logger.warning(f"v2学習をスキップ: {reason}")
        return V2TrainingResult(
            model_id=None, dataset_id=dataset_id, n_events=n_events,
            n_resolved=n_resolved, positive_rate=None, skipped_reason=reason)

    inputs = validation.training_inputs(
        events, folds[-1], window_sessions=window_sessions,
        feature_cols=list(FEATURE_COLS))
    if len(inputs.events) == 0:
        reason = "最新foldの学習イベントが0件です"
        return V2TrainingResult(
            model_id=None, dataset_id=dataset_id, n_events=n_events,
            n_resolved=n_resolved, positive_rate=None, skipped_reason=reason)

    positive_rate = float(inputs.events["label"].astype(int).mean())
    model_id = f"v2-{clock.now():%Y%m%dT%H%M%S}-{dataset_id[:8]}"

    def _meta():
        return ms.ModelMeta(
            model_id=model_id, trained_at=clock.now(),
            training_window_sessions=window_sessions,
            symbols=sorted(ohlcv_by_symbol),
            label_definition="net_return>0",
            feature_cols=list(FEATURE_COLS),
            positive_rate=positive_rate,
            fold_results=[],
            dataset_id=dataset_id,
            # どう作られたラベルで学習したかを固定する。昇格検査が
            # 評価実行のラベル契約と突き合わせる（外部レビューR07/R13）
            label_contract_id=ds.make_label_contract_id(policy_conf, costs),
        )

    saved_id = ms.train_as_candidate(
        lambda: _fit_candidate(inputs.events, inputs.weights), _meta,
        base_dir=base_dir)

    result = V2TrainingResult(
        model_id=saved_id, dataset_id=dataset_id, n_events=n_events,
        n_resolved=n_resolved, positive_rate=positive_rate,
        skipped_reason=None if saved_id else "候補モデルの学習・保存に失敗しました")
    _save_metrics(result, window_sessions, trigger)
    if saved_id:
        logger.warning(
            f"v2候補モデルを作成: {saved_id}（学習{len(inputs.events)}件 / "
            f"正例率{positive_rate:.3f} / dataset={dataset_id}）"
        )
    return result
