"""分割と漏れの遮断 — どのイベントを学習に使ってよいかを決める。

現行の ml_model._fit() は連結した行に TimeSeriesSplit を当てており、
銘柄Aの2026年が学習側・銘柄Bの2024年が検証側に入る分割が成立する
（レビューF02）。本モジュールは fold 境界を**全銘柄共通のカレンダー日付**で
切り、学習側へ未来が入る経路を purge 以外も含めて塞ぐ。

**モデルは扱わない。** 分割・除外・重み・前処理の統計量までを担当し、
学習・推論・指標計算は段階C後半に置く。lightgbm / sklearn を import しない。
"""
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """1つの分割。境界は日付であり行番号ではない。

    train_end は**学習締切**＝この分割で学習を実行する時点である。
    「この日までに判断されたイベント」ではなく
    **「この日までに判断され、かつこの日までにラベルが確定したイベント」**だけが
    学習候補になる（split_events の docstring を参照）。

    val_start / val_end はこのfoldの検証期間。片方向walk-forward専用のため
    常に train_start <= train_end < val_start <= val_end が成り立つ。
    この不変条件は __post_init__ で強制する。成り立たないFoldは
    purge・embargoの契約が定義できないので、黙って通さず例外にする。
    """
    index: int
    train_start: date
    train_end: date
    val_start: date
    val_end: date

    def __post_init__(self) -> None:
        if not (self.train_start <= self.train_end):
            raise ValueError(
                f"train_start <= train_end でなければならない: "
                f"{self.train_start} > {self.train_end}")
        if not (self.val_start <= self.val_end):
            raise ValueError(
                f"val_start <= val_end でなければならない: "
                f"{self.val_start} > {self.val_end}")
        if not (self.train_end < self.val_start):
            # 学習期間が検証期間を跨ぐ・後ろまで伸びる分割は片方向walk-forwardでは
            # 作れない。ここを通すと「未来で学習して過去を検証する」分割が
            # 静かに成立してしまう（外部レビューR23）。
            raise ValueError(
                f"片方向walk-forwardでは train_end < val_start が必要: "
                f"train_end={self.train_end} val_start={self.val_start}")


def sessions_of(events: pd.DataFrame) -> list:
    """全銘柄共通の判断セッション（decision_at の重複なし昇順）。

    銘柄ごとに別々の境界を使うと、同じ暦日が fold の両側に現れうる。
    分割の基準は常にこの共通セッション列にする。
    """
    values = pd.to_datetime(events["decision_at"].dropna()).dt.date.unique()
    return sorted(values)


def calendar_folds(events: pd.DataFrame, n_splits: int = 5) -> list:
    """拡大窓の walk-forward 分割を日付で作る。

    セッション列を n_splits+1 個の連続した区画に分け、fold k は区画 0..k を
    学習、区画 k+1 を検証にする。銘柄数や1日あたりの候補数が変わっても
    境界の日付は変わらない（行番号に依存しないため）。
    """
    sessions = sessions_of(events)
    n_chunks = n_splits + 1
    if len(sessions) < n_chunks:
        raise ValueError(
            f"分割に足りるセッションがありません: {len(sessions)}日 < {n_chunks}区画"
        )

    bounds = np.linspace(0, len(sessions), n_chunks + 1).astype(int)
    folds = []
    for k in range(n_splits):
        train_lo, train_hi = bounds[0], bounds[k + 1]
        val_lo, val_hi = bounds[k + 1], bounds[k + 2]
        if val_hi <= val_lo:
            continue
        folds.append(Fold(
            index=k,
            train_start=sessions[train_lo],
            train_end=sessions[train_hi - 1],
            val_start=sessions[val_lo],
            val_end=sessions[val_hi - 1],
        ))
    return folds
