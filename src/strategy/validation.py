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
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.dataset import STATUS_RESOLVED, uniqueness_weights
from src.strategy.indicators import FEATURE_COLS


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


def _resolved(events: pd.DataFrame) -> pd.DataFrame:
    """ラベルが確定したイベントだけを残す。

    未成熟・未約定・欠損はラベルが無く指標も計算できないため、学習にも
    検証にも使わない（損失0として扱わないのと同じ理由）。
    """
    return events[events["status"] == STATUS_RESOLVED]


def split_events(events: pd.DataFrame, fold: Fold, *,
                 embargo_sessions: int = 0) -> tuple:
    """fold を適用して (学習イベント, 検証イベント) を返す。

    **purge**: 学習側から `label_end_at > fold.train_end` のイベントを除外する。

    `train_end` は学習締切＝この分割で学習を実行する時点である。判断日だけを
    締切で切ると、「判断は締切前だがラベルの確定は締切後」というイベントが
    学習に入る。それは学習実行時点では観測できない情報であり、
    walk-forward が答えようとしている「その時点で何を知り得たか」を壊す。

    `train_end < val_start` を Fold が保証しているので、この条件は
    「ラベルが検証期間へ食い込む学習イベントを落とす」という従来の purge
    （spec §7 の `label_end_at >= val_start` を除外）を必ず含む。厳しいほうを採る。

    **embargo**: 既定で無効。有効にすると `val_start` の**直前** embargo_sessions
    セッションぶんの判断日を学習側から外す。purge を通した後でも、検証開始の
    直前にある学習イベントは系列相関で検証期間の結果と相関するため、
    境界に空白セッションを置きたい場合に使う。

    片方向walk-forwardでは `val_end` より後の判断日が学習側に入ることは
    構造的に無いので、「検証期間の後ろを外す」向きの embargo は実装しない
    （到達不能なコードになる）。両側を学習に使う分割を将来採用するなら、
    purge の契約から分けて設計し直すこと。
    """
    resolved = _resolved(events)

    val = resolved[
        (resolved["decision_at"] >= fold.val_start)
        & (resolved["decision_at"] <= fold.val_end)
    ]

    train = resolved[
        (resolved["decision_at"] >= fold.train_start)
        & (resolved["decision_at"] <= fold.train_end)
    ]
    # purge = 学習締切での実現可能性。
    # decision_at だけを train_end で切ると、判断は締切前でもラベルが締切後に
    # 確定するイベントが学習に入る。例: train_end=1/5, val_start=1/12 のとき、
    # 1/5判断・1/8確定のラベルは「1/5時点では誰も知り得ない」のに通ってしまう
    # （外部レビューR06）。学習締切とは学習を実行する時点のことなので、
    # ラベル確定日も同じ締切で切る。
    #
    # train_end < val_start が Fold で保証されているため、この条件は
    # 従来の purge（label_end_at < val_start）を必ず含む。より厳しいほうだけ残す。
    train = train[train["label_end_at"] <= fold.train_end]

    if embargo_sessions > 0:
        # 検証開始の直前セッションを学習側から落とす（前方ギャップ）。
        # 基準は purge 適用後の学習集合自身のセッション列にする。
        # 全解決済みイベントの全期間を基準にすると、embargo_sessions が
        # 保有期間（holding）以下のとき、対象セッションは既にpurgeで
        # 消えており無言のno-opになる（最終ブランチレビュー指摘）。
        train_sessions = sessions_of(train)
        embargoed = set(train_sessions[-embargo_sessions:])
        if embargoed:
            train = train[~train["decision_at"].isin(embargoed)]

    return train.reset_index(drop=True), val.reset_index(drop=True)


def inner_folds(train_events: pd.DataFrame, n_splits: int = 3) -> list:
    """外側foldの学習側をさらに分割した内側foldを作る。

    **外側foldは最終評価専用で一切触らない。** early stopping・閾値選択・
    確率校正・学習窓選択・戦略選択はすべてこの内側foldで行う。
    現行は early stopping に使った検証データでそのままCV指標を出しており、
    報告値が楽観に寄っている（ml_model.py:147-160）。

    引数は split_events() が返した学習イベントであること。外側の検証期間は
    そこに含まれていないため、内側foldがそれに触れることは構造的にない。
    """
    return calendar_folds(train_events, n_splits=n_splits)


def training_weights(train_events: pd.DataFrame) -> np.ndarray:
    """学習イベント集合に対して一意性重みを計算し直す。

    **purge後の学習集合に対して呼ぶこと。** イベント表全体で一度計算した重みを
    各foldへ流すと、検証側イベントの終了時点が学習側の重みへ影響する（spec §7）。
    段階B後半で sample_weight を列として保存しなかったのはこのためで、
    ここが正しい呼び出し口になる。
    """
    return uniqueness_weights(train_events)


@dataclass(frozen=True)
class Preprocessor:
    """学習側だけで求めた前処理の統計量。

    検証側・本番側はこの統計量で変換するだけで、自分では fit し直さない。
    分割器だけを直しても全データで fit すればリークは残る（spec §7 経路3）。
    """
    means: pd.Series
    stds: pd.Series
    positive_rate: float
    n_fitted: int


def fit_preprocessor(train_events: pd.DataFrame,
                     feature_cols: Optional[list] = None) -> Preprocessor:
    """**学習イベントだけ**から前処理の統計量を求める。

    標準化の平均・標準偏差、欠損補完に使う平均、クラス比率をここで固定する。
    検証側の値は一切見ない。閾値選択と確率校正も段階C後半で同じ規約に従う。

    学習集合が空の場合、meansをNaN・positive_rateもNaNで返す
    （n_fitted=0と併せて判定できるが、値そのものが「fit不能」と分かるように
    NaNにする。0.0/1.0を返すと統計的に正常なfitと見分けがつかず、
    0行で学習したことに下流が気づけない — 最終ブランチレビュー指摘）。
    """
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    X = train_events.reindex(columns=cols).astype("float64")
    if len(train_events) == 0:
        nan_series = pd.Series(float("nan"), index=cols)
        return Preprocessor(
            means=nan_series, stds=nan_series,
            positive_rate=float("nan"), n_fitted=0,
        )
    means = X.mean()
    stds = X.std(ddof=0)
    # 分散0の列はゼロ除算になるため1として扱う（変換後は全て0になる）
    stds = stds.mask((stds == 0) | stds.isna(), 1.0)
    labels = pd.to_numeric(train_events.get("label"), errors="coerce")
    positive_rate = float(labels.mean()) if labels is not None and len(labels) else 0.0
    return Preprocessor(
        means=means.fillna(0.0),
        stds=stds,
        positive_rate=positive_rate,
        n_fitted=len(train_events),
    )


def apply_preprocessor(pre: Preprocessor, events: pd.DataFrame,
                       feature_cols: Optional[list] = None) -> pd.DataFrame:
    """学習側の統計量で変換する（欠損は学習側の平均で埋める）。"""
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    X = events.reindex(columns=cols).astype("float64")
    X = X.fillna(pre.means)
    return (X - pre.means) / pre.stds


def apply_training_window(train_events: pd.DataFrame,
                          window_sessions: Optional[int] = None) -> pd.DataFrame:
    """学習窓を適用する。None なら拡大窓、整数なら直近 N セッションの移動窓。

    現行の週次学習は load_ohlcv(sym) の既定値そのままで直近500行だけを読んでおり、
    **意図した選択ではない**（レビューF07）。窓を明示パラメータにして、
    拡大窓と移動窓を内側foldで比較して選べるようにする。

    窓は**セッション（暦日）で切る**。行数で切ると、1日あたりの候補数が
    銘柄数や相場つきで変わったときに実期間が動いてしまう。
    """
    if window_sessions is None:
        return train_events.reset_index(drop=True)
    if window_sessions <= 0:
        raise ValueError(f"window_sessions は正の整数: {window_sessions}")

    sessions = sessions_of(train_events)
    if len(sessions) <= window_sessions:
        return train_events.reset_index(drop=True)

    cutoff = sessions[-window_sessions]
    return train_events[
        train_events["decision_at"] >= cutoff
    ].reset_index(drop=True)


@dataclass(frozen=True)
class TrainingInputs:
    """このfoldで学習に使ってよい入力一式。

    **学習に入る入力はすべてこの関数を通す。** そうしておけば「学習側へ
    未来が入っていないか」の検査が1箇所で済む。
    """
    events: pd.DataFrame
    weights: np.ndarray
    preprocessor: Preprocessor
    fold: Fold


def training_inputs(events: pd.DataFrame, fold: Fold, *,
                    window_sessions: Optional[int] = None,
                    embargo_sessions: int = 0,
                    feature_cols: Optional[list] = None) -> TrainingInputs:
    """foldの学習締切で固定された入力一式を返す。

    適用の順序に意味がある。purge → 学習窓 → 重み → 前処理 の順で、
    **絞り込みが終わった集合に対して重みと統計量を求める**。順序を逆にすると、
    後から落とすイベントが重みや平均に混ざる。

    spec §14 の段階C完了条件「そのfoldの学習締切で固定されたモデルについて、
    外側foldの値を変えてもモデル・前処理・重み・閾値が変わらない」は、
    この関数の出力が変わらないことで検証できる。
    """
    train, _ = split_events(events, fold, embargo_sessions=embargo_sessions)
    train = apply_training_window(train, window_sessions)
    weights = training_weights(train)
    pre = fit_preprocessor(train, feature_cols=feature_cols)
    return TrainingInputs(events=train, weights=weights, preprocessor=pre, fold=fold)
