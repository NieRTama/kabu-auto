"""昇格済み候補モデルのshadow記録 — signal_scan から呼ばれる観察専用の配線。

**候補は発注に繋がらない。** 本モジュールは `shadow.compare()` /
`shadow.record_shadow()` を呼んで記録するだけで、戻り値で signal_scan の
判断を変えない。候補が未昇格（`model_store.load_current()` が None）なら
何もしない。

1回のスキャンにつき次の3段で使う:

  1. `prepare(current_model)`   … ループに入る前に**一度だけ**。候補モデルの
     読み込みはここだけで行う（銘柄ごとに読み直さない）。
  2. `collect(batch, symbol, df)` … 銘柄ごとに特徴量1行を溜める（DBへは書かない）。
  3. `flush(batch)`             … ループを抜けた後に**一度だけ**。まとめて記録する。

**`flush()` を銘柄ごとに呼んではいけない。** `shadow.record_shadow()` は
同一 `evaluation_run_id` の既存行を削除してから挿入する（run単位の置換・
src/strategy/shadow.py:96-103）ため、銘柄ごとに呼ぶと前の銘柄の記録が
毎回消え、最後の1銘柄しか残らない。
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from src.backtest import execution
from src.core import clock
from src.strategy import dataset as ds
from src.strategy import ml_model
from src.strategy import model_store as ms
from src.strategy import policy
from src.strategy.indicators import FEATURE_COLS

# 観察用の**仮の**基準点。チューニングされた閾値ではない。
# 候補の評価で fold 別に選ばれた閾値は 0.35/0.41/0.41/0.35/1.0 とばらつきが
# 大きく、単一の「正しい」値を選べる状況にない
# （docs/kabu-auto-ml-real-data-comparison_20260921.md）。二値分類の標準的な
# 基準点として 0.5 を置き、観察期間を通して同じ基準で記録を揃える。
SHADOW_THRESHOLD = 0.5

# 現行（legacy pickle）モデルのメタが読めないときのID
LEGACY_MODEL_ID_UNKNOWN = "legacy-unknown"


class _LegacyModelProbaAdapter:
    """shadow.compare() の current 引数用。判断ロジックは一切持たない。

    sklearn の `LGBMClassifier.predict(X)` はクラスラベル(0/1)を返すため、
    そのまま渡すと確率のつもりでラベルを記録してしまう（実機の実測値:
    predict -> [1] / predict_proba -> [[0.4211, 0.5789]]）。
    `predict_proba(X)[:, 1]` を `shadow.compare()` が要求する
    `predict(X) -> 正例確率の1次元配列` の形へ合わせるだけの薄い変換。
    """

    def __init__(self, model):
        self._model = model

    def predict(self, X):
        proba = np.asarray(self._model.predict_proba(X), dtype=float)
        if proba.ndim != 2 or proba.shape[1] < 2:
            # 単一クラスしか見なかったモデル等。既定値で埋めると「現行が
            # 何を予測したか」を捏造することになるので、その場で落とす
            # （呼び出し側が握り潰し、その日のshadow記録だけを見送る）
            raise ValueError(
                "現行モデルの predict_proba() が2クラスの確率を返しません: "
                f"shape={proba.shape}")
        return proba[:, 1]


@dataclass
class ShadowBatch:
    """1回の signal_scan 分のshadow記録。ループ中は溜めるだけで書かない。

    `event_ids` と `rows` は同じ順で1銘柄1要素ずつ増える。
    """
    evaluation_run_id: str
    candidate: object
    candidate_model_id: str
    current: Optional[object]
    current_model_id: Optional[str]
    label_contract_id: str
    threshold: float = SHADOW_THRESHOLD
    event_ids: list = field(default_factory=list)
    rows: list = field(default_factory=list)


def _legacy_model_id(model) -> Optional[str]:
    """現行モデルの記録用ID。未ロードなら None。

    legacy の pickle モデルには model_id が無い。`ml_model` が本体と対で書く
    サイドカー `models/lgb_model.meta.json` の sha256 先頭12桁で版を表す。
    週次再学習で中身が入れ替わるため、単に "legacy" と記録すると数ヶ月後に
    「どの現行と比べたのか」を復元できない。

    メタが読めなくても**例外にしない**。shadowの都合で本来の判断を止めない。
    """
    if model is None:
        return None
    try:
        meta_path = Path(ml_model.MODEL_PATH).with_suffix(".meta.json")
        digest = json.loads(meta_path.read_text(encoding="utf-8"))["sha256"]
        return f"legacy-{digest[:12]}"
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning(
            f"現行モデルのメタを読めません（IDは不明として記録します）: {e}")
        return LEGACY_MODEL_ID_UNKNOWN


def prepare(current_model, *, base_dir: str = "models") -> Optional[ShadowBatch]:
    """このスキャンのshadowバッチを作る。候補が未昇格なら None。

    **候補モデルの読み込みはここだけ。** 銘柄ごとに `load_current()` を
    呼び直すと、ディスクI/Oが無駄なだけでなく、スキャンの途中で昇格が
    起きた場合に同じスキャンの記録へ別のモデルが混ざる。
    """
    loaded = ms.load_current(base_dir=base_dir)
    if loaded is None:
        return None            # 未昇格は正常。静かにスキップする
    candidate, meta = loaded

    if list(meta.feature_cols) != list(FEATURE_COLS):
        # 学習時と現在で列が違うモデルに黙って推論させると、誤った数字が
        # 「正常な観察結果」として残る（Knowledge.md §10 desyncガード）
        logger.error(
            f"shadow記録を行いません: 候補 {meta.model_id} の特徴量定義が"
            f"現行と一致しません（候補={list(meta.feature_cols)} / "
            f"現行={list(FEATURE_COLS)}）")
        return None

    label_contract_id = ds.make_label_contract_id(
        policy.config_from_settings(), execution.config_from_settings())
    if meta.label_contract_id and meta.label_contract_id != label_contract_id:
        # 実績（PredictionOutcome）を後から結合するときのキーが変わる。
        # 記録は続けるが、あとで気づけるようにログへ残す
        logger.warning(
            f"shadow記録のラベル契約が候補の学習時と異なります: "
            f"学習時={meta.label_contract_id} / 現在={label_contract_id}"
            "（退出ポリシーかコストの設定が変わっています。記録は続けます）")

    return ShadowBatch(
        evaluation_run_id=f"shadow-{clock.today().isoformat()}",
        candidate=candidate,
        candidate_model_id=meta.model_id,
        current=(_LegacyModelProbaAdapter(current_model)
                 if current_model is not None else None),
        current_model_id=_legacy_model_id(current_model),
        label_contract_id=label_contract_id,
    )
