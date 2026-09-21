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
import numpy as np

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
