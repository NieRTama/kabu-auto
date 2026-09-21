# 昇格済み候補モデルのshadow記録を実発注経路へ配線する 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 昇格済み候補モデル `v2-20260921T123704-0589b6e1` の判断を、`signal_scan()` の実発注経路と**同じ入力**に対して毎営業日16:20に計算し、`shadow_comparisons` テーブルへ観察記録として残す（実発注には一切影響しない）。

**Architecture:** `src/services/shadow_recording.py` を新設し、`prepare()`（スキャン開始前に候補を1回だけロード）→ `collect()`（銘柄ごとに特徴量1行を溜める）→ `flush()`（ループ後に1回だけ `shadow.compare()` + `shadow.record_shadow()`）の3段構成にする。`src/services/trading.py` への変更は「この3箇所を呼ぶ追加行」と「import 1行」だけで、既存の判断・発注ロジックは1行も書き換えない。呼び出しはすべて try/except で隔離し、shadow記録の失敗が買い/売り/様子見の判断や発注を妨げないようにする。

**Tech Stack:** Python 3.11 / pandas / numpy / LightGBM 4.1.0（`lgb.Booster`）/ SQLAlchemy（SQLite）/ loguru / pytest

**Spec:**
- `docs/kabu-auto-ml-real-data-comparison_20260921.md`（実データ5モデル比較。AUCがchanceと区別できないこと・fold別閾値が0.35〜1.0でばらつくことの根拠）
- `docs/superpowers/plans/2026-09-21-model-promotion-workflow.md`（train/evaluate/promote の実装計画。今回の候補はこの経路で昇格した）
- 計画作成ブリーフ `shadow-wiring-planning-brief.md`（スクラッチパッド。内容は本計画の「Global Constraints」「設計上の確定事項」へ転記済みなので、本計画だけ読めば実装できる）

---

## Global Constraints

- **`signal_scan()` の既存ロジック（`sig = gen_signal(sym, df, self.model)` 以降の分岐・発注判断）は1行も変更しない。** shadow記録は追加だけで行い、既存の戻り値・制御フローに影響を与えない。
- shadow記録処理で例外が発生しても**シグナルスキャン全体を落とさない**。1銘柄のshadow記録失敗が、その銘柄・他銘柄の本来の判断（買い/売り/様子見）や paper執行を妨げてはならない。try/except で囲み、失敗はログに残すだけにする。
- **候補モデルが無い（未昇格）ときは何もしない。** `model_store.load_current()` が `None` を返すケースはエラーにせず静かにスキップする。
- shadow記録はDBへの追加書き込みのみ（`predictions` / `shadow_comparisons` テーブル）。取引関連テーブル（`trades` / `positions` / `orders` / `order_intents`）・`config.yaml`・`models/lgb_model.pkl`（現行の運用モデル本体）・`models/current.json` には一切触れない。
- **候補モデルは `signal_scan()` の呼び出しごとに一度だけ読み込む**（銘柄ごとに `model_store.load_current()` を呼び直さない）。ディスクI/Oの無駄と、実行途中で昇格が起きた場合の一貫性の両面から。
- **`shadow.record_shadow()` は1スキャンにつき1回だけ呼ぶ。** 同一 `evaluation_run_id` の既存行を delete→insert で置換する仕様（`src/strategy/shadow.py:96-103`）のため、銘柄ごとに呼ぶと前の銘柄の記録が毎回消え、最後の1銘柄しか残らない。
- 新規フラグを `config.yaml` に足さない。「候補が未昇格なら何もしない」が唯一のOFF条件である。
- ファイルはUTF-8 BOM無し・LFで書く。`core.autocrlf=true` のため既存ファイルの作業ツリー上はCRLFだが、gitのobjectはLFに正規化される（Knowledge.md §1）。コミット前に `git diff --stat` を見て、意図しない全行差分が出ていないことを確認する。
- テストは `pytest tests/<file>.py -v` で実行する。**ネットワークへ出るテストを書かない。**
- 作業ブランチは `main` から切った `feature/shadow-wiring`。各タスクの終わりに1コミットする。
- コミットメッセージ末尾に `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` を付ける。

---

## 設計上の確定事項（実装者が迷わないための決定）

1. **shadow記録の `evaluation_run_id` は日次で新しく作る**: `f"shadow-{clock.today().isoformat()}"`（例 `shadow-2026-09-22`）。候補を評価したときの `evaluation_run_id`（`promote_workflow evaluate` が発行したもの）とは意味が違う（あちらは「評価」、こちらは「その日の並行記録バッチ」）。`record_shadow()` の「同一 run の既存行を削除してから挿入する」仕様と、1日1回 `signal_scan` が走る前提が噛み合う（同じ日に複数回走っても上書きで整合する）。
2. **shadow比較の対象は `signal_scan` がループする全銘柄**（rule閾値を超えてMLへ進んだ銘柄に限らない）。目的は「候補が現行とどれだけ違う判断をするか」を広く観察することなので絞り込まない。ただし `_is_fresh_for_new_candidate(sym)` で除外された銘柄・`len(df) < 30` で除外された銘柄は現行判断そのものがスキップされているので、shadowも同様にスキップする（既存の `continue` より後にコードを置けば自然にそうなる）。
3. **`threshold` は 0.5 固定**。候補モデルの評価fold別閾値は 0.35/0.41/0.41/0.35/1.0 とばらつきが大きく、単一の「正しい」値を選べる状況にない。0.5 は標準的な二値分類の基準点として採用し、コードのコメントで「チューニングされた閾値ではなく観察用の仮の基準」と明記する。
4. **候補が無いときは shadow記録処理全体をスキップする**（ループに入る前に判定し、無駄なdf処理をしない）。
5. shadow記録用の特徴量行は、`gen_signal()` が内部で使っているのと**同じ `df`**（`signal_scan` の599行目時点で既にロード済み）から作る。二重にOHLCVを取得しない。
6. **shadowが見る特徴量1行は、現行モデルが見ているのと同一の行にする。** `gen_signal` → `ml_model.predict_proba()` は `build_features(df)`（＝`_add_feature_columns(df).dropna(subset=FEATURE_COLS)`）の**最終行**を使う。したがって shadow側は `build_feature_frame(df)`（行を落とさず `feature_valid` 列で示す）を使い、`feature_valid == True` の行だけに絞った上での最終行を取る。この2つは同じ行になる（dropna と `feature_valid` マスクは同じ条件）。
7. **現行（legacy pickle）モデルの記録用IDは `f"legacy-{sha256[:12]}"`。** `models/lgb_model.pkl` は週次再学習（`ml_retrain`）で中身が入れ替わるため、単に `"legacy"` と記録すると数ヶ月後に「どの現行と比べたのか」を復元できない（Knowledge.md §10「実績の保存キーは『どう作られたか』を含める」）。sha256 はサイドカー `models/lgb_model.meta.json` から読む。読めないときは `"legacy-unknown"` とし、記録自体は続ける（shadowの都合で判断を止めない）。
8. **候補メタの `feature_cols` が現行の `FEATURE_COLS` と食い違ったら shadow記録を行わない**（fail-closed、Knowledge.md §10 desyncガード）。黙って別の列で推論して「正常な観察結果」に見せない。
9. `evaluation_runs` テーブルには shadow バッチの行を作らない（段階Eの `record_shadow()` と同じ扱い。`predictions` / `shadow_comparisons` の `evaluation_run_id` 列だけで参照する）。

---

## File Structure

| ファイル | 責務 | 変更 |
| --- | --- | --- |
| `src/services/shadow_recording.py` | shadow記録の配線一式（候補のロード・現行モデルのアダプタ・特徴量1行の抽出・バッチ記録）。**記録だけを行い、発注判断を返さない。** | 新規 |
| `src/services/trading.py` | `signal_scan()` から上記を3箇所で呼ぶ（import 1行＋呼び出し3箇所）。既存ロジックは不変。 | 修正 |
| `tests/test_shadow_wiring.py` | 上記の単体テスト・異常系テスト・`signal_scan()` 統合テスト。 | 新規 |
| `docs/運用Runbook.md` | 「shadow記録を読む（昇格後の観察）」節を追記。 | 修正 |

`src/strategy/shadow.py` は**変更しない**（`tests/test_shadow.py::TestNotWiredToOrdering` が「shadow.py は trading / execution / order を import しない」を固定しており、配線は呼ぶ側に置く）。

---

## 既存コードの正確な参照（実装者が推測しないための一覧・すべて実在確認済み）

### `src/services/trading.py`
- `TradingServices.__init__(self, client, risk, order_mgr, model=None)`（`:233`）。`self.model` は `main.py:223` の `ml_model.load()` の戻り値＝`lgb.LGBMClassifier` または `None`。
- `TradingServices.signal_scan(self) -> None`（`:567`）。
  - `:584` `codes = watchlist_store.get_codes()`
  - `:585` `for sym in codes:`
  - `:586` `try:`
  - `:587` `if not self._is_fresh_for_new_candidate(sym):` … `:595` `continue`
  - `:596` `df = load_ohlcv(sym)` / `:597-598` `if len(df) < 30: continue`
  - `:599` `sig = gen_signal(sym, df, self.model)`
  - `:600` `_save_signal(sig, data_as_of=self._bar_states[sym].last_bar_session)`
  - `:601-602` `if sig.action not in ("BUY", "SELL"): continue`
  - `:636-637` `except Exception as e:` / `logger.error(f"シグナルスキャンエラー: {sym} {e}")`
  - `:639-645` 全銘柄が鮮度不足だったときのWARNINGアラート
- import 群は `:12-33`（`:31` `from src.risk import liquidity` / `:32` `from src.strategy import ml_model`）。

### `src/strategy/shadow.py`（全文が短い。実装前に一読すること）
- `compare(event_ids: list, features: pd.DataFrame, *, current, candidate, threshold: float) -> list[ShadowComparison]`（`:45`）。`current` / `candidate` は**どちらも `.predict(features) -> 正例確率の1次元配列` を持つオブジェクト**であること（`_predict()` が `np.asarray(model.predict(features), dtype=float)` を呼ぶ・`:39-42`）。`current` は `None` 可（未昇格扱い）。
- `record_shadow(comparisons, *, evaluation_run_id, candidate_model_id, current_model_id, threshold, label_contract_id) -> int`（`:79`）。同一 `evaluation_run_id` の既存比較行を削除してから挿入する（`:126-127`）。
- `load_shadow_comparisons(evaluation_run_id: str) -> pd.DataFrame`（`:151`）
- `disagreement_summary(comparisons: list) -> dict`（`:177`）。キーは `both_take` / `both_skip` / `only_current` / `only_candidate` / `n` / `agreement_rate`。
- 定数 `AGREEMENT_BOTH_TAKE="both_take"` / `AGREEMENT_BOTH_SKIP="both_skip"` / `AGREEMENT_ONLY_CURRENT="only_current"` / `AGREEMENT_ONLY_CANDIDATE="only_candidate"`（`:22-25`）。

### 型の不一致（実機で確認済み・アダプタが必要な理由）
`self.model`（`lgb.LGBMClassifier`）に同じ1行の入力を与えた実測値:

```
model.predict(X)        -> [1]                      # クラスラベル
model.predict_proba(X)  -> [[0.4211066 0.5788934]]  # (n, 2) の確率
```

`shadow.compare()` は `.predict()` を呼ぶので、素の `self.model` を渡すと**確率のつもりでクラスラベル（0/1）を記録してしまう**。`predict_proba(X)[:, 1]` を `predict()` の形に合わせる薄いアダプタが必須（Task 1）。
一方、候補モデル（`model_store.load_current()` が返す `lgb.Booster` / `ConstantModel`）は `predict(X) -> 正例確率の1次元配列` を既に持つ（`model_store.load_model()` の docstring `:229-233`）。実測: `Booster.predict(X) -> array([0.13284307])`。**候補側にアダプタは不要。**

### `src/strategy/model_store.py`
- `load_current(*, base_dir: str = "models") -> Optional[tuple]`（`:338`）。`(model, ModelMeta)` または `None`（未昇格）。
- `ModelMeta`（`:45`）の使うフィールド: `model_id: str` / `feature_cols: list` / `label_contract_id: Optional[str]` / `model_kind: str`。
- `save_candidate(model, meta, *, base_dir="models") -> Path`（`:147`）/ `set_current(model_id, *, base_dir="models", previous_model_id=None)`（`:289`）— テストで候補を用意するのに使う。
- `ConstantModel`（`:198`）: `predict(X)` が `np.full(len(X), probability)` を返す。`is_constant=True` / `constant_probability` を持つオブジェクトを `save_candidate()` に渡すと定数モデルとして保存され、読み戻すと `ConstantModel` になる（`_extract_artifact()` `:123-125`、`_load_from_dir()` `:219-221`）。

### `src/strategy/indicators.py`
- `FEATURE_COLS`（`:123-127`、10列）: `ma_cross_sm`, `ma_cross_ml`, `rsi`, `macd`, `macd_hist`, `bb_pct`, `volume_ratio`, `price_momentum_5`, `price_momentum_20`, `returns`。**legacy も v2候補も同じ列**なので特徴量の不一致は原則起きない（起きたら Task 2 のガードが止める）。
- `build_feature_frame(df) -> pd.DataFrame`（`:109`）: 日付インデックス保持・`feature_valid` 列付き。
- `build_features(df) -> pd.DataFrame`（`:95`）: `_add_feature_columns(df).dropna(subset=FEATURE_COLS)`。legacy推論が使う。
- `ma_long` の既定は75なので、**特徴量が有効になるには約75本以上の日足が必要**（テストデータは120本にする）。

### `src/strategy/ml_model.py`
- `MODEL_PATH = Path("models/lgb_model.pkl")`（`:27`）。サイドカーは `MODEL_PATH.with_suffix(".meta.json")`（`_meta_path()` `:30-32`。privateなので呼ばず、`MODEL_PATH` から組み立てる）。中身の実例:
  ```json
  {"sha256": "6958ed7c114b609a1175e26703a0c60867c2a5620df1fde301da36022a71cb3a",
   "trained_at": "2026-09-18T11:14:50.350372", "n_samples": 25243, ...}
  ```
- `predict_proba(model: lgb.LGBMClassifier, df) -> float`（`:295`）。内部で `build_features()` 済みの最終行に `model.predict_proba(latest)[0][1]` を当てる。

### `src/strategy/dataset.py`
- `make_event_id(symbol: str, decision_at: date) -> str`（`:80`）→ `f"{symbol}:{YYYYMMDD}"`。
- `make_label_contract_id(policy_conf, costs, *, peak_basis="previous") -> str`（`:94`、SHA256先頭12桁）。
- `build_events()` は `decision_at = feat.index[i].date()`（`:297`）としており、**判断日は特徴量フレームのインデックス日付**。shadowも同じ規約にする。

### 設定の読み出し口
- `src/strategy/policy.py:103` `config_from_settings() -> PolicyConfig`
- `src/backtest/execution.py:43` `config_from_settings() -> CostConfig`
- 本番プロセスは起動時に `risk_profile_store.load("risk_profile.json")`（`main.py:48`）で high_risk を config へ適用済み。実測: 適用後の `make_label_contract_id(...)` は `b379db8dd396` で、**昇格済み候補 `v2-20260921T123704-0589b6e1` のメタの `label_contract_id` と一致する**（未適用だと `a769a9859a51`）。テストは risk_profile を読まないので、期待値はテスト内で同じ関数を呼んで求めること（固定文字列を書かない）。

### `src/core/clock.py`
`clock.today()` / `clock.now()`（JST naive）。`trading.clock` も `shadow_recording.clock` も同じモジュールオブジェクトなので、テストで `patch.object(trading.clock, "today", ...)` すれば両方に効く。

---

## Task 1: `_LegacyModelProbaAdapter`（新規モジュールの骨格）

**Files:**
- Create: `src/services/shadow_recording.py`
- Test: `tests/test_shadow_wiring.py`

**Interfaces:**
- Consumes: `src/strategy/shadow.py` の `compare()`（`current` 引数の契約 `.predict(X) -> 1次元の正例確率配列`）
- Produces:
  - `SHADOW_THRESHOLD: float = 0.5`
  - `LEGACY_MODEL_ID_UNKNOWN: str = "legacy-unknown"`
  - `class _LegacyModelProbaAdapter: __init__(self, model)` / `predict(self, X) -> np.ndarray`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_shadow_wiring.py` を新規作成する。

```python
"""昇格済み候補モデルのshadow記録の配線（src/services/shadow_recording.py）

**観察のみ。実発注には一切影響しない。** 記録の失敗が本来の判断
（買い/売り/様子見）や paper執行を妨げないことも、このファイルで固定する。
"""
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.backtest import execution
from src.core import clock
from src.core import config as cfg
from src.data import database as db
from src.data.bar_status import BarStatus
from src.data.database import OHLCV, Signal, get_session
from src.services import shadow_recording
from src.services import trading
from src.strategy import dataset as ds
from src.strategy import model_store as ms
from src.strategy import policy
from src.strategy import shadow
from src.strategy.indicators import FEATURE_COLS, build_features
from src.strategy.signal import Signal as TradeSignal


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


class _FakeClassifier:
    """sklearn LGBMClassifier の「形」だけを持つ最小の贋物。

    predict() はクラスラベル、predict_proba() は (n, 2) の確率を返す。
    実機の LGBMClassifier で確認した挙動と同じ
    （predict -> [1] / predict_proba -> [[0.4211, 0.5789]]）。
    """

    def __init__(self, positives):
        self._p = np.asarray(positives, dtype=float)

    def predict(self, X):
        return (self._p[:len(X)] >= 0.5).astype(int)

    def predict_proba(self, X):
        p = self._p[:len(X)]
        return np.column_stack([1.0 - p, p])


class _OneColumnProbaClassifier:
    """単一クラスしか見なかった分類器（predict_proba が (n, 1) を返す）"""

    def predict_proba(self, X):
        return np.ones((len(X), 1), dtype=float)


class _FixedProbaModel:
    """候補モデル役。predict() が正例確率をそのまま返す（Booster と同じ契約）"""

    def __init__(self, probabilities):
        self._p = np.asarray(probabilities, dtype=float)

    def predict(self, X):
        return self._p[:len(X)]


class TestLegacyModelProbaAdapter:
    def test_predict_returns_probabilities_not_class_labels(self):
        """素の predict() はラベルを返す。確率として記録してはいけない"""
        model = _FakeClassifier([0.5789, 0.1])
        X = pd.DataFrame({"f": [1.0, 2.0]})

        assert list(model.predict(X)) == [1, 0]
        got = shadow_recording._LegacyModelProbaAdapter(model).predict(X)
        assert list(got) == pytest.approx([0.5789, 0.1])

    def test_rejects_a_single_class_probability_matrix(self):
        """確率が2列そろわないときは既定値で埋めず例外にする

        黙って0.5等で埋めると「現行が何を予測したか」が捏造される
        （Knowledge.md §10「欠落を既定値で埋めると未結線が正常な結果に化ける」）。
        """
        adapter = shadow_recording._LegacyModelProbaAdapter(
            _OneColumnProbaClassifier())
        with pytest.raises(ValueError):
            adapter.predict(pd.DataFrame({"f": [1.0]}))

    def test_satisfies_the_contract_shadow_compare_requires(self):
        comparisons = shadow.compare(
            ["7203:20260910"], pd.DataFrame({"f": [1.0]}),
            current=shadow_recording._LegacyModelProbaAdapter(
                _FakeClassifier([0.8])),
            candidate=_FixedProbaModel([0.2]),
            threshold=shadow_recording.SHADOW_THRESHOLD)

        assert comparisons[0].current_probability == pytest.approx(0.8)
        assert comparisons[0].current_takes is True
        assert comparisons[0].candidate_takes is False
        assert comparisons[0].agreement == shadow.AGREEMENT_ONLY_CURRENT

    def test_threshold_is_the_observation_default(self):
        assert shadow_recording.SHADOW_THRESHOLD == 0.5
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `pytest tests/test_shadow_wiring.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.services.shadow_recording'`）

- [ ] **Step 3: 最小実装を書く**

`src/services/shadow_recording.py` を新規作成する。

```python
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
```

- [ ] **Step 4: テストを実行して成功を確認する**

Run: `pytest tests/test_shadow_wiring.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: コミット**

```bash
git add src/services/shadow_recording.py tests/test_shadow_wiring.py
git commit -m "$(cat <<'EOF'
feat(shadow): 現行モデルをshadow.compare()の契約へ合わせるアダプタを追加

LGBMClassifier.predict() はクラスラベルを返すため、そのまま
shadow.compare() へ渡すと確率のつもりでラベルを記録してしまう。
predict_proba()[:, 1] を predict() の形へ変換する薄いアダプタを置く。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `prepare()` — 候補のロードと記録の前提を1回だけ決める

**Files:**
- Modify: `src/services/shadow_recording.py`
- Test: `tests/test_shadow_wiring.py`

**Interfaces:**
- Consumes: Task 1 の `_LegacyModelProbaAdapter` / `SHADOW_THRESHOLD` / `LEGACY_MODEL_ID_UNKNOWN`、`model_store.load_current()`、`dataset.make_label_contract_id()`、`policy.config_from_settings()`、`execution.config_from_settings()`
- Produces:
  - `@dataclass class ShadowBatch`: `evaluation_run_id: str` / `candidate: object` / `candidate_model_id: str` / `current: Optional[object]` / `current_model_id: Optional[str]` / `label_contract_id: str` / `threshold: float = SHADOW_THRESHOLD` / `event_ids: list` / `rows: list`
  - `prepare(current_model, *, base_dir: str = "models") -> Optional[ShadowBatch]`
  - `_legacy_model_id(model) -> Optional[str]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_shadow_wiring.py` の末尾に追記する。

```python
class _ConstantCandidate:
    """model_store.save_candidate() が定数モデルとして保存できる最小の形。

    読み戻すと model_store.ConstantModel になり、
    predict(X) -> np.full(len(X), probability) を返す。
    """

    is_constant = True

    def __init__(self, probability):
        self.constant_probability = float(probability)


def _current_contract() -> str:
    """いま設定から決まるラベル契約ID（固定文字列を書かない）"""
    return ds.make_label_contract_id(policy.config_from_settings(),
                                     execution.config_from_settings())


def _promote_constant_candidate(models_dir, *, model_id="v2-test-0001",
                                probability=0.7, feature_cols=None):
    """tmp配下に候補を保存して現行（＝昇格済み）に設定する"""
    meta = ms.ModelMeta(
        model_id=model_id,
        trained_at=clock.now(),
        symbols=["7203"],
        label_definition="net_return>0",
        feature_cols=list(FEATURE_COLS) if feature_cols is None else feature_cols,
        label_contract_id=_current_contract(),
    )
    ms.save_candidate(_ConstantCandidate(probability), meta,
                      base_dir=str(models_dir))
    ms.set_current(model_id, base_dir=str(models_dir))
    return model_id


class TestPrepare:
    def test_returns_none_when_no_candidate_is_promoted(self, isolated_db, tmp_path):
        """未昇格は異常ではない。静かにスキップする"""
        assert shadow_recording.prepare(
            None, base_dir=str(tmp_path / "models")) is None

    def test_loads_the_promoted_candidate(self, isolated_db, tmp_path):
        models_dir = tmp_path / "models"
        _promote_constant_candidate(models_dir)

        batch = shadow_recording.prepare(None, base_dir=str(models_dir))

        assert batch is not None
        assert batch.candidate_model_id == "v2-test-0001"
        assert batch.candidate.predict(pd.DataFrame({"f": [1.0]})) == \
            pytest.approx([0.7])
        assert batch.threshold == 0.5
        assert batch.label_contract_id == _current_contract()
        assert batch.event_ids == []
        assert batch.rows == []

    def test_run_id_is_one_batch_per_day(self, isolated_db, tmp_path):
        models_dir = tmp_path / "models"
        _promote_constant_candidate(models_dir)

        with patch.object(clock, "today", return_value=date(2026, 9, 10)):
            batch = shadow_recording.prepare(None, base_dir=str(models_dir))

        assert batch.evaluation_run_id == "shadow-2026-09-10"

    def test_wraps_the_current_model_in_the_probability_adapter(
            self, isolated_db, tmp_path):
        models_dir = tmp_path / "models"
        _promote_constant_candidate(models_dir)

        batch = shadow_recording.prepare(_FakeClassifier([0.9]),
                                         base_dir=str(models_dir))

        assert isinstance(batch.current, shadow_recording._LegacyModelProbaAdapter)
        assert batch.current.predict(pd.DataFrame({"f": [1.0]})) == \
            pytest.approx([0.9])

    def test_no_current_model_is_recorded_as_such(self, isolated_db, tmp_path):
        """現行モデル未ロード（学習前）でも候補だけは記録できる"""
        models_dir = tmp_path / "models"
        _promote_constant_candidate(models_dir)

        batch = shadow_recording.prepare(None, base_dir=str(models_dir))

        assert batch.current is None
        assert batch.current_model_id is None

    def test_refuses_to_record_when_the_feature_definition_disagrees(
            self, isolated_db, tmp_path):
        """特徴量定義が食い違う候補は黙って推論しない（fail-closed）"""
        models_dir = tmp_path / "models"
        _promote_constant_candidate(models_dir, feature_cols=["rsi", "macd"])

        assert shadow_recording.prepare(None, base_dir=str(models_dir)) is None


class TestLegacyModelId:
    def test_is_none_without_a_current_model(self):
        assert shadow_recording._legacy_model_id(None) is None

    def test_uses_the_sha256_of_the_saved_pickle(self, tmp_path, monkeypatch):
        """週次再学習で中身が入れ替わるので、版を sha256 で区別する"""
        monkeypatch.chdir(tmp_path)
        meta_path = tmp_path / "models" / "lgb_model.meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            '{"sha256": "6958ed7c114b609a1175e26703a0c60867c2a5620df1fde301da36'
            '022a71cb3a", "trained_at": "2026-09-18T11:14:50.350372"}',
            encoding="utf-8")

        got = shadow_recording._legacy_model_id(_FakeClassifier([0.5]))

        assert got == "legacy-6958ed7c114b"

    def test_falls_back_when_the_sidecar_is_missing(self, tmp_path, monkeypatch):
        """メタが読めなくても記録自体は続ける（判断を止めない）"""
        monkeypatch.chdir(tmp_path)

        got = shadow_recording._legacy_model_id(_FakeClassifier([0.5]))

        assert got == shadow_recording.LEGACY_MODEL_ID_UNKNOWN
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `pytest tests/test_shadow_wiring.py -v -k "Prepare or LegacyModelId"`
Expected: FAIL（`AttributeError: module 'src.services.shadow_recording' has no attribute 'prepare'`）

- [ ] **Step 3: 実装を書く**

`src/services/shadow_recording.py` の import 群を次で置き換える（`import numpy as np` の行を含めて差し替える）。

```python
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
```

`_LegacyModelProbaAdapter` の**後ろ**に追記する。

```python
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
```

- [ ] **Step 4: テストを実行して成功を確認する**

Run: `pytest tests/test_shadow_wiring.py -v`
Expected: PASS（13 passed）

- [ ] **Step 5: コミット**

```bash
git add src/services/shadow_recording.py tests/test_shadow_wiring.py
git commit -m "$(cat <<'EOF'
feat(shadow): 昇格済み候補をスキャン開始時に一度だけ読み込む prepare() を追加

未昇格なら None を返して静かにスキップする。特徴量定義が現行と
食い違う候補は fail-closed で記録しない。現行モデルの版は
lgb_model.meta.json の sha256 先頭12桁で表す。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `collect()` / `flush()` — 銘柄ごとに溜めて、最後に1回だけ記録する

**Files:**
- Modify: `src/services/shadow_recording.py`
- Test: `tests/test_shadow_wiring.py`

**Interfaces:**
- Consumes: Task 2 の `ShadowBatch`、`indicators.build_feature_frame()`、`dataset.make_event_id()`、`shadow.compare()` / `record_shadow()` / `disagreement_summary()`
- Produces:
  - `collect(batch: ShadowBatch, symbol: str, df: pd.DataFrame) -> bool`（記録対象に加えたら True、特徴量が揃わずスキップしたら False）
  - `flush(batch: ShadowBatch) -> int`（保存件数）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_shadow_wiring.py` の末尾に追記する。

```python
def _ohlcv_frame(*, periods=120, end=date(2026, 9, 10)) -> pd.DataFrame:
    """合成の日足（日付インデックス）。

    ma_long の既定が75本なので、最終行で全特徴量が有効になるよう120本入れる。
    """
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=periods)
    close = np.array([1000.0 + 50.0 * np.sin(i / 7.0) + i * 0.5
                      for i in range(periods)])
    return pd.DataFrame({
        "open": close - 3.0,
        "high": close + 6.0,
        "low": close - 6.0,
        "close": close,
        "volume": np.array([100_000 + 300 * i for i in range(periods)]),
    }, index=idx)


def _batch(*, current=None, current_model_id=None,
           run_id="shadow-2026-09-10", candidate_probability=0.7):
    return shadow_recording.ShadowBatch(
        evaluation_run_id=run_id,
        candidate=_FixedProbaModel([candidate_probability] * 16),
        candidate_model_id="v2-test-0001",
        current=current,
        current_model_id=current_model_id,
        label_contract_id="lc_test",
    )


class TestCollect:
    def test_event_id_is_the_symbol_and_the_decision_session(self, isolated_db):
        batch = _batch()

        assert shadow_recording.collect(batch, "7203", _ohlcv_frame()) is True
        assert batch.event_ids == ["7203:20260910"]

    def test_uses_the_same_feature_row_the_current_model_sees(self, isolated_db):
        """gen_signal が使う build_features() の最終行と同一であること

        別の行を渡すと「同じ入力に対する2つの判断」ではなくなる。
        """
        df = _ohlcv_frame()
        batch = _batch()
        shadow_recording.collect(batch, "7203", df)

        expected = build_features(df)[list(FEATURE_COLS)].iloc[[-1]]
        assert list(batch.rows[0].columns) == list(FEATURE_COLS)
        assert batch.rows[0].to_numpy() == pytest.approx(expected.to_numpy())

    def test_skips_a_symbol_whose_features_are_not_ready(self, isolated_db):
        """助走期間が足りない銘柄は記録しない（欠損を0で埋めない）"""
        batch = _batch()

        assert shadow_recording.collect(batch, "7203",
                                        _ohlcv_frame(periods=40)) is False
        assert batch.event_ids == []
        assert batch.rows == []

    def test_accumulates_every_symbol(self, isolated_db):
        batch = _batch()
        shadow_recording.collect(batch, "7203", _ohlcv_frame())
        shadow_recording.collect(batch, "9984", _ohlcv_frame())

        assert batch.event_ids == ["7203:20260910", "9984:20260910"]
        assert len(batch.rows) == 2


class TestFlush:
    def test_writes_one_row_per_collected_symbol(self, isolated_db):
        batch = _batch()
        shadow_recording.collect(batch, "7203", _ohlcv_frame())
        shadow_recording.collect(batch, "9984", _ohlcv_frame())

        n = shadow_recording.flush(batch)

        assert n == 2
        got = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert set(got["event_id"]) == {"7203:20260910", "9984:20260910"}
        assert set(got["candidate_model_id"]) == {"v2-test-0001"}
        assert got["candidate_probability"].tolist() == pytest.approx([0.7, 0.7])
        assert set(got["threshold"]) == {0.5}
        assert set(got["label_contract_id"]) == {"lc_test"}

    def test_records_both_sides_when_a_current_model_exists(self, isolated_db):
        batch = _batch(
            current=shadow_recording._LegacyModelProbaAdapter(
                _FakeClassifier([0.9, 0.9])),
            current_model_id="legacy-6958ed7c114b")
        shadow_recording.collect(batch, "7203", _ohlcv_frame())
        shadow_recording.collect(batch, "9984", _ohlcv_frame())

        shadow_recording.flush(batch)

        got = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert set(got["current_model_id"]) == {"legacy-6958ed7c114b"}
        assert got["current_probability"].tolist() == pytest.approx([0.9, 0.9])
        assert set(got["agreement"]) == {shadow.AGREEMENT_BOTH_TAKE}

    def test_records_the_unpromoted_current_as_its_own_state(self, isolated_db):
        batch = _batch()
        shadow_recording.collect(batch, "7203", _ohlcv_frame())

        shadow_recording.flush(batch)

        got = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert got["current_model_id"].isna().all()
        assert got["current_probability"].isna().all()
        assert set(got["agreement"]) == {shadow.AGREEMENT_ONLY_CANDIDATE}

    def test_nothing_collected_writes_nothing(self, isolated_db):
        assert shadow_recording.flush(_batch()) == 0
        assert len(shadow.load_shadow_comparisons("shadow-2026-09-10")) == 0

    def test_flushing_the_same_run_twice_replaces_the_rows(self, isolated_db):
        """同じ日に2回走っても上書きで整合する（run単位の置換）"""
        first = _batch()
        shadow_recording.collect(first, "7203", _ohlcv_frame())
        shadow_recording.collect(first, "9984", _ohlcv_frame())
        shadow_recording.flush(first)

        second = _batch(candidate_probability=0.2)
        shadow_recording.collect(second, "7203", _ohlcv_frame())
        shadow_recording.collect(second, "9984", _ohlcv_frame())
        shadow_recording.flush(second)

        got = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert len(got) == 2
        assert got["candidate_probability"].tolist() == pytest.approx([0.2, 0.2])

    def test_one_flush_per_scan_keeps_every_symbol(self, isolated_db):
        """銘柄ごとに flush してはいけないことを、結果の差で示す

        record_shadow() は同一 evaluation_run_id の既存行を削除してから
        挿入する。銘柄ごとに flush すると最後の1銘柄しか残らない。
        """
        per_symbol = _batch()
        shadow_recording.collect(per_symbol, "7203", _ohlcv_frame())
        shadow_recording.flush(per_symbol)
        per_symbol.event_ids.clear()
        per_symbol.rows.clear()
        shadow_recording.collect(per_symbol, "9984", _ohlcv_frame())
        shadow_recording.flush(per_symbol)

        wrong = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert set(wrong["event_id"]) == {"9984:20260910"}  # 7203が消える

        batched = _batch(run_id="shadow-2026-09-11")
        shadow_recording.collect(batched, "7203", _ohlcv_frame())
        shadow_recording.collect(batched, "9984", _ohlcv_frame())
        shadow_recording.flush(batched)

        right = shadow.load_shadow_comparisons("shadow-2026-09-11")
        assert set(right["event_id"]) == {"7203:20260910", "9984:20260910"}
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `pytest tests/test_shadow_wiring.py -v -k "Collect or Flush"`
Expected: FAIL（`AttributeError: module 'src.services.shadow_recording' has no attribute 'collect'`）

- [ ] **Step 3: 実装を書く**

`src/services/shadow_recording.py` の import 群に2行足す（`import numpy as np` の直後に `import pandas as pd`、`from src.strategy import policy` の直後に `from src.strategy import shadow`、`from src.strategy.indicators import FEATURE_COLS` を `FEATURE_COLS, build_feature_frame` へ変更）。結果は次のとおり。

```python
import numpy as np
import pandas as pd
from loguru import logger

from src.backtest import execution
from src.core import clock
from src.strategy import dataset as ds
from src.strategy import ml_model
from src.strategy import model_store as ms
from src.strategy import policy
from src.strategy import shadow
from src.strategy.indicators import FEATURE_COLS, build_feature_frame
```

ファイル末尾（`prepare()` の後ろ）に追記する。

```python
def collect(batch: ShadowBatch, symbol: str, df: pd.DataFrame) -> bool:
    """この銘柄の特徴量1行をバッチへ溜める。DBへは書かない。

    使う行は **gen_signal が現行モデルへ渡すのと同じ行**にする。
    `ml_model.predict_proba()` は `build_features(df)`（＝欠損行を落とした
    フレーム）の最終行を使うので、こちらは行を落とさない
    `build_feature_frame(df)` の `feature_valid` が立った行の最終行を取る。
    同じ条件なので同じ行になる。

    特徴量が1行も揃わない銘柄（助走期間が足りない等）は記録しない。
    欠損を0等で埋めると、未結線や不足が「正常な観察結果」に化ける
    （Knowledge.md §10）。
    """
    frame = build_feature_frame(df)
    valid = frame[frame["feature_valid"]]
    if valid.empty:
        logger.warning(
            f"shadow記録をスキップ: {symbol} の特徴量が揃いません"
            f"（{len(df)}本）")
        return False
    row = valid.iloc[[-1]]
    decision_at = row.index[-1].date()
    batch.event_ids.append(ds.make_event_id(symbol, decision_at))
    batch.rows.append(row[list(FEATURE_COLS)].astype("float64"))
    return True


def flush(batch: ShadowBatch) -> int:
    """溜めた分をまとめて記録する。保存件数を返す。

    **1回のスキャンにつき一度だけ呼ぶこと。** `shadow.record_shadow()` は
    同一 `evaluation_run_id` の既存行を削除してから挿入するため、銘柄ごとに
    呼ぶと前の銘柄の記録が毎回消える。
    """
    if not batch.event_ids:
        return 0
    # 銘柄をまたいで日付インデックスが重複するので行番号へ振り直す。
    # shadow.compare() は event_ids と features を位置で対応付ける
    features = pd.concat(batch.rows, ignore_index=True)
    comparisons = shadow.compare(
        batch.event_ids, features,
        current=batch.current, candidate=batch.candidate,
        threshold=batch.threshold)
    n = shadow.record_shadow(
        comparisons,
        evaluation_run_id=batch.evaluation_run_id,
        candidate_model_id=batch.candidate_model_id,
        current_model_id=batch.current_model_id,
        threshold=batch.threshold,
        label_contract_id=batch.label_contract_id)
    summary = shadow.disagreement_summary(comparisons)
    logger.info(
        f"shadow並行記録: run={batch.evaluation_run_id} "
        f"候補={batch.candidate_model_id} 現行={batch.current_model_id} "
        f"{n}件 一致率={summary['agreement_rate']:.3f} "
        f"(both_take={summary['both_take']} both_skip={summary['both_skip']} "
        f"only_current={summary['only_current']} "
        f"only_candidate={summary['only_candidate']})")
    return n
```

- [ ] **Step 4: テストを実行して成功を確認する**

Run: `pytest tests/test_shadow_wiring.py -v`
Expected: PASS（23 passed）

- [ ] **Step 5: コミット**

```bash
git add src/services/shadow_recording.py tests/test_shadow_wiring.py
git commit -m "$(cat <<'EOF'
feat(shadow): 銘柄ごとに溜めて1回だけ記録する collect()/flush() を追加

現行モデルが見るのと同じ特徴量行（build_features の最終行）を使う。
record_shadow() は run 単位の置換なので、銘柄ごとに呼ぶと最後の
1銘柄しか残らない。その差をテストで固定した。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `signal_scan()` への配線（既存ロジックは1行も変えない）

**Files:**
- Modify: `src/services/trading.py:31-32`（import 1行追加）, `src/services/trading.py:584-585`（prepare）, `:600-601`（collect）, `:638-639`（flush）
- Test: `tests/test_shadow_wiring.py`

**Interfaces:**
- Consumes: Task 2/3 の `shadow_recording.prepare()` / `collect()` / `flush()`
- Produces: なし（`signal_scan()` のシグネチャ・戻り値は変わらない）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_shadow_wiring.py` の末尾に追記する。

```python
def _fresh_states(symbols, session_date=date(2026, 9, 10)):
    return {s: BarStatus(symbol=s, last_bar_session=session_date,
                         observed_at=datetime(2026, 9, 10, 16, 20),
                         is_final=True, state="fresh")
            for s in symbols}


class TestSignalScanWiring:
    """配線をソースで固定する（既存ロジックを変えていないことを含む）"""

    def _source(self):
        import inspect
        return inspect.getsource(trading.TradingServices.signal_scan)

    def test_existing_decision_line_is_unchanged(self):
        src = self._source()
        assert "sig = gen_signal(sym, df, self.model)" in src
        assert "_save_signal(sig, data_as_of=self._bar_states[sym].last_bar_session)" in src

    def test_the_scan_never_reassigns_the_operating_model(self):
        """候補を self.model へ入れない（shadowは観察のみ）"""
        assert "self.model =" not in self._source()

    def test_calls_prepare_once_before_the_loop(self):
        src = self._source()
        assert src.index("shadow_recording.prepare(self.model)") < \
            src.index("for sym in codes:")

    def test_collects_before_the_action_filter(self):
        """HOLDの銘柄も記録対象にする（絞り込まない）"""
        src = self._source()
        assert src.index("shadow_recording.collect(shadow_batch, sym, df)") < \
            src.index('if sig.action not in ("BUY", "SELL"):')

    def test_flushes_once_after_the_loop(self):
        src = self._source()
        assert src.index("shadow_recording.flush(shadow_batch)") > \
            src.index("シグナルスキャンエラー")


class TestShadowFailureDoesNotAffectTrading:
    """shadow記録の失敗が本来の判断・発注を妨げないこと"""

    def _services(self, model=None):
        risk = MagicMock()
        risk.validate_buy.return_value = (True, "ok")
        risk.calc_position_size.return_value = 100
        svc = trading.TradingServices(client=MagicMock(), risk=risk,
                                      order_mgr=MagicMock(), model=model)
        svc.trading_conf = dict(svc.trading_conf)
        svc.trading_conf["mode"] = "paper"
        return svc

    def _scan(self, svc, symbols):
        svc._bar_states = _fresh_states(symbols)
        with patch.object(trading.clock, "now",
                          return_value=datetime(2026, 9, 10, 16, 20)), \
             patch.object(trading.clock, "today", return_value=date(2026, 9, 10)), \
             patch.object(trading.watchlist_store, "get_codes",
                          return_value=list(symbols)), \
             patch.object(trading.watchlist_store, "get_sectors",
                          return_value={}), \
             patch.object(trading.TradingScheduler, "is_maintenance_window",
                          return_value=False), \
             patch.object(trading, "load_ohlcv", return_value=_ohlcv_frame()), \
             patch.object(trading, "gen_signal",
                          side_effect=lambda sym, df, model: TradeSignal(
                              symbol=sym, action="BUY", rule_score=0.6,
                              ml_score=0.2, combined_score=0.4)), \
             patch.object(trading.liquidity, "check_liquidity",
                          return_value=(True, "")):
            svc.signal_scan()

    def test_a_failing_collect_does_not_block_the_paper_order(self, isolated_db):
        svc = self._services()
        with patch.object(shadow_recording, "prepare",
                          return_value=_batch()), \
             patch.object(shadow_recording, "collect",
                          side_effect=RuntimeError("shadow boom")):
            self._scan(svc, ["7203"])

        assert svc.order_mgr.buy.called

    def test_a_failing_collect_does_not_stop_the_other_symbols(self, isolated_db):
        svc = self._services()
        with patch.object(shadow_recording, "prepare",
                          return_value=_batch()), \
             patch.object(shadow_recording, "collect",
                          side_effect=RuntimeError("shadow boom")):
            self._scan(svc, ["7203", "9984"])

        with get_session() as session:
            saved = {r.symbol for r in session.scalars(select(Signal)).all()}
        assert saved == {"7203", "9984"}

    def test_a_failing_prepare_does_not_stop_the_scan(self, isolated_db):
        svc = self._services()
        with patch.object(shadow_recording, "prepare",
                          side_effect=RuntimeError("load boom")):
            self._scan(svc, ["7203"])

        assert svc.order_mgr.buy.called

    def test_a_failing_flush_does_not_stop_the_scan(self, isolated_db):
        svc = self._services()
        with patch.object(shadow_recording, "prepare",
                          return_value=_batch()), \
             patch.object(shadow_recording, "flush",
                          side_effect=RuntimeError("save boom")):
            self._scan(svc, ["7203"])

        assert svc.order_mgr.buy.called

    def test_no_candidate_means_no_shadow_calls(self, isolated_db):
        """未昇格なら collect も flush も呼ばれない"""
        svc = self._services()
        with patch.object(shadow_recording, "prepare", return_value=None), \
             patch.object(shadow_recording, "collect") as collect, \
             patch.object(shadow_recording, "flush") as flush:
            self._scan(svc, ["7203"])

        assert not collect.called
        assert not flush.called
        assert svc.order_mgr.buy.called
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `pytest tests/test_shadow_wiring.py -v -k "Wiring or FailureDoesNot"`
Expected: FAIL（`ValueError: substring not found` — `signal_scan` にまだ shadow の呼び出しが無い）

- [ ] **Step 3: `src/services/trading.py` へ3箇所＋importを足す**

(a) import（`:31` と `:32` の間に1行）:

```python
from src.risk import liquidity
from src.services import shadow_recording
from src.strategy import ml_model
```

(b) ループの手前（`:584` `codes = watchlist_store.get_codes()` の直後）:

```python
        codes = watchlist_store.get_codes()
        # ─── shadow記録の準備（観察のみ・発注には一切影響しない）───
        # 昇格済み候補があれば、以降のループで「現行と同じ入力に対する候補の
        # 判断」を溜める。候補が未昇格なら None で、何も記録しない。
        # 候補モデルの読み込みはスキャン1回につきここだけ（銘柄ごとに
        # 読み直さない）。
        try:
            shadow_batch = shadow_recording.prepare(self.model)
        except Exception as e:
            logger.error(
                f"shadow記録の準備に失敗しました（売買判断には影響しません）: {e}")
            shadow_batch = None
        for sym in codes:
```

(c) `_save_signal(...)`（`:600`）の直後、`if sig.action not in ("BUY", "SELL"):` の**前**:

```python
                _save_signal(sig, data_as_of=self._bar_states[sym].last_bar_session)
                if shadow_batch is not None:
                    # 本来の判断（sig）には一切関与しない。ここで例外を外へ
                    # 出すと、この銘柄の以降の処理（paper執行）まで止まって
                    # しまうので、内側で必ず捕まえる
                    try:
                        shadow_recording.collect(shadow_batch, sym, df)
                    except Exception as e:
                        logger.error(
                            f"shadow記録に失敗しました（{sym}・売買判断には"
                            f"影響しません）: {e}")
                if sig.action not in ("BUY", "SELL"):
                    continue
```

(d) ループを抜けた直後（`:637` `logger.error(f"シグナルスキャンエラー: {sym} {e}")` の後、`:639` の `if codes and excluded_count == len(codes):` の**前**）:

```python
            except Exception as e:
                logger.error(f"シグナルスキャンエラー: {sym} {e}")

        # 溜めたshadow記録をまとめて1回で保存する（銘柄ごとに保存すると
        # run単位の置換で前の銘柄が消える）
        if shadow_batch is not None:
            try:
                shadow_recording.flush(shadow_batch)
            except Exception as e:
                logger.error(
                    f"shadow記録の保存に失敗しました（売買判断には影響しません）: {e}")

        if codes and excluded_count == len(codes):
```

- [ ] **Step 4: テストを実行して成功を確認する**

Run: `pytest tests/test_shadow_wiring.py -v`
Expected: PASS（33 passed）

- [ ] **Step 5: 既存の取引系テストが緑のままであることを確認する**

Run: `pytest tests/test_signal_freshness_gate.py tests/test_paper_execution_v2.py tests/test_select_latest_signals.py tests/test_engine_version_wiring.py tests/test_shadow.py -v`
Expected: すべて PASS（1本も落ちないこと。落ちたら配線が既存挙動を変えている）

- [ ] **Step 6: コミット**

```bash
git add src/services/trading.py tests/test_shadow_wiring.py
git commit -m "$(cat <<'EOF'
feat(shadow): signal_scan へ候補モデルのshadow記録を配線する

既存の判断（gen_signal 以降）は1行も変えず、prepare/collect/flush を
追加するだけ。3箇所すべて try/except で隔離し、shadow記録の失敗が
売買判断・paper執行・他銘柄の処理を妨げないことをテストで固定した。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 統合テスト — 実DB・実OHLCV・実候補モデルで `shadow_comparisons` を確かめる

**Files:**
- Test: `tests/test_shadow_wiring.py`

**Interfaces:**
- Consumes: Task 4 までの全て
- Produces: なし（テストのみ）

モックで差し替えるのは「ウォッチリスト・現在時刻・メンテナンス判定」だけにする。OHLCVは実際にDBへ入れて `load_ohlcv()` に読ませ、候補モデルは実際に `model_store` で保存・昇格させ、記録は実際の `shadow_comparisons` テーブルを読んで確かめる。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_shadow_wiring.py` の末尾に追記する。

```python
def _seed_ohlcv(symbol, *, periods=120, end=date(2026, 9, 10)):
    """合成の日足をDBへ入れる（load_ohlcv 経由で実際に読ませる）"""
    frame = _ohlcv_frame(periods=periods, end=end)
    with get_session() as session:
        for ts, row in frame.iterrows():
            session.add(OHLCV(
                symbol=symbol, date=ts.date(),
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=int(row["volume"]), adjusted_close=float(row["close"])))
        session.commit()


class TestSignalScanRecordsShadow:
    """signal_scan を実際に走らせ、DBに入った行の件数と中身を確かめる"""

    def _run(self, svc, symbols):
        svc._bar_states = _fresh_states(symbols)
        with patch.object(trading.clock, "now",
                          return_value=datetime(2026, 9, 10, 16, 20)), \
             patch.object(trading.clock, "today", return_value=date(2026, 9, 10)), \
             patch.object(trading.watchlist_store, "get_codes",
                          return_value=list(symbols)), \
             patch.object(trading.watchlist_store, "get_sectors",
                          return_value={}), \
             patch.object(trading.TradingScheduler, "is_maintenance_window",
                          return_value=False):
            svc.signal_scan()

    def _services(self, model=None):
        return trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                       order_mgr=MagicMock(), model=model)

    def test_records_one_row_per_scanned_symbol(self, isolated_db, tmp_path,
                                                monkeypatch):
        _seed_ohlcv("7203")
        _seed_ohlcv("9984")
        _promote_constant_candidate(tmp_path / "models")
        # prepare() は既定の相対パス "models" を見る。実リポジトリを汚さない
        # よう、cfg.load / db.init を済ませた後に作業ディレクトリを移す
        monkeypatch.chdir(tmp_path)

        self._run(self._services(), ["7203", "9984"])

        got = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert len(got) == 2
        assert set(got["event_id"]) == {"7203:20260910", "9984:20260910"}
        assert set(got["candidate_model_id"]) == {"v2-test-0001"}
        assert got["candidate_probability"].tolist() == pytest.approx([0.7, 0.7])
        assert set(got["threshold"]) == {0.5}
        assert set(got["label_contract_id"]) == {_current_contract()}
        # 現行モデル未ロード（model=None）なので「現行が無かった」状態が残る
        assert got["current_model_id"].isna().all()
        assert set(got["agreement"]) == {shadow.AGREEMENT_ONLY_CANDIDATE}

    def test_records_the_current_models_probability_too(self, isolated_db,
                                                        tmp_path, monkeypatch):
        _seed_ohlcv("7203")
        _promote_constant_candidate(tmp_path / "models")
        monkeypatch.chdir(tmp_path)

        self._run(self._services(model=_FakeClassifier([0.9])), ["7203"])

        got = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert len(got) == 1
        # ラベル(1)ではなく確率(0.9)が入っていること＝アダプタが効いている
        assert got["current_probability"].tolist() == pytest.approx([0.9])
        assert got["current_model_id"].tolist() == \
            [shadow_recording.LEGACY_MODEL_ID_UNKNOWN]
        assert got["agreement"].tolist() == [shadow.AGREEMENT_BOTH_TAKE]

    def test_the_candidate_never_touches_the_signal(self, isolated_db, tmp_path,
                                                    monkeypatch):
        """保存されたシグナルが、候補の有無で変わらないこと"""
        _seed_ohlcv("7203")
        monkeypatch.chdir(tmp_path)
        svc = self._services()
        self._run(svc, ["7203"])
        with get_session() as session:
            without = [(r.action, r.combined_score)
                       for r in session.scalars(select(Signal)).all()]
            session.query(Signal).delete()
            session.commit()

        _promote_constant_candidate(tmp_path / "models")
        self._run(self._services(), ["7203"])

        with get_session() as session:
            with_candidate = [(r.action, r.combined_score)
                              for r in session.scalars(select(Signal)).all()]
        assert with_candidate == without
        assert len(shadow.load_shadow_comparisons("shadow-2026-09-10")) == 1

    def test_no_candidate_records_nothing_and_still_saves_signals(
            self, isolated_db, tmp_path, monkeypatch):
        _seed_ohlcv("7203")
        monkeypatch.chdir(tmp_path)   # models/current.json が無い状態

        self._run(self._services(), ["7203"])

        assert len(shadow.load_shadow_comparisons("shadow-2026-09-10")) == 0
        with get_session() as session:
            assert session.scalar(select(Signal)) is not None

    def test_a_symbol_without_enough_history_is_skipped(self, isolated_db,
                                                        tmp_path, monkeypatch):
        """特徴量が揃わない銘柄は記録しない（他の銘柄は記録する）"""
        _seed_ohlcv("7203")
        _seed_ohlcv("9984", periods=40)
        _promote_constant_candidate(tmp_path / "models")
        monkeypatch.chdir(tmp_path)

        self._run(self._services(), ["7203", "9984"])

        got = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert set(got["event_id"]) == {"7203:20260910"}

    def test_running_twice_in_a_day_stays_consistent(self, isolated_db, tmp_path,
                                                     monkeypatch):
        _seed_ohlcv("7203")
        _seed_ohlcv("9984")
        _promote_constant_candidate(tmp_path / "models")
        monkeypatch.chdir(tmp_path)

        self._run(self._services(), ["7203", "9984"])
        self._run(self._services(), ["7203", "9984"])

        assert len(shadow.load_shadow_comparisons("shadow-2026-09-10")) == 2
        with get_session() as session:
            preds = list(session.scalars(select(db.Prediction)).all())
        assert len(preds) == 2
        assert {p.purpose for p in preds} == {"shadow"}
        assert {p.fold_index for p in preds} == {-1}

    def test_no_outcome_rows_are_written(self, isolated_db, tmp_path, monkeypatch):
        """実績（勝敗）は予測時点で未確定。ここでは書かない"""
        _seed_ohlcv("7203")
        _promote_constant_candidate(tmp_path / "models")
        monkeypatch.chdir(tmp_path)

        self._run(self._services(), ["7203"])

        with get_session() as session:
            assert list(session.scalars(select(db.PredictionOutcome)).all()) == []

    def test_no_orders_are_created(self, isolated_db, tmp_path, monkeypatch):
        """shadowは記録だけ。注文・建玉・約定を一切作らない"""
        _seed_ohlcv("7203")
        _promote_constant_candidate(tmp_path / "models")
        monkeypatch.chdir(tmp_path)
        svc = self._services()

        self._run(svc, ["7203"])

        assert not svc.order_mgr.buy.called
        assert not svc.order_mgr.sell.called
        assert not svc.order_mgr.sell_market.called
        with get_session() as session:
            assert list(session.scalars(select(db.Trade)).all()) == []
            assert list(session.scalars(select(db.Position)).all()) == []
            assert list(session.scalars(select(db.OrderIntent)).all()) == []
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `pytest tests/test_shadow_wiring.py -v -k "SignalScanRecordsShadow"`
Expected: 最初の実行では FAIL する可能性がある（実装の抜けが無ければ PASS）。**失敗した場合は Knowledge.md §「実装計画書のテストコードは値が噛み合っているか実行前に検算する」に従い、テストデータの本数（120本＝ma75の助走に足りるか）・日付（2026-09-10は木曜・非休場）・`_fresh_states` の整合を先に確認してから実装を疑うこと。**

- [ ] **Step 3: 落ちたテストがあれば実装を直す**

想定される原因と対処:
- `shadow_comparisons` が0件 → `monkeypatch.chdir` が `cfg.load`/`db.init` より前に効いている。`isolated_db` フィクスチャを先に受け取り、その後で `chdir` すること。
- `event_id` の日付がずれる → `_seed_ohlcv` の `end` と `_fresh_states` の `session_date` と `clock.now` のパッチが同じ 2026-09-10 を指しているか確認する。
- 銘柄が1件しか入らない → `flush()` をループ内で呼んでいる（Task 4(d) の位置を確認）。

- [ ] **Step 4: 全テストが緑であることを確認する**

Run: `pytest tests/test_shadow_wiring.py -v`
Expected: PASS（41 passed）

Run: `pytest -q`
Expected: 既存テストが1本も落ちていないこと（落ちたものがあれば、それは配線による退行なので直す）

- [ ] **Step 5: 作業ツリーのファイルが規約どおりか確認する**

Run:
```bash
python -c "import sys; p='src/services/shadow_recording.py'; d=open(p,'rb').read(); print(p, 'BOM' if d.startswith(b'\xef\xbb\xbf') else 'no-BOM', 'CRLF' if b'\r\n' in d else 'LF')"
```
Expected: `src/services/shadow_recording.py no-BOM LF`

Run: `git diff --stat main`
Expected: `src/services/trading.py` の差分が十数行程度であること（全行差分になっていたら改行コードを壊している）

- [ ] **Step 6: コミット**

```bash
git add tests/test_shadow_wiring.py
git commit -m "$(cat <<'EOF'
test(shadow): signal_scan の統合テストで shadow_comparisons の中身を確認

実DBのOHLCVを load_ohlcv 経由で読ませ、実際に保存・昇格させた候補で
記録させる。銘柄数ぶんの行・確率・閾値・ラベル契約・一致区分に加え、
注文系テーブルに1行も増えないことを固定した。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: 運用Runbookへの追記（観察の手順と期間の目安）

**Files:**
- Modify: `docs/運用Runbook.md`（末尾の「### 5. 戻す（ロールバック）」の後ろに追記）

**Interfaces:**
- Consumes: `shadow.load_shadow_comparisons()` / `shadow.disagreement_summary()` / `ShadowComparison`
- Produces: なし

- [ ] **Step 1: Runbookへ節を追記する**

`docs/運用Runbook.md` の末尾（`### 5. 戻す（ロールバック）` のコード例の後）に、次をそのまま追記する。

````markdown

### 6. shadow記録を読む（昇格後の観察）

**昇格済みの候補モデルは実発注に使っていない。** `signal_scan`（平日16:20）が
毎日、現行モデルと同じ入力に対する候補の判断を `shadow_comparisons` テーブルへ
記録するだけで、買い/売り/様子見の判断は従来どおり現行モデル（legacy）が決める。
候補が未昇格（`models/current.json` が無い）なら何も記録されない。

- 記録のID（`evaluation_run_id`）は**日ごと**に `shadow-YYYY-MM-DD`
- 判定閾値は **0.5固定**。チューニングされた値ではなく観察用の仮の基準
- 1日分の記録件数 ≒ その日スキャンした銘柄数（鮮度不足で除外された銘柄と、
  日足が75本に満たない銘柄は入らない）

1日分を見る:

```bash
python -c "from src.core import config as cfg; from src.core import risk_profile as rp; from src.data import database as db; cfg.load('config.yaml'); rp.load('risk_profile.json'); db.init(); from src.strategy import shadow; print(shadow.load_shadow_comparisons('shadow-2026-09-24').to_string())"
```

期間をまとめて集計する（日付範囲を書き換えて使う）:

```bash
python -c "import pandas as pd; from src.core import config as cfg; from src.core import risk_profile as rp; from src.data import database as db; cfg.load('config.yaml'); rp.load('risk_profile.json'); db.init(); from src.strategy import shadow; rows=[r for d in pd.bdate_range('2026-09-22','2026-12-30') for r in shadow.load_shadow_comparisons(f'shadow-{d.date()}').to_dict('records')]; print(shadow.disagreement_summary([shadow.ShadowComparison(event_id=r['event_id'], current_probability=r['current_probability'], candidate_probability=r['candidate_probability'], current_takes=bool(r['current_takes']), candidate_takes=bool(r['candidate_takes']), agreement=r['agreement']) for r in rows]))"
```

出力の見方:

| キー | 意味 |
| --- | --- |
| `both_take` | 現行も候補も「採る」と判断した件数 |
| `both_skip` | どちらも見送った件数 |
| `only_current` | 現行だけが採ろうとした件数 |
| `only_candidate` | 候補だけが採ろうとした件数 |
| `agreement_rate` | 一致率（`(both_take + both_skip) / n`） |

**読み方の注意**:

- **これは勝ち負けの記録ではない。** 「どちらが当たったか」は実績
  （`PredictionOutcome`）が要るが、現時点では保存していない。最大保有10営業日
  経たないと確定しないため、今回の配線には含めていない。いま読めるのは
  「候補が現行とどれだけ違う判断をしたか」だけである。
- `only_current` / `only_candidate` が極端に片側へ寄っているときは、能力の差
  ではなく**閾値の差**を見ている可能性が高い（候補の評価fold別閾値は
  0.35〜1.0とばらついていた）。
- 昇格済み候補の評価AUCは平均0.5032で、chanceと統計的に区別できていない
  （`docs/kabu-auto-ml-real-data-comparison_20260921.md`）。数字が出たこと
  自体は運用へ入れる理由にならない。
- **観察期間の目安は最低6ヶ月、できれば1年。** 1日あたりの記録は銘柄数ぶん
  （数十件）しかなく、日足スイングの決着は1件あたり最大10営業日かかる。
  数週間の記録で判断しない。
- 記録が0件の日が続くときは、①候補が昇格されているか（`models/current.json`）、
  ②`signal_scan` が動いているか（ログの「シグナルスキャン開始」）、
  ③ログに「shadow記録を行いません」「shadow記録の準備に失敗しました」が
  出ていないか、の順に見る。
````

- [ ] **Step 2: 追記した節の内容が実装と食い違っていないことを確かめる**

Run:
```bash
python -c "from src.core import config as cfg; from src.core import risk_profile as rp; from src.data import database as db; cfg.load('config.yaml'); rp.load('risk_profile.json'); db.init(); from src.strategy import shadow; print(shadow.load_shadow_comparisons('shadow-2026-09-24').to_string())"
```
Expected: 例外を出さずに空のDataFrame（まだ記録が無いため）を表示すること。**本番DBを読むだけで書き込みは起きない**が、不安なら先に `data/` のバックアップを取ってから実行する（Knowledge.md §3）。

- [ ] **Step 3: コミット**

```bash
git add docs/運用Runbook.md
git commit -m "$(cat <<'EOF'
docs(runbook): shadow記録の読み方と観察期間の目安を追記

昇格済み候補は実発注に使わず、shadow_comparisons へ日次で並行記録する
運用であることと、load_shadow_comparisons/disagreement_summary の
使い方・注意点（勝敗ではない・最低6ヶ月）をまとめた。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: 最終確認**

Run: `pytest -q`
Expected: 全件 PASS

Run: `git log --oneline main..HEAD`
Expected: 6コミット（Task 1〜6）

---

## 今回作らないもの（将来の課題）

意図的にスコープ外とした。**次に着手するときのために理由ごと残す。**

1. **shadow記録した判断の実績確定**（`PredictionOutcome` への保存、
   `evaluation.save_outcomes()`）。`max_holding_sessions`（10営業日）分の
   時間が経ってから初めて意味を持つため、今回は作らない。結合キーは
   `(label_contract_id, event_id)` で、両方とも今回の記録に入っている。
   なおラベル契約IDは設定（退出ポリシー・コスト・risk_profile）が変わると
   別物になるので、実績を付けるときは**記録時の契約IDで**照合すること。
2. **shadow記録の結果を見て次に何をするかの判断ロジック**（昇格取り消し・
   別モデルの試行等）。人間が Runbook の手順で見る運用とし、自動化しない。
3. **ダッシュボードへの可視化**。`load_shadow_comparisons()` を直接叩けば
   確認できるので必須にしない。
4. **`evaluation_runs` テーブルへのshadowバッチの行**。今回は
   `predictions` / `shadow_comparisons` の `evaluation_run_id` 列だけで参照する
   （段階Eの `record_shadow()` と同じ扱い）。期間集計を頻繁にやるように
   なったら、日次の run を列挙できる索引として足すことを検討する。
5. **観察対象を「rule閾値を超えた候補」に絞ること**。今回は広く観察する
   ために全銘柄を対象にしている。件数が問題になるなら絞り込みを検討する
   （1日数十件・年間1万件強の見込みなので当面は不要）。
