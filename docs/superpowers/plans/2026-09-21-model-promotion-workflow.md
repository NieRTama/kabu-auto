# 候補モデル昇格ワークフロー（学習→評価→昇格）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `v2_training.train_v2()` が作る候補モデルを、`evaluation.run_evaluation()` で評価し `promotion.promote()` で昇格させるまでの一連の運用手順を、人間が手で叩く単一CLI `scripts/promote_workflow.py`（train / evaluate / promote の3サブコマンド）として実装する。

**Architecture:** 既存の段階A〜Eの関数（`v2_training` / `evaluation` / `promotion` / `model_store`）を**一切変更せず**、その上に薄いCLI層だけを載せる。CLIは (1) `main.py` と同じ順序での設定読み込み（`cfg.load()` → `watchlist_store.load()` → `risk_profile_store.load()` → `db.init()`）、(2) 3つの既存関数への正しい引数の受け渡し、(3) 人間が判断するための結果表示、の3つだけを担う。昇格可否の判断ロジックは既存の `promotion.check_promotable()` に完全に委ね、CLI側には二重チェックを作らない。

**Tech Stack:** Python 3.11 / argparse / pandas 2.1.4 / SQLAlchemy 2.0.23 / LightGBM 4.1.0 / pytest（すべて既存の `requirements.txt` の範囲内。**新規依存の追加は無い**）

**Spec:**
- `C:\Users\garnet\AppData\Local\Temp\claude\c--Users-garnet-kabu-auto\a32ccf89-56a0-4441-97d0-53b7122ec7f8\scratchpad\promotion-workflow-planning-brief.md`（計画作成ブリーフ＝本計画の要件定義）
- `docs/kabu-auto-ml-real-data-comparison_20260921.md`（実データでの5モデル比較・3戦略比較。risk_profile 適用の必須性の根拠）

## 背景（なぜこれを作るのか・実装者が前提として知るべきこと）

- 実データでの5モデル比較（独立6期間）の結果、現行モデル相当（LightGBM）のAUCは **0.5018** で、0.5と統計的に区別できない（t検定 **p=0.7231**）。「weighted_blend が一貫して優れている」という以前の主張は**撤回済み**（`docs/kabu-auto-ml-real-data-comparison_20260921.md`）。
- ユーザーの意図は「候補モデルを実際に昇格させたい」だが、`promotion.check_promotable()` が要求する `evaluation_run_id`（実績が確定した評価記録）を作る経路が存在しない。そこで**「今すぐ昇格」ではなく「昇格ワークフローを先に作る」**ことで合意した。
- **本計画のスコープに「実際に昇格を実行すること」は含まない。** このワークフローが完成しても、実際に昇格するかどうかは評価結果を見てからユーザーが別途判断する。実装者は `promote` サブコマンドを**作る**だけで、本番の `models/` に対して**実行してはならない**。
- `engine_version: v2` への切替自体は実際の発注ロジックに影響しない（週次再学習の経路とダッシュボードの手動バックテストにしか関係しない）。本計画は engine_version には一切触らない。

## Global Constraints

すべてのタスクの要件に、この節が暗黙に含まれる。

- **スケジューラ（`src/core/scheduler.py`）へは絶対に登録しない。** `src/services/trading.py` の `TradingServices.ml_retrain()` も変更しない。人間が手動で叩くツールに限定する。
- `promote` サブコマンドは `--reason` と `--decided-by` を**必須のコマンドライン引数**にし、空文字（空白のみを含む）なら実行前に拒否する。
- `promote` は `check_promotable()` の検査に通らなければ現行モデルを一切変更せず例外で終わる、という既存の安全機構を**そのまま活かす**。CLI側で二重チェックを作り込まない。
- 本番の `config.yaml` と取引関連DBテーブル（`trades` / `positions` / `orders` / `signals` 等）には一切触れない。触れてよいのは `models/candidates/`・`models/current.json`（`promote` 実行時のみ）・評価系DBテーブル（`evaluation_runs` / `predictions` / `prediction_outcomes`）・`model_metrics`・`model_promotions` のみ。
- `risk_profile.json` の適用を **`main.py` と同じ順序**（`cfg.load()` → `risk_profile_store.load("risk_profile.json")`）で必ず行う。この適用漏れは実データ分析v2で結論を覆した重大な誤りだった（イベント総数が 5,484件 → 12,033件 と2.2倍ずれた。`docs/kabu-auto-ml-real-data-comparison_20260921.md` §1）。
- 新規のDB書き込み（`ModelMetrics` / `evaluation_runs` / `predictions` / `prediction_outcomes` / `model_promotions`）は既存のテーブル・関数をそのまま使い、**スキーマ変更をしない**。
- ファイルは **UTF-8 BOM無し・LF**。
- テストは `pytest tests/<file>.py -v` で実行する。**ネットワークへ出るテストを書かない。**
- 実装のコミットメッセージ末尾に `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` を付ける。

## File Structure

| ファイル | 役割 | 新規/変更 |
|---|---|---|
| `.gitignore` | 生成物（イベント表・候補モデル・現行参照）の追跡漏れを是正する | 変更 |
| `scripts/promote_workflow.py` | CLI本体。3サブコマンドと設定読み込み（`_bootstrap`）だけを持つ。判断ロジックは持たない | 新規 |
| `tests/test_promote_workflow.py` | CLIの単体テスト＋train→evaluate→promote の統合テスト | 新規 |
| `docs/運用Runbook.md` | 手動での学習・評価・昇格の手順を追記（末尾に節を追加） | 変更 |

`scripts/__init__.py` は既に存在するため、CLIは `python -m scripts.promote_workflow ...` で起動する（`python scripts/promote_workflow.py` だと `sys.path[0]` が `scripts/` になり `import src` が失敗する）。

## 実装者が参照する既存コードの正確な情報

**すべて実在確認済み。推測で書き換えないこと。**

### `src/strategy/v2_training.py`
- `train_v2(ohlcv_by_symbol: dict, *, policy_conf, costs, window_sessions: Optional[int] = None, trigger: str = "weekly_schedule", base_dir: str = "models") -> V2TrainingResult`（`src/strategy/v2_training.py:66-153`）
- `V2TrainingResult`（`:30-38`）: `model_id: Optional[str]` / `dataset_id: str` / `n_events: int` / `n_resolved: int` / `positive_rate: Optional[float]` / `skipped_reason: Optional[str] = None`
- `MIN_RESOLVED_EVENTS = 200`（`:27`）未達だと `model_id=None` で返る（**例外にはならない**）。
- 内部で `ds.save_events(events, dataset_id)` を **base_dir 省略**で呼ぶ（`:89`）ため、既定の相対パス `data/datasets/` へ書き込む。**これは本番の実運用として正しい動作**（段階F・Task2 のレビューで問題視されたのは「テストが誤ってこの経路を通りリポジトリを汚す」ケース）。ただし `.gitignore` に `data/datasets/` が無いのでTask 1で是正する。
- 学習に使うモデルクラスは `evaluation.CurrentLightGBM`（`_fit_candidate`、`:41-46`）。
- 保存するメタの `feature_cols` は `list(FEATURE_COLS)`、`label_contract_id` は `ds.make_label_contract_id(policy_conf, costs)`（`:124-137`）。

### `src/strategy/evaluation.py`
- `run_evaluation(events: pd.DataFrame, *, model_factories: Optional[dict] = None, n_splits: int = 5, window_sessions: Optional[int] = None, feature_cols: Optional[list] = None, evaluation_run_id: Optional[str] = None, persist: bool = True) -> dict`（`:754-851`）
  - 戻り値は `{"evaluation_run_id", "fold_results", "summary", "degraded_reasons"}`（**`degraded_reasons` も返る**）。
  - `summary` は `pd.DataFrame`。列は `fold_index` / `model_id` / `n_train` / `n_val` / `train_positive_rate` / `threshold` に加え、`compute_metrics()`（`:394-436`）の全キー = `n` / `positive_rate` / `roc_auc` / `average_precision` / `log_loss` / `brier` / `brier_vs_constant` / `log_loss_vs_constant`。`roc_auc` と `average_precision` は検証側が片側クラスだと `None`。
  - `persist=True` のとき `save_predictions()` / `save_outcomes(events)` / `save_evaluation_run()` を呼ぶ（`:810-828`）。**本ツールは評価記録を残すのが目的なので `persist=True`（既定）のまま使う。** 前回の分析スクリプトが `persist=False` だったのはオフライン分析用。
  - `save_evaluation_run(..., model_id=(list(factories)[0] if len(factories) == 1 else None), ...)`（`:825`）。つまり **`model_factories` を1件だけ渡せば `EvaluationRun.model_id` に候補のIDが入る。**
- `default_model_factories() -> dict`（`:370-382`）は `{model_id: モデルクラス}`（**呼び出し可能なクラスそのもの**が value）。
- **候補を評価する際は `model_factories={candidate_model_id: CurrentLightGBM}` のように、候補の実際の model_id をキーにして渡すこと。** `check_promotable()` が `load_prediction_details(evaluation_run_id, model_id=model_id)` でこの鍵と突き合わせるため、`"current_lightgbm"` のような汎用名で保存すると予測明細が見つからず永久に昇格できない。
- `CurrentLightGBM`（`:361-367`）は `src.strategy.evaluation` から直接 import 可能。
- `capture_run_config(events, *, n_splits, window_sessions, feature_cols) -> RunConfig`（`:655-689`）。`RunConfig.label_contract_id` は events の `label_contract_id` 列が単一値のときだけその値、複数値なら `None`。
- `save_predictions(predictions, evaluation_run_id, model_id, *, purpose=PURPOSE_VALIDATION) -> int`（`:41-98`）。`predictions` は `event_id` / `label_contract_id` / `raw_probability` / `calibrated_probability` / `fold_index` 列が必須。
- `save_outcomes(events) -> int`（`:101-141`）。`events["status"] == STATUS_RESOLVED` の行だけ保存する。
- `save_evaluation_run(evaluation_run_id, run_config, *, purpose, model_id, n_folds, n_predictions, degraded_reasons=None) -> None`（`:692-726`）。`degraded_reasons` が空でなければ `EvaluationRun.degraded = 1` になる。
- `PURPOSE_VALIDATION = "validation"`（`:37`）/ `PURPOSE_SHADOW = "shadow"`（`:38`）
- 評価対象の events は**過去データ**（walk-forward の各fold検証区間は既に確定済みの実績を持つ）なので、実績確定を待つ必要はない。`check_promotable` の「実績が1件も確定していない」拒否条件は自然に満たされる。

### `src/strategy/promotion.py`
- `check_promotable(model_id: str, *, evaluation_run_id: Optional[str], degraded: bool, base_dir: str, expected_feature_cols: list) -> PromotionCheck`（`:28-142`）。`PromotionCheck.ok: bool` / `PromotionCheck.blockers: list`。
- `promote(model_id: str, *, evaluation_run_id: Optional[str], decided_by: str, reason: str, degraded: bool, base_dir: str, expected_feature_cols: list) -> int`（`:284-349`）。`decided_by` / `reason` が空文字なら `ValueError`。検査に落ちると `ValueError("昇格できません: " + " / ".join(check.blockers))`。成功時は `ModelPromotion.id` を返す。
- 拒否条件（`:53-142`）: 候補として保存されていない / 既に現行と同じ model_id / 引数 `degraded=True` / `evaluation_run_id` なし / 評価実行の記録が無い / **保存済み記録が degraded** / モデルにラベル契約IDが無い / ラベル契約の不一致 / 予測明細が無い（または別モデルの実行） / **実績が1件も確定していない** / shadow記録のみ / 特徴量定義の不一致。
- 引数の `degraded` は**補助的な早期拒否**にすぎず、真偽は保存済みの `EvaluationRun.degraded` から読む（`:74-82`）。**CLIは常に `degraded=False` を渡してよい。**
- `PROMOTION_PENDING = "pending"` / `PROMOTION_COMMITTED = "committed"` / `PROMOTION_FAILED = "failed"`（`:147-149`）。

### `src/strategy/model_store.py`
- `read_meta(model_id: str, *, base_dir: str = "models") -> ModelMeta`（`:191-195`）。存在しなければ `FileNotFoundError`。
- `ModelMeta`（`:45-68`）の使用フィールド: `model_id` / `trained_at: datetime` / `training_window_sessions: Optional[int]` / `feature_cols: list` / `dataset_id: Optional[str]` / `label_contract_id: Optional[str]`。
- `candidate_dir(model_id, base_dir="models") -> Path` = `Path(base_dir) / "candidates" / model_id`（`:71-72`）。
- `read_current(*, base_dir="models") -> Optional[CurrentRef]`（`:273-287`）。`CurrentRef` は `model_id` / `previous_model_id` / `switched_at` を持つ。
- `CURRENT_REF = "current.json"` / `CANDIDATES_DIR = "candidates"` / `MODEL_FILE = "model.txt"` / `META_FILE = "meta.json"`（`:33-37`）。

### 設定・データ取得
- `src/core/config.py`: `load(path="config.yaml") -> dict` / `get_section(section: str) -> dict`
- `src/core/watchlist.py`: `load(path="watchlists.json", legacy_path="watchlist.json") -> dict`（`:92`）/ `get_all_codes() -> list[str]`（`:204`。全リストの銘柄を重複除去して返す）
- `src/core/risk_profile.py`: `load(path="risk_profile.json") -> str`（`:161-201`）。`config.yaml` の `trading` / `strategy` 節をアクティブなプロファイル（現状 `high_risk`）で**プロセス内メモリ上でのみ**上書きし、プロファイル名を返す。`_persist()` は呼ばれないので `risk_profile.json` 自体は書き換わらない。
- `src/data/database.py`: `init() -> None`（`:598`）/ `get_session()`（`:667`）
- `src/data/market_data.py`: `load_ohlcv(symbol: str, limit: int = 500, price_basis: str = "adjusted") -> pd.DataFrame`（`:206`）。`limit=None` を渡すと SQLAlchemy の `.limit(None)` になり LIMIT 句が付かない（＝全期間）。
- `src/strategy/policy.py`: `config_from_settings() -> PolicyConfig`（`:103-117`）。`PolicyConfig(stop_loss_pct, breakeven_trigger_pct, trailing_stop_pct, sell_threshold, max_holding_sessions)`。`stop_loss_pct` / `breakeven_trigger_pct` / `trailing_stop_pct` は `trading` 節、`sell_threshold` / `tb_max_holding` は `strategy` 節から読む。
- `src/backtest/execution.py`: `config_from_settings() -> CostConfig`（`:43-49`）。`CostConfig(slippage_pct, commission_pct)` を `backtest` 節から読む。
- `src/strategy/indicators.py`: `FEATURE_COLS`（`:123`）。`promote` の `expected_feature_cols` にはこれを渡す（`train_v2` が `meta.feature_cols = list(FEATURE_COLS)` を書くため）。

### 参考（**変更してはならない**）
- `src/services/trading.py:317-377` の `TradingServices.ml_retrain()` の v2 分岐。ウォッチリスト全銘柄のOHLCV取得（`load_ohlcv(sym)` → `len(df) < 200` で除外）・`policy.config_from_settings()` / `execution.config_from_settings()` の呼び方・`window_sessions = cfg.get_section("backtest").get("retrain_window_sessions", None)` の読み方の参考になる。**このメソッド自体は1行も変更しない。**

### 設計上の確定事項（実装者が迷わないための決定）

1. **`--ohlcv-limit` の既定値は 0（＝全期間、`limit=None`）。** `ml_retrain()` の `load_ohlcv(sym)` 既定値500本は、`validation.apply_training_window()` のdocstringが明記する通り「意図しない切り詰め」（レビューF07）であり、v2が是正しようとしていた対象そのものである。新規ワークフローの既定を同じ500本にすると同じ問題を持ち込むことになるため、既定は全期間とする。分析報告書（`docs/kabu-auto-ml-real-data-comparison_20260921.md`）の5モデル比較・3戦略比較もすべて全期間（2019-06-19〜2026-09-18・62銘柄）で行っており、候補モデルをその分析結果と比較可能にする意味でも全期間が既定として妥当。`--ohlcv-limit N`（N>0）を指定すれば直近N本に絞ることもできる（動作確認や高速な試行用）。
2. **`ModelMetrics.trigger` は `String(20)`**（`src/data/database.py:404`）。CLIから起動した学習の trigger は `"manual_workflow"`（15文字）にする。`"weekly_schedule"`（週次）とも legacy の `"manual"` とも区別できる。
3. **`evaluate` の events は再構築せず、学習時に保存されたイベント表を読み直す。** `ms.read_meta(model_id).dataset_id` → `ds.load_events(dataset_id)`。こうすると候補が学習したものと完全に同一のイベント表で評価でき、`label_contract_id` の一致も構造的に保証される（`check_promotable` のラベル契約検査を確実に通せる）。
4. **`evaluate` の `window_sessions` / `feature_cols` は `ModelMeta` から取る**（`meta.training_window_sessions` / `list(meta.feature_cols)`）。config を読み直すと、学習後に設定が変わっていた場合に候補と評価の条件がずれる。
5. **`evaluate` の `--n-splits` 既定は 5。** `train_v2` が `validation.calendar_folds(events, n_splits=5)` をハードコードしている（`src/strategy/v2_training.py:104`）ため、5 にしておくと候補の学習時と同じ fold 構造になる。

---

## Task 1: `.gitignore` の是正（生成物の追跡漏れ）

**Files:**
- Modify: `.gitignore:26-31`（`models/lgb_model.meta.json` の行の直後に追記）

**Interfaces:**
- Consumes: なし
- Produces: `data/datasets/`・`models/candidates/`・`models/current.json` が git 管理外になる。Task 2以降のテストと実運用がリポジトリを汚さない前提を作る。

理由: `train_v2()` は `data/datasets/<dataset_id>.csv.gz` を、`save_candidate()` は `models/candidates/<model_id>/{model.txt,meta.json}` を、`promote()` は `models/current.json` を書く。現状 `.gitignore` にはこのいずれも無く（`models/*.pkl` と `models/lgb_model.meta.json` だけがある）、CLIを1回叩くだけで `git status` が汚れて誤コミットの温床になる。`models/current.json` を無視するのは `models/lgb_model.meta.json` と全く同じ理由（昇格のたびに書き換わる運用状態であり、追跡すると常時dirtyになる）。

- [ ] **Step 1: 現状を確認する（追跡されていないことの確認）**

Run:
```bash
git check-ignore -v data/datasets/x.csv.gz models/candidates/v2-x/meta.json models/current.json
```
Expected: 何も出力されず終了コード1（＝どれも無視されていない）

- [ ] **Step 2: `.gitignore` に追記する**

`.gitignore` の以下の行（現状の26-31行目付近）:

```
models/*.pkl
# lgb_model.meta.json は週次再学習のたびに本番プロセスが書き換えるため、
# 追跡し続けると git status が常時dirtyになる（誤コミットの温床）。
# 2026-09-14に一度スナップショットとしてコミット済み（047fffd）で、
# 過去の履歴はそちらに残る。以降は追跡しない。
models/lgb_model.meta.json
```

の直後に、次の6行を挿入する:

```
# v2の生成物（イベント表・候補モデル・現行参照）。
# data/datasets/ は train_v2() が dataset_id ごとに .csv.gz を書き足す。
# models/candidates/ は候補1件につき1ディレクトリ（model.txt / meta.json）。
# models/current.json は現行モデルの参照で、昇格のたびに書き換わる運用状態。
# lgb_model.meta.json と同じ理由で追跡しない。
data/datasets/
models/candidates/
models/current.json
```

- [ ] **Step 3: 無視されることを確認する**

Run:
```bash
git check-ignore -v data/datasets/x.csv.gz models/candidates/v2-x/meta.json models/current.json
```
Expected: 3行とも `.gitignore:<行番号>:<パターン>	<パス>` の形で出力され、終了コード0

Run:
```bash
git status --porcelain
```
Expected: `.gitignore` の変更1行（` M .gitignore`）のみ

- [ ] **Step 4: コミット**

```bash
git add .gitignore
git commit -m "$(cat <<'EOF'
chore: v2の生成物(data/datasets・models/candidates・current.json)を.gitignoreへ追加

train_v2()/save_candidate()/promote() が書き出す生成物がいずれも未追跡のまま
git status に現れる状態だった。models/lgb_model.meta.json と同じ理由で除外する。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: CLIの骨格・設定読み込み（`_bootstrap`）・`train` サブコマンド

**Files:**
- Create: `scripts/promote_workflow.py`
- Test: `tests/test_promote_workflow.py`

**Interfaces:**
- Consumes: Task 1 の `.gitignore`（テストが生成物を作るため）
- Produces:
  - `scripts/promote_workflow.TRIGGER_MANUAL: str = "manual_workflow"`
  - `scripts/promote_workflow.EXIT_OK: int = 0` / `EXIT_ERROR: int = 1`
  - `scripts/promote_workflow._bootstrap(config_path: str = "config.yaml") -> str`（アクティブなリスクプロファイル名を返す）
  - `scripts/promote_workflow._collect_ohlcv(limit: Optional[int]) -> tuple[dict, list]`（`({symbol: DataFrame}, スキップ理由の文字列リスト)`）
  - `scripts/promote_workflow.cmd_train(args) -> int`
  - `scripts/promote_workflow.build_parser() -> argparse.ArgumentParser`
  - `scripts/promote_workflow.main(argv: Optional[list] = None) -> int`
  - Task 3 は `build_parser()` に `evaluate` サブパーサを、Task 4 は `promote` サブパーサを追加する。共通オプション `--config` / `--base-dir` は**親パーサ**に付く（＝サブコマンド名より前に書く）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_promote_workflow.py` を新規作成する（全文）:

```python
"""scripts/promote_workflow.py（候補モデルの学習→評価→昇格CLI）のテスト

このCLIは**人間が手で叩くときだけ動く**。自動昇格を実装しないという段階Eの
契約（src/strategy/promotion.py のdocstring）を、スケジューラへ登録されて
いないことのテストで固定する。
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from scripts import promote_workflow as pw
from src.core import config as cfg
from src.data import database as db


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """本番DBもリポジトリも汚さないための隔離。

    dataset.save_events() は base_dir 既定値（"data/datasets"、相対パス）で
    呼ばれる（train_v2() は base_dir をモデル保存にしか渡さない）。chdir
    せずに実行するとリポジトリ直下の data/datasets/ へ .csv.gz が生成され
    続けるため、cfg.load / db.init の後に tmp_path へ chdir して相対パス
    書き込みを閉じ込める（tests/test_v2_training.py と同じ対処）。
    """
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _ohlcv(n=300, start_price=1000.0, seed=0):
    """合成OHLCV（tests/test_v2_training.py の _ohlcv と同じ生成則）"""
    rng = np.random.default_rng(seed)
    start = date(2025, 1, 6)
    rows, p = [], start_price
    for i in range(n):
        p *= 1 + rng.normal(0, 0.01)
        rows.append({"date": start + timedelta(days=i), "open": p,
                     "high": p * 1.01, "low": p * 0.99, "close": p,
                     "volume": 1_000_000})
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


class TestSafety:
    def test_not_registered_in_scheduler_or_trading_services(self):
        """人手専用。自動実行される経路を1つも作らないこと"""
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for rel in ("src/core/scheduler.py", "src/services/trading.py", "main.py"):
            text = (root / rel).read_text(encoding="utf-8")
            assert "promote_workflow" not in text, rel


class TestBootstrap:
    def test_loads_config_in_the_same_order_as_main_py(self, monkeypatch):
        """cfg.load → watchlist.load → risk_profile.load → db.init の順

        risk_profile.load() は cfg.load() が読んだ trading/strategy 節を
        high_risk の値で上書きする。順序が崩れると config.yaml の素の値で
        ラベルが作られ、本番と別のタスクを学習してしまう
        （docs/kabu-auto-ml-real-data-comparison_20260921.md §1）。
        """
        from src.core import config as cfg_mod
        from src.core import risk_profile as rp_mod
        from src.core import watchlist as wl_mod
        from src.data import database as db_mod

        calls = []

        def fake_rp_load(path="risk_profile.json"):
            calls.append(f"risk_profile.load:{path}")
            return "high_risk"

        monkeypatch.setattr(cfg_mod, "load",
                            lambda path="config.yaml": calls.append(f"cfg.load:{path}"))
        monkeypatch.setattr(wl_mod, "load",
                            lambda path="watchlists.json", legacy_path="watchlist.json":
                            calls.append(f"watchlist.load:{path}"))
        monkeypatch.setattr(rp_mod, "load", fake_rp_load)
        monkeypatch.setattr(db_mod, "init", lambda: calls.append("db.init"))

        assert pw._bootstrap("config.yaml") == "high_risk"
        assert calls == [
            "cfg.load:config.yaml",
            "watchlist.load:watchlists.json",
            "risk_profile.load:risk_profile.json",
            "db.init",
        ]


class TestCollectOhlcv:
    def test_skips_symbols_with_fewer_than_200_bars(self, monkeypatch):
        """特徴量の助走が足りない銘柄は落とす（ml_retrain と同じ基準）"""
        from src.core import watchlist as wl_mod
        from src.data import market_data

        frames = {"7203": _ohlcv(n=300), "6758": _ohlcv(n=199),
                  "9984": _ohlcv(n=200)}
        monkeypatch.setattr(wl_mod, "get_all_codes",
                            lambda: ["7203", "6758", "9984"])
        monkeypatch.setattr(market_data, "load_ohlcv",
                            lambda symbol, limit=500: frames[symbol])

        got, skipped = pw._collect_ohlcv(500)
        assert sorted(got) == ["7203", "9984"]
        assert skipped == ["6758(199本)"]

    def test_passes_the_limit_through_and_keeps_going_on_failure(self, monkeypatch):
        """limit をそのまま渡す。1銘柄の読み込み失敗で全体を止めない"""
        from src.core import watchlist as wl_mod
        from src.data import market_data

        seen = []

        def fake_load(symbol, limit=500):
            seen.append((symbol, limit))
            if symbol == "6758":
                raise RuntimeError("DB読み込み失敗")
            return _ohlcv(n=300)

        monkeypatch.setattr(wl_mod, "get_all_codes", lambda: ["7203", "6758"])
        monkeypatch.setattr(market_data, "load_ohlcv", fake_load)

        got, skipped = pw._collect_ohlcv(None)
        assert seen == [("7203", None), ("6758", None)]
        assert list(got) == ["7203"]
        assert skipped == ["6758(読み込み失敗: DB読み込み失敗)"]


class TestTrain:
    def test_builds_policy_conf_after_the_risk_profile_is_applied(
            self, isolated_db, tmp_path, monkeypatch):
        """risk_profile 適用**後**の値で policy_conf / costs を作ること

        適用漏れだと sell_threshold が config.yaml の素の値のままになり、
        ラベル定義そのものが本番と別物になる（分析報告書 §1: イベント総数が
        5,484件→12,033件と2.2倍ずれた）。_bootstrap の中で trading/strategy
        節が書き換わる状況を再現し、その後に policy_conf が作られることを固定する。
        """
        from src.strategy import v2_training

        def fake_bootstrap(config_path="config.yaml"):
            cfg.get_section("trading")["stop_loss_pct"] = -0.10
            cfg.get_section("strategy")["sell_threshold"] = -0.08
            cfg.get_section("backtest")["retrain_window_sessions"] = None
            cfg.get_section("backtest")["slippage_pct"] = 0.001
            return "high_risk"

        monkeypatch.setattr(pw, "_bootstrap", fake_bootstrap)
        monkeypatch.setattr(pw, "_collect_ohlcv",
                            lambda limit: ({"7203": _ohlcv()}, []))

        captured = {}

        def fake_train_v2(ohlcv_by_symbol, **kwargs):
            captured["ohlcv"] = ohlcv_by_symbol
            captured.update(kwargs)
            return v2_training.V2TrainingResult(
                model_id="v2-20260921T120000-abcdef12", dataset_id="ds000001",
                n_events=300, n_resolved=250, positive_rate=0.47)

        monkeypatch.setattr(v2_training, "train_v2", fake_train_v2)

        base = str(tmp_path / "models")
        assert pw.main(["--base-dir", base, "train"]) == 0
        assert captured["policy_conf"].stop_loss_pct == pytest.approx(-0.10)
        assert captured["policy_conf"].sell_threshold == pytest.approx(-0.08)
        assert captured["costs"].slippage_pct == pytest.approx(0.001)
        assert captured["window_sessions"] is None
        assert captured["trigger"] == "manual_workflow"
        assert captured["base_dir"] == base
        assert list(captured["ohlcv"]) == ["7203"]

    def test_ohlcv_limit_zero_means_full_history(
            self, isolated_db, tmp_path, monkeypatch):
        """--ohlcv-limit 0 は limit=None（LIMIT句なし＝全期間）へ変換する"""
        from src.strategy import v2_training

        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        seen = {}

        def fake_collect(limit):
            seen["limit"] = limit
            return {"7203": _ohlcv()}, []

        monkeypatch.setattr(pw, "_collect_ohlcv", fake_collect)
        monkeypatch.setattr(
            v2_training, "train_v2",
            lambda ohlcv, **kw: v2_training.V2TrainingResult(
                model_id="v2-x", dataset_id="ds1", n_events=1, n_resolved=1,
                positive_rate=0.5))

        assert pw.main(["--base-dir", str(tmp_path / "models"),
                        "train", "--ohlcv-limit", "0"]) == 0
        assert seen["limit"] is None

    def test_returns_error_when_no_candidate_was_produced(
            self, isolated_db, tmp_path, monkeypatch):
        """決着イベント不足は例外にならず model_id=None で返る。終了コード1にする"""
        from src.strategy import v2_training

        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(pw, "_collect_ohlcv",
                            lambda limit: ({"7203": _ohlcv()}, []))
        monkeypatch.setattr(
            v2_training, "train_v2",
            lambda ohlcv, **kw: v2_training.V2TrainingResult(
                model_id=None, dataset_id="ds000001", n_events=300,
                n_resolved=51, positive_rate=None,
                skipped_reason="決着したイベントが不足しています: 51件 < 200件"))

        assert pw.main(["--base-dir", str(tmp_path / "models"), "train"]) == 1

    def test_returns_error_when_no_symbol_has_enough_history(
            self, isolated_db, tmp_path, monkeypatch):
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(pw, "_collect_ohlcv", lambda limit: ({}, ["7203(10本)"]))

        assert pw.main(["--base-dir", str(tmp_path / "models"), "train"]) == 1
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `pytest tests/test_promote_workflow.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'scripts.promote_workflow'`

- [ ] **Step 3: `scripts/promote_workflow.py` を実装する**

新規作成（全文）:

```python
"""候補モデルの学習→評価→昇格を人手で回すCLI。

段階A〜Eで `v2_training.train_v2()`（候補の保存）・
`evaluation.run_evaluation()`（評価記録の保存）・`promotion.promote()`
（現行の切替）は揃ったが、この3つを繋ぐ運用手順がどこにも無かった。
本スクリプトがその手順を1本のCLIにする。

**このスクリプトをスケジューラ（src/core/scheduler.py）へ登録しては
ならない。** 自動昇格を実装しないというのが段階Eの契約
（src/strategy/promotion.py のdocstring）であり、本スクリプトは人間が
手で叩くときだけ動く。判断ロジックもここには置かない。昇格可否は
`promotion.check_promotable()` が唯一の判断者である。

使い方:

    python -m scripts.promote_workflow train
    python -m scripts.promote_workflow evaluate <model_id>
    python -m scripts.promote_workflow promote <model_id> <evaluation_run_id> \\
        --reason "..." --decided-by "..."

共通オプション（--config / --base-dir）はサブコマンド名より**前**に書く:

    python -m scripts.promote_workflow --base-dir models train
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    # Windowsのcp932環境で日本語出力が UnicodeEncodeError で落ちるのを防ぐ
    # （scripts/gen_graph_notes.py と同じ対処）
    sys.stdout.reconfigure(encoding="utf-8")


# ModelMetrics.trigger は String(20)（src/data/database.py:404）。
# 週次の "weekly_schedule" とも legacy の "manual" とも区別できる名前にする。
TRIGGER_MANUAL = "manual_workflow"

EXIT_OK = 0
EXIT_ERROR = 1


def _bootstrap(config_path: str = "config.yaml") -> str:
    """main.py と同じ順序で設定を読み、DBを初期化する。アクティブなプロファイル名を返す。

    **順序を変えてはならない。** `risk_profile_store.load()` は `cfg.load()`
    が読んだ `trading` / `strategy` 節をアクティブプロファイル（現状
    high_risk）の値でプロセス内メモリ上だけ上書きする。この適用を飛ばすと
    `policy.config_from_settings()` が本番と別の閾値を返し、
    `dataset.build_events_multi()` が作るラベルそのものが本番と別物になる
    （docs/kabu-auto-ml-real-data-comparison_20260921.md §1: 適用漏れで
    イベント総数が 5,484件 → 12,033件 と2.2倍ずれ、結論が覆った）。

    `risk_profile.load()` は `_persist()` を呼ばないので `risk_profile.json`
    自体は書き換わらない（読むだけ）。
    """
    from src.core import config as cfg
    from src.core import risk_profile as risk_profile_store
    from src.core import watchlist as watchlist_store
    from src.data import database as db

    cfg.load(config_path)
    watchlist_store.load("watchlists.json")
    profile = risk_profile_store.load("risk_profile.json")
    db.init()
    return profile


# ─── train ────────────────────────────────────────────────────────────────


def _collect_ohlcv(limit: Optional[int]) -> tuple:
    """ウォッチリスト全銘柄のOHLCVを集める。

    `TradingServices.ml_retrain()` の v2 分岐（src/services/trading.py:323-333）
    と同じ集め方にする。200本未満の銘柄は特徴量の助走が足りないので落とし、
    1銘柄の読み込み失敗で全体を止めない。

    戻り値: `({symbol: DataFrame}, スキップ理由の文字列リスト)`
    """
    from src.core import watchlist as watchlist_store
    from src.data import market_data

    ohlcv_by_symbol: dict = {}
    skipped: list = []
    for symbol in watchlist_store.get_all_codes():
        try:
            df = market_data.load_ohlcv(symbol, limit=limit)
        except Exception as e:
            skipped.append(f"{symbol}(読み込み失敗: {e})")
            continue
        if len(df) < 200:
            skipped.append(f"{symbol}({len(df)}本)")
            continue
        ohlcv_by_symbol[symbol] = df
    return ohlcv_by_symbol, skipped


def cmd_train(args) -> int:
    """ウォッチリスト全銘柄の実データで候補モデルを学習・保存する。

    **運用モデルは変更しない。** `train_v2()` は候補を保存して終わる。
    """
    from src.backtest import execution
    from src.core import config as cfg
    from src.strategy import policy
    from src.strategy import v2_training

    profile = _bootstrap(args.config)
    limit = None if args.ohlcv_limit == 0 else args.ohlcv_limit
    ohlcv_by_symbol, skipped = _collect_ohlcv(limit)
    if not ohlcv_by_symbol:
        print("学習できる銘柄がありません（200本以上のOHLCVを持つ銘柄が0件）")
        if skipped:
            print(f"スキップ: {', '.join(skipped)}")
        return EXIT_ERROR

    # policy_conf / costs は _bootstrap() の risk_profile 適用**後**に作る。
    # 順序を逆にすると config.yaml の素の値でラベルを作ってしまう。
    policy_conf = policy.config_from_settings()
    costs = execution.config_from_settings()
    window_sessions = cfg.get_section("backtest").get("retrain_window_sessions", None)

    print(f"リスクプロファイル: {profile}")
    print(f"対象銘柄: {len(ohlcv_by_symbol)}件 / OHLCV本数上限: "
          f"{'全期間' if limit is None else limit}")
    if skipped:
        print(f"スキップ{len(skipped)}件: {', '.join(skipped)}")
    print(f"policy_conf: {policy_conf}")
    print(f"costs: {costs}")
    print(f"window_sessions: {window_sessions}（Noneは拡大窓＝切らない）")

    result = v2_training.train_v2(
        ohlcv_by_symbol,
        policy_conf=policy_conf,
        costs=costs,
        window_sessions=window_sessions,
        trigger=TRIGGER_MANUAL,
        base_dir=args.base_dir,
    )

    print(f"dataset_id: {result.dataset_id}")
    print(f"イベント {result.n_events}件 / 決着 {result.n_resolved}件"
          f"（必要 {v2_training.MIN_RESOLVED_EVENTS}件）")
    if result.model_id is None:
        print(f"候補モデルは作られませんでした: {result.skipped_reason}")
        return EXIT_ERROR

    print(f"正例率: {result.positive_rate:.4f}")
    print(f"候補モデル: {result.model_id}")
    print("運用モデルは変更していません。次は評価です:")
    print(f"  python -m scripts.promote_workflow evaluate {result.model_id}")
    return EXIT_OK


# ─── CLI ──────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promote_workflow",
        description="候補モデルの学習・評価・昇格（人手運用専用。"
                    "スケジューラへ登録しないこと）")
    parser.add_argument("--config", default="config.yaml",
                        help="設定ファイル（既定: config.yaml）")
    parser.add_argument("--base-dir", default="models",
                        help="モデルの保存先（既定: models）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser(
        "train", help="ウォッチリスト全銘柄で候補モデルを学習・保存する")
    p_train.add_argument(
        "--ohlcv-limit", type=int, default=0,
        help="銘柄あたりのOHLCV本数。0で全期間（既定: 0＝全期間。"
             "500等を指定すると直近N本に絞れるが、週次再学習の"
             "500本既定は意図しない切り詰め＝レビューF07の対象であり、"
             "本ワークフローでは踏襲しない）")
    p_train.set_defaults(func=cmd_train)

    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: テストを実行して通ることを確認する**

Run: `pytest tests/test_promote_workflow.py -v`
Expected: 8 passed

- [ ] **Step 5: CLIのヘルプが出ることを手で確認する**

Run: `python -m scripts.promote_workflow --help`
Expected: usage が表示され、`train` サブコマンドが列挙される（終了コード0）

Run: `python -m scripts.promote_workflow`
Expected: `error: the following arguments are required: command` で終了コード2

- [ ] **Step 6: リポジトリが汚れていないことを確認する**

Run: `git status --porcelain`
Expected: `?? scripts/promote_workflow.py` と `?? tests/test_promote_workflow.py` の2行のみ（`data/datasets/` や `models/candidates/` が出ないこと）

- [ ] **Step 7: コミット**

```bash
git add scripts/promote_workflow.py tests/test_promote_workflow.py
git commit -m "$(cat <<'EOF'
feat(promote_workflow): 候補モデル学習CLI（trainサブコマンド）を追加

main.py と同じ順序（cfg.load → watchlist.load → risk_profile.load → db.init）
で設定を読んでから policy_conf/costs を作る。risk_profile 適用漏れは
ラベル定義そのものを本番と別物にする（分析報告書 §1）。
スケジューラへは登録しない（自動昇格を実装しない契約）。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `evaluate` サブコマンド

**Files:**
- Modify: `scripts/promote_workflow.py`（`import pandas as pd` の追加、`cmd_train` の直後に `evaluate` 節を追加、`build_parser()` にサブパーサを追加）
- Test: `tests/test_promote_workflow.py`（`TestEvaluate` クラスを追加）

**Interfaces:**
- Consumes: Task 2 の `_bootstrap(config_path) -> str` / `EXIT_OK` / `EXIT_ERROR` / `build_parser()`
- Produces:
  - `scripts/promote_workflow._mean_or_none(series) -> Optional[float]`
  - `scripts/promote_workflow.cmd_evaluate(args) -> int`
  - `evaluate <model_id> [--n-splits N]` サブコマンド。標準出力の最後から2ブロック目に `evaluation_run_id: <id>` の行を必ず1行出す（Task 5の統合テストがこの行を読む）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_promote_workflow.py` の末尾（`class TestTrain` の後）に追加する:

```python
class TestEvaluate:
    def _meta(self):
        from datetime import datetime

        from src.strategy import model_store as ms

        return ms.ModelMeta(
            model_id="v2-20260921T120000-abcdef12",
            trained_at=datetime(2026, 9, 21, 12, 0, 0),
            training_window_sessions=None,
            feature_cols=["rsi", "ma_dev"],
            dataset_id="ds000001",
            label_contract_id="lc0001",
        )

    def test_uses_the_candidate_model_id_as_the_factory_key(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        """model_factories の鍵は候補の model_id 自身であること

        check_promotable() は load_prediction_details(run_id, model_id=候補ID)
        で突き合わせる（src/strategy/promotion.py:106）。"current_lightgbm"
        のような汎用名で保存すると予測明細が見つからず永久に昇格できない。
        """
        from src.strategy import dataset as ds
        from src.strategy import evaluation
        from src.strategy import model_store as ms

        meta = self._meta()
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(ms, "read_meta",
                            lambda model_id, base_dir="models": meta)

        stored = tmp_path / "ds000001.csv.gz"
        stored.write_bytes(b"")
        events = pd.DataFrame({"label_contract_id": ["lc0001", "lc0001"]})
        monkeypatch.setattr(
            ds, "dataset_path",
            lambda dataset_id, base_dir="data/datasets": stored)
        monkeypatch.setattr(
            ds, "load_events",
            lambda dataset_id, base_dir="data/datasets": events)

        captured = {}

        def fake_run_evaluation(ev, **kwargs):
            captured["events"] = ev
            captured.update(kwargs)
            return {
                "evaluation_run_id": "20260921T130000-0badf00d",
                "fold_results": [],
                "summary": pd.DataFrame([
                    {"fold_index": 0, "model_id": meta.model_id, "n_val": 100,
                     "roc_auc": 0.5123, "brier": 0.2476,
                     "brier_vs_constant": -0.0004},
                ]),
                "degraded_reasons": [],
            }

        monkeypatch.setattr(evaluation, "run_evaluation", fake_run_evaluation)

        assert pw.main(["evaluate", meta.model_id]) == 0
        assert captured["model_factories"] == {
            meta.model_id: evaluation.CurrentLightGBM}
        assert captured["persist"] is True
        assert captured["n_splits"] == 5
        assert captured["window_sessions"] is None
        assert captured["feature_cols"] == ["rsi", "ma_dev"]
        assert "evaluation_run_id: 20260921T130000-0badf00d" in capsys.readouterr().out

    def test_reports_degraded_reasons_and_still_succeeds(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        """degraded は評価の失敗ではない。記録は残し、昇格できない旨を伝える"""
        from src.strategy import dataset as ds
        from src.strategy import evaluation
        from src.strategy import model_store as ms

        meta = self._meta()
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(ms, "read_meta",
                            lambda model_id, base_dir="models": meta)
        stored = tmp_path / "ds000001.csv.gz"
        stored.write_bytes(b"")
        monkeypatch.setattr(
            ds, "dataset_path", lambda dataset_id, base_dir="data/datasets": stored)
        monkeypatch.setattr(
            ds, "load_events", lambda dataset_id, base_dir="data/datasets":
            pd.DataFrame({"label_contract_id": ["lc0001"]}))
        monkeypatch.setattr(
            evaluation, "run_evaluation",
            lambda ev, **kw: {
                "evaluation_run_id": "run-x", "fold_results": [],
                "summary": pd.DataFrame([{"fold_index": 0, "roc_auc": None,
                                          "brier": 0.25,
                                          "brier_vs_constant": 0.0}]),
                "degraded_reasons": ["fold 0 model=v2-x: モデルが定数に縮退"]})

        assert pw.main(["evaluate", meta.model_id]) == 0
        out = capsys.readouterr().out
        assert "degraded: 1件" in out
        assert "モデルが定数に縮退" in out
        assert "AUC平均: None" in out

    def test_warns_when_the_label_contract_does_not_match(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        from src.strategy import dataset as ds
        from src.strategy import evaluation
        from src.strategy import model_store as ms

        meta = self._meta()
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(ms, "read_meta",
                            lambda model_id, base_dir="models": meta)
        stored = tmp_path / "ds000001.csv.gz"
        stored.write_bytes(b"")
        monkeypatch.setattr(
            ds, "dataset_path", lambda dataset_id, base_dir="data/datasets": stored)
        monkeypatch.setattr(
            ds, "load_events", lambda dataset_id, base_dir="data/datasets":
            pd.DataFrame({"label_contract_id": ["OTHER"]}))
        monkeypatch.setattr(
            evaluation, "run_evaluation",
            lambda ev, **kw: {
                "evaluation_run_id": "run-x", "fold_results": [],
                "summary": pd.DataFrame([{"fold_index": 0, "roc_auc": 0.5,
                                          "brier": 0.25,
                                          "brier_vs_constant": 0.0}]),
                "degraded_reasons": []})

        assert pw.main(["evaluate", meta.model_id]) == 0
        assert "ラベル契約が一致しません" in capsys.readouterr().out

    def test_errors_when_the_candidate_is_missing(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        from src.strategy import model_store as ms

        def raise_missing(model_id, base_dir="models"):
            raise FileNotFoundError(f"メタが見つかりません: {base_dir}/candidates/{model_id}/meta.json")

        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(ms, "read_meta", raise_missing)

        assert pw.main(["evaluate", "v2-nope"]) == 1
        assert "候補モデルが見つかりません" in capsys.readouterr().out

    def test_errors_when_the_saved_event_table_is_missing(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        """イベント表が無ければ評価しない（無言で作り直して別データを評価しない）"""
        from src.strategy import dataset as ds
        from src.strategy import model_store as ms

        meta = self._meta()
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(ms, "read_meta",
                            lambda model_id, base_dir="models": meta)
        monkeypatch.setattr(
            ds, "dataset_path",
            lambda dataset_id, base_dir="data/datasets":
            tmp_path / "missing" / "ds000001.csv.gz")

        assert pw.main(["evaluate", meta.model_id]) == 1
        out = capsys.readouterr().out
        assert "イベント表がありません" in out
        assert "ds000001" in out

    def test_errors_when_no_fold_produced_a_result(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        from src.strategy import dataset as ds
        from src.strategy import evaluation
        from src.strategy import model_store as ms

        meta = self._meta()
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(ms, "read_meta",
                            lambda model_id, base_dir="models": meta)
        stored = tmp_path / "ds000001.csv.gz"
        stored.write_bytes(b"")
        monkeypatch.setattr(
            ds, "dataset_path", lambda dataset_id, base_dir="data/datasets": stored)
        monkeypatch.setattr(
            ds, "load_events", lambda dataset_id, base_dir="data/datasets":
            pd.DataFrame({"label_contract_id": ["lc0001"]}))
        monkeypatch.setattr(
            evaluation, "run_evaluation",
            lambda ev, **kw: {"evaluation_run_id": "run-x", "fold_results": [],
                              "summary": pd.DataFrame(),
                              "degraded_reasons": []})

        assert pw.main(["evaluate", meta.model_id]) == 1
        assert "fold結果が0件" in capsys.readouterr().out
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `pytest tests/test_promote_workflow.py::TestEvaluate -v`
Expected: 6件すべて FAIL。最初のエラーは `argparse` の `error: argument command: invalid choice: 'evaluate'` による `SystemExit: 2`

- [ ] **Step 3: `import pandas as pd` を追加する**

`scripts/promote_workflow.py` の import 部を次のように変更する。

変更前:
```python
import argparse
import sys
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
```

変更後:
```python
import argparse
import sys
from typing import Optional

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
```

- [ ] **Step 4: `evaluate` 節を実装する**

`scripts/promote_workflow.py` の `cmd_train` の直後（`# ─── CLI ───` の直前）に次を挿入する:

```python
# ─── evaluate ─────────────────────────────────────────────────────────────


def _mean_or_none(series) -> Optional[float]:
    """NaN を除いた平均。全て NaN なら None を返す。

    `roc_auc` / `average_precision` は検証側が片側クラスだと None になる
    （src/strategy/evaluation.py:429-430）。その fold を 0 とみなして平均すると
    成績を過小評価するため、除いて平均する。
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def cmd_evaluate(args) -> int:
    """候補モデルと同じ構成（CurrentLightGBM）でwalk-forward評価し、記録を残す。

    評価対象のイベント表は**学習時に保存されたもの**を読み直す。作り直すと
    設定やデータの更新で別のイベント表になり、候補が学習したものと違う
    ラベル契約で評価してしまう（check_promotable がラベル契約の不一致で拒否する）。
    """
    from src.strategy import dataset as ds
    from src.strategy import evaluation
    from src.strategy import model_store as ms

    _bootstrap(args.config)

    try:
        meta = ms.read_meta(args.model_id, base_dir=args.base_dir)
    except FileNotFoundError as e:
        print(f"候補モデルが見つかりません: {e}")
        return EXIT_ERROR

    path = ds.dataset_path(meta.dataset_id)
    if not path.exists():
        print(f"イベント表がありません: {path}"
              f"（model_id={args.model_id} の dataset_id={meta.dataset_id}）")
        print("先に train を実行してください（イベント表は学習時に保存されます）")
        return EXIT_ERROR
    events = ds.load_events(meta.dataset_id)

    contracts = sorted(set(events["label_contract_id"].dropna().astype(str)))
    print(f"候補: {args.model_id}")
    print(f"dataset_id: {meta.dataset_id} / イベント {len(events)}件")
    print(f"ラベル契約: events={contracts} model={meta.label_contract_id}")
    if meta.label_contract_id not in contracts:
        print("警告: ラベル契約が一致しません。この評価記録では昇格できません")

    # **候補の model_id をキーにする。** check_promotable() は
    # load_prediction_details(evaluation_run_id, model_id=model_id) で
    # この鍵と突き合わせる（src/strategy/promotion.py:106）。"current_lightgbm"
    # のような汎用名で保存すると予測明細が見つからず永久に昇格できない。
    # window_sessions / feature_cols はメタから取る（学習時の条件を再現する）。
    out = evaluation.run_evaluation(
        events,
        model_factories={args.model_id: evaluation.CurrentLightGBM},
        n_splits=args.n_splits,
        window_sessions=meta.training_window_sessions,
        feature_cols=list(meta.feature_cols),
        persist=True,
    )

    summary = out["summary"]
    if summary.empty:
        print("fold結果が0件でした（分割できるイベントがありません）")
        return EXIT_ERROR

    print("")
    print(summary.to_string(index=False))
    print("")
    print(f"AUC平均: {_mean_or_none(summary['roc_auc'])}")
    print(f"Brier平均: {_mean_or_none(summary['brier'])}")
    print(f"Brier vs 定数(平均): {_mean_or_none(summary['brier_vs_constant'])}")
    print("（Brier vs 定数が正なら定数モデルより良い。"
          "AUC 0.5 は予測力が無いのと区別できない）")

    reasons = out["degraded_reasons"]
    if reasons:
        print(f"degraded: {len(reasons)}件。この評価記録では昇格できません")
        for reason in reasons:
            print(f"  - {reason}")

    print("")
    print(f"evaluation_run_id: {out['evaluation_run_id']}")
    print("昇格する場合（昇格するかどうかは評価結果を見てから判断すること）:")
    print(f"  python -m scripts.promote_workflow promote {args.model_id} "
          f"{out['evaluation_run_id']} --reason \"...\" --decided-by \"...\"")
    return EXIT_OK
```

- [ ] **Step 5: `build_parser()` にサブパーサを追加する**

`build_parser()` の `p_train.set_defaults(func=cmd_train)` の直後、`return parser` の直前に挿入する:

```python
    p_eval = sub.add_parser(
        "evaluate", help="候補モデルをwalk-forwardで評価し評価記録を残す")
    p_eval.add_argument("model_id", help="train が表示した候補のmodel_id")
    p_eval.add_argument(
        "--n-splits", type=int, default=5,
        help="walk-forwardの分割数（既定: 5＝train_v2 と同じfold構造）")
    p_eval.set_defaults(func=cmd_evaluate)
```

- [ ] **Step 6: テストを実行して通ることを確認する**

Run: `pytest tests/test_promote_workflow.py -v`
Expected: 14 passed

- [ ] **Step 7: コミット**

```bash
git add scripts/promote_workflow.py tests/test_promote_workflow.py
git commit -m "$(cat <<'EOF'
feat(promote_workflow): evaluateサブコマンドを追加

model_factories の鍵に候補の model_id 自身を使う。check_promotable() は
load_prediction_details(run_id, model_id=候補ID) で突き合わせるため、
汎用名で保存すると予測明細が見つからず昇格できない。
評価対象のイベント表は学習時に保存されたものを読み直し、ラベル契約の
一致を構造的に保証する。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `promote` サブコマンド

**Files:**
- Modify: `scripts/promote_workflow.py`（`cmd_evaluate` の直後に `promote` 節を追加、`build_parser()` にサブパーサを追加）
- Test: `tests/test_promote_workflow.py`（`TestPromote` クラスを追加）

**Interfaces:**
- Consumes: Task 2 の `_bootstrap` / `EXIT_OK` / `EXIT_ERROR` / `build_parser()`
- Produces:
  - `scripts/promote_workflow.cmd_promote(args) -> int`
  - `promote <model_id> <evaluation_run_id> --reason R --decided-by D [--dry-run]` サブコマンド

**設計上の注意（実装者向け）:** `--dry-run` は既存の `promotion.check_promotable()` をそのまま呼ぶだけであり、CLI側に判断ロジックを持たせるものではない（Global Constraints の「二重チェックを作り込まない」に反しない）。空文字チェックだけは `promotion.promote()` より前にCLIで行う。`promote()` 自身も空文字を `ValueError` にするが、(a) 空白のみの文字列（`"   "`）は `promote()` を通ってしまう、(b) DB接続やモデル読み込みより前に分かりやすく落とせる、の2点のため。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_promote_workflow.py` の末尾に追加する:

```python
class TestPromote:
    _MODEL = "v2-20260921T120000-abcdef12"
    _RUN = "20260921T130000-0badf00d"

    def _args(self, *extra):
        return ["promote", self._MODEL, self._RUN,
                "--reason", "AUC・Brierを確認し現行より悪化がないため",
                "--decided-by", "garnet", *extra]

    def test_missing_reason_or_decided_by_exits_with_argparse_error(self):
        """--reason / --decided-by は必須引数。argparse が終了コード2で落とす"""
        with pytest.raises(SystemExit) as e:
            pw.main(["promote", self._MODEL, self._RUN, "--decided-by", "garnet"])
        assert e.value.code == 2

        with pytest.raises(SystemExit) as e:
            pw.main(["promote", self._MODEL, self._RUN, "--reason", "良さそう"])
        assert e.value.code == 2

    def test_blank_reason_is_rejected_before_anything_runs(
            self, monkeypatch, capsys):
        """空白のみの理由は promote() を呼ぶ前にCLIで拒否する

        promotion.promote() の `if not reason` は "   " を通してしまう。
        """
        from src.strategy import promotion

        called = []
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": called.append("bootstrap"))
        monkeypatch.setattr(promotion, "promote",
                            lambda *a, **kw: called.append("promote"))

        assert pw.main(["promote", self._MODEL, self._RUN,
                        "--reason", "   ", "--decided-by", "garnet"]) == 1
        assert called == []
        assert "--reason が空です" in capsys.readouterr().out

    def test_blank_decided_by_is_rejected_before_anything_runs(
            self, monkeypatch, capsys):
        from src.strategy import promotion

        called = []
        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": called.append("bootstrap"))
        monkeypatch.setattr(promotion, "promote",
                            lambda *a, **kw: called.append("promote"))

        assert pw.main(["promote", self._MODEL, self._RUN,
                        "--reason", "良い", "--decided-by", "  "]) == 1
        assert called == []
        assert "--decided-by が空です" in capsys.readouterr().out

    def test_forwards_arguments_to_promotion_promote(
            self, tmp_path, monkeypatch, capsys):
        from src.strategy import model_store as ms
        from src.strategy import promotion
        from src.strategy.indicators import FEATURE_COLS

        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        refs = iter([
            None,
            ms.CurrentRef(model_id=self._MODEL, previous_model_id=None,
                          switched_at=None),
        ])
        monkeypatch.setattr(ms, "read_current",
                            lambda base_dir="models": next(refs))

        captured = {}

        def fake_promote(model_id, **kwargs):
            captured["model_id"] = model_id
            captured.update(kwargs)
            return 7

        monkeypatch.setattr(promotion, "promote", fake_promote)

        base = str(tmp_path / "models")
        assert pw.main(["--base-dir", base, *self._args()]) == 0
        assert captured["model_id"] == self._MODEL
        assert captured["evaluation_run_id"] == self._RUN
        assert captured["decided_by"] == "garnet"
        assert captured["reason"] == "AUC・Brierを確認し現行より悪化がないため"
        assert captured["degraded"] is False
        assert captured["base_dir"] == base
        assert captured["expected_feature_cols"] == list(FEATURE_COLS)
        assert "promotion_id=7" in capsys.readouterr().out

    def test_blockers_are_reported_and_current_is_untouched(
            self, monkeypatch, capsys):
        """check_promotable に落ちたら promote() が ValueError を投げる。
        CLIはその文面をそのまま出して終了コード1にする（再実装しない）。
        """
        from src.strategy import model_store as ms
        from src.strategy import promotion

        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        monkeypatch.setattr(ms, "read_current", lambda base_dir="models": None)

        def raise_blocked(model_id, **kwargs):
            raise ValueError(
                "昇格できません: 実績が1件も確定していません（予測だけでは"
                "成績を測れません） / shadow記録だけでは昇格できません")

        monkeypatch.setattr(promotion, "promote", raise_blocked)

        assert pw.main(self._args()) == 1
        out = capsys.readouterr().out
        assert "実績が1件も確定していません" in out
        assert "shadow記録だけでは昇格できません" in out

    def test_dry_run_only_checks_and_never_promotes(self, monkeypatch, capsys):
        from src.strategy import promotion

        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        called = []
        monkeypatch.setattr(promotion, "promote",
                            lambda *a, **kw: called.append("promote"))
        monkeypatch.setattr(
            promotion, "check_promotable",
            lambda model_id, **kw: promotion.PromotionCheck(
                ok=False, blockers=["未評価です（evaluation_run_id がありません）"]))

        assert pw.main(self._args("--dry-run")) == 1
        assert called == []
        assert "未評価です" in capsys.readouterr().out

    def test_dry_run_reports_ok_without_changing_current(
            self, monkeypatch, capsys):
        from src.strategy import promotion

        monkeypatch.setattr(pw, "_bootstrap",
                            lambda config_path="config.yaml": "high_risk")
        called = []
        monkeypatch.setattr(promotion, "promote",
                            lambda *a, **kw: called.append("promote"))
        monkeypatch.setattr(
            promotion, "check_promotable",
            lambda model_id, **kw: promotion.PromotionCheck(ok=True, blockers=[]))

        assert pw.main(self._args("--dry-run")) == 0
        assert called == []
        assert "昇格可能です" in capsys.readouterr().out
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `pytest tests/test_promote_workflow.py::TestPromote -v`
Expected: 7件すべて FAIL。`argparse` の `invalid choice: 'promote'` による `SystemExit: 2`（必須引数テストはたまたま code==2 で通る可能性があるため、この時点で `-v` の結果を確認し `test_forwards_arguments_to_promotion_promote` が確実に FAIL していることを見ること）

- [ ] **Step 3: `promote` 節を実装する**

`scripts/promote_workflow.py` の `cmd_evaluate` の直後（`# ─── CLI ───` の直前）に挿入する:

```python
# ─── promote ──────────────────────────────────────────────────────────────


def cmd_promote(args) -> int:
    """評価記録を根拠に候補を現行へ昇格する。

    **昇格可否の判断は promotion.check_promotable() が唯一の判断者である。**
    CLI側に二重チェックを作らない。検査に落ちた場合 promotion.promote() は
    現行を一切変更せずに ValueError を投げるので、その文面をそのまま出す。
    """
    from src.strategy import model_store as ms
    from src.strategy import promotion
    from src.strategy.indicators import FEATURE_COLS

    # 空文字は promotion.promote() も ValueError にするが、空白のみの文字列は
    # 通ってしまう。DB接続やモデル読み込みより前に、CLIの言葉で落とす。
    reason = args.reason.strip()
    decided_by = args.decided_by.strip()
    if not reason:
        print("--reason が空です。なぜ昇格するのかを必ず書いてください")
        return EXIT_ERROR
    if not decided_by:
        print("--decided-by が空です。誰の判断かを必ず書いてください")
        return EXIT_ERROR

    _bootstrap(args.config)

    # train_v2() は meta.feature_cols に list(FEATURE_COLS) を書く
    # （src/strategy/v2_training.py:131）。同じものを期待値として渡す。
    expected_feature_cols = list(FEATURE_COLS)

    if args.dry_run:
        check = promotion.check_promotable(
            args.model_id,
            evaluation_run_id=args.evaluation_run_id,
            degraded=False,
            base_dir=args.base_dir,
            expected_feature_cols=expected_feature_cols,
        )
        if check.ok:
            print("昇格可能です（--dry-run なので現行は変更していません）")
            return EXIT_OK
        print("昇格できません:")
        for blocker in check.blockers:
            print(f"  - {blocker}")
        return EXIT_ERROR

    before = ms.read_current(base_dir=args.base_dir)
    print(f"現行: {before.model_id if before else '(未昇格)'}")

    try:
        promotion_id = promotion.promote(
            args.model_id,
            evaluation_run_id=args.evaluation_run_id,
            decided_by=decided_by,
            reason=reason,
            # 引数の degraded は補助的な早期拒否にすぎず、真偽は
            # check_promotable() が保存済みの EvaluationRun.degraded から読む
            # （src/strategy/promotion.py:74-82）。CLIは常に False を渡す。
            degraded=False,
            base_dir=args.base_dir,
            expected_feature_cols=expected_feature_cols,
        )
    except ValueError as e:
        print(str(e))
        return EXIT_ERROR

    after = ms.read_current(base_dir=args.base_dir)
    print(f"昇格しました: promotion_id={promotion_id}")
    print(f"現行: {after.model_id if after else '(未昇格)'} / "
          f"previous: {after.previous_model_id if after else None}")
    return EXIT_OK
```

- [ ] **Step 4: `build_parser()` にサブパーサを追加する**

`build_parser()` の `p_eval.set_defaults(func=cmd_evaluate)` の直後、`return parser` の直前に挿入する:

```python
    p_promote = sub.add_parser(
        "promote", help="評価記録を根拠に候補を現行へ昇格する")
    p_promote.add_argument("model_id", help="昇格させる候補のmodel_id")
    p_promote.add_argument("evaluation_run_id",
                           help="evaluate が表示した evaluation_run_id")
    p_promote.add_argument("--reason", required=True,
                           help="なぜ昇格するのか（必須・空文字不可）")
    p_promote.add_argument("--decided-by", required=True,
                           help="誰の判断か（必須・空文字不可）")
    p_promote.add_argument(
        "--dry-run", action="store_true",
        help="check_promotable() の判定だけを表示し、現行は変更しない")
    p_promote.set_defaults(func=cmd_promote)
```

- [ ] **Step 5: テストを実行して通ることを確認する**

Run: `pytest tests/test_promote_workflow.py -v`
Expected: 21 passed

- [ ] **Step 6: コミット**

```bash
git add scripts/promote_workflow.py tests/test_promote_workflow.py
git commit -m "$(cat <<'EOF'
feat(promote_workflow): promoteサブコマンドを追加

--reason / --decided-by を必須にし、空白のみの文字列もCLIで拒否する
（promotion.promote() の `if not reason` は "   " を通してしまう）。
昇格可否の判断は check_promotable() に完全に委ね、CLI側に二重チェックを
作らない。--dry-run は check_promotable() をそのまま呼ぶだけ。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 一気通貫の統合テストと運用Runbookへの手順追記

**Files:**
- Test: `tests/test_promote_workflow.py`（`TestIntegration` クラスを追加）
- Modify: `docs/運用Runbook.md`（末尾に節を追加）

**Interfaces:**
- Consumes: Task 2〜4 の `cmd_train` / `cmd_evaluate` / `cmd_promote`、`pw.main()`
- Produces: 実装完了。以降の変更は無い。

**統合テストの設計（実装者向けの重要な注意）:**

合成OHLCV（`_ohlcv(n=300)` × 2銘柄）から作れる決着イベントは **51件**しかない（`tests/test_v2_training.py:103-110` に実測が記録されている）。本番既定の `MIN_RESOLVED_EVENTS=200` に届かせるには `n>=1500` が必要で、`train_v2()` 1回に4〜5分かかる（`dataset.simulate_event()` が実測O(n^2)）。そのため、既存テストと同じく `v2_training.MIN_RESOLVED_EVENTS` を一時的に 10 へ下げる（**本番の値は変えない**）。

さらに、51件のイベントで `run_evaluation()` を通すと、内側foldの校正が identity へ縮退するなどして `degraded_reasons` がほぼ確実に付く。すると `EvaluationRun.degraded=1` になり `check_promotable()` が「保存済みの実行記録が degraded です」で**必ず**拒否する。したがって統合テストは2本に分ける:

1. `train` → `evaluate` の実データ経路（候補ディレクトリ・`EvaluationRun`・`Prediction` が候補IDで残ること）
2. `train` → クリーンな評価記録の作成（既存の `save_predictions` / `save_outcomes` / `save_evaluation_run` を使う。`tests/test_model_promotion.py:48-70` の `_recorded_evaluation` と同じ考え方）→ `promote` で `models/current.json` が更新されること

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_promote_workflow.py` の末尾に追加する:

```python
# 合成OHLCV(n=300×2銘柄)の決着イベントは51件しかなく本番既定の200件に
# 届かない。届かせるには n>=1500 が要り train_v2() 1回で4〜5分かかる
# （tests/test_v2_training.py:103-110 の実測）。本番のしきい値は変えず、
# テストだけ下げる。
_TEST_MIN_RESOLVED_EVENTS = 10


def _bars():
    return {"7203": _ohlcv(seed=1), "9984": _ohlcv(seed=2, start_price=500.0)}


def _run_train(tmp_path, monkeypatch) -> str:
    """実データ経路で train を走らせ、できた候補の model_id を返す"""
    from src.strategy import v2_training

    monkeypatch.setattr(pw, "_bootstrap",
                        lambda config_path="config.yaml": "high_risk")
    monkeypatch.setattr(pw, "_collect_ohlcv", lambda limit: (_bars(), []))
    monkeypatch.setattr(v2_training, "MIN_RESOLVED_EVENTS",
                        _TEST_MIN_RESOLVED_EVENTS)

    base = str(tmp_path / "models")
    assert pw.main(["--base-dir", base, "train"]) == 0
    candidates = sorted((tmp_path / "models" / "candidates").iterdir())
    assert len(candidates) == 1
    return candidates[0].name


def _record_clean_evaluation(model_id, meta, run_id="run-integration-1"):
    """degraded でない評価記録を1件作る（既存の保存関数だけを使う）。

    51件の決着イベントで run_evaluation() を通すと内側foldの校正が identity へ
    縮退して degraded_reasons が付き、check_promotable() が必ず拒否する。
    昇格の経路そのものを固定したいので、ここは実イベント表から作った
    予測・実績・実行記録の3点セットを degraded 無しで保存する
    （tests/test_model_promotion.py の _recorded_evaluation と同じ考え方）。
    """
    from src.strategy import dataset as ds
    from src.strategy import evaluation
    from src.strategy.indicators import FEATURE_COLS

    events = ds.load_events(meta.dataset_id)
    resolved = events[events["status"] == ds.STATUS_RESOLVED].head(20)
    assert len(resolved) > 0

    preds = pd.DataFrame({
        "event_id": resolved["event_id"].astype(str).values,
        "label_contract_id": resolved["label_contract_id"].astype(str).values,
        "raw_probability": 0.6,
        "calibrated_probability": 0.55,
        "fold_index": 0,
    })
    evaluation.save_predictions(preds, run_id, model_id)
    evaluation.save_outcomes(events)
    run_config = evaluation.capture_run_config(
        events, n_splits=5, window_sessions=None,
        feature_cols=list(FEATURE_COLS))
    evaluation.save_evaluation_run(
        run_id, run_config, purpose=evaluation.PURPOSE_VALIDATION,
        model_id=model_id, n_folds=5, n_predictions=len(preds),
        degraded_reasons=[])
    return run_id


class TestIntegration:
    def test_train_then_evaluate_records_predictions_under_the_candidate_id(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        """train → evaluate で、候補ID自身の予測明細と実行記録が残ること"""
        from sqlalchemy import select

        from src.data.database import EvaluationRun, Prediction, get_session

        base = str(tmp_path / "models")
        model_id = _run_train(tmp_path, monkeypatch)
        capsys.readouterr()

        assert pw.main(["--base-dir", base, "evaluate", model_id]) == 0
        out = capsys.readouterr().out
        run_id = out.split("evaluation_run_id: ")[1].splitlines()[0].strip()

        with get_session() as session:
            run = session.scalar(select(EvaluationRun).where(
                EvaluationRun.evaluation_run_id == run_id))
            preds = list(session.scalars(select(Prediction).where(
                Prediction.evaluation_run_id == run_id)).all())

        assert run is not None
        # model_factories が1件なので EvaluationRun.model_id に候補IDが入る
        assert run.model_id == model_id
        assert preds
        assert {p.model_id for p in preds} == {model_id}

    def test_promote_switches_models_current_json(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        """昇格すると models/current.json が候補を指し、記録が committed になる"""
        from sqlalchemy import select

        from src.data.database import ModelPromotion, get_session
        from src.strategy import model_store as ms
        from src.strategy import promotion

        base = str(tmp_path / "models")
        model_id = _run_train(tmp_path, monkeypatch)
        capsys.readouterr()

        meta = ms.read_meta(model_id, base_dir=base)
        run_id = _record_clean_evaluation(model_id, meta)

        assert ms.read_current(base_dir=base) is None
        assert pw.main([
            "--base-dir", base, "promote", model_id, run_id,
            "--reason", "AUC・Brierを確認し現行より悪化がないため",
            "--decided-by", "garnet"]) == 0

        ref = ms.read_current(base_dir=base)
        assert ref is not None
        assert ref.model_id == model_id

        with get_session() as session:
            row = session.scalar(select(ModelPromotion).where(
                ModelPromotion.model_id == model_id))
        assert row.state == promotion.PROMOTION_COMMITTED
        assert row.evaluation_run_id == run_id
        assert row.decided_by == "garnet"
        assert row.reason == "AUC・Brierを確認し現行より悪化がないため"
        assert row.previous_model_id is None

    def test_promote_is_refused_when_the_evaluation_is_missing(
            self, isolated_db, tmp_path, monkeypatch, capsys):
        """評価記録が無ければ現行は変わらない（安全機構が生きていること）"""
        from src.strategy import model_store as ms

        base = str(tmp_path / "models")
        model_id = _run_train(tmp_path, monkeypatch)
        capsys.readouterr()

        assert pw.main([
            "--base-dir", base, "promote", model_id, "run-does-not-exist",
            "--reason", "とりあえず上げたい",
            "--decided-by", "garnet"]) == 1
        assert ms.read_current(base_dir=base) is None
        assert "評価実行の記録がありません" in capsys.readouterr().out
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `pytest tests/test_promote_workflow.py::TestIntegration -v`
Expected: 3件すべて FAIL（`NameError: name '_TEST_MIN_RESOLVED_EVENTS' is not defined` など、Step 1でヘルパを書く前に実行した場合）。Step 1 を書いた直後に実行するなら、まだ実装が無いわけではないので**このステップでは PASS してもよい**。PASS した場合はその旨を記録し、Step 3 へ進む（Task 5 はテストの追加とドキュメント整備であり、実装コードの追加は無い）。

- [ ] **Step 3: 統合テストが通ることを確認する（時間がかかる）**

Run: `pytest tests/test_promote_workflow.py::TestIntegration -v`
Expected: 3 passed（`train_v2` と `run_evaluation` を実データ経路で走らせるため、合計1〜3分かかりうる）

FAIL した場合の切り分け:
- `候補モデルは作られませんでした` → `MIN_RESOLVED_EVENTS` の monkeypatch が効いていない
- `イベント表がありません` → `isolated_db` の `monkeypatch.chdir(tmp_path)` が効いておらず、`data/datasets/` の相対パスがずれている
- `ラベル契約が一致しません` → `_record_clean_evaluation` が `meta.dataset_id` ではなく別のイベント表を読んでいる

- [ ] **Step 4: ファイル全体のテストを実行する**

Run: `pytest tests/test_promote_workflow.py -v`
Expected: 24 passed

- [ ] **Step 5: 既存テストが壊れていないことを確認する**

Run: `pytest tests/test_v2_training.py tests/test_evaluation.py tests/test_model_promotion.py tests/test_engine_version_wiring.py -v`
Expected: すべて passed（本計画は既存モジュールを1行も変更していないので、壊れるはずがない。壊れたなら余計な変更を入れている）

- [ ] **Step 6: リポジトリが汚れていないことを確認する**

Run: `git status --porcelain`
Expected: `M tests/test_promote_workflow.py` のみ（`data/datasets/` や `models/candidates/` が出ないこと。出たら Task 1 の `.gitignore` が効いていないか、テストの chdir が漏れている）

- [ ] **Step 7: 運用Runbookに手順を追記する**

`docs/運用Runbook.md` の末尾（「Obsidianグラフノートの再生成」節の最後の行の後）に、次を追記する:

````markdown

---

## 候補モデルの学習・評価・昇格（手動）

**このツールはスケジューラに登録されていない。人間が叩いたときだけ動く。**
週次再学習（`ml_retrain`）が作る候補とは独立に、任意のタイミングで候補を
作り・評価し・昇格できる。

### 1. 候補モデルを学習する

```bash
python -m scripts.promote_workflow train
```

- ウォッチリスト全リストの銘柄（200本以上のOHLCVがあるもの）で学習する
- `risk_profile.json`（現状 high_risk）を適用した実効値でラベルを作る
- 既定は全期間を使う（分析報告書と同条件）。直近N本に絞りたいときは
  `--ohlcv-limit N`（週次再学習の既定500本は意図しない切り詰めのため
  踏襲しない）
- **運用モデルは変更されない。** 候補が `models/candidates/<model_id>/` に
  保存されるだけ
- 決着イベントが200件に届かないと候補は作られず終了コード1になる

### 2. 候補を評価する

```bash
python -m scripts.promote_workflow evaluate <model_id>
```

- 学習時に保存されたイベント表（`data/datasets/<dataset_id>.csv.gz`）を
  読み直し、walk-forward（5分割）で評価する
- fold別のAUC・Brierと、その平均を表示する
- 最後に `evaluation_run_id: <id>` を表示する。これが昇格の根拠になる
- `degraded: N件` と出た評価記録では昇格できない（縮退した成績は根拠に
  できない）

**読み方の注意**: AUC 0.5 は「予測力が無い」のと区別できない。実データでの
5モデル比較では現行相当モデルのAUCは 0.5018（p=0.7231）だった
（`docs/kabu-auto-ml-real-data-comparison_20260921.md`）。数字が出たこと
自体は昇格の理由にならない。

### 3. 昇格できるか確認する（現行は変えない）

```bash
python -m scripts.promote_workflow promote <model_id> <evaluation_run_id> \
    --reason "..." --decided-by "..." --dry-run
```

拒否理由は**全件**表示される。1つ直しては再実行、を繰り返さずに済む。

### 4. 昇格する

```bash
python -m scripts.promote_workflow promote <model_id> <evaluation_run_id> \
    --reason "AUC・Brierを確認し現行より悪化がないため" --decided-by "garnet"
```

- `--reason` と `--decided-by` は必須。空文字・空白のみは拒否される
- 検査に通らなければ**現行を一切変更せず**終了コード1で終わる
- 成功すると `models/current.json` が新しい候補を指し、`model_promotions`
  に `state=committed` の記録が残る

### 5. 戻す（ロールバック）

CLIには実装していない。Pythonから `promotion.rollback()` を呼ぶ
（判断者と理由は戻す方向でも必須）:

```bash
python -c "from src.core import config as cfg; from src.data import database as db; from src.core import risk_profile as rp; cfg.load('config.yaml'); rp.load('risk_profile.json'); db.init(); from src.strategy import promotion; print(promotion.rollback(decided_by='garnet', reason='昇格後に成績が悪化したため'))"
```
````

- [ ] **Step 8: Runbookの追記を目視確認する**

Run: `python -c "print(open('docs/運用Runbook.md', encoding='utf-8').read()[-2500:])"`
Expected: 追記した節が文字化けせずに表示され、コードブロックが閉じている

- [ ] **Step 9: コミット**

```bash
git add tests/test_promote_workflow.py docs/運用Runbook.md
git commit -m "$(cat <<'EOF'
test(promote_workflow): train→evaluate→promote の統合テストとRunbook手順を追加

合成データ(決着51件)では run_evaluation が degraded になり
check_promotable が必ず拒否するため、promote の経路は既存の保存関数で
作ったクリーンな評価記録で固定する。models/current.json が実際に
切り替わることと、評価記録が無ければ現行が変わらないことを確認する。

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## 完了条件

すべて満たしたら完了。

1. `pytest tests/test_promote_workflow.py -v` が 24 passed
2. `pytest tests/test_v2_training.py tests/test_evaluation.py tests/test_model_promotion.py tests/test_engine_version_wiring.py -v` が全 passed（既存の回帰なし）
3. `git status --porcelain` が空（生成物が1つも追跡されていない）
4. `git diff --stat main...HEAD` に `src/` 配下のファイルが**1つも含まれない**（既存モジュールを変更していない）
5. `grep -rn "promote_workflow" src/ main.py` が**何も返さない**（スケジューラ・本体から呼ばれていない）
6. `python -m scripts.promote_workflow --help` / `... train --help` / `... evaluate --help` / `... promote --help` がすべて終了コード0

## スコープ外（やらないこと）

- **実際の昇格の実行**。ツールを作るだけ。本番の `models/` に対して `promote` を実行しない
- `src/services/trading.py` の `ml_retrain()` の変更（週次再学習の経路は触らない）
- `config.yaml` の `engine_version` の変更
- `src/strategy/shadow.py` の結線（shadow運用は別スコープ）
- `rollback` のサブコマンド化（Runbookに手順だけ書く）
- `docs/詳細設計書.md` / `docs/概要設計書.md` の更新（本計画は既存モジュールの設計を変えないため。更新するとObsidian vaultへの同期義務が発生する）
