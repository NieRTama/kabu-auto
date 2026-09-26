# ponytail-audit 指摘7件の削減 実装計画

**Goal:** 2026-09-26 の ponytail-audit で挙がった過剰実装7件を、挙動を変えずに削る。

**実行方式:** 計画=Opus、実装=Sonnet サブエージェント。1タスク=1エージェント=1コミット。順番に実行し、各タスク完了をレビューしてから次へ進む。

## Global Constraints

- 本番稼働中（live・実資金）。**本番プロセスの再起動・停止はしない**。`data/kabu_auto.db`・`config.yaml`・`models/` に触れない。
- 挙動を変えない。テスト以外の本番ロジックの変更は、各タスクで明示した範囲に限る。
- ファイルは UTF-8（BOM無し）・LF。既存のコメントの言語（日本語）に合わせる。
- テスト実行は `python -m pytest <files> -q`。各タスクでは、変更したファイルと、削除したシンボルを参照していたテストファイルを実行する。
- コミットメッセージは日本語、Conventional Commits 形式（`refactor(...)` / `chore(...)` / `test(...)`）。末尾は `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`。
- 無関係なファイルはステージしない（`git add <path>` で個別に追加する）。
- 過去の計画書 `docs/superpowers/plans/*.md` は履歴なので編集しない。

---

### Task 1: `isolated_db` フィクスチャを conftest に集約

**Files:** `tests/conftest.py`、下記の30ファイル

現状、35ファイルが同名のフィクスチャを個別に定義している。

1. `tests/conftest.py` に標準版を追加する（`import pytest` / `import src.core.config as cfg` / `import src.data.database as db`）:
   ```python
   @pytest.fixture
   def isolated_db(tmp_path):
       cfg.load("config.yaml")
       cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
       db.init()
       try:
           yield tmp_path
       finally:
           db._engine = None
           db._Session = None
   ```
2. **完全一致の標準版（yield＋teardown、18ファイル）→ 定義を削除する**:
   test_bought_today, test_fill_recording, test_lots_fifo, test_max_positions_and_candidate_notional, test_order_integration, test_order_intent, test_paper_execution_v2, test_pnl_report, test_position_sizing_held_value, test_reserved_unresolved, test_risk_manager_price_cache, test_risk_snapshot, test_schema_version, test_sector_concentration, test_select_latest_signals, test_sync_on_startup, test_trailing_stop, test_unrealized_pnl_live_price
3. **`return tmp_path` 版（teardown無し、7ファイル）→ 定義を削除する**（conftest 版は teardown が付くが、本番DB接続を残さない方向なので安全側）:
   test_corporate_actions, test_dataset, test_evaluation, test_model_promotion, test_shadow, test_signal_freshness_gate, test_walkforward
4. **`dash._auth_required = False` を足している版（5ファイル）→ 上書き用の短いフィクスチャに置き換える**:
   test_backtest_history, test_dashboard_positions, test_engine_version_wiring, test_model_latest_skip_null, test_report_journal
   ```python
   @pytest.fixture
   def isolated_db(isolated_db):
       dash._auth_required = False
       return isolated_db
   ```
5. **独自処理を持つ5ファイルには触らない**: test_backtest_threshold_override（OHLCV投入）、test_position_drift_reconcile（halt隔離）、test_position_sizing_message（autouse）、test_promote_workflow と test_v2_training（chdir）。
6. 削除によって未使用になった import（`cfg` / `db` / `pytest` など）を各ファイルから消す。判定は `python -m pyflakes tests/<file>` が使えればそれで行い、使えなければ grep で確認する。**他で使っている import は残す。**
7. 対象30ファイルと conftest を実行して全件パスを確認する。
8. コミット: `test: isolated_db フィクスチャを conftest に集約`

### Task 2: 本番から呼ばれない関数7つを削除

**Files:** 下記の本番ファイル、それらを参照するテスト、`docs/詳細設計書.md`

| 関数 | 場所 |
|---|---|
| `recompute_metrics` | src/strategy/evaluation.py:872 |
| `inner_validation_event_ids` | src/strategy/evaluation.py:966 |
| `split_factor_between` | src/data/market_data.py:151 |
| `promotion_history` | src/strategy/promotion.py:215 |
| `is_daily_loss_limit_reached` | src/risk/manager.py:237（`RiskManager` のメソッド） |
| `get_schema_version` | src/data/database.py:644 |
| `require_api_password` | src/core/config.py:77 |

1. 着手前に `grep -rnw <name> src scripts main.py frontend` を実行し、参照が定義行だけであることを再確認する。1件でも参照があればその関数は削除せず、報告する。
2. 関数本体を削除する。削除で未使用になった import やモジュール定数も削除する。
3. 各関数を検証しているテスト（テストメソッド／クラス）を削除する。ファイルが空になる場合はファイルごと削除する。
4. `docs/詳細設計書.md` の該当関数への言及を削除、または「（2026-09-26 削除: 本番未使用）」へ書き換える。
5. 影響したテストファイルを実行して全件パスを確認する。
6. コミット: `refactor: 本番から呼ばれない関数7つを削除`

### Task 3: `make_rule_only` / `make_weighted_blend` を削除

**Files:** src/backtest/walkforward.py:190-215、参照しているテスト（test_walkforward.py）

1. 参照が定義とテストだけであることを grep で再確認する（`scripts/` にも無いこと）。
2. 2関数と、それを検証しているテストを削除する。未使用になった import も消す。
3. `python -m pytest tests/test_walkforward.py -q` を実行する。
4. コミット: `refactor(walkforward): 未使用の戦略ファクトリ2つを削除`

### Task 4: WebSocket 配信登録メソッド3つを削除

**Files:** src/api/kabu_client.py:158-169、参照しているテスト、`docs/詳細設計書.md`

1. `register_push` / `unregister_push` / `unregister_all` の参照を grep で再確認する（本番に無いこと）。
2. 3メソッドと、それを検証しているテストを削除する。
3. `docs/詳細設計書.md` の `register_push` への言及を修正する。
4. kabu_client 関連のテストを実行する。
5. コミット: `refactor(kabu_client): 未使用のプッシュ配信登録メソッドを削除`

### Task 5: `AlertProvider` Protocol を削除

**Files:** src/core/alerts.py

1. `class AlertProvider(Protocol)` を削除し、`AlertProvider` の型注釈を `DiscordWebhookProvider` に置き換える（`build_providers() -> list[DiscordWebhookProvider]`、`_send_one(provider: DiscordWebhookProvider, ...)` など）。
2. 未使用になった `Protocol` の import を消す。
3. `build_providers` の docstring から「将来プロバイダを追加する場合は…」の段落を削除する（1行の説明だけを残す）。`config.yaml` のコメントは触らない。
4. `grep -rn AlertProvider src tests main.py` の結果が0件になることを確認する。
5. `python -m pytest tests/test_alerts.py tests/test_alert_levels.py -q` を実行する。
6. コミット: `refactor(alerts): 実装が1つだけのAlertProvider Protocolを削除`

### Task 6: `pydantic-settings` を requirements から削除

1. `grep -rn "pydantic_settings\|pydantic-settings" --include=*.py .` の結果が0件であることを確認する。
2. `requirements.txt` から該当の1行を削除する（パッケージのアンインストールはしない）。
3. コミット: `chore(deps): 未使用の pydantic-settings を削除`

### Task 7: スケジューラのタイムゾーンを pytz から zoneinfo へ統一

**Files:** src/core/scheduler.py、pytz を import しているテスト3ファイル、requirements.txt

1. `src/core/scheduler.py` の `import pytz` / `TZ = pytz.timezone("Asia/Tokyo")` を `from zoneinfo import ZoneInfo` / `TZ = ZoneInfo("Asia/Tokyo")` に置き換える。
2. **`TZ` の全使用箇所を確認する**（src 全体と tests で `scheduler.TZ` / `TZ.` を grep）。`TZ.localize(...)` や `TZ.normalize(...)` など pytz 固有の API があれば、`dt.replace(tzinfo=TZ)` などの等価な書き方に直す。`datetime.now(TZ)` と `astimezone(TZ)` はそのまま動く。
3. pytz を import しているテスト3ファイルも同じように zoneinfo へ置き換える。pytz 固有 API の置き換えも同様に行う。
4. `grep -rn pytz src tests main.py scripts` の結果が0件になったら、`requirements.txt` から `pytz==...` の行を削除する（pandas と APScheduler が推移的に依存するためインストールは残る。アンインストールはしない）。
5. `python -m pytest tests/ -q -k "scheduler or market or holiday or clock"` と、変更したテストファイルを実行する。加えて `python -c "from src.core.scheduler import TradingScheduler, TZ; s=TradingScheduler(); print(TZ)"` で import と生成を確認する（`start()` は呼ばない）。
6. コミット: `refactor(scheduler): タイムゾーンをzoneinfoへ統一しpytzの直接依存を削除`

---

## 完了条件

7タスク完了後、コントローラ（Opus）が以下を行う。

- フルテストスイート `python -m pytest tests -q` を実行し、全件パスを確認する。
- 本番への反映（再起動）はユーザーの判断に委ねる。
