# kabu-auto 運用Runbook（緊急対応・手動復旧手順）

ライブ運用中の異常時に「何を止め、何を確認し、どう復旧するか」を辿るための手順書（レビュー P2-4 対応で作成）。
判断に迷う場合は、まず新規発注を止める（§1）。

前提:
- ダッシュボード: `http://<host>:8080/`
- 緊急操作系API（停止/解除/緊急決済/逆指値）は `X-Emergency-Token` ヘッダー必須。
  トークンは起動時にコンソールへ一度だけ表示される（または `config.yaml` の
  `dashboard.emergency_token`）。
- 取引停止状態は `data/trading_halt.json` に永続化され、再起動を跨いで維持される。

目次:
- 緊急対応: §1 初動（取引停止） / §2 緊急全ポジション決済
- 起動と認証: §3 起動・停止・再起動 / §4 ログイン認証切れ / §5 API障害
- 整合性: §6 未解決注文 / §7 建玉ドリフト / §8 取引再開
- 最終手段と保険: §9 完全手動決済 / §10 ブローカー側逆指値
- 復旧: §11 端末故障・DB破損
- 参照: §12 通知の見分け方 / §13 チェックリスト / §14 Obsidianグラフノート / §15 候補モデルの学習・評価・昇格

---

## 1. 初動：取引停止（共通）

新規発注を全停止し、未約定BUYをキャンセルする（損切り・緊急決済は停止中も実行される）。

ダッシュボードの「取引停止」ボタン、または:

```bash
curl -X POST http://localhost:8080/api/halt \
  -H "X-Emergency-Token: <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"reason":"手動停止：要調査"}'
```

同時に全ポジションを成行決済する場合は `{"close_positions": true}` を付ける。

状態確認: `GET /api/halt` または `GET /api/status`（`can_place_order` が false になる）。

---

## 2. 緊急全ポジション決済

```bash
curl -X POST http://localhost:8080/api/emergency_close -H "X-Emergency-Token: <TOKEN>"
```

挙動（`OrderManager.close_all_positions`）:
- live/semi_live はブローカー `/positions` を正本として成行決済する（ローカルDBは正本にしない）。
- `/positions` の取得に失敗した場合は自動決済せず、critical アラートを出して中断する
  （実建玉を把握できない状態では決済しない）。→ §9 の手動決済へ。

---

## 3. 起動・停止・再起動

### 3.1 起動の仕組み（2026-10-15 時点）

| 項目 | 内容 |
|---|---|
| 起動元 | タスクスケジューラ `\kabu\kabu-auto`（実行ユーザー garnet、`RunLevel: Highest`） |
| トリガー | garnet のログオン時（2026-09-25 に `Boot` から変更）。`AutoAdminLogon: 0` のため、PC再起動後は Windows へのログオンまで起動しない |
| 実行内容 | `conhost.exe --headless cmd.exe /c scripts\run_autostart.bat`（2026-09-27〜。コンソール窓を表示しない） |
| 起動スクリプト | `scripts\run_autostart.bat` が `CONFIRM_LIVE_TRADING=true` を設定してから `python main.py` を実行する。`set X=Y && cmd` の1行書きは値に末尾スペースが入り弾かれるため、set と起動は別行 |
| 異常終了時 | タスクの `RestartCount:3` / `RestartInterval:1分` |
| 死活監視 | `\kabu\kabu-auto-watchdog`（5分毎、`scripts\watchdog.py`）。`data/kabu_auto.lock` のPIDが死んでいればタスクを再実行し Discord へ通知する。再起動は30分に1回まで |
| 日次運用概要 | `\kabu\kabu-ops-daily`（毎日18:30、2026-11-14まで）が `log/ops_daily/ops_daily_YYYY-MM-DD.md` を作成する |

経緯: 2026-09-09 に深夜のWindows Update自動再起動でプロセスが停止し、約5.5時間停止したことを受けて自動起動タスクを再有効化した。

### 3.2 PC再起動後に人がすること

- Windows にログオンする（ログオンで kabu-auto が起動する）。
- kabuステーションへのログインは平日8:30の完全自動ログイン（§4.1）が行う。失敗した場合は手動でログインする。
- `python main.py` を手動実行しない。自動起動済みのプロセスがロックを保持しているため、
  「多重起動を検知したため起動を中断します」で終了する（安全機構の仕様）。
  起動状態はダッシュボード http://localhost:8080/ で確認する。

### 3.3 停止・再起動

- `RunLevel: Highest` で起動したプロセスの停止とタスク設定の変更には管理者権限が要る。
  通常権限の `Stop-Process` は `Access is denied` になる。
- 停止: `Start-Process taskkill -Verb RunAs -ArgumentList '/PID','<pid>','/T','/F' -Wait`（UACの承認が必要）。
  PID は `data/kabu_auto.lock` の値、または `(Get-NetTCPConnection -LocalPort 8080 -State Listen).OwningProcess`。
- 再起動: 停止後に `Start-ScheduledTask -TaskPath '\kabu\' -TaskName kabu-auto`。
  タスクは `MultipleInstances: IgnoreNew` のため、旧プロセスが生きている間の起動要求は無視される
  （`LastTaskResult 0x800710E0`）。
- 通常権限で登録した予約タスクからは停止できない（2026-10-14 に予約再起動が実行されなかった実例）。
- 意図的に長時間止める場合は、先に `data/watchdog.pause` を作成する（作成しないと5分以内に watchdog が再起動し通知が出る）。
- 再起動後は `data/kabu_auto.lock` のPIDが変わったこと、`log/kabu_auto_YYYY-MM-DD.log` に「kabu-auto 起動完了」が出たことで確認する。

### 3.4 注意事項

- `config.yaml` の `trading.mode` が `live`/`semi_live` の場合、無人再起動でも人の確認なしにライブ発注が再開する（PC停止後もlive運用を継続するための仕様）。mode を変更する作業中に再起動を挟むと、そのタイミングで反映される。
- `config.yaml` は起動時に一度だけ読む。設定やコードを変更した場合は再起動するまで反映されない。
- 3回連続で起動に失敗した場合（`RestartCount:3` を使い切った場合）の専用通知は無い。平日8:45のハートビート（§12）が届かないことで検知する。

---

## 4. ログイン認証切れ

**症状**: 🔴「kabuステーションの再ログインが必要です」が届く。API が 401 を返す。

- kabuステーションの認証はPCを起動したままでも日をまたぐと切れる（アプリ自身が `Code 10016: ログイン認証の有効期間が切れました` を持つ）。毎営業日1回発生する。
- 8:30のトークン更新が成功した後、場中にセッションが切れることもある（2026-09-02 に発生）。認証状態は実際のAPI応答で判定しているため、場中の401も検知・通知される。

### 4.1 自動での復帰

- 平日8:30 に完全自動ログイン（`broker_full_login`）が kabuステーションの起動・ID/パスワード入力・2段階認証（Gmail API で受信したワンタイムパスワード）を行う。1日3回まで（`broker_full_login_max_attempts_per_day`）。実体は WSL2 側の `kabusapi-auto-login-template`。
- 認証切れ中は `auth_recovery` が5分間隔で再接続を試みる（時間制限なし）。
- 復帰が場中かつ14時前（`catchup_deadline_hour`）なら、逃した朝の発注を実行する（キャッチアップ）。
- 認証切れの間は新規発注が止まる（損切り・緊急決済は動き続ける）。

### 4.2 人の対応

- kabuステーションにログインする。ログイン後5分以内に自動で復帰する。
- 即時に再接続する場合は Discord で `@kabu-auto reconnect`。
- 完全自動ログインを手動で再実行する場合は Discord で `@kabu-auto full_login`。
- Gmail 認証が失効している場合（OAuth が「テスト」ステータスのため約7日で失効）は Discord で `gmail_auth`、または `scripts\gmail_reauth.bat`。

### 4.3 kabuステーション自体が起動していない場合

- 認証切れが続くと kabu-auto がアプリを起動する（1日3回まで、`max_launch_attempts_per_day`）。
- 手動で起動する場合は Discord で `@kabu-auto launch`（起動のみ。ログインは§4.1/§4.2）。

### 4.4 復旧しない場合

- kabuステーションのログ（`%APPDATA%\KabuS\Log\KabuS_YYYYMMDD.log`）に `ログイン処理エラー。結果コード:[-1]` が出ていればサーバー側の障害（2026-08-31 夜に発生）。時間を置くか公式の障害情報を確認する。
- 完全自動ログインが「前面がログイン窓ではないため入力を中断します」で失敗する場合は、ログイン窓の上に他アプリのポップアップ（2026-10-15 は Dropbox）が重なっていないか確認する。

### 4.5 パスキー必須化（外部要因）

三菱UFJ eスマート証券はログイン時のパスキー必須化を「2026年10月末目標」と告知している（実施日は未確定）。必須化後は §4.1 の完全自動ログインが動作しない可能性がある。パスキーを登録しない場合は、ログイン後の確認画面で「リスク同意」と「登録できない理由」の回答が10月末までに必要。

---

## 5. API障害（認証切れ以外の接続不可）

症状: トークン更新失敗、`/orders`・`/positions`・`/wallet` のタイムアウト/例外。

対応:
1. §1 で取引を停止する。
2. kabuステーション本体（PC常駐アプリ）の起動・ログイン状態、`base_url`、ネットワークを確認する。
3. 復旧したら `/orders`・`/positions` をダッシュボードで照会し、DBとの整合を確認する（§7）。
4. 起動時の preflight（`src/core/preflight.py`）は live/semi_live で API疎通失敗時に起動を中断する（fail-closed）。接続が安定してから再起動する。

照合ジョブ（15秒毎の `reconcile_orders`）は API障害時に例外を捕捉して次回再試行する。未約定注文は OPEN のまま残り、復旧後の照合で収束する。

### 5.1 16:20 のシグナルスキャンが空振りした（🟡「新規シグナル0件」）

症状: 🟡「新規シグナル0件 — 全N銘柄が鮮度不足で新規候補から除外されました（batch=…）」。翌朝 9:05 の発注対象（BUY・SELL シグナル）が作られていない。

1. `log/kabu_auto_YYYY-MM-DD.log` の 16:00 前後で「データ更新: … 状態=fresh」「データ更新完了」を確認する。データ更新自体が失敗していれば、先に `POST /api/data/update` で取り直す。
2. シグナルスキャンを手動で実行する（ダッシュボードの認証が必要。live は Signal を保存するだけで発注しない）:

```bash
curl -X POST http://localhost:8080/api/signal_scan -H "X-API-Token: <TOKEN>"
```

3. ログの「シグナルスキャン開始… batch=…」以降で候補が保存されたことを確認する。

再起動直後でも 1.15.11.a 以降は DB の最終足から鮮度を復元するため、16:00〜16:20 の間の再起動では起きない。

---

## 6. 約定不明・部分約定・残注文（UNKNOWN / CANCEL_FAILED）

状態の意味（`src/execution/order_status.py`）:
- `UNKNOWN`: 起動同期/照合でブローカーに該当注文が見つからない（約定/失効/取り逃しの可能性）。
- `CANCEL_FAILED`: キャンセル要求が失敗/拒否（実口座に注文が残存している可能性）。
- `PARTIALLY_FILLED`: 一部約定・残数は生存（追って約定 or タイムアウトキャンセル）。
- `PARTIALLY_FILLED_DONE`: 部分約定のままブローカー側で確定終了（残数は約定しない）。

未解決注文（UNKNOWN/CANCEL_FAILED）が残る間は新規発注が抑止され、取引停止スイッチも解除できない。

手順:
1. 証券会社の取引サイト / kabuステーションで実際の注文・建玉を確認する。
2. 実口座の状態に合わせてDBを是正する:
   - 実際は約定していた → ブローカー実態に合わせて建玉を確認する（次回 `/positions` 照合でドリフト検知される。§7）。
   - 実際は失効/取消だった → 当該 Trade を解消する（CANCELLED 相当へ）。
3. 未解決がゼロになったら取引停止を解除する（§8）。

---

## 7. 建玉ドリフト（DBとブローカー実建玉の不一致）

検知: 定期照合 `reconcile_positions_with_broker()` がDBの Position とブローカー `/positions`（LeavesQty）を比較し、差異があれば critical ログとアラートを出し、kill switch を作動させて新規発注を止める（fail-closed）。照合時点で生きている注文がある銘柄は、その回の比較から外す（2026-10-13 の誤検知対策）。

対応:
1. アラートの差分（`DB=○○株 ブローカー=○○株`）を確認する。
2. ブローカー実態を正としてDBの Position を是正する。
3. 取り逃した約定が原因なら、`/orders` 照合で Fill を取り込めるか確認する。
4. 整合後、未解決ゼロを確認して取引停止を解除する（§8）。

---

## 8. 取引再開（停止解除）

```bash
curl -X DELETE http://localhost:8080/api/halt -H "X-Emergency-Token: <TOKEN>"
```

- 未解決注文（UNKNOWN/CANCEL_FAILED）が1件でも残っていると 409 で解除を拒否する。先に §6 を完了する。
- 合計ドローダウン上限（実現損失＋含み損）到達で自動停止した場合は、損失要因を確認し、必要なら日次リセット（翌営業日 8:25 の `risk_reset`）まで待つか建玉を整理する。

---

## 9. 緊急決済が失敗したとき（完全手動）

`/positions` 取得失敗や API 全断で自動決済できない場合:
1. 証券会社の取引サイト / 電話注文で建玉を成行決済する。
2. kabu-auto は §1 で停止したままにする（誤発注防止）。
3. 市場と口座が落ち着いてから、DBを実態に合わせて是正し、§7→§8 で復旧する。

---

## 10. 損切りの保険（ブローカー側逆指値ストップ）

アプリの損切り監視（`stop_loss_check`）が止まっても証券会社側で発動する逆指値成行を発注できる:

```bash
curl -X POST http://localhost:8080/api/stop_loss \
  -H "X-Emergency-Token: <TOKEN>" -H "Content-Type: application/json" \
  -d '{"symbol":"7203"}'
```

トリガー価格は保有建玉の平均取得単価 × (1 + stop_loss_pct) で算出される。

---

## 11. 端末故障・DB破損からの復旧

### 11.1 DBが壊れた（同じPCが動作している場合）

```powershell
# 1) まず止める（§1）
# 2) 壊れたDBを退避してから、日次バックアップの最新を戻す
Move-Item data\kabu_auto.db data\kabu_auto.db.broken
Copy-Item data\backups\kabu_auto_YYYY-MM-DD.db data\kabu_auto.db
```

- `data/kabu_auto.db-wal` / `-shm` も一緒に退避する（古いWALが残ると整合が壊れる）。
- 戻したのは最後のバックアップ時点のデータで、それ以降の約定・シグナルは含まれない。§7（建玉ドリフト）を実施してから再開する。

### 11.2 PCごと壊れた（別PCで復元する）

コードはGitHub、状態は `scripts/backup_state.ps1` が作成したZIP（既定 `OneDrive\kabu-auto\`）にある。

1. ZIP内の `RESTORE.md` を開く（バックアップ時点のコミットハッシュ入りの手順）。
2. GitHubからclone → そのコミットへ合わせる → `pip install -r requirements.txt`
3. ZIPの中身をリポジトリ直下へ上書き展開する（`MANIFEST.txt` のSHA256で照合できる）。
4. `.env` を含めずに取得していた場合は `.env.example` から再設定する。
5. paper で起動して健全性を確認してから live へ戻す。
6. §7（建玉ドリフト）を実施する。復元直後はDBと実口座が食い違いやすく、そのまま live にすると kill switch で発注停止になる。

### 11.3 バックアップの取り方（平常時）

```powershell
powershell -File scripts\backup_state.ps1              # 既定: OneDrive へ7世代
powershell -File scripts\backup_state.ps1 -NoSecrets   # .env / auth.json を含めない
```

本体を止めずに実行できる（DBはオンラインバックアップAPIで読む）。
詳細は [README.md](../README.md) の「バックアップと復元」を参照。

---

## 12. 通知の見分け方

| 記号 | 意味 | 対応 |
|------|------|------|
| 🔴 | 要対応 | 再ログイン要求・未解決注文・整合性違反。すぐ対処する |
| 🟡 | 注意 | 損失上限接近・取引停止中。状況を確認する |
| 🟢 | 実行報告 | 約定・接続回復。対応不要 |
| ⚪ | 定期 | ハートビート（平日8:45）。届かない場合は異常 |

⚪が届かない日は、プロセスが停止している可能性がある。

よくある 🔴 の内訳:

| 通知 | 対処 |
|------|------|
| 再ログインが必要です | §4 |
| 完全自動ログインに失敗しました | §4.4 |
| 直近N分でエラーがM件 | ログを確認する。多くは認証切れの派生 |
| 直近N分で警告がM件 | リトライが失敗し続けている。接続先・認証を確認する |
| 場中にデータ取得が途絶 | §5。ジョブ停止・接続断を確認する |
| 起動以来一度も取得できていません | 認証切れのまま起動したか接続先の誤り。§4 → §5 |
| 建玉ドリフト検知 | §7 |
| 未解決注文が残っています | §6 |
| 🟡 新規シグナル0件（鮮度不足） | §5.1 |

Discord コマンドの一覧は `@kabu-auto help`。当日の運用概要は `@kabu-auto summary`。

---

## 13. チェックリスト（異常検知アラートを受けたら）

1. [ ] §1 で新規発注を停止したか
2. [ ] ブローカー実態（注文・建玉）を直接確認したか
3. [ ] DBとブローカーの差異を是正したか
4. [ ] 未解決注文がゼロになったか
5. [ ] 損失要因を記録したか（再発防止）
6. [ ] 取引停止を解除したか（§8）

---

## 14. Obsidianグラフノートの再生成

コードやドキュメントを更新してpushしたら、vaultのグラフノートも再生成する。

```bash
python scripts/gen_graph_notes.py
```

- 出力先は `Tama vault/Claude/graph/`（モジュール・節・index）
- 生成物はgit管理外の派生データで、再生成すれば復元できる
- iCloudがオフラインだと出力先の親が見えず中断する。その場合は同期を待って再実行する
- ノート名がvault内の既存ノートと衝突した場合も中断する。表示された名前を確認する

---

## 15. 候補モデルの学習・評価・昇格（手動）

このツールはスケジューラに登録されていない。手動で実行したときだけ動く。
週次再学習（`ml_retrain`）が作る候補とは独立に、任意のタイミングで候補を作成・評価・昇格できる。

### 15.1 候補モデルを学習する

```bash
python -m scripts.promote_workflow train
```

- ウォッチリスト全リストの銘柄（200本以上のOHLCVがあるもの）で学習する
- `risk_profile.json` のアクティブプロファイルを適用した実効値でラベルを作る
- 既定は全期間を使う（分析報告書と同条件）。直近N本に絞る場合は `--ohlcv-limit N`
  （週次再学習の既定500本の切り詰めは踏襲しない）
- 運用モデルは変更されない。候補が `models/candidates/<model_id>/` に保存されるだけ
- 決着イベントが200件に届かないと候補は作られず終了コード1になる

### 15.2 候補を評価する

```bash
python -m scripts.promote_workflow evaluate <model_id>
```

- 学習時に保存されたイベント表（`data/datasets/<dataset_id>.csv.gz`）を読み直し、walk-forward（5分割）で評価する
- fold別のAUC・Brierと、その平均を表示する
- 最後に `evaluation_run_id: <id>` を表示する。これが昇格の根拠になる
- `degraded: N件` と出た評価記録では昇格できない

AUC 0.5 は予測力が無い状態と区別できない。実データでの5モデル比較では、現行相当モデルのAUCは 0.5018（p=0.7231）だった（`docs/kabu-auto-ml-real-data-comparison_20260921.md`）。数値が出たこと自体は昇格の根拠にならない。

### 15.3 昇格できるか確認する（現行は変えない）

```bash
python -m scripts.promote_workflow promote <model_id> <evaluation_run_id> \
    --reason "..." --decided-by "..." --dry-run
```

拒否理由は全件表示される。

### 15.4 昇格する

```bash
python -m scripts.promote_workflow promote <model_id> <evaluation_run_id> \
    --reason "AUC・Brierを確認し現行より悪化がないため" --decided-by "garnet"
```

- `--reason` と `--decided-by` は必須。空文字・空白のみは拒否される
- 検査に通らなければ現行を変更せず終了コード1で終わる
- 成功すると `models/current.json` が新しい候補を指し、`model_promotions` に `state=committed` の記録が残る

### 15.5 戻す（ロールバック）

CLIには未実装。Pythonから `promotion.rollback()` を呼ぶ（判断者と理由は必須）:

```bash
python -c "from src.core import config as cfg; from src.data import database as db; from src.core import risk_profile as rp; cfg.load('config.yaml'); rp.load('risk_profile.json'); db.init(); from src.strategy import promotion; print(promotion.rollback(decided_by='garnet', reason='昇格後に成績が悪化したため'))"
```

### 15.6 shadow記録を読む（昇格後の観察）

昇格済みの候補モデルは実発注に使っていない。`signal_scan`（平日16:20）が毎日、現行モデルと同じ入力に対する候補の判断を `shadow_comparisons` テーブルへ記録するだけで、買い/売り/様子見の判断は現行モデル（legacy）が決める。候補が未昇格（`models/current.json` が無い）なら何も記録されない。

- 記録のID（`evaluation_run_id`）は日ごとに `shadow-YYYY-MM-DD`
- 判定閾値は 0.5固定（チューニングされた値ではない観察用の基準）
- 1日分の記録件数 ≒ その日スキャンした銘柄数（鮮度不足で除外された銘柄と、日足が75本に満たない銘柄は入らない）

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

読み方の注意:

- 勝ち負けの記録ではない。「どちらが当たったか」には実績（`PredictionOutcome`）が要るが、現時点では保存していない（最大保有10営業日経過まで確定しないため、今回の配線に含めていない）。読めるのは「候補が現行とどれだけ違う判断をしたか」だけ。
- `only_current` / `only_candidate` が片側へ偏っている場合、能力の差ではなく閾値の差を反映している可能性がある（候補の評価fold別閾値は 0.35〜1.0 に分布していた）。
- 昇格済み候補の評価AUCは平均0.5032で、chanceと統計的に区別できていない（`docs/kabu-auto-ml-real-data-comparison_20260921.md`）。
- 観察期間の目安は最低6ヶ月、可能なら1年（観察起点 2026-09-28）。1日あたりの記録は銘柄数ぶん（数十件）で、日足スイングの決着は1件あたり最大10営業日かかる。
- 記録が0件の日が続く場合は、①候補が昇格されているか（`models/current.json`）、②`signal_scan` が動いているか（ログの「シグナルスキャン開始」）、③ログに「shadow記録を行いません」「shadow記録の準備に失敗しました」が出ていないか、の順に確認する。
