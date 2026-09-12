# Kabuステーション完全自動ログイン統合 設計書

- 日付: 2026-09-11
- 対象: kabu-auto（本体）+ kabusapi-auto-login-template（WSL2/Docker側、`~/projects/kabusapi-auto-login-template` に導入済み）

## 背景・目的

kabu-auto は現状、Kabuステーション（KabuS.exe）の**起動のみ**を自動化しており（`src/core/broker_launcher.py`）、
2段階認証（ワンタイムパスワード入力）は意図的に自動化していない
（`auto_launch_broker: false`、コメント「認証（2段階認証）は自動化しない——専用認証アプリでの承認は人が行う」）。

今回、外部リポジトリ [kabusapi-auto-login-template](https://github.com/nakanishi1337/kabusapi-auto-login-template) を
WSL2 + Docker Engine環境に導入し（環境構築済み。WSL2 Ubuntu 26.04、Docker Engine、リポジトリclone、`.env`雛形まで完了）、
Gmail API経由でワンタイムパスワードを自動取得・自動入力する仕組みを利用可能にした。

**本設計は、この「2段階認証を自動化しない」という既存方針を意図的に転換し、
Kabuステーションの起動〜ログイン〜2段階認証入力までを完全自動化した上で、
kabu-auto本体の既存スケジューラ・Discordコマンド経路の両方から呼べるように統合するためのもの。**
（ユーザー確認済み: 「意図した転換（完全自動ログインへ）」「両方組み合わせたい」）

## 前提（環境構築で確認済みの事実）

- Kabuステーション本体は `%LOCALAPPDATA%\kabuStation\KabuS.exe` に導入済み
- 既存kabu-autoは Kabuステーション API に **`http://localhost:18080/kabusapi` へ直接接続**している
  （Windows上で直接動くプロセスのため）。テンプレートのnginxプロキシ（28080）・`kabu-proxy`コンテナは
  「WSL内のDockerコンテナから取引する」用途のものであり、**本統合では不要**。
- テンプレートの `scripts/kabustation/login_kabustation.ps1` は、**既存のKabuS.exeプロセスを
  問答無用でkillしてから再起動する**設計。既存kabu-autoの監視は「プロセスの生死」自体でなく
  「API接続できるか（401等）」を見ているため（`main.py`, `probe_running()`の用途はDiscord案内文言と
  auto_launch_broker分岐のみ）、致命的な誤検知の可能性は低いが、再起動中の一時的な接続断が
  `health_check`（15分毎エラー率）や`auth_recovery_check`（5分間隔）のノイズになりうる点は
  導入後に実測確認する。
- 既存のジョブスケジューラは `src/core/scheduler.py` の `add_job` 方式（cron/interval）。
  Discordコマンドは `main.py` に `_cmd_xxx` 関数として実装され、既存の `launch` コマンド
  （`_cmd_launch`）は「認証は認証アプリで人が行う」という前提のまま維持する。

## アーキテクチャ概要

新モジュール `src/core/broker_full_login.py` を新設する（既存 `broker_launcher.py` とは別ファイル。
「プロセスを起動するだけ」という既存の責務を保ったまま、責務が異なる「起動+ログイン+2FA自動入力」を
分離するため）。

テンプレート側リポジトリ（`kabusapi-auto-login-template`）には、`kabu-proxy` を起動しない
専用スクリプト `scripts/kabustation/run_login_only.sh` を追加する
（既存の `run_kabustation_api.sh` から `kabu-proxy` 起動と `run_nginx.ps1` 呼び出しを除いたもの。
既存kabu-autoは直接18080に接続するためプロキシ不要）。

- `src/core/scheduler.py`: 新規cronジョブ `broker_full_login`（平日06:45、既存の `risk_reset`(8:25)より前）
- `main.py`: 新規Discordコマンド `full_login`（手動トリガー用。既存 `launch` コマンドは変更しない）
- `config.yaml`: `runtime.broker_full_login_enabled`（既定 `false`）、実行時刻等の設定を追加

## データフロー

1. スケジューラ（06:45）または Discordコマンド `full_login` が `broker_full_login.run(manual: bool)` を呼ぶ
2. 内部で以下を `subprocess.run(..., timeout=180)` により同期実行:
   ```
   wsl -d Ubuntu -- bash -lc "cd ~/projects/kabusapi-auto-login-template && ./scripts/kabustation/run_login_only.sh"
   ```
   `run_login_only.sh` の内部フロー:
   `docker compose up -d onetime_password` → `login_kabustation.ps1`（起動+パスワード入力）→
   `get_onetime_password.py`（Gmail APIでOTP取得）→ `input_onetime_password.ps1`（OTP自動入力）
3. 終了コード・標準出力/標準エラーをパースして成功/失敗を判定する
4. 結果を既存の通知フォーマット（🟢実行報告 / 🔴要対応）で送信する
5. その後「実際にAPI接続できたか」の確認は、既存の `auth_recovery_check`（5分間隔）にそのまま委ねる
   （本モジュールは変更しない。責務分離を保つ）

## 安全策

- **二重起動防止**: `broker_launcher.py` と同様のロック・クールダウンを `broker_full_login.py` にも実装する。
  同時に2回走ると `login_kabustation.ps1` の「既存プロセスをkillしてから再起動」が競合するため必須。
- **タイムアウト必須**: `subprocess.run(timeout=180)`。WSL側スクリプトがハングしても
  kabu-auto本体プロセスを巻き込まない。タイムアウト時は🔴通知。
- **段階導入**: `config.yaml` の `broker_full_login_enabled: false` を既定にし、
  実際に動かして安全性を確認してから `true` に切り替える（Knowledge.md §9の型を踏襲）。
- **既存監視への影響確認**: 再起動中（最大1〜2分程度見込み）の一時的な接続断が、
  `health_check` のエラー率監視やハートビートに乗ってノイズにならないか、
  導入後の実運用で実測する。ノイズになる場合は、既知の再起動時間帯を監視側で抑止する
  （Knowledge.md §15「既知の状態フラグで抑止する」の型を検討）。

## テスト方針

- `broker_full_login.py`: 外部コマンド（`wsl` 呼び出し）をモックしたユニットテストで、
  クールダウン・日次上限・タイムアウト処理・成功/失敗判定を検証する
  （既存 `tests/test_broker_launcher.py` のパターンを踏襲）。
- `scheduler.py`: 新規ジョブが正しい曜日・時刻で登録されるかのテスト
  （既存 `tests/test_reconcile_scheduling.py` 等に倣う）。
- `main.py`: 新規Discordコマンド `full_login` のテスト（既存 `launch` コマンドのテストパターンに倣う）。
- 実機統合テスト（実際にWSL経由でKabuステーションへのログインが通るか）は、
  パスワード・Gmail API認証情報が要るため自動テスト化できない。
  **ユーザー立ち会いのもとで別途手動確認する。**

## ドキュメント更新

- `config.yaml` の「2段階認証は自動化しない」コメントを、今回の方針転換を反映して更新する
  （転換前の方針・転換した理由が分かる形で残す）。
- CLAUDE.mdの取り決めに従い、`docs/詳細設計書.md` / `docs/概要設計書.md` に今回の統合を追記し、
  push と同じタイミングでObsidian vaultへも同期する。
- テンプレート側リポジトリに追加する `run_login_only.sh` にもコメントを残す。

## スコープ外（今回やらないこと）

- `kabusapi-auto-login-template` の `trade` コンテナ・`docker-compose` を使った取引実行の統合
  （既存kabu-autoは独立したWindows常駐プロセスとして稼働を続ける。テンプレートは
  「起動+ログイン自動化」の部分のみ利用する）。
- Gmail APIのOTP抽出ロジック（`get_onetime_password.py` の正規表現）の改善。
  現状のテンプレート実装をそのまま利用し、誤爆が実運用で問題になった場合に別途対応する。
