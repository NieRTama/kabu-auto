# kabu-auto

日本株（東京証券取引所）向けの完全自動スイングトレードシステム。  
**ルールベース＋機械学習ハイブリッド戦略**により、auカブコム証券の kabuステーションAPI を使って数日〜数週間単位のポジションを自動管理する。

---

## 主な機能

| 機能 | 概要 |
|------|------|
| 自動売買 | ルールスコア＋LightGBM の合成スコアで買い・売りシグナルを生成し、自動発注 |
| ペーパートレード | 注文は実行せずシグナル・損益をシミュレート。実装確認に最適 |
| MLモデル自動再学習 | 毎週末に過去データで LightGBM を再学習（トリプルバリア法でラベリング） |
| バックテスト | 過去データでウォークフォワード方式の戦略検証（ルックアヘッドバイアスなし） |
| Webダッシュボード | ブラウザで損益・ポジション・取引履歴・シグナルをリアルタイム確認 |
| 損益可視化 | 日次損益棒グラフ・累積損益折れ線・MTD/YTD・シャープレシオ・最大ドローダウン |
| MLモデル精度追跡 | 各学習ごとに精度・AUC・特徴量重要度を記録・グラフ表示 |
| リスク管理 | 損切り・最大保有銘柄数・セクター集中制限・1日注文数上限・当日損失上限 |
| 緊急決済 | ダッシュボードのボタン1つで全ポジション成行決済 |
| アラート通知 | 大損失・API切断などを LINE Notify / メールで通知 |
| 自動バックアップ | SQLite DBを日次で `data/backups/` へ自動コピー |

---

## システム構成

```
kabu-auto/
├── src/
│   ├── api/           # kabuステーションAPI クライアント（REST + WebSocket）
│   ├── data/          # OHLCV取得・SQLite管理（WALモード）
│   ├── strategy/
│   │   ├── indicators.py   # テクニカル指標（MA/RSI/BB/MACD）
│   │   ├── labeling.py     # トリプルバリア法ラベリング
│   │   ├── ml_model.py     # LightGBM 学習・推論
│   │   └── signal.py       # ルール＋ML 合成シグナル生成
│   ├── execution/     # 発注・ポジション管理・約定確認
│   ├── risk/          # リスク管理（損切り・集中制限）
│   ├── backtest/      # ウォークフォワードバックテストエンジン
│   ├── dashboard/     # FastAPI バックエンド（16エンドポイント）
│   └── core/          # 設定・ログ・スケジューラ・アラート
├── frontend/
│   └── index.html     # ダッシュボードUI（Chart.js）
├── models/            # 学習済み LightGBM モデル（.pkl）
├── data/              # SQLite DB・ログ・バックアップ
├── docs/              # 概要設計書・詳細設計書
├── config.yaml        # 設定ファイル
├── requirements.txt
└── main.py            # エントリポイント
```

---

## 必要環境

| 項目 | 要件 |
|------|------|
| OS | **Windows**（kabuステーションが Windows 専用アプリのため） |
| Python | 3.11 以上 |
| Git | リポジトリ取得・更新に使用（[git-scm.com](https://git-scm.com/download/win) からインストール） |
| 証券口座 | auカブコム証券（kabuステーション API の利用申込が必要） |
| PC設定 | **スリープ無効化**（電源オプション → スリープしない）必須 |

> テクニカル指標の計算は `pandas` のみで実装しており、`pandas-ta` には依存しない
> （`pandas-ta==0.3.14b0` は Python 3.11 向けに PyPI で配布されていないため）。

---

## セットアップ

### 1. kabuステーションの準備

1. auカブコム証券で口座開設
2. kabuステーション をダウンロード・インストール
3. kabuステーションを起動し API を有効化（ツール → kabuステーションAPI）
4. PC のスリープを無効化（システム異常の原因になるため）

### 2. パッケージのインストール

```bash
pip install -r requirements.txt
```

### 3. config.yaml の編集

```yaml
kabu_station:
  password: "kabuステーションのログインパスワード"

trading:
  mode: "paper"          # まずペーパートレードで動作確認
  watchlist:
    - "7203"             # トヨタ自動車
    - "9984"             # ソフトバンクグループ
    - "6758"             # ソニーグループ

alerts:
  line_notify_token: ""  # LINE Notify トークン（任意）
  email: ""              # アラート送信先メール（任意）

dashboard:
  emergency_token: ""    # 空欄で起動時にランダム生成・ログに表示
```

### 4. 起動

```bash
python main.py
```

起動後、ブラウザで `http://localhost:8080` を開くとダッシュボードが表示される。  
緊急決済トークンはログ（`data/kabu_auto.log`）に出力される。

### 5. 初回データ取得とモデル学習

過去データは毎日16:00に自動取得されるが、起動直後にすぐ試したい場合は API で手動実行できる。

```bash
# ① ウォッチリスト全銘柄の過去データを取得（yfinance、数分かかる）
curl -X POST http://localhost:8080/api/data/update

# ② 取得したデータでMLモデルを学習
curl -X POST http://localhost:8080/api/model/retrain

# ③ 学習結果を確認
curl http://localhost:8080/api/model/latest
```

PowerShell の場合は `Invoke-WebRequest -Uri <URL> -Method POST` を使用する。

---

## 取引モード

| モード | 説明 |
|--------|------|
| `paper` | ペーパートレード — 注文は実行されずシグナル・損益をシミュレート |
| `live` | 本番 — kabuステーションAPI 経由で実際に発注 |

`config.yaml` の `trading.mode` で切り替える。**必ずペーパーモードで動作確認してから本番へ。**

ライブモードで起動するには、環境変数 `CONFIRM_LIVE_TRADING=true` の設定が必要です。

```bash
CONFIRM_LIVE_TRADING=true python main.py
```

未設定の場合は起動時にエラーメッセージを表示して終了します。

---

## Webダッシュボード

`http://localhost:8080` でアクセス。自動で10秒ごとに更新される。

### 表示内容

**損益サマリ（上段）**

| 項目 | 説明 |
|------|------|
| 累積損益 (realized) | 確定済み損益の累計 |
| 含み損益 (unrealized) | 保有銘柄の評価損益（最新OHLCV価格ベース） |
| 今日の損益 | 当日の確定損益 |
| MTD | 月初来損益 |
| YTD | 年初来損益 |

**パフォーマンス指標（下段）**

| 項目 | 説明 |
|------|------|
| 勝率 | 勝ちトレード数 ÷ 全トレード数 |
| 勝ち/負け | 勝ちトレード数 / 負けトレード数 |
| 最大DD | 最大ドローダウン（累積損益ピーク比） |
| シャープレシオ | 日次損益の平均/標準偏差 × √252 |
| 保有数 | 現在の保有銘柄数 |
| システム状態 | paper/live・WebSocket接続状態 |

**グラフ**
- 累積損益チャート（折れ線・90日）
- 日次損益チャート（棒グラフ・緑/赤）

**テーブル**
- 現在ポジション（銘柄・数量・平均取得単価・現在値・含み損益・リターン%）
- 取引履歴（銘柄・売買・価格・数量・損益・リターン%・日時）
- シグナル履歴（銘柄・スコア・理由・日時）

### 緊急全ポジション決済

ダッシュボード右上の「緊急全決済」ボタンを押すと、全保有銘柄を成行で即時決済する。  
本番モードでは `X-Emergency-Token` ヘッダーが必要（起動ログを参照）。

---

## バックテスト

ダッシュボードの「バックテスト実行」パネルから、または API 経由で実行できる。

```bash
curl -X POST http://localhost:8080/api/backtest/run \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["7203", "6758"], "start_date": "2023-01-01", "end_date": "2023-12-31"}'
```

- **ウォークフォワード方式**: 各営業日時点で入手可能なデータのみ使用（ルックアヘッドバイアスなし）
- バックテスト期間前のデータのみで ML モデルを事前学習
- 結果は SQLite に保存され、ダッシュボードの「バックテスト結果」タブで確認できる

---

## ML モデルの精度追跡

毎週末の自動再学習または手動再学習のたびに以下を記録する。

| 指標 | 内容 |
|------|------|
| accuracy | テストセット精度 |
| auc | ROC-AUC スコア |
| n_estimators | 最適木の本数（CV 平均） |
| 特徴量重要度 | 上位特徴量ランキング |

ダッシュボードの「MLモデル精度」タブで精度推移をグラフ表示できる。

---

## リスク管理の設定

`config.yaml` の `trading` セクションで調整できる。

```yaml
trading:
  max_positions: 5           # 最大同時保有銘柄数
  max_position_ratio: 0.20   # 1銘柄最大投資額（総資金比 20%）
  stop_loss_pct: -0.05       # 損切りライン（-5%）
  max_sector_ratio: 0.40     # 同一セクター集中率上限（40%）
  order_timeout_seconds: 300 # 未約定注文の自動キャンセル（秒）
  daily_order_limit: 100     # 1日の最大注文数
  max_daily_loss: 30000      # 当日損失上限（円）。0 で無効
```

---

## 戦略パラメータの調整

`config.yaml` の `strategy` セクションで調整できる。

```yaml
strategy:
  ma_short: 5          # 短期移動平均（日）
  ma_mid: 25           # 中期移動平均
  ma_long: 75          # 長期移動平均
  rsi_period: 14       # RSI 期間
  rsi_oversold: 30     # RSI 売られすぎ閾値
  rsi_overbought: 70   # RSI 買われすぎ閾値
  bb_period: 20        # ボリンジャーバンド期間
  bb_std: 2.0          # ボリンジャーバンド標準偏差倍率
  ml_weight: 0.5       # ML シグナルの重み（0〜1）
  rule_weight: 0.5     # ルールベースシグナルの重み（0〜1）
  buy_threshold: 0.25  # 買いシグナル合成スコア閾値
  sell_threshold: -0.25 # 売りシグナル合成スコア閾値
```

---

## 主要スケジュール

| 時刻 | 処理 |
|------|------|
| 08:30 | kabuステーション API トークン自動更新 |
| 09:05 | 前日シグナルを元に発注（ライブモード: SELL優先 → BUY） |
| 09:00〜15:30（5分毎） | 損切りチェック・シグナルスキャン |
| 15:35 | シグナルスキャン・翌営業日の売買候補をスキャン |
| 16:00 | 日次OHLCVデータ更新 |
| 17:00 | SQLite 日次バックアップ |
| 毎週末 | LightGBM モデル自動再学習 |

---

## APIエンドポイント一覧

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/` | ダッシュボード HTML |
| GET | `/api/status` | システム稼働状態 |
| GET | `/api/positions` | 現在ポジション（含み損益付き） |
| GET | `/api/trades` | 取引履歴 |
| GET | `/api/signals` | シグナル履歴 |
| GET | `/api/pnl_summary` | 損益サマリ（基本） |
| GET | `/api/pnl_chart` | 累積損益チャート用データ |
| GET | `/api/pnl/daily` | 日次損益リスト（棒グラフ用） |
| GET | `/api/pnl/enhanced_summary` | 損益サマリ（MTD/YTD/Sharpe/DD） |
| POST | `/api/emergency_close` | 緊急全ポジション決済 |
| GET | `/api/model/metrics` | モデル精度履歴 |
| GET | `/api/model/latest` | 最新モデルの学習結果 |
| POST | `/api/model/retrain` | MLモデル手動再学習トリガー |
| POST | `/api/data/update` | 過去データを手動取得（yfinance） |
| GET | `/api/watchlist` | ウォッチリスト銘柄一覧 |
| POST | `/api/backtest/run` | バックテスト実行 |
| GET | `/api/backtest/runs` | バックテスト結果一覧 |
| GET | `/api/backtest/{run_id}` | バックテスト結果詳細 |

---

## テスト・CI

ユニットテストは `pytest` で実行できる。

```bash
pytest tests/ -v
```

GitHub Actions により、プッシュ・プルリクエスト時に自動で全テストが実行される（`.github/workflows/test.yml`）。

---

## 注意事項

- **投資は自己責任**。本ツールの利用による損失に対して作者は責任を負いません。
- kabuステーションAPIには**1日の注文数制限**があります（`daily_order_limit` で管理）。
- スイングトレードでは翌朝の**窓開けリスク**（ストップ高/安）により、損切りラインを大幅に超えて約定する場合があります。
- 本番移行前に必ずペーパーモードで**数週間以上**の動作確認を行ってください。
- データベースバックアップは `data/backups/` に日次保存されます。定期的に外部メディアへコピーしてください。
- 毎週土曜深夜の kabuステーション**システムメンテナンス時間帯**は自動的に稼働を停止します。
- `config.yaml` は起動時に一度だけ読み込まれます。編集後は **`main.py` の再起動**が必要です。
