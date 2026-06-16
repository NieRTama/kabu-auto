# kabu-auto

日本株（東証）向けの完全自動売買ツール。ルールベース＋機械学習ハイブリッド戦略によるスイングトレード（数日〜数週間）。

## 必要環境

- **OS**: Windows（kabuステーションがWindows専用のため）
- **Python**: 3.11以上
- **証券会社口座**: auカブコム証券（kabuステーション申込が必要）

## セットアップ

### 1. kabuステーション の準備

1. auカブコム証券で口座開設
2. kabuステーションをダウンロード・インストール
3. kabuステーションを起動してAPIを有効化（ツール → kabuステーションAPI）
4. **PC のスリープを無効化**（電源オプション → スリープしない）

### 2. Pythonパッケージのインストール

```bash
pip install -r requirements.txt
```

### 3. 設定ファイルの編集

`config.yaml` を編集:

```yaml
kabu_station:
  password: "kabuステーションのログインパスワード"

trading:
  mode: "paper"  # まずペーパートレードで確認

alerts:
  line_notify_token: "LINE Notifyのトークン"  # 任意
```

### 4. ウォッチリストをconfig.yamlに追加

`trading` セクションに `watchlist` キーを追加:

```yaml
trading:
  mode: "paper"
  watchlist:
    - "7203"   # トヨタ自動車
    - "9984"   # ソフトバンクグループ
    - "6758"   # ソニーグループ
```

### 5. 初回データ取得とモデル学習

```bash
python -c "
from src.core import config as cfg
cfg.load()
from src.data import database as db
db.init()
from src.data.market_data import update_symbol
from src.strategy import ml_model
from src.data.market_data import load_ohlcv

for sym in ['7203', '6758', '9984']:
    update_symbol(sym, years=3)

df = load_ohlcv('7203')
ml_model.train(df)
print('完了')
"
```

### 6. 起動

```bash
python main.py
```

ダッシュボード: http://localhost:8080

## 取引モード

| モード | 説明 |
|--------|------|
| `paper` | ペーパートレード（注文は実行されずログのみ） |
| `live`  | 本番（実際の資金で発注） |

**必ずペーパーモードで動作確認してから本番に切り替えてください。**

## ディレクトリ構成

```
kabu-auto/
├── src/
│   ├── api/          # kabuステーションAPIクライアント
│   ├── data/         # DB管理・市場データ取得
│   ├── strategy/     # テクニカル指標・MLモデル・シグナル生成
│   ├── execution/    # 発注・ポジション管理
│   ├── risk/         # リスク管理
│   ├── dashboard/    # FastAPIダッシュボード
│   └── core/         # 設定・ログ・スケジューラ・アラート
├── frontend/         # ダッシュボードUI
├── models/           # 学習済みMLモデル
├── data/             # SQLiteDB・ログ・バックアップ
├── config.yaml       # 設定ファイル
├── requirements.txt
└── main.py           # エントリポイント
```

## リスク管理の設定

`config.yaml` で調整可能:

```yaml
trading:
  max_positions: 5           # 最大同時保有銘柄数
  max_position_ratio: 0.20   # 1銘柄最大投資額（総資金比）
  stop_loss_pct: -0.05       # 損切りライン（-5%）
  max_sector_ratio: 0.40     # 同一セクター集中率上限
  order_timeout_seconds: 300 # 未約定注文の自動キャンセル（秒）
```

## 注意事項

- **投資は自己責任**。本ツールの利用による損失に対して作者は責任を負いません。
- kabuステーションAPIには**1日の注文数制限**があります。
- スイングトレードでは翌朝の**窓開けリスク**（ストップ高/安）により損切り価格を大幅に超えて約定することがあります。
- **定期的にバックアップを確認**してください（`data/backups/` フォルダ）。
