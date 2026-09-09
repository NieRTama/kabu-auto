# ML評価基盤の是正 設計書（段階A〜E）

作成日: 2026-09-10
対象: `docs/kabu-auto-detailed-review_20260910.md` の F01〜F09、および段階D（ポートフォリオwalk-forward）・段階E（モデル昇格とshadow）
スコープ外: F10〜F15（リスク価格・プロファイル適用・認証・ログマスキング・DB制約）は別プランとして切り出す

---

## 1. 目的

現状、MLが利益に寄与しているかを**判定できない**。保存済みAUCは0.5085だが、
その数値が意味を持つための前提（時系列の分離、データの鮮度、ラベルの定義、
判断時点と執行時点の分離）が揃っていないため、良い数字が出ても運用への
再現性を評価できない。

本設計の目的は「MLを効かせること」ではなく、**MLが効いているかを測れる状態を作ること**である。
効果測定はこの基盤が入ってから行う。

## 2. 精査結果

レビューはzipスナップショットを対象としていたが、現行リポジトリ（`main` @ 932f50f）と
突き合わせた結果、**F01〜F09は全て現存する**。以後のコミットは発注系の修正が中心で、
本設計の対象領域には及んでいない。

| ID | 現行コードでの確認箇所 |
|---|---|
| F01 | `src/data/market_data.py:87-90` `end = date.today()` を排他境界のまま `yf.download` へ渡す。`src/services/trading.py:346` `signal_scan` は最終足の日付を検査しない |
| F02 | `src/strategy/ml_model.py:117-140` `pd.concat(..., ignore_index=True)` 後に `TimeSeriesSplit` |
| F03 | purge/embargo はリポジトリ全体に実装が無い |
| F04 | `src/backtest/engine.py:113-190` 同一の `close_price` でスコア生成と約定を行う |
| F05 | `src/strategy/labeling.py:55-75` `end = min(i+max_holding, n-1)` により未成熟イベントにもラベルが付く |
| F06 | `src/strategy/indicators.py:11-16` `loss.replace(0, nan)` で単調上昇時にRSIがNaN。`indicators.py:83` の `dropna` 後に行番号で保有期間を数える |
| F07 | `src/services/trading.py:181` `load_ohlcv(sym)` は既定500行 |
| F08 | `src/strategy/signal.py:78` ML欠落時にルール重みがそのまま残る |
| F09 | `src/services/trading.py:189` `self.model = train_multi(...)` の直接代入 |

### レビュー記述の補正

1. **F02は既知の負債である。** `ml_model.py:127-133` のdocstring自身が
   「全体としての時系列順序は厳密ではない（略）将来の改善課題とする」と明記している。
   新規発見ではなく、明示的に先送りされた負債の回収として扱う。
2. **F10の前半は修正済み。** `src/risk/manager.py:117-126` は既にリアルタイム価格関数を
   優先し、失敗時のみDB終値へ落ちる（コミット c2a0630）。残る問題は戻り値が素のfloatで
   鮮度・由来を判別できない点のみ。この項目は別プランへ送る。
3. **テスト網羅の実態。** レビューは「テスト一式は共有外」としたが、実リポジトリには60本超ある。
   ただし `indicators` / `labeling` / `market_data` の単体テストは**存在しない**。
   F01・F05・F06は回帰テストがゼロであり、ここが最大の穴である。

### レビューに無い追加の指摘

`src/services/trading.py:311-314` の `stop_loss_check` は paper モードで日足終値を用いて
損切りを判定している。F04（同一終値での判断・約定）は**バックテストだけでなく
paper運用にも及んでいる**ため、段階Dの是正範囲に paper 経路を含める。

## 3. 設計方針

F04（同一終値で判定・約定）とF05（ラベルが運用規則と不一致）は別々の欠陥に見えるが、
根は一つである。「売買判断 → 執行 → 退出」の規則が、ラベル生成・バックテスト・実運用の
3箇所に別々の実装として散らばっており、片方を直しても他方とずれる。

したがって**規則を1箇所に集約する**設計を採る。個別修正（既存モジュールのin-place改修）は
工数では有利だが、直した直後は一致していても次に退出条件を触った時に静かにずれるため採らない。
検証系を完全に別パッケージへ分離する案も、運用と検証が別実装になり
「バックテストで良かったものが運用で再現しない」という問題そのものを再生産するため採らない。

## 4. 全体構造

```
src/strategy/
  policy.py      新規  ExitPolicy — 売買規則の唯一の実装。3者が共有する
  dataset.py     新規  イベント表の生成・保存・読込
  validation.py  新規  カレンダー分割 + purge/embargo + 入れ子CV
  indicators.py  改修  端点を仕様化、日付を保持、マスクで管理
  labeling.py    改修  policy.py を呼ぶ薄い層に縮小
  ml_model.py    改修  validation.py を使う。候補保存と昇格の分離
  signal.py      改修  strategy_version を明示
src/backtest/
  engine.py      据置  旧単一銘柄エンジン。無改造で残す
  execution.py   新規  執行仮定（翌営業日寄り・悲観的ギャップ・未約定）
  portfolio.py   新規  現金・保有・銘柄間の資金競合
  walkforward.py 新規  ポートフォリオ walk-forward エンジン
src/data/
  market_data.py 改修  取得境界・鮮度検査・PriceQuote
```

各モジュールの責務境界:

| モジュール | 何をするか | 何に依存するか |
|---|---|---|
| `policy.py` | エントリと以降の足から退出時点・価格・理由を決める純粋関数 | 設定のみ（DB・ネットワークに触らない） |
| `dataset.py` | イベント表の生成・`csv.gz` への保存・`dataset_id` の払い出し | `policy.py` `indicators.py` `market_data.py` |
| `validation.py` | 日付境界でのfold分割、purge、入れ子CV、予測明細の収集 | `dataset.py` |
| `execution.py` | 注文をいつ・いくらで約定させるかの仮定 | `policy.py` |
| `portfolio.py` | 現金・保有・セクター上限・銘柄間の資金競合 | なし（純粋な状態機械） |
| `walkforward.py` | 日次ループの進行と再学習スケジュール | 上記すべて |

## 5. 段階A: データ土台（F01）

### 取得境界

`fetch_ohlcv` に渡す `end` を `end + 1日` にする。yfinance公式仕様の `end` は排他境界であり、
現行実装は営業日Tの引け後に実行してもTの足を取得できない。

### 鮮度検査

```python
@dataclass(frozen=True)
class DataFreshness:
    symbol: str
    last_bar_session: date | None
    as_of_session: date
    state: str  # "fresh" | "stale" | "missing"
```

- `as_of_session` は `src/core/market_calendar.is_business_day` を用いて直近営業日から確定する
- `update_symbol` は取得後の `last_bar_session` を返す
- `signal_scan` は `state != "fresh"` の銘柄を**新規候補から除外**し、理由を記録する
- 更新結果は全体成功・部分成功・失敗と対象銘柄数を記録する

既存ポジションの退出管理は候補生成の阻害と分離する。鮮度不足でも退出は止めない。

### 判断時点の記録

`Signal` に `generated_at`（シグナルを作った時刻）と `data_as_of`（使ったデータの最終営業日）を
**別々に**保存する。現行は生成日時のみで、古い足から作られたシグナルを区別できない。

### 調整済み価格と実約定価格の分離

`upsert_ohlcv` は現在 `adjusted_close` に `close` と同じ値を書いている（`market_data.py:66,78`）。
取得は `auto_adjust=True` なので両者は同一の調整済み系列である。

本段階では**スキーマ変更を行わず**、`dataset_id` に取得時点を記録することで
「コード変更による成績差」と「データ改訂による成績差」を分離できる状態にする。
権利落ち・分割イベントの明示的な保存は段階Dの完了後に別途判断する。

## 6. 段階B: 特徴量・ラベル・共有ポリシー（F05・F06）

### ExitPolicy

```python
@dataclass(frozen=True)
class ExitDecision:
    exit_at: date | None
    exit_price: float | None
    reason: str      # STOP_LOSS / TRAILING / SIGNAL_SELL / TIME_BARRIER / UNRESOLVED
    resolved: bool
```

`ExitPolicy.evaluate(entry, bars, scores) -> ExitDecision` を
**ラベル生成・バックテスト・運用の3者が呼ぶ。**

引数の意味を明示する。

| 引数 | 内容 |
|---|---|
| `entry` | `symbol` / `entry_at` / `entry_price` / `quantity` |
| `bars` | `entry_at` 以降の日足（`date` / `open` / `high` / `low` / `close`）を時系列昇順で |
| `scores` | `date -> combined_score` の対応。`SIGNAL_SELL` 判定に使う。ラベル生成時に売りシグナルを考慮しないなら空辞書を渡す |

`ExitDecision.exit_price` は**コスト控除前の約定価格**とする。
手数料・スリッページの控除は `execution.py` が一元的に行い、`net_return` は
`dataset.py` がその控除後の値として算出する。ポリシーに価格以外の会計を持ち込まない。

悲観規約（3者共通・4項目）:

1. 同一日に損切り線と利確線の両方へ到達した場合は**損切りを採用する**。
   日足では到達順を復元できないため、不利側先行を仮定する。
2. 寄りが既に損切り線を割っている場合は `min(open, stop)` で約定する。
   現行エンジンは損切り線ちょうどで約定できる前提になっており、ギャップダウンで楽観に振れる。
3. `max_holding` までに決着せず足が尽きた場合は `resolved=False` とし、
   **ラベル対象から除外する**。現行は最終リターンの符号でラベルを付けており、
   未来1行しかないサンプルにもラベル1が付く。
4. 引けで判断し翌営業日の寄りで執行する。`entry_price = open × (1 + slip)`。

規則が1箇所にあるため、退出条件を変更するとラベル・バックテスト・運用が同時に追随する。

### RSIの端点仕様

| 条件 | 値 |
|---|---|
| `loss == 0` かつ `gain > 0` | 100 |
| `gain == 0` かつ `loss > 0` | 0 |
| 両方 0 | 50 |

### 欠損の扱い

`build_features` の `dropna(subset=FEATURE_COLS)` を廃止し、**日付インデックスを保持したまま
`feature_valid` マスク列**を付ける。現行は `dropna` 後のDataFrameを `reset_index(drop=True)` して
行番号を「N営業日」として数えているため、途中欠損があるとラベルの保有期間が実日数とずれる。

学習への採否はマスクで管理し、時間軸の連続性は市場系列側で保つ。

### イベント表

1行 = 1候補。

| 列 | 役割 |
|---|---|
| `symbol` | 対象銘柄 |
| `decision_at` | 売買判断の時点（Tの引け） |
| `feature_as_of` | 特徴量に使った情報の最終時点 |
| `entry_at` | 想定した執行時点（T+1の寄り） |
| `label_end_at` | ラベル確定に使った最後の時点 |
| `label` / `net_return` | 予測対象（コスト控除後） |
| `exit_reason` | ExitDecision の理由 |
| `sample_weight` | 学習用重み（既存の average uniqueness を流用） |
| `dataset_id` / `feature_version` / `strategy_version` | 再現用識別子 |

永続化は `data/datasets/<dataset_id>.csv.gz`。pandas標準で読み書きでき**依存追加が不要**である
（`requirements.txt` に pyarrow が無いため parquet は採らない）。メタ情報のみ新テーブル `Dataset` に記録する。

## 7. 段階C: 学習・選択・最終評価の分離（F02・F03・F07）

### 分割

fold境界は**全銘柄共通のカレンダー日付**で切る。行番号ではない。
`TimeSeriesSplit` を連結行に適用する現行方式では、銘柄Aの2026年が学習側・
銘柄Bの2024年が検証側に入る分割が成立する。

### purge と embargo

学習側から `label_end_at >= 検証開始日` のイベントを除外する（purge）。
片方向walk-forwardでは検証後の未来がそのfoldの学習に入らないため、
**後方embargoは既定で無効**とし、設定で有効化のみ可能にする。

### 入れ子構造

- **外側fold** = 最終評価専用。一切触らない
- **内側fold** = early stopping・閾値選択・確率校正

現行は early stopping に使った検証データでそのままCV指標を出しているため、
報告値が楽観に寄っている（`ml_model.py:147-160`）。

### 予測明細の保存

サンプル単位で `symbol / decision_at / y_true / p / fold / model_id` を保存する。
新テーブル `PredictionDetail`。これがあると**AUCも売買判断も後から再計算できる**ため、
「AUC 0.5085が何を意味するか分からない」という現状が再発しない。

### 比較対象（5つ固定）

定数確率 / 多数派予測 / ロジスティック回帰 / 小さいLightGBM / 現行LightGBM。
**深層モデルはこの段階では候補にしない。** 同じ入力・同じ分割で追加価値を示せるかを先に確かめる。

### 学習窓（F07）

`training_window_sessions` として明示パラメータ化し、実際の開始・終了日と銘柄ごとの採用件数を
保存する。拡大窓と一定期間の移動窓を同じ分割で比較して選ぶ。
現行の「既定500行」は意図した選択ではなく `load_ohlcv` の既定値である。

### 記録する指標

- fold別・銘柄別・期間別の件数、正例率、除外件数
- ROC-AUC、PR系指標、log loss、Brier と**定数モデルとの差**
- 予測確率のヒストグラムと校正曲線（校正は独立した過去期間で学習）
- 取引に採用した上位群のコスト控除後成績
- 重要特徴の安定性

## 8. 段階D: ポートフォリオ walk-forward（F04・F08）

### 日次ループ（5フェーズ）

1. 前日までに決まった注文の執行（`execution.py`）
2. 保有と現金の更新（`portfolio.py`）
3. `ExitPolicy` による退出処理
4. 日末評価（NAV記録）
5. 翌日の候補生成

**Tの終値の情報はT+1以降の注文にしか使えない。**

### 扱う条件

複数銘柄の資金競合、セクター上限、未約定、部分約定、出来高制約。
運用と同じ週次再学習スケジュールを再現する。
現行は開始前に一度だけ学習し、テスト期間中は再学習しない（`engine.py:79-94`）。

### 実行条件のスナップショット

実行開始時に `run_id / strategy_version / config_hash / dataset_id / model_id /
code_version / execution_model_version` を保存する。
現行 `BacktestRun` は閾値とコストしか持たず、保存済み44実行のうち37実行はコストがNULLで再現できない。

結果は最終リターンだけでなく、日次NAV・保有・注文候補・除外理由・予測確率まで辿れる形にする。

### 例外の扱い

`engine.py:154-155` の `except Exception: pass` を廃止する。
推論例外が1件でも発生した実行には `degraded=true` を立て、比較とモデル昇格から除外する。

### 戦略バージョンの比較（F08）

同一条件で3案を比較する。

1. 既存の加重合成（`rule_score * rule_weight + ml_score * ml_weight`）
2. 縮尺を明示したルール単独
3. ルールで候補を作り、MLで買う・見送るの順位を決める

**まず3を試す。** ルールが候補としたイベントに対してのみ「この取引はコスト控除後に価値があるか」を
学習するため、全日付を機械的に買い・売りへ変換せず、現行ルールへの追加効果を測定しやすい。

期待値は `p × 平均利益 − (1−p) × 平均損失 − コスト` で扱い、利益側と損失側が非対称なら
採用確率の境界は0.5にならない。それぞれの推定値は学習期間だけで求める。

モデル失敗時は「ルール単独の定義済み戦略へ切替」または「新規候補生成を停止」の
どちらかを**設定として明示する**。現行は暗黙にルール重みだけが残る。

### paper経路の是正

`src/services/trading.py:311-314` の `stop_loss_check` が paper モードで日足終値を使う点も
同じ執行仮定（`execution.py`）へ寄せる。

## 9. 段階E: 候補モデルの昇格と shadow（F09）

学習成功はモデル更新ではなく**候補の生成**である。

- 候補は `models/candidates/<model_id>/` に置く
- `models/current/` は評価を通るまで保持する。
  現行の実体は `models/lgb_model.pkl` + `models/lgb_model.meta.json`（`ml_model.MODEL_PATH`）である。
  初回の `v2` 起動時にこれを `models/current/` へ複製し、**元のファイルは削除しない**。
  `legacy` へ戻したときに現行運用がそのまま動く必要があるため
- 保存形式は pickle を廃し、LightGBMネイティブ形式 + JSONメタとする（pickleは任意コード実行の経路）
- メタには `model_id` / 学習窓 / 銘柄集合 / ラベル定義 / 特徴量定義 / クラス比率 /
  fold別結果 / コード版 / 依存関係版 / `dataset_id` を保存する
- 途中書き込みや学習失敗で現行版が失われないようにし、前のモデルへ戻せるようにする

### shadow運用

同じ入力に対し現行と候補の判断を並行記録する。**候補は発注に繋がない。**
差が何によって生じたかを、見送った候補の結果も含めて記録する。

## 10. 段階投入と後方互換

運用は止めない。

- `config.yaml` に `strategy.engine_version: legacy | v2` を追加。**既定は `legacy`**

  この設定が切り替えるのは次の3点だけである。それ以外の経路は値に関わらず従来どおり動く。

  | 対象 | `legacy` | `v2` |
  |---|---|---|
  | バックテストの実行主体 | `backtest/engine.py` | `backtest/walkforward.py` |
  | 学習データの生成元 | `labeling.build_training_set` | `dataset.build_events` |
  | CV分割 | `TimeSeriesSplit` | `validation.calendar_split` |

- 新モジュールは追加のみ。`engine.py` と発注経路は段階Eのshadowまで無改造
- DBは加算マイグレーションのみ。新テーブル `Dataset` / `PredictionDetail`、
  `BacktestRun` への追加列は全てnullable。既存44実行の記録は消さない
- 旧エンジンの過去結果は旧エンジン由来と分かる形で保管し、新エンジン結果と混ぜない

各段階の完了時点で**既存テスト60本超が全て通ること**を、段階投入が安全である証拠とする。

## 11. データモデルの変更

| 変更 | 内容 | 互換性 |
|---|---|---|
| 新テーブル `Dataset` | `dataset_id` / 生成日時 / 銘柄集合 / 期間 / `feature_version` / `strategy_version` / ファイルパス / SHA256 | 加算のみ |
| 新テーブル `PredictionDetail` | `model_id` / `symbol` / `decision_at` / `y_true` / `p` / `fold` | 加算のみ |
| `BacktestRun` に列追加 | `strategy_version` / `config_hash` / `dataset_id` / `model_id` / `code_version` / `execution_model_version` / `degraded` | 全てnullable |
| `Signal` に列追加 | `data_as_of` | nullable |
| `ModelMetrics` に列追加 | `model_id` / `positive_rate` / `training_window_sessions` | 全てnullable |

ファイル実体（`data/datasets/*.csv.gz`、`models/candidates/*`）はDBに入れない。

## 12. テスト戦略

`indicators` / `labeling` / `market_data` は現在テストがゼロである。
実装は別モデルが行うため、**期待挙動を先にテストで固定する（TDD）**。

新規テストファイル:

| ファイル | 主な対象 |
|---|---|
| `tests/test_indicators.py` | RSI端点3ケース、欠損マスク、将来行追加で過去特徴が変わらないこと |
| `tests/test_market_data_freshness.py` | 引け後のT日更新、更新失敗・部分更新、stale銘柄の候補除外 |
| `tests/test_policy.py` | 悲観規約4項目、未解決イベント、ギャップダウン約定 |
| `tests/test_dataset.py` | 列の充足、`label_end_at` の正しさ、`dataset_id` の再現性 |
| `tests/test_validation.py` | 日付境界分割、purge、外側foldの不可侵、予測明細の保存 |
| `tests/test_walkforward.py` | T+1執行、資金競合、セクター上限、degraded伝播 |
| `tests/test_model_promotion.py` | 候補と現行の分離、学習失敗時の現行保持、ロールバック |

`labeling.py` は `policy.py` を呼ぶ薄い層に縮小されるため、専用のテストファイルは作らない。
トリプルバリアの挙動は `test_policy.py`、サンプル重みとイベント表の整合は `test_dataset.py` が担う。

レビュー §10 の表（20ケース）をそのままケースへ落とす。

## 13. 完了条件

段階ごとに満たすべき条件を定める。

| 段階 | 完了条件 |
|---|---|
| A | 全シグナルが `data_as_of` を持ち、どの営業日の足から作られたか辿れる。`state != "fresh"` の銘柄が新規候補として採用されない |
| B | 将来データを追加しても過去の特徴量が変わらない。未成熟ラベルを学習しない。ラベルの終了イベントが実行シミュレーションと一致する |
| C | 外側foldの値を変えても学習済みモデルと内側で選んだ閾値が変わらない。予測明細から指標を再計算できる |
| D | Tの終値で買う条件が成立してもT+1以降にのみ約定する。degraded実行が比較から除外される |
| E | 学習失敗で現行モデルが失われない。shadowの候補が発注に繋がらない |

**全体の合格条件は本設計では固定しない。** 「AUCが0.55を超えたら採用」「paperで1ヶ月成功したら合格」
といった一律基準は置かない。売買回数が少なければ期間を延ばし、比較の不確実性が大きければ保留する。
利益目標・許容ドローダウン・学習窓の最終値はユーザーの資金と運用目的に依存する未確定事項である。

## 14. スコープ外

- F10〜F15（リスク価格の鮮度、プロファイル適用の原子性、初期設定の保護、
  ログアウトとAPIトークン、ログのマスキング、DB制約と監査メタデータ）は別プランとする
- 深層モデルへの移行
- 権利落ち・分割イベントの明示的な保存（段階D完了後に判断）
- 日中データの導入（`ExitPolicy` の悲観規約で差が大きい戦略が出た場合に検討）

## 15. リスクと未確定事項

1. **ラベル定義の変更で学習データ量が減る。** 未成熟イベントを除外するため、
   現行19,458サンプルから減少する。段階Bの完了時点で実数を確認し、
   不足するなら学習窓の拡大（F07）で補う。
2. **新旧エンジンの二重化期間がある。** 段階投入の要求上、`engine.py` と `walkforward.py` が
   並存する。段階Eの完了後に旧エンジンの廃止を別途判断する。
3. **改善後の成績は本設計では保証できない。** 是正は「測れる状態を作る」ことであり、
   MLが有効であることを示すものではない。段階Cの比較で追加価値が示せなければ、
   ルール単独運用を選ぶ判断もあり得る。
