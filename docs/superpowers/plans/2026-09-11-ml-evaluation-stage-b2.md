# ML評価基盤 段階B後半（イベント表とラベル）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 「1行＝1候補」のイベント表を作り、コスト控除後の純収益からラベルを確定させる。未成熟・未約定・欠損を別ステータスにして学習対象から外す。

**Architecture:** `dataset.py` が候補生成からラベル確定までを組み立てる。候補は**バージョン固定のルールだけ**で作り（MLの予測を候補生成に使わない＝循環を断つ）、退出は段階B前半の `policy.run_session_series()` に、約定とコストは `execution.py` に委ねる。イベント表は正規化した内容のハッシュを `dataset_id` とし、`data/datasets/<dataset_id>.csv.gz` に保存する。既存の `labeling.py` は legacy 経路が依存しているため**挙動を変えない**。

**Tech Stack:** Python 3.11 / pandas 2.1.4 / numpy 1.26.2 / SQLAlchemy 2.0.23 / pytest / hashlib / gzip

**Spec:** `docs/superpowers/specs/2026-09-10-ml-evaluation-foundation-design.md`（§6・§7・§10・§11・§12・§14）

**前提:** 段階B前半（`docs/superpowers/plans/2026-09-11-ml-evaluation-stage-b1.md`）が完了していること。本計画は `policy.HoldingState` / `policy.Observation` / `policy.run_session_series` / `execution.Fill` / `execution.entry_fill` / `execution.exit_fill` / `execution.net_return` / `indicators.build_feature_frame` に依存する。

## Global Constraints

- 日時は **JST naive**。現在時刻は `src/core/clock.now()` / `clock.today()` を使い、`datetime.now()` を直接呼ばない。
- 新規のDB列・テーブルはすべて nullable。`_migrate_add_missing_columns()` が既存テーブルへの列追加を、`create_all` が新規テーブルを自動で作る。
- **既存の公開関数の挙動を変えない。** `labeling.build_training_set()` / `labeling.triple_barrier_labels()` / `indicators.build_features()` / `signal.compute_rule_score()` の戻り値は本計画の前後で同一であること。`src/backtest/engine.py`・`src/services/trading.py`・`src/strategy/ml_model.py` は変更しない。
- **候補生成にMLの予測を使わない。** ラベルを作るためにMLの予測が要る循環を断つため（spec §6）。
- **`dataset.py` は学習・評価を行わない。** イベント表を作って保存するだけ。分割・重み再計算・学習は段階C（`validation.py`）の担当。
- ファイルは UTF-8 **BOM無し**・LF で保存する。確認は `git show <rev>:<path>` でコミット済みblobに対して行う。
- テストは `pytest tests/<file>.py -v` で実行する。ネットワークへ出るテストを書かない。
- コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` を付ける。実装者自身のモデル名を書かない。

---

## 本計画における spec からの意図的な差分（2点）

実装者が「計画が spec と食い違っている」と判断して勝手に直さないよう、先に明示する。

### 差分1: `labeling.py` を縮小しない

spec §4 の構造表は `labeling.py` を「`policy.py` と過去検証アダプタを呼ぶ薄い層に縮小」としているが、**本計画では `labeling.py` を一切変更しない**（モジュールdocstringへの追記のみ）。

理由: spec §10 が「既存の公開関数 `build_features()` / `build_training_set()` の挙動は変更しない」「`legacy` は旧評価方式へ戻す指定」と定めており、`build_training_set()` は `ml_model.train()` / `train_multi()` の legacy 経路が依存している。縮小すると `engine.py` 経由の legacy バックテストが壊れる。spec §10 の互換保証のほうが束縛力が強いと判断する。

v2 経路は `labeling.py` を使わず `dataset.py` を使う。両者は `strategy.engine_version` で切り替わる（spec §10 の切替表：学習データの生成元 `legacy` = `labeling.build_training_set` / `v2` = `dataset.build_events`）。

### 差分2: イベント表に `sample_weight` 列を持たせない

spec §6 のイベント表は `sample_weight`（一意性重み）を列として挙げているが、**列としては保存しない**。代わりに `dataset.uniqueness_weights(events)` 関数を提供し、段階Cが fold ごとに呼ぶ。

理由: spec §7 が「イベント表に一度だけ計算した一意性重みをそのまま各foldへ流すと、検証側イベントの終了時点が学習側の重みへ影響する。**purge後の学習イベント集合で再計算する**」と定めている。列として保存すると、その値を素通しで使う実装を誘発し、まさにその漏れを作る。再計算に必要な情報（`entry_at` / `label_end_at`）はイベント表に含めるため、機能は失われない。

---

## File Structure

| ファイル | 責務 |
|---|---|
| `src/strategy/dataset.py`（新規） | 候補生成・1候補のシミュレーション・イベント表の組み立て・正規化と `dataset_id`・保存と読込・一意性重みの算出 |
| `src/data/database.py`（改修） | `Dataset` テーブルの追加（メタ情報のみ。イベント実体はDBに入れない） |
| `src/strategy/labeling.py`（改修） | モジュールdocstringに legacy 専用である旨を追記（**挙動は変更しない**） |
| `tests/test_dataset.py`（新規） | 列の充足、ラベル契約、`label_end_at` の正しさ、未成熟の別ステータス化、`dataset_id` の再現性、保存と読込、一意性重み、legacy統合ケース |

---

## Task 1: イベント表のスキーマと状態区分

**Files:**
- Create: `src/strategy/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `src/strategy/indicators.FEATURE_COLS`
- Produces:
  - 状態定数 `STATUS_RESOLVED` / `STATUS_IMMATURE` / `STATUS_UNFILLED` / `STATUS_INVALID_FEATURES`
  - 版定数 `FEATURE_VERSION` / `STRATEGY_VERSION` / `EXECUTION_MODEL_VERSION`
  - `META_COLUMNS: list[str]` — メタ列の固定順
  - `EVENT_COLUMNS: list[str]` — `META_COLUMNS + FEATURE_COLS`（イベント表の列順はこれで固定する）
  - `NOMINAL_QUANTITY: int`
  - `make_event_id(symbol: str, decision_at: date) -> str`
  - `make_label_contract_id(policy_conf: policy.PolicyConfig, costs: execution.CostConfig, *, peak_basis: str = PEAK_BASIS_PREVIOUS) -> str` — ラベルの作られ方の契約ID（SHA256先頭12桁）。イベント表の `label_contract_id` 列に入り、段階C2で実績（`PredictionOutcome`）の保存キーの一部になる

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_dataset.py` を新規作成する。

```python
"""イベント表とラベル（src/strategy/dataset.py）のテスト

1行=1候補。候補はバージョン固定のルールだけで作り（MLの予測を候補生成に
使わない＝循環を断つ）、退出はpolicy、約定とコストはexecutionに委ねる。
未成熟・未約定・欠損は別ステータスにして学習対象から外す（spec §6）。
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest import execution
from src.core import config as cfg
from src.strategy import dataset
from src.strategy import indicators
from src.strategy import policy


@pytest.fixture(autouse=True)
def _load_config():
    cfg.load("config.yaml")


class TestSchema:
    def test_status_values_are_distinct(self):
        """4つの状態区分が互いに異なる（未成熟を損失0に潰さないため）"""
        values = {
            dataset.STATUS_RESOLVED,
            dataset.STATUS_IMMATURE,
            dataset.STATUS_UNFILLED,
            dataset.STATUS_INVALID_FEATURES,
        }
        assert len(values) == 4

    def test_event_columns_start_with_meta_then_features(self):
        """列順は固定。メタ列のあとに特徴量列が続く"""
        assert dataset.EVENT_COLUMNS == dataset.META_COLUMNS + list(indicators.FEATURE_COLS)

    def test_meta_columns_cover_spec_requirements(self):
        """spec §6 が要求する列が揃っている"""
        required = {
            "event_id", "label_contract_id", "symbol", "decision_at",
            "feature_as_of", "entry_at",
            "label_end_at", "status", "label", "net_return", "exit_reason",
            "feature_version", "strategy_version", "execution_model_version",
        }
        assert required <= set(dataset.META_COLUMNS)

    def test_sample_weight_is_not_a_column(self):
        """一意性重みは列に持たない（fold内で再計算する。本計画の差分2）"""
        assert "sample_weight" not in dataset.EVENT_COLUMNS


class TestMakeEventId:
    def test_is_deterministic(self):
        a = dataset.make_event_id("7203", date(2026, 9, 10))
        b = dataset.make_event_id("7203", date(2026, 9, 10))
        assert a == b

    def test_differs_by_symbol_and_session(self):
        base = dataset.make_event_id("7203", date(2026, 9, 10))
        assert dataset.make_event_id("9984", date(2026, 9, 10)) != base
        assert dataset.make_event_id("7203", date(2026, 9, 11)) != base


class TestLabelContractId:
    """ラベル契約ID。このクラスだけで完結するようヘルパを局所に置く
    （_policy_conf / _costs は後続タスクのテストで定義される）。"""

    @staticmethod
    def _p(stop=-0.07, breakeven=0.02, trailing=0.04,
           sell_thr=-0.25, max_holding=10):
        return policy.PolicyConfig(
            stop_loss_pct=stop, breakeven_trigger_pct=breakeven,
            trailing_stop_pct=trailing, sell_threshold=sell_thr,
            max_holding_sessions=max_holding)

    @staticmethod
    def _c(slip=0.0, comm=0.0):
        return execution.CostConfig(slippage_pct=slip, commission_pct=comm)

    def test_same_settings_give_the_same_id(self):
        a = dataset.make_label_contract_id(self._p(), self._c())
        b = dataset.make_label_contract_id(self._p(), self._c())
        assert a == b
        assert len(a) == 12

    def test_different_costs_give_a_different_id(self):
        """コストが違えば同じ銘柄・同じ日でもラベルは別物になる

        この2つが同じIDになると、別コストで評価をやり直したときに
        過去runの実績を上書きしてしまう（外部レビューR07）。
        """
        free = dataset.make_label_contract_id(self._p(), self._c())
        costly = dataset.make_label_contract_id(
            self._p(), self._c(slip=0.001, comm=0.001))
        assert free != costly

    def test_different_exit_policy_gives_a_different_id(self):
        base = dataset.make_label_contract_id(self._p(), self._c())
        assert dataset.make_label_contract_id(
            self._p(stop=-0.03), self._c()) != base
        assert dataset.make_label_contract_id(
            self._p(max_holding=5), self._c()) != base
        assert dataset.make_label_contract_id(
            self._p(sell_thr=-0.5), self._c()) != base

    def test_id_takes_no_data_argument(self):
        """契約IDはラベルの定義だけを表し、対象データには依存しない

        dataset_id（内容ハッシュ）を実績キーに使うと、銘柄を1つ足すだけで
        過去の実績と結び付かなくなる。銘柄・日付・件数を引数に取らないこと
        自体を契約として固定する。
        """
        import inspect
        params = set(inspect.signature(dataset.make_label_contract_id).parameters)
        assert params == {"policy_conf", "costs", "peak_basis"}
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.strategy.dataset'`

- [ ] **Step 3: 実装を書く**

`src/strategy/dataset.py` を新規作成する。

```python
"""イベント表の生成 — 1行が1つの売買候補に対応する。

学習・検証・バックテストが共有する「何を予測するのか」をこのモジュールが確定させる。
候補は**バージョン固定のルールだけ**で生成し、MLの予測は使わない。scores に
今回学習するML自身を含めると、ラベルを作るためにそのMLの予測が必要になる循環が
生じるため（spec §6）。MLは当面エントリー候補の選別に限定する。

退出は src/strategy/policy.py に、約定とコストの控除は
src/backtest/execution.py に委ねる。本モジュールはそれらを組み立てて
「この候補はコスト控除後に得だったか」を確定させるだけで、学習も評価も行わない
（分割・重み再計算・学習は段階Cの validation.py が担当する）。

**未成熟・未約定・欠損は別ステータスにする。** 従来の labeling.py は
max_holding に満たない末尾のイベントにも最終リターンの符号でラベルを付けており、
未来1行しかないサンプルにラベル1が付いていた（レビューF05）。
"""
import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.indicators import FEATURE_COLS

# ─── 状態区分 ──────────────────────────────────────────────────────────
# 未成熟・未約定・欠損を「損失0」に潰さないため、ラベルとは別に持つ。
STATUS_RESOLVED = "resolved"                  # 退出まで決着した（ラベルを付ける唯一の状態）
STATUS_IMMATURE = "immature"                  # 足が尽きて決着しなかった
STATUS_UNFILLED = "unfilled"                  # 約定できなかった（翌営業日の足が無い）
STATUS_INVALID_FEATURES = "invalid_features"  # 特徴量が揃っていない

# ─── 再現用の版 ────────────────────────────────────────────────────────
# 定義を変えたらここを上げる。dataset_id（内容ハッシュ）と併せて、
# 「コード変更による差」と「データ改訂による差」を分離するために持つ。
FEATURE_VERSION = "f1"                      # indicators.FEATURE_COLS の定義
STRATEGY_VERSION = "rule_only_v1"           # 候補生成はルールのみ
EXECUTION_MODEL_VERSION = "t1_open_v1"      # Tの引けで判断しT+1の寄りで執行
LABEL_VERSION = "l1"                        # トリプルバリアの定義そのもの

# 純収益率は数量に依存しない（買い代金・売り代金・手数料が同じ係数で伸縮し、
# 比を取ると約分される）。Fill の型を満たすための名目値として持つ。
NOMINAL_QUANTITY = 100

META_COLUMNS = [
    "event_id",
    "label_contract_id",        # ラベルの作られ方（退出ポリシー＋コスト＋版）
    "symbol",
    "decision_at",              # 売買判断の時点（Tの引け）
    "feature_as_of",            # 特徴量に使った情報の最終時点
    "entry_at",                 # 想定した執行時点（T+1の寄り）
    "label_end_at",             # ラベル確定に使った最後の時点
    "status",
    "label",                    # status==resolved のときのみ 0/1。他は NaN
    "net_return",               # コスト控除後の純収益率
    "exit_reason",
    "entry_price",
    "exit_price",
    "sessions_held",
    "rule_score",
    "feature_version",
    "strategy_version",
    "execution_model_version",
]

EVENT_COLUMNS = META_COLUMNS + list(FEATURE_COLS)


def make_event_id(symbol: str, decision_at: date) -> str:
    """イベントの一意識別子。

    銘柄と判断セッションの組で一意になる。内容ハッシュ（dataset_id）を
    安定させるため、乱数やタイムスタンプを混ぜず決定的に作る。

    **これは「どの判断を指すか」であって「ラベルがどう作られたか」ではない。**
    同じ銘柄・同じ日でも、退出ポリシーやコストを変えれば実績ラベルは別物になる。
    実績の保存キーには make_label_contract_id() と組で使うこと
    （外部レビューR07）。
    """
    return f"{symbol}:{pd.Timestamp(decision_at).strftime('%Y%m%d')}"


def make_label_contract_id(policy_conf: "policy.PolicyConfig",
                           costs: "execution.CostConfig", *,
                           peak_basis: str = PEAK_BASIS_PREVIOUS) -> str:
    """ラベルの作られ方を一意に決める契約ID（SHA256の先頭12桁）。

    同じ (symbol, decision_at) でも、損切り幅・トレーリング・売り閾値・
    最大保有期間・スリッページ・手数料・執行モデルが違えば `label` と
    `net_return` は別の値になる。実績を `event_id` だけで保存すると、
    別のコストで評価をやり直したときに**過去runの実績を上書きしてしまい、
    保存済みの指標が後から変わる**（外部レビューR07）。

    `dataset_id`（内容ハッシュ）ではなくパラメータのハッシュにする理由:
    dataset_id は銘柄を1つ増やしただけでも変わるため、shadow運用のように
    データが継ぎ足されていく用途では過去の実績と結び付かなくなる。
    契約IDは「ラベルの定義」だけを表し、対象データの増減では変わらない。

    順序が確定した JSON にしてからハッシュする。dict の反復順や float の
    既定表現に依存すると、同じ設定で違うIDが出る。
    """
    payload = {
        "label_version": LABEL_VERSION,
        "execution_model_version": EXECUTION_MODEL_VERSION,
        "peak_basis": peak_basis,
        "stop_loss_pct": round(float(policy_conf.stop_loss_pct), 8),
        "breakeven_trigger_pct": round(float(policy_conf.breakeven_trigger_pct), 8),
        "trailing_stop_pct": round(float(policy_conf.trailing_stop_pct), 8),
        "sell_threshold": round(float(policy_conf.sell_threshold), 8),
        "max_holding_sessions": int(policy_conf.max_holding_sessions),
        "slippage_pct": round(float(costs.slippage_pct), 8),
        "commission_pct": round(float(costs.commission_pct), 8),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
```

`import json` と `import hashlib` をファイル先頭へ足す（`hashlib` は
`compute_dataset_id` で既に使う）。

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS（10件）

- [ ] **Step 5: BOM確認とコミット**

Run: `head -c 3 src/strategy/dataset.py | xxd`（`2222 22` を確認。`efbb bf` なら下記で除去）

```python
for p in ["src/strategy/dataset.py", "tests/test_dataset.py"]:
    with open(p, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        with open(p, "wb") as f:
            f.write(data[3:])
```

```bash
git add src/strategy/dataset.py tests/test_dataset.py
git commit -m "$(cat <<'EOF'
feat(strategy): イベント表のスキーマと状態区分を追加

1行=1候補。未成熟・未約定・欠損を「損失0」に潰さず別ステータスにする
（従来はmax_holdingに満たない末尾のイベントにも符号でラベルが付いていた）。
一意性重みは列に持たない。foldごとに再計算しないと検証側イベントの
終了時点が学習側の重みへ漏れるため（spec §7）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: ルールだけで候補を生成する

**Files:**
- Modify: `src/strategy/dataset.py`（`rule_scores` と `find_candidates` を追加）
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `src/strategy/signal.compute_rule_score`、Task 1 の定数
- Produces:
  - `rule_scores(feat: pd.DataFrame) -> pd.Series` — 各セッションのルールスコア（`feat` の index を保持）
  - `find_candidates(feat: pd.DataFrame, buy_threshold: float) -> list[int]` — 買い候補となるセッションの位置インデックス

**背景:** `signal.compute_rule_score(df)` は `df.iloc[-1]` と `df.iloc[-2]` しか見ない（`src/strategy/signal.py:24-67`）。したがって各セッションのスコアは**2行の窓**を渡せば求まる。既存の実装をそのまま再利用し、ルールを二重に書かない。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_dataset.py` の末尾に追記する。冒頭の import に `from datetime import timedelta` を足す。

```python
def _ohlcv(n: int, start_price: float = 1000.0) -> pd.DataFrame:
    """日付インデックス・昇順・重複なしの単一銘柄OHLCVを作る"""
    start = date(2025, 1, 6)
    rows = []
    price = start_price
    for i in range(n):
        price *= 1 + 0.002 * ((i % 7) - 3)
        rows.append({
            "date": start + timedelta(days=i),
            "open": price, "high": price * 1.01, "low": price * 0.99,
            "close": price, "volume": 100000,
        })
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


class TestRuleScores:
    def test_returns_one_score_per_session(self):
        feat = indicators.build_feature_frame(_ohlcv(120))
        scores = dataset.rule_scores(feat)
        assert len(scores) == len(feat)
        assert list(scores.index) == list(feat.index)

    def test_scores_are_within_rule_range(self):
        feat = indicators.build_feature_frame(_ohlcv(120))
        scores = dataset.rule_scores(feat).dropna()
        assert ((scores >= -1.0) & (scores <= 1.0)).all()

    def test_matches_signal_module_for_a_single_session(self):
        """既存の compute_rule_score と同じ値になる（ルールを二重に書いていない）"""
        from src.strategy import signal as signal_mod

        feat = indicators.build_feature_frame(_ohlcv(120))
        scores = dataset.rule_scores(feat)
        i = 100
        expected = signal_mod.compute_rule_score(feat.iloc[i - 1:i + 1])
        assert scores.iloc[i] == pytest.approx(expected)

    def test_first_session_has_no_score(self):
        """前日が無い先頭セッションはスコアを出さない（compute_rule_scoreが2行必要）"""
        feat = indicators.build_feature_frame(_ohlcv(120))
        scores = dataset.rule_scores(feat)
        assert pd.isna(scores.iloc[0])


class TestFindCandidates:
    def test_selects_sessions_at_or_above_threshold(self, monkeypatch):
        feat = indicators.build_feature_frame(_ohlcv(120))
        fake = pd.Series([np.nan] * len(feat), index=feat.index)
        fake.iloc[50] = 0.30
        fake.iloc[60] = 0.20
        fake.iloc[70] = 0.25
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        got = dataset.find_candidates(feat, buy_threshold=0.25)
        assert got == [50, 70]

    def test_excludes_sessions_with_invalid_features(self, monkeypatch):
        """特徴量が揃っていないセッションは候補にしない"""
        feat = indicators.build_feature_frame(_ohlcv(120))
        fake = pd.Series([0.99] * len(feat), index=feat.index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        got = dataset.find_candidates(feat, buy_threshold=0.25)
        valid_positions = [i for i, v in enumerate(feat["feature_valid"]) if bool(v)]
        assert got == valid_positions

    def test_returns_empty_when_nothing_reaches_threshold(self, monkeypatch):
        feat = indicators.build_feature_frame(_ohlcv(120))
        fake = pd.Series([0.01] * len(feat), index=feat.index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        assert dataset.find_candidates(feat, buy_threshold=0.25) == []
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.dataset' has no attribute 'rule_scores'`

- [ ] **Step 3: 実装を追加**

`src/strategy/dataset.py` の末尾に追加する。import に `from src.strategy.signal import compute_rule_score` を足す。

```python
def rule_scores(feat: pd.DataFrame) -> pd.Series:
    """各セッションのルールスコアを返す（feat の index を保持する）。

    signal.compute_rule_score() は df.iloc[-1] と df.iloc[-2] しか見ないため、
    2行の窓を渡せばそのセッションのスコアが求まる。ルールを二重に書かず
    既存実装をそのまま再利用する。前日が無い先頭セッションは NaN。
    """
    values = [float("nan")]
    for i in range(1, len(feat)):
        values.append(compute_rule_score(feat.iloc[i - 1:i + 1]))
    return pd.Series(values, index=feat.index, name="rule_score")


def find_candidates(feat: pd.DataFrame, buy_threshold: float) -> list[int]:
    """買い候補となるセッションの位置インデックスを返す。

    **MLの予測を使わない。** ラベルを作るためにMLの予測が要る循環を断つため、
    候補生成はバージョン固定のルールだけで行う（spec §6）。
    特徴量が揃っていないセッションは候補にしない。
    """
    scores = rule_scores(feat)
    valid = feat["feature_valid"] if "feature_valid" in feat.columns else pd.Series(
        True, index=feat.index)
    out = []
    for i in range(len(feat)):
        score = scores.iloc[i]
        if pd.isna(score) or not bool(valid.iloc[i]):
            continue
        if score >= buy_threshold:
            out.append(i)
    return out
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS（13件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/dataset.py tests/test_dataset.py
git commit -m "$(cat <<'EOF'
feat(strategy): ルールだけで候補を生成する処理を追加

候補生成にMLの予測を使わない。scoresに今回学習するML自身を含めると
ラベルを作るためにそのMLの予測が必要になる循環が生じるため。
compute_rule_scoreはlatestとprevしか見ないので2行の窓で再利用し、
ルールを二重に書かない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 1候補を執行・退出までシミュレートする

**Files:**
- Modify: `src/strategy/dataset.py`（`EventOutcome` と `simulate_event` を追加）
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: Task 1・2 の定数と関数、`policy.HoldingState` / `policy.Observation` / `policy.run_session_series` / `policy.PolicyConfig` / `policy.PEAK_BASIS_PREVIOUS`、`execution.entry_fill` / `execution.exit_fill` / `execution.net_return` / `execution.CostConfig`
- Produces:
  - `EventOutcome`（frozen dataclass）: `status: str`, `entry_at: Optional[date]`, `label_end_at: Optional[date]`, `label: Optional[int]`, `net_return: Optional[float]`, `exit_reason: Optional[str]`, `entry_price: Optional[float]`, `exit_price: Optional[float]`, `sessions_held: int`
  - `simulate_event(feat: pd.DataFrame, i: int, policy_conf, costs, *, peak_basis=PEAK_BASIS_PREVIOUS) -> EventOutcome`

**ラベル契約（spec §6）:**

- `label = 1` は、仮定した執行と退出に基づく**コスト控除後の純収益が正**の場合
- **`net_return == 0` は `label = 0` に含める**（境界を明示する）
- 未成熟・未約定は `label` を付けず別ステータスにする。**損失0として扱わない**
- 最大保有期間は営業日数で数える

**執行の並び:** `i` の引けで判断 → `i+1` の寄りで約定 → `i+1` 以降の足に退出ポリシーを逐次適用。エントリー当日（`i+1`）も退出判定の対象にする（実運用も朝に買ってその日のうちに損切り監視が走るため）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_dataset.py` の末尾に追記する。冒頭の import に `from src.backtest import execution` と `from src.strategy import policy` を足す。

```python
def _policy_conf(stop=-0.07, breakeven=0.02, trailing=0.04,
                 sell_thr=-0.25, max_holding=10):
    return policy.PolicyConfig(
        stop_loss_pct=stop, breakeven_trigger_pct=breakeven,
        trailing_stop_pct=trailing, sell_threshold=sell_thr,
        max_holding_sessions=max_holding,
    )


def _costs(slip=0.0, comm=0.0):
    return execution.CostConfig(slippage_pct=slip, commission_pct=comm)


def _frame(bars: list[dict]) -> pd.DataFrame:
    """simulate_event に渡す最小のフレーム（日付index・OHLC・feature_valid）"""
    df = pd.DataFrame(bars).set_index("date")
    df.index = pd.to_datetime(df.index)
    df["feature_valid"] = True
    return df


class TestSimulateEvent:
    def test_stop_loss_gives_label_zero(self):
        """損切りで終わった候補は label=0、純収益は負"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 2), "open": 1000, "high": 1005, "low": 900, "close": 910},
        ]
        out = dataset.simulate_event(_frame(bars), 0, _policy_conf(), _costs())
        assert out.status == dataset.STATUS_RESOLVED
        assert out.entry_at == date(2026, 9, 2)
        assert out.label_end_at == date(2026, 9, 2)
        assert out.exit_reason == policy.STOP_LINE
        assert out.entry_price == pytest.approx(1000.0)
        assert out.exit_price == pytest.approx(930.0)   # 基準線 1000*0.93
        assert out.net_return == pytest.approx(-0.07)
        assert out.label == 0

    def test_zero_net_return_is_label_zero(self):
        """純収益0は label=0 に含める（境界の明示）"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 2), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 3), "open": 1000, "high": 1005, "low": 995, "close": 1000},
        ]
        # max_holding=1 で満了 → 翌営業日(9/3)の寄り1000で退出。入りも1000なので0%
        out = dataset.simulate_event(
            _frame(bars), 0, _policy_conf(max_holding=1), _costs())
        assert out.status == dataset.STATUS_RESOLVED
        assert out.exit_reason == policy.TIME_LIMIT
        assert out.net_return == pytest.approx(0.0)
        assert out.label == 0

    def test_profitable_exit_gives_label_one(self):
        """トレーリングで利益が残れば label=1"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 2), "open": 1000, "high": 1200, "low": 1150, "close": 1190},
            {"date": date(2026, 9, 3), "open": 1190, "high": 1195, "low": 1100, "close": 1110},
        ]
        out = dataset.simulate_event(_frame(bars), 0, _policy_conf(), _costs())
        assert out.status == dataset.STATUS_RESOLVED
        assert out.exit_reason == policy.TRAILING
        assert out.label_end_at == date(2026, 9, 3)
        assert out.exit_price == pytest.approx(1152.0)  # ピーク1200 * 0.96
        assert out.net_return == pytest.approx(0.152)
        assert out.label == 1

    def test_immature_when_bars_run_out(self):
        """最大保有期間まで足が届かない候補は未成熟。ラベルを付けない"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 2), "open": 1000, "high": 1005, "low": 995, "close": 1000},
        ]
        out = dataset.simulate_event(
            _frame(bars), 0, _policy_conf(max_holding=10), _costs())
        assert out.status == dataset.STATUS_IMMATURE
        assert out.label is None
        assert out.net_return is None
        assert out.label_end_at is None

    def test_unfilled_when_no_entry_bar(self):
        """翌営業日の足が無ければエントリーできない（未約定）"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
        ]
        out = dataset.simulate_event(_frame(bars), 0, _policy_conf(), _costs())
        assert out.status == dataset.STATUS_UNFILLED
        assert out.label is None
        assert out.entry_at is None

    def test_unfilled_when_market_exit_has_no_next_bar(self):
        """満了の成行退出に必要な翌営業日の足が無ければ未約定"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 2), "open": 1000, "high": 1005, "low": 995, "close": 1000},
        ]
        # max_holding=1 → 9/2 に満了意図が出るが、翌足が無い
        out = dataset.simulate_event(
            _frame(bars), 0, _policy_conf(max_holding=1), _costs())
        assert out.status == dataset.STATUS_UNFILLED
        assert out.label is None

    def test_invalid_features_are_not_simulated(self):
        """特徴量が揃っていない判断セッションはシミュレートしない"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 2), "open": 1000, "high": 1005, "low": 900, "close": 910},
        ]
        feat = _frame(bars)
        feat.iloc[0, feat.columns.get_loc("feature_valid")] = False
        out = dataset.simulate_event(feat, 0, _policy_conf(), _costs())
        assert out.status == dataset.STATUS_INVALID_FEATURES
        assert out.label is None

    def test_entry_uses_next_open_not_decision_close(self):
        """判断した日の終値では約定しない（F04の回帰防止）"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1005},
            {"date": date(2026, 9, 2), "open": 980, "high": 1005, "low": 900, "close": 910},
        ]
        out = dataset.simulate_event(_frame(bars), 0, _policy_conf(), _costs())
        assert out.entry_price == pytest.approx(980.0)   # 翌日の寄り
        assert out.entry_price != pytest.approx(1005.0)  # 判断日の終値ではない

    def test_costs_are_reflected_in_net_return(self):
        """スリッページと手数料が純収益に反映される"""
        bars = [
            {"date": date(2026, 9, 1), "open": 1000, "high": 1005, "low": 995, "close": 1000},
            {"date": date(2026, 9, 2), "open": 1000, "high": 1005, "low": 900, "close": 910},
        ]
        free = dataset.simulate_event(_frame(bars), 0, _policy_conf(), _costs())
        charged = dataset.simulate_event(
            _frame(bars), 0, _policy_conf(), _costs(slip=0.001, comm=0.001))
        assert charged.net_return < free.net_return
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.dataset' has no attribute 'simulate_event'`

- [ ] **Step 3: 実装を追加**

`src/strategy/dataset.py` の末尾に追加する。import に次を足す。

```python
from src.backtest import execution
from src.strategy import policy
```

```python
@dataclass(frozen=True)
class EventOutcome:
    """1候補をシミュレートした結果。

    label は status == STATUS_RESOLVED のときだけ 0/1 を持つ。
    未成熟・未約定・欠損では None にして、損失0として扱わない（spec §6）。
    """
    status: str
    entry_at: Optional[date]
    label_end_at: Optional[date]
    label: Optional[int]
    net_return: Optional[float]
    exit_reason: Optional[str]
    entry_price: Optional[float]
    exit_price: Optional[float]
    sessions_held: int


def _observation(feat: pd.DataFrame, i: int) -> policy.Observation:
    row = feat.iloc[i]
    return policy.Observation(
        session=feat.index[i].date(),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        score=None,  # 退出の売りシグナルは段階Dで結線する（ラベルは執行と退出だけで決める）
    )


def _unresolved(status: str, entry_at: Optional[date] = None,
                sessions_held: int = 0) -> EventOutcome:
    return EventOutcome(
        status=status, entry_at=entry_at, label_end_at=None, label=None,
        net_return=None, exit_reason=None, entry_price=None, exit_price=None,
        sessions_held=sessions_held,
    )


def simulate_event(feat: pd.DataFrame, i: int,
                   policy_conf: policy.PolicyConfig,
                   costs: execution.CostConfig, *,
                   peak_basis: str = policy.PEAK_BASIS_PREVIOUS) -> EventOutcome:
    """位置 i のセッションを判断時点として、執行から退出までをシミュレートする。

    並びは「i の引けで判断 → i+1 の寄りで約定 → i+1 以降の足へ退出ポリシーを
    逐次適用」。エントリー当日（i+1）も退出判定の対象にする。実運用も朝に買った
    その日のうちに損切り監視が走るため。

    ラベルは**コスト控除後の純収益が正なら1、それ以外は0**とする。
    0は0側に含める。決着しなかった候補にはラベルを付けず、別ステータスにする。
    """
    if "feature_valid" in feat.columns and not bool(feat["feature_valid"].iloc[i]):
        return _unresolved(STATUS_INVALID_FEATURES)

    entry_idx = i + 1
    if entry_idx >= len(feat):
        return _unresolved(STATUS_UNFILLED)

    entry_bar = _observation(feat, entry_idx)
    entry = execution.entry_fill(entry_bar, NOMINAL_QUANTITY, costs)

    state = policy.HoldingState(
        symbol=str(feat.attrs.get("symbol", "")),
        entry_at=entry.at,
        avg_cost=entry.price,
        quantity=entry.quantity,
        peak_price=entry.price,
        sessions_held=0,
    )
    observations = [_observation(feat, k) for k in range(entry_idx, len(feat))]
    final, intent, _ = policy.run_session_series(
        state, observations, policy_conf, peak_basis=peak_basis)

    if intent is None:
        # 足が尽きて決着しなかった。未来1行しかないサンプルにラベルを付けない
        return _unresolved(STATUS_IMMATURE, entry_at=entry.at,
                           sessions_held=final.sessions_held)

    # 意図が出たのは entry_idx から数えて sessions_held 本目の足
    intent_idx = entry_idx + final.sessions_held - 1
    next_idx = intent_idx + 1
    exit_bar = _observation(feat, intent_idx)
    next_bar = _observation(feat, next_idx) if next_idx < len(feat) else None

    fill = execution.exit_fill(intent, exit_bar, next_bar, NOMINAL_QUANTITY, costs)
    if fill is None:
        return _unresolved(STATUS_UNFILLED, entry_at=entry.at,
                           sessions_held=final.sessions_held)

    ret = execution.net_return(entry, fill, costs)
    return EventOutcome(
        status=STATUS_RESOLVED,
        entry_at=entry.at,
        label_end_at=fill.at,
        label=1 if ret > 0 else 0,
        net_return=ret,
        exit_reason=intent.reason,
        entry_price=entry.price,
        exit_price=fill.price,
        sessions_held=final.sessions_held,
    )
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS（22件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/dataset.py tests/test_dataset.py
git commit -m "$(cat <<'EOF'
feat(strategy): 1候補の執行から退出までのシミュレーションを追加

Tの引けで判断しT+1の寄りで約定、以降の足へ退出ポリシーを逐次適用する。
ラベルはコスト控除後の純収益が正なら1、0は0側に含める。
足が尽きて決着しなかった候補と約定できなかった候補にはラベルを付けず
別ステータスにする（損失0として扱わない）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: 銘柄ぶん・複数銘柄ぶんのイベント表を組み立てる

**Files:**
- Modify: `src/strategy/dataset.py`（`build_events` と `build_events_multi` を追加）
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: Task 1〜3 のすべて、`indicators.build_feature_frame`
- Produces:
  - `build_events(symbol: str, ohlcv: pd.DataFrame, policy_conf, costs, *, buy_threshold: Optional[float] = None, peak_basis=PEAK_BASIS_PREVIOUS) -> pd.DataFrame` — `EVENT_COLUMNS` の順に並んだイベント表
  - `build_events_multi(ohlcv_by_symbol: dict[str, pd.DataFrame], policy_conf, costs, **kwargs) -> pd.DataFrame`

**注意:** `build_events_multi` は銘柄ごとに `build_events` を呼んでから連結する。生のOHLCVを連結してから処理すると移動平均・RSI・退出判定が銘柄境界をまたいで壊れる（既存 `ml_model.train_multi()` の docstring と同じ理由）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_dataset.py` の末尾に追記する。

```python
class TestBuildEvents:
    def test_columns_and_order_are_fixed(self, monkeypatch):
        feat_len = 120
        ohlcv = _ohlcv(feat_len)
        fake = pd.Series([0.99] * feat_len, index=indicators.build_feature_frame(ohlcv).index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        events = dataset.build_events("7203", ohlcv, _policy_conf(), _costs())
        assert list(events.columns) == dataset.EVENT_COLUMNS

    def test_every_row_carries_identity_and_versions(self, monkeypatch):
        ohlcv = _ohlcv(120)
        fake = pd.Series([0.99] * 120, index=indicators.build_feature_frame(ohlcv).index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        events = dataset.build_events("7203", ohlcv, _policy_conf(), _costs())
        assert len(events) > 0
        assert (events["symbol"] == "7203").all()
        assert (events["feature_version"] == dataset.FEATURE_VERSION).all()
        assert (events["strategy_version"] == dataset.STRATEGY_VERSION).all()
        assert (events["execution_model_version"] == dataset.EXECUTION_MODEL_VERSION).all()
        assert events["event_id"].is_unique

    def test_decision_at_precedes_entry_at(self, monkeypatch):
        """判断は執行より前。同じセッションで判断して約定しない（F04）"""
        ohlcv = _ohlcv(120)
        fake = pd.Series([0.99] * 120, index=indicators.build_feature_frame(ohlcv).index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        events = dataset.build_events("7203", ohlcv, _policy_conf(), _costs())
        filled = events[events["entry_at"].notna()]
        assert len(filled) > 0
        assert (filled["decision_at"] < filled["entry_at"]).all()

    def test_label_end_at_is_not_before_entry_at(self, monkeypatch):
        ohlcv = _ohlcv(120)
        fake = pd.Series([0.99] * 120, index=indicators.build_feature_frame(ohlcv).index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        events = dataset.build_events("7203", ohlcv, _policy_conf(), _costs())
        resolved = events[events["status"] == dataset.STATUS_RESOLVED]
        assert len(resolved) > 0
        assert (resolved["label_end_at"] >= resolved["entry_at"]).all()

    def test_only_resolved_rows_have_labels(self, monkeypatch):
        """未成熟・未約定にラベルが付いていない（spec §14 段階B完了条件）"""
        ohlcv = _ohlcv(120)
        fake = pd.Series([0.99] * 120, index=indicators.build_feature_frame(ohlcv).index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        events = dataset.build_events("7203", ohlcv, _policy_conf(), _costs())
        unresolved = events[events["status"] != dataset.STATUS_RESOLVED]
        assert unresolved["label"].isna().all()
        resolved = events[events["status"] == dataset.STATUS_RESOLVED]
        assert resolved["label"].notna().all()
        assert set(resolved["label"].unique()) <= {0, 1}

    def test_tail_sessions_are_not_labelled(self, monkeypatch):
        """末尾の候補は決着に必要な足が無いのでラベルが付かない。

        足の残り本数により immature（決着しなかった）にも unfilled（約定
        できなかった）にもなり得るが、いずれもラベルは付けない。
        """
        ohlcv = _ohlcv(120)
        fake = pd.Series([0.99] * 120, index=indicators.build_feature_frame(ohlcv).index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        events = dataset.build_events(
            "7203", ohlcv, _policy_conf(max_holding=10), _costs())
        last = events.sort_values("decision_at").iloc[-1]
        assert last["status"] != dataset.STATUS_RESOLVED
        assert pd.isna(last["label"])

    def test_features_are_carried_on_each_row(self, monkeypatch):
        ohlcv = _ohlcv(120)
        fake = pd.Series([0.99] * 120, index=indicators.build_feature_frame(ohlcv).index)
        monkeypatch.setattr(dataset, "rule_scores", lambda _f: fake)

        events = dataset.build_events("7203", ohlcv, _policy_conf(), _costs())
        assert events[list(indicators.FEATURE_COLS)].notna().all().all()


class TestBuildEventsMulti:
    def test_concatenates_per_symbol(self, monkeypatch):
        ohlcv_a, ohlcv_b = _ohlcv(120), _ohlcv(120, start_price=500.0)
        monkeypatch.setattr(
            dataset, "rule_scores",
            lambda f: pd.Series([0.99] * len(f), index=f.index))

        events = dataset.build_events_multi(
            {"7203": ohlcv_a, "9984": ohlcv_b}, _policy_conf(), _costs())
        assert set(events["symbol"].unique()) == {"7203", "9984"}
        assert events["event_id"].is_unique

    def test_skips_symbols_with_insufficient_data(self, monkeypatch):
        """データが足りない銘柄は黙ってスキップし、他の銘柄を止めない"""
        monkeypatch.setattr(
            dataset, "rule_scores",
            lambda f: pd.Series([0.99] * len(f), index=f.index))

        events = dataset.build_events_multi(
            {"7203": _ohlcv(120), "0000": _ohlcv(5)}, _policy_conf(), _costs())
        assert set(events["symbol"].unique()) == {"7203"}
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.dataset' has no attribute 'build_events'`

- [ ] **Step 3: 実装を追加**

`src/strategy/dataset.py` の末尾に追加する。import に `from loguru import logger` と `from src.core import config as cfg` と `from src.strategy.indicators import build_feature_frame` を足す（`FEATURE_COLS` は既にimport済み）。

```python
def build_events(symbol: str, ohlcv: pd.DataFrame,
                 policy_conf: policy.PolicyConfig,
                 costs: execution.CostConfig, *,
                 buy_threshold: Optional[float] = None,
                 peak_basis: str = policy.PEAK_BASIS_PREVIOUS) -> pd.DataFrame:
    """単一銘柄のOHLCVからイベント表を作る（1行=1候補）。

    ohlcv は単一銘柄の時系列（日付インデックス・重複なし・昇順）であること。
    複数銘柄は build_events_multi() を使う。
    """
    if buy_threshold is None:
        buy_threshold = cfg.get_section("strategy").get("buy_threshold", 0.25)

    feat = build_feature_frame(ohlcv)
    feat.attrs["symbol"] = symbol
    scores = rule_scores(feat)
    candidates = find_candidates(feat, buy_threshold)

    # ラベル契約IDは銘柄・日付に依存しないので、ループの外で一度だけ作る
    label_contract_id = make_label_contract_id(
        policy_conf, costs, peak_basis=peak_basis)

    rows = []
    for i in candidates:
        outcome = simulate_event(feat, i, policy_conf, costs, peak_basis=peak_basis)
        decision_at = feat.index[i].date()
        row = {
            "event_id": make_event_id(symbol, decision_at),
            "label_contract_id": label_contract_id,
            "symbol": symbol,
            "decision_at": decision_at,
            # 特徴量は判断セッションまでの情報だけで作られるため同じ時点になる
            "feature_as_of": decision_at,
            "entry_at": outcome.entry_at,
            "label_end_at": outcome.label_end_at,
            "status": outcome.status,
            "label": outcome.label,
            "net_return": outcome.net_return,
            "exit_reason": outcome.exit_reason,
            "entry_price": outcome.entry_price,
            "exit_price": outcome.exit_price,
            "sessions_held": outcome.sessions_held,
            "rule_score": float(scores.iloc[i]),
            "feature_version": FEATURE_VERSION,
            "strategy_version": STRATEGY_VERSION,
            "execution_model_version": EXECUTION_MODEL_VERSION,
        }
        for col in FEATURE_COLS:
            row[col] = float(feat[col].iloc[i])
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)[EVENT_COLUMNS]


def build_events_multi(ohlcv_by_symbol: dict, policy_conf: policy.PolicyConfig,
                       costs: execution.CostConfig, **kwargs) -> pd.DataFrame:
    """複数銘柄のイベント表を作って連結する。

    **銘柄ごとに build_events() を呼んでから連結する。** 生のOHLCVを連結して
    から処理すると、移動平均・RSI・退出判定が銘柄境界をまたいで壊れる
    （ml_model.train_multi() と同じ理由）。
    データ不足などで失敗した銘柄はスキップし、他の銘柄を止めない。
    """
    parts = []
    for symbol, ohlcv in ohlcv_by_symbol.items():
        try:
            events = build_events(symbol, ohlcv, policy_conf, costs, **kwargs)
        except Exception as e:
            logger.warning(f"イベント表の生成をスキップ: {symbol} {e}")
            continue
        if len(events) > 0:
            parts.append(events)

    if not parts:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.concat(parts, ignore_index=True)[EVENT_COLUMNS]
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS（32件）

- [ ] **Step 5: 全体回帰を確認してコミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

```bash
git add src/strategy/dataset.py tests/test_dataset.py
git commit -m "$(cat <<'EOF'
feat(strategy): 銘柄ごとのイベント表の組み立てを追加

列順を固定し、各行に識別子と特徴量・戦略・執行モデルの版を持たせる。
複数銘柄は銘柄ごとに作ってから連結する（生のOHLCVを連結してから
処理すると指標と退出判定が銘柄境界をまたいで壊れるため）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 正規化と `dataset_id`、保存と読込

**Files:**
- Modify: `src/strategy/dataset.py`（正規化・ハッシュ・入出力を追加）
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: Task 4 の `EVENT_COLUMNS`
- Produces:
  - `normalize_for_hash(events: pd.DataFrame) -> str`
  - `compute_dataset_id(events: pd.DataFrame) -> str` — 正規化した内容のSHA256先頭16桁
  - `save_events(events: pd.DataFrame, dataset_id: str, base_dir: str = "data/datasets") -> Path`
  - `load_events(dataset_id: str, base_dir: str = "data/datasets") -> pd.DataFrame`
  - `file_sha256(path: Path) -> str`

**書式の固定（spec §6）:** 日付はISO文字列、欠損値は空文字、浮動小数点は固定精度、列順と並び順を固定する。`dataset_id` は**正規化した内容のハッシュ**であり、取得日時を含む採取履歴IDとは別に持つ。gzipは `mtime=0` を指定してバイト列を決定的にする（既定では書き込み時刻が埋め込まれ、同一内容でもファイルが変わる）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_dataset.py` の末尾に追記する。

```python
def _sample_events(monkeypatch, symbol="7203", n=120, start_price=1000.0):
    ohlcv = _ohlcv(n, start_price=start_price)
    monkeypatch.setattr(
        dataset, "rule_scores",
        lambda f: pd.Series([0.99] * len(f), index=f.index))
    return dataset.build_events(symbol, ohlcv, _policy_conf(), _costs())


class TestDatasetId:
    def test_same_content_gives_same_id(self, monkeypatch):
        """同一内容なら同じ dataset_id になる（spec §6）"""
        a = _sample_events(monkeypatch)
        b = _sample_events(monkeypatch)
        assert dataset.compute_dataset_id(a) == dataset.compute_dataset_id(b)

    def test_row_order_does_not_change_id(self, monkeypatch):
        """並び順は正規化で固定されるのでIDに影響しない"""
        events = _sample_events(monkeypatch)
        shuffled = events.sample(frac=1.0, random_state=7).reset_index(drop=True)
        assert dataset.compute_dataset_id(events) == dataset.compute_dataset_id(shuffled)

    def test_different_content_gives_different_id(self, monkeypatch):
        a = _sample_events(monkeypatch, symbol="7203")
        b = _sample_events(monkeypatch, symbol="9984", start_price=500.0)
        assert dataset.compute_dataset_id(a) != dataset.compute_dataset_id(b)

    def test_id_is_short_hex(self, monkeypatch):
        did = dataset.compute_dataset_id(_sample_events(monkeypatch))
        assert len(did) == 16
        assert all(c in "0123456789abcdef" for c in did)

    def test_label_dtype_does_not_change_id(self, monkeypatch):
        """欠損の有無でlabelのdtypeが変わってもIDは変わらない。

        dtype推論に任せると、たまたま全件resolvedの回だけint64になって
        "1"と書かれ、欠損がある回の"1.0000000000"と別のハッシュになる。
        """
        events = _sample_events(monkeypatch)
        as_int = events.copy()
        as_int["label"] = as_int["label"].astype("object")
        as_float = events.copy()
        as_float["label"] = as_float["label"].astype("float64")
        assert dataset.compute_dataset_id(as_int) == dataset.compute_dataset_id(as_float)


class TestSaveLoad:
    def test_roundtrip_preserves_content_hash(self, monkeypatch, tmp_path):
        """保存して読み直しても dataset_id が変わらない"""
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        path = dataset.save_events(events, did, base_dir=str(tmp_path))
        assert path.exists()

        loaded = dataset.load_events(did, base_dir=str(tmp_path))
        assert dataset.compute_dataset_id(loaded) == did

    def test_roundtrip_preserves_columns_and_row_count(self, monkeypatch, tmp_path):
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        dataset.save_events(events, did, base_dir=str(tmp_path))
        loaded = dataset.load_events(did, base_dir=str(tmp_path))
        assert list(loaded.columns) == dataset.EVENT_COLUMNS
        assert len(loaded) == len(events)

    def test_roundtrip_preserves_status_and_label_semantics(self, monkeypatch, tmp_path):
        """読み直してもラベル無しのステータスにラベルが生えない"""
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        dataset.save_events(events, did, base_dir=str(tmp_path))
        loaded = dataset.load_events(did, base_dir=str(tmp_path))
        unresolved = loaded[loaded["status"] != dataset.STATUS_RESOLVED]
        assert unresolved["label"].isna().all()

    def test_written_bytes_are_deterministic(self, monkeypatch, tmp_path):
        """同一内容なら書き出したバイト列も同じ（gzipのmtimeを固定している）"""
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        p1 = dataset.save_events(events, did, base_dir=str(tmp_path / "a"))
        p2 = dataset.save_events(events, did, base_dir=str(tmp_path / "b"))
        assert dataset.file_sha256(p1) == dataset.file_sha256(p2)

    def test_file_name_is_the_dataset_id(self, monkeypatch, tmp_path):
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        path = dataset.save_events(events, did, base_dir=str(tmp_path))
        assert path.name == f"{did}.csv.gz"
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.dataset' has no attribute 'normalize_for_hash'`

- [ ] **Step 3: 実装を追加**

`src/strategy/dataset.py` の末尾に追加する。import に `from pathlib import Path` を足す（`hashlib` は既にimport済み）。

```python
_DATE_COLUMNS = ("decision_at", "feature_as_of", "entry_at", "label_end_at")
_FLOAT_FORMAT = "%.10f"
# 浮動小数点として書き出す列。dtype推論に任せると、たまたま欠損が無い回だけ
# int64 になって "1" と書かれ、欠損がある回の "1.0000000000" と別のハッシュに
# なってしまう。明示的に float へ寄せて表現を固定する。
_FLOAT_COLUMNS = ("label", "net_return", "entry_price", "exit_price", "rule_score")


def _normalized_frame(events: pd.DataFrame) -> pd.DataFrame:
    """ハッシュ・保存の双方で使う正規化。

    列順・並び順・日付表現・数値のdtypeを固定する。これをしないと、
    同じ内容でも生成の順序や dtype 推論の差で別のIDになる。
    """
    out = events.reindex(columns=EVENT_COLUMNS).copy()
    out = out.sort_values(["symbol", "decision_at"]).reset_index(drop=True)
    for col in _DATE_COLUMNS:
        out[col] = out[col].map(
            lambda d: "" if pd.isna(d) else pd.Timestamp(d).strftime("%Y-%m-%d"))
    for col in _FLOAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in FEATURE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def normalize_for_hash(events: pd.DataFrame) -> str:
    """正規化したCSV文字列を返す（欠損は空文字、浮動小数点は固定精度）。"""
    return _normalized_frame(events).to_csv(
        index=False, float_format=_FLOAT_FORMAT, na_rep="", lineterminator="\n")


def compute_dataset_id(events: pd.DataFrame) -> str:
    """正規化した内容のSHA256（先頭16桁）。

    **内容由来のIDであり、採取日時は含めない。** 同一内容なら同じIDになる。
    「いつ取ったか」は Dataset テーブルの採取履歴ID（collection_id）が持つ。
    こうしておくと「コード変更による成績差」と「データ改訂による成績差」を
    分離できる（spec §5）。
    """
    return hashlib.sha256(normalize_for_hash(events).encode("utf-8")).hexdigest()[:16]


def dataset_path(dataset_id: str, base_dir: str = "data/datasets") -> Path:
    return Path(base_dir) / f"{dataset_id}.csv.gz"


def save_events(events: pd.DataFrame, dataset_id: str,
                base_dir: str = "data/datasets") -> Path:
    """イベント表を data/datasets/<dataset_id>.csv.gz へ保存する。

    pandas 標準で読み書きでき依存追加が要らないため csv.gz を採る
    （requirements.txt に pyarrow が無いので parquet は採らない）。
    gzip は既定で書き込み時刻をヘッダへ埋めるため、mtime=0 を指定して
    同一内容なら同一バイト列になるようにする。
    """
    path = dataset_path(dataset_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    _normalized_frame(events).to_csv(
        path, index=False, float_format=_FLOAT_FORMAT, na_rep="",
        lineterminator="\n", compression={"method": "gzip", "mtime": 0})
    return path


def load_events(dataset_id: str, base_dir: str = "data/datasets") -> pd.DataFrame:
    """保存したイベント表を読み込む（日付列を date に戻す）。"""
    path = dataset_path(dataset_id, base_dir)
    out = pd.read_csv(path, compression="gzip")
    for col in _DATE_COLUMNS:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.date
        out[col] = out[col].where(out[col].notna(), None)
    return out.reindex(columns=EVENT_COLUMNS)


def file_sha256(path: Path) -> str:
    """ファイルのSHA256（完全性の確認用。識別子は dataset_id を使う）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS（42件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/dataset.py tests/test_dataset.py
git commit -m "$(cat <<'EOF'
feat(strategy): イベント表の正規化・内容ハッシュ・保存と読込を追加

dataset_idは正規化した内容のハッシュにし、採取日時を含めない。
同一内容なら同じIDになるので「コード変更による成績差」と
「データ改訂による成績差」を分離できる。
列順・並び順・日付表現・浮動小数点精度・欠損表現を固定し、gzipの
mtimeも0に固定してバイト列を決定的にする。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Dataset テーブルにメタ情報を記録する

**Files:**
- Modify: `src/data/database.py`（`Dataset` モデルを追加）
- Modify: `src/strategy/dataset.py`（`save_dataset_meta` を追加）
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: Task 5 の `compute_dataset_id` / `save_events` / `file_sha256`
- Produces:
  - `Dataset` モデル: `dataset_id` / `collection_id` / `generated_at` / `symbols_json` / `period_start` / `period_end` / `feature_version` / `strategy_version` / `execution_model_version` / `file_path` / `file_sha256` / `input_ohlcv_sha256` / `n_events` / `n_resolved`
  - `input_ohlcv_hash(ohlcv_by_symbol: dict) -> str`
  - `save_dataset_meta(events, dataset_id, path, input_hash, *, collection_id=None) -> int`（保存した行のid）

**注意:** イベントの実体はDBに入れない（ファイルのまま持つ）。DBが持つのはメタ情報だけ。新テーブルは `create_all` が作るため、マイグレーション作業は不要。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_dataset.py` の末尾に追記する。冒頭の import に `from sqlalchemy import select` と `from src.data import database as db` と `from src.data.database import get_session` を足す。

```python
@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


class TestInputOhlcvHash:
    def test_same_input_gives_same_hash(self):
        a = {"7203": _ohlcv(50)}
        b = {"7203": _ohlcv(50)}
        assert dataset.input_ohlcv_hash(a) == dataset.input_ohlcv_hash(b)

    def test_different_input_gives_different_hash(self):
        a = {"7203": _ohlcv(50)}
        b = {"7203": _ohlcv(51)}
        assert dataset.input_ohlcv_hash(a) != dataset.input_ohlcv_hash(b)


class TestSaveDatasetMeta:
    def test_records_identity_and_provenance(self, monkeypatch, isolated_db, tmp_path):
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        path = dataset.save_events(events, did, base_dir=str(tmp_path / "ds"))
        input_hash = dataset.input_ohlcv_hash({"7203": _ohlcv(120)})

        dataset.save_dataset_meta(events, did, path, input_hash)

        with get_session() as session:
            row = session.scalar(select(db.Dataset))
        assert row.dataset_id == did
        assert row.file_sha256 == dataset.file_sha256(path)
        assert row.input_ohlcv_sha256 == input_hash
        assert row.n_events == len(events)
        assert row.feature_version == dataset.FEATURE_VERSION
        assert row.strategy_version == dataset.STRATEGY_VERSION
        assert row.execution_model_version == dataset.EXECUTION_MODEL_VERSION

    def test_collection_id_differs_between_runs(self, monkeypatch, isolated_db, tmp_path):
        """内容が同じでも採取履歴IDは実行ごとに変わる（dataset_idとは別物）"""
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        path = dataset.save_events(events, did, base_dir=str(tmp_path / "ds"))
        input_hash = dataset.input_ohlcv_hash({"7203": _ohlcv(120)})

        dataset.save_dataset_meta(events, did, path, input_hash)
        dataset.save_dataset_meta(events, did, path, input_hash)

        with get_session() as session:
            rows = list(session.scalars(select(db.Dataset)).all())
        assert len(rows) == 2
        assert rows[0].dataset_id == rows[1].dataset_id       # 内容は同じ
        assert rows[0].collection_id != rows[1].collection_id  # 採取は別

    def test_records_period_and_symbols(self, monkeypatch, isolated_db, tmp_path):
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        path = dataset.save_events(events, did, base_dir=str(tmp_path / "ds"))
        dataset.save_dataset_meta(events, did, path, "x")

        with get_session() as session:
            row = session.scalar(select(db.Dataset))
        assert row.period_start == events["decision_at"].min()
        assert row.period_end == events["decision_at"].max()
        assert "7203" in row.symbols_json

    def test_counts_resolved_events(self, monkeypatch, isolated_db, tmp_path):
        events = _sample_events(monkeypatch)
        did = dataset.compute_dataset_id(events)
        path = dataset.save_events(events, did, base_dir=str(tmp_path / "ds"))
        dataset.save_dataset_meta(events, did, path, "x")

        with get_session() as session:
            row = session.scalar(select(db.Dataset))
        expected = int((events["status"] == dataset.STATUS_RESOLVED).sum())
        assert row.n_resolved == expected
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — `AttributeError: module 'src.data.database' has no attribute 'Dataset'`

- [ ] **Step 3: モデルを追加**

`src/data/database.py` の `class CorporateAction` の直後に追加する。`Text` は既にこのファイルで使われている型なので import 済みであることを確認すること（`from sqlalchemy import (...)` の並びに `Text` が無ければ足す）。

```python
class Dataset(Base):
    """イベント表（学習データ）のメタ情報。

    イベントの実体はDBに入れず data/datasets/<dataset_id>.csv.gz に置く。
    ここが持つのは「どの入力から・いつ・どの版で作ったか」だけ。

    dataset_id は**正規化した内容のハッシュ**で、同一内容なら同じ値になる。
    「いつ取ったか」は collection_id が別に持つ。こう分けておくと、
    コード変更による成績差とデータ改訂による成績差を分離できる。
    """
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True)
    dataset_id = Column(String(64), nullable=False)   # 内容ハッシュ
    collection_id = Column(String(64))                # 採取履歴ID（実行ごとに変わる）
    generated_at = Column(DateTime, default=clock.now)
    symbols_json = Column(Text)
    period_start = Column(Date)
    period_end = Column(Date)
    feature_version = Column(String(32))
    strategy_version = Column(String(32))
    execution_model_version = Column(String(32))
    file_path = Column(String(255))
    file_sha256 = Column(String(64))
    input_ohlcv_sha256 = Column(String(64))           # 入力OHLCVの内容ハッシュ
    n_events = Column(Integer)
    n_resolved = Column(Integer)

    __table_args__ = (Index("ix_datasets_dataset_id", "dataset_id"),)
```

- [ ] **Step 4: 実装を追加**

`src/strategy/dataset.py` の末尾に追加する。import に `import json` と `from src.core import clock` を足す。

```python
def input_ohlcv_hash(ohlcv_by_symbol: dict) -> str:
    """入力OHLCVの内容ハッシュ。

    dataset_id だけでは「その時点の入力値」を復元できない。どのOHLCVから
    作ったかをこのハッシュで固定しておくと、データ改訂の有無を後から判別できる。
    """
    h = hashlib.sha256()
    for symbol in sorted(ohlcv_by_symbol):
        df = ohlcv_by_symbol[symbol]
        h.update(symbol.encode("utf-8"))
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        normalized = df[cols].copy()
        normalized.index = pd.to_datetime(normalized.index).strftime("%Y-%m-%d")
        h.update(normalized.to_csv(float_format=_FLOAT_FORMAT,
                                   lineterminator="\n").encode("utf-8"))
    return h.hexdigest()


def save_dataset_meta(events: pd.DataFrame, dataset_id: str, path,
                      input_hash: str, *, collection_id: Optional[str] = None) -> int:
    """Dataset テーブルにメタ情報を1行書く。書いた行のidを返す。"""
    from src.data.database import Dataset, get_session

    if collection_id is None:
        collection_id = f"{clock.now():%Y%m%dT%H%M%S}-{dataset_id}"

    symbols = sorted(events["symbol"].unique().tolist()) if len(events) else []
    period_start = events["decision_at"].min() if len(events) else None
    period_end = events["decision_at"].max() if len(events) else None

    with get_session() as session:
        row = Dataset(
            dataset_id=dataset_id,
            collection_id=collection_id,
            symbols_json=json.dumps(symbols, ensure_ascii=False),
            period_start=period_start,
            period_end=period_end,
            feature_version=FEATURE_VERSION,
            strategy_version=STRATEGY_VERSION,
            execution_model_version=EXECUTION_MODEL_VERSION,
            file_path=str(path),
            file_sha256=file_sha256(Path(path)),
            input_ohlcv_sha256=input_hash,
            n_events=len(events),
            n_resolved=int((events["status"] == STATUS_RESOLVED).sum()) if len(events) else 0,
        )
        session.add(row)
        session.commit()
        return row.id
```

- [ ] **Step 5: テストを実行して成功を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS（49件）

- [ ] **Step 6: 全体回帰とコミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと（新テーブルは `create_all` が作るので既存DBのテストも通る）

```bash
git add src/data/database.py src/strategy/dataset.py tests/test_dataset.py
git commit -m "$(cat <<'EOF'
feat(data,strategy): イベント表のメタ情報を記録するDatasetテーブルを追加

イベント実体はファイルのまま持ち、DBには来歴だけを置く。
内容ハッシュ(dataset_id)と採取履歴ID(collection_id)を分けて持つので、
同一内容の再生成と実際のデータ改訂を区別できる。入力OHLCVの内容
ハッシュも記録し、その時点の入力値を後から判別できるようにする。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: 一意性重みを fold 内で再計算できるようにする

**Files:**
- Modify: `src/strategy/dataset.py`（`uniqueness_weights` を追加）
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: Task 1 の `STATUS_RESOLVED`
- Produces: `uniqueness_weights(events: pd.DataFrame) -> np.ndarray` — 引数と同じ長さの重み配列

**背景（spec §7）:** イベント表に一度だけ計算した一意性重みをそのまま各foldへ流すと、**検証側イベントの終了時点が学習側の重みへ影響する**。そのため重みは列として保存せず（本計画の差分2）、purge後の学習イベント集合に対してこの関数を呼んで再計算する。呼ぶのは段階Cの `validation.py`。

**重なりの数え方:** `entry_at` から `label_end_at` までを**暦日**で数える。全イベントが同じ暦を共有するため相対的な重なりは保たれる。厳密なセッション単位にする必要が出たら段階Cで分割器と一緒に見直す。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_dataset.py` の末尾に追記する。

```python
def _events_with_spans(spans: list[tuple]) -> pd.DataFrame:
    """(entry_at, label_end_at) の並びから最小のイベント表を作る"""
    rows = []
    for k, (entry, end) in enumerate(spans):
        rows.append({
            "event_id": f"X:{k}",
            "symbol": "7203",
            "entry_at": entry,
            "label_end_at": end,
            "status": dataset.STATUS_RESOLVED if end is not None else dataset.STATUS_IMMATURE,
        })
    return pd.DataFrame(rows)


class TestUniquenessWeights:
    def test_isolated_events_get_weight_one(self):
        """重ならないイベントは重み1"""
        events = _events_with_spans([
            (date(2026, 9, 1), date(2026, 9, 2)),
            (date(2026, 9, 10), date(2026, 9, 11)),
        ])
        w = dataset.uniqueness_weights(events)
        assert w == pytest.approx([1.0, 1.0])

    def test_fully_overlapping_events_get_half(self):
        """完全に重なる2件はそれぞれ重み0.5"""
        events = _events_with_spans([
            (date(2026, 9, 1), date(2026, 9, 3)),
            (date(2026, 9, 1), date(2026, 9, 3)),
        ])
        w = dataset.uniqueness_weights(events)
        assert w == pytest.approx([0.5, 0.5])

    def test_partial_overlap_is_between(self):
        """一部だけ重なるイベントの重みは0.5と1.0の間"""
        events = _events_with_spans([
            (date(2026, 9, 1), date(2026, 9, 4)),
            (date(2026, 9, 3), date(2026, 9, 6)),
        ])
        w = dataset.uniqueness_weights(events)
        assert all(0.5 < x < 1.0 for x in w)

    def test_unresolved_events_get_zero(self):
        """決着していないイベントは重み0（学習に効かせない）"""
        events = _events_with_spans([
            (date(2026, 9, 1), date(2026, 9, 3)),
            (date(2026, 9, 1), None),
        ])
        w = dataset.uniqueness_weights(events)
        assert w[1] == pytest.approx(0.0)

    def test_length_matches_input(self):
        events = _events_with_spans([
            (date(2026, 9, 1), date(2026, 9, 3)),
            (date(2026, 9, 2), date(2026, 9, 5)),
            (date(2026, 9, 9), None),
        ])
        assert len(dataset.uniqueness_weights(events)) == 3

    def test_removing_an_event_changes_remaining_weights(self):
        """fold内で再計算する意味があること＝集合が変われば重みも変わる（spec §7）"""
        full = _events_with_spans([
            (date(2026, 9, 1), date(2026, 9, 3)),
            (date(2026, 9, 1), date(2026, 9, 3)),
        ])
        subset = _events_with_spans([
            (date(2026, 9, 1), date(2026, 9, 3)),
        ])
        assert dataset.uniqueness_weights(full)[0] != pytest.approx(
            dataset.uniqueness_weights(subset)[0])

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(columns=["entry_at", "label_end_at", "status"])
        assert len(dataset.uniqueness_weights(empty)) == 0
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.dataset' has no attribute 'uniqueness_weights'`

- [ ] **Step 3: 実装を追加**

`src/strategy/dataset.py` の末尾に追加する。import に `from collections import Counter` を足す。

```python
def uniqueness_weights(events: pd.DataFrame) -> np.ndarray:
    """イベント期間の重なりに基づく一意性重み（López de Prado の average uniqueness）。

    **この関数は fold ごとに、purge 後の学習イベント集合に対して呼ぶこと。**
    イベント表全体で一度だけ計算した値を各foldへ流すと、検証側イベントの
    終了時点が学習側の重みへ影響する（spec §7）。そのため重みは列として
    保存せず、必要なときにこの関数で計算する。

    重なりは entry_at 〜 label_end_at を暦日で数える。全イベントが同じ暦を
    共有するため相対的な重なりは保たれる。決着していない（label_end_at が無い）
    イベントは重み0にして学習に効かせない。
    """
    n = len(events)
    if n == 0:
        return np.zeros(0)

    spans = []
    for _, row in events.iterrows():
        entry, end = row.get("entry_at"), row.get("label_end_at")
        if entry is None or end is None or pd.isna(entry) or pd.isna(end):
            spans.append(None)
            continue
        spans.append((pd.Timestamp(entry), pd.Timestamp(end)))

    concurrency = Counter()
    for span in spans:
        if span is None:
            continue
        for day in pd.date_range(span[0], span[1], freq="D"):
            concurrency[day] += 1

    weights = np.zeros(n)
    for k, span in enumerate(spans):
        if span is None:
            continue
        days = pd.date_range(span[0], span[1], freq="D")
        weights[k] = float(np.mean([1.0 / concurrency[d] for d in days]))
    return weights
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS（56件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/dataset.py tests/test_dataset.py
git commit -m "$(cat <<'EOF'
feat(strategy): 一意性重みをfold内で再計算する関数を追加

イベント表全体で一度だけ計算した重みを各foldへ流すと、検証側イベントの
終了時点が学習側の重みへ影響する。列として保存せず、purge後の学習集合に
対して呼ぶ関数として提供する。決着していないイベントは重み0にする。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `labeling.py` を legacy 専用と明示する

**Files:**
- Modify: `src/strategy/labeling.py`（モジュールdocstringのみ。**挙動は変更しない**）
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: なし
- Produces: なし（ドキュメントのみ）

**背景:** spec §4 は `labeling.py` を薄い層へ縮小するとしていたが、spec §10 の互換保証（`build_training_set()` の挙動を変えない／`legacy` は旧評価方式へ戻す指定）が優先される。本計画の冒頭「差分1」を参照。実装者が誤って縮小しないよう、モジュール自身にも根拠を残す。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_dataset.py` の末尾に追記する。

```python
class TestLegacyLabelingUnchanged:
    """labeling.py は legacy 経路が依存しているため挙動を変えない（本計画の差分1）"""

    def test_build_training_set_still_returns_three_parts(self):
        from src.strategy import labeling

        X, y, w = labeling.build_training_set(_ohlcv(200))
        assert len(X) == len(y) == len(w)
        assert list(X.columns) == list(indicators.FEATURE_COLS)
        assert set(y.unique()) <= {0, 1}

    def test_triple_barrier_labels_still_available(self):
        from src.strategy import labeling

        feat = indicators.build_features(_ohlcv(200)).reset_index(drop=True)
        labels, t_ends = labeling.triple_barrier_labels(
            feat, pt_mult=2.0, sl_mult=2.0, max_holding=10)
        assert len(labels) == len(feat)
        assert len(t_ends) == len(feat)

    def test_docstring_marks_module_as_legacy(self):
        """v2経路はdataset.pyを使うことがモジュール自身に書かれている"""
        from src.strategy import labeling

        assert "dataset.py" in (labeling.__doc__ or "")
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_dataset.py::TestLegacyLabelingUnchanged -v`
Expected: `test_docstring_marks_module_as_legacy` が FAIL。他の2件はPASS（既存挙動が保たれている証拠）。

- [ ] **Step 3: docstring を追記する**

`src/strategy/labeling.py` のモジュールdocstringの**末尾**に次の段落を追加する。既存の本文・関数・挙動には一切触れない。

```
**このモジュールは legacy 経路専用である（段階B後半以降）。**
v2 経路のラベル生成は src/strategy/dataset.py が担当する。dataset.py は
退出を policy.py に、約定とコストを backtest/execution.py に委ね、未成熟・
未約定を別ステータスにして学習対象から外す（本モジュールは max_holding に
満たない末尾のイベントにも最終リターンの符号でラベルを付ける）。

本モジュールを縮小・変更しないこと。ml_model.train()/train_multi() の
legacy 経路と src/backtest/engine.py がこの挙動に依存しており、
strategy.engine_version=legacy で旧評価方式へ戻せることが段階投入の前提
（設計書 §10）になっている。
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS（59件）

- [ ] **Step 5: 挙動が変わっていないことを確認**

Run: `git diff --stat src/strategy/labeling.py`
Expected: docstring の行のみが変更されていること（関数定義に差分が無いこと）

Run: `pytest tests/test_ml_train_multi.py tests/test_ml_model_save.py -v`
Expected: PASS（legacy 経路の回帰が無いこと）

- [ ] **Step 6: 全体回帰とコミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

```bash
git add src/strategy/labeling.py tests/test_dataset.py
git commit -m "$(cat <<'EOF'
docs(strategy): labeling.pyがlegacy専用であることを明示

設計書§4は薄い層への縮小としていたが、§10の互換保証
（build_training_setの挙動を変えない／legacyで旧評価方式へ戻せる）が
優先される。ml_model.train_multiとengine.pyがこの挙動に依存するため
縮小しない。v2経路はdataset.pyを使う旨をモジュール自身に残す。
挙動の変更は無し（docstringのみ）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## 段階B 完了条件の確認

spec §14 の段階B完了条件を、前半・後半あわせて検証する。

- [ ] **確認1: 将来データを追加しても過去の特徴量が変わらない**（前半で実装済み）

Run: `pytest tests/test_indicators.py::TestBuildFeatureFrame::test_appending_future_rows_does_not_change_past_features -v`
Expected: PASS

- [ ] **確認2: 未成熟ラベルを学習しない**

Run: `pytest tests/test_dataset.py::TestBuildEvents::test_only_resolved_rows_have_labels -v` と `pytest tests/test_dataset.py::TestBuildEvents::test_tail_sessions_are_immature_not_labelled -v`
Expected: PASS

- [ ] **確認3: ラベルの終了イベントが実行シミュレーションと一致する**

Run: `pytest tests/test_dataset.py::TestSimulateEvent -v`
Expected: PASS（9件）。`label_end_at` が `exit_fill` の約定日と一致し、`net_return` が同じ約定価格から算出されていること

- [ ] **確認4: 当日の高値を遡ってストップへ使わない**（前半で実装済み）

Run: `pytest tests/test_policy.py::TestStepPeakOrdering -v`
Expected: PASS

- [ ] **確認5: 同一内容なら同じ dataset_id になる**

Run: `pytest tests/test_dataset.py::TestDatasetId -v`
Expected: PASS（4件）

- [ ] **確認6: legacy 経路に回帰が無い**

Run: `pytest tests/ -q`
Expected: 段階B後半の着手前と同じ結果（新規テスト59件ぶんだけ増える）

- [ ] **確認7: 学習データ量の増減内訳を測る**（spec §16 リスク1）

イベント表を実データで1回生成し、次を記録する。判断材料であり合否ではない。

```python
# 例: 生成したイベント表の内訳を出す
counts = events["status"].value_counts()
print(counts)
print("resolved の正例率:", events[events["status"] == "resolved"]["label"].mean())
```

Expected: `resolved` / `immature` / `unfilled` / `invalid_features` の件数と、`resolved` の正例率が得られること。従来（19,458サンプル）との差は、未成熟の除外・RSI修正・候補定義の変更・学習窓の変更が同時に効くため、**方向を事前に断定しない**。

---

## 次の段階

段階C（`src/strategy/validation.py`）は、本計画が作ったイベント表を入力として、全銘柄共通のカレンダー日付での fold 分割・`label_end_at >= 検証開始日` の purge・fold 内での `uniqueness_weights()` 再計算・前処理の学習側のみ fit・入れ子CV・予測明細の保存を実装する。`Prediction` / `PredictionOutcome` テーブルもそこで追加する（spec §7・§11）。
