# ML評価基盤 段階B前半（退出ポリシーと執行アダプタ）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 売買規則（いつ退出するか）を `policy.py` の1箇所に集約し、約定の仮定を `execution.py` に分離する。あわせて特徴量の端点と欠損の扱いを仕様化する。

**Architecture:** 「判断」と「約定」を分ける。`policy.py` はその時点までに観測できた情報だけを受け取り逐次に状態を進めて**退出意図**を返す純粋関数群（DB・ネットワーク・将来足に触らない）。`execution.py` がその意図を約定へ変換し、コストを一元的に控除する。この2つを後半のイベント表生成・段階Dのバックテスト・段階Eの実運用アダプタが共有する。既存の `build_features()` / `labeling.py` / `engine.py` は本計画では一切変更しない。

**Tech Stack:** Python 3.11 / pandas 2.1.4 / numpy 1.26.2 / pytest / dataclasses

**Spec:** `docs/superpowers/specs/2026-09-10-ml-evaluation-foundation-design.md`（§3・§4・§6・§10・§12・§14）

## Global Constraints

- 日時は **JST naive**。現在時刻は `src/core/clock.now()` / `clock.today()` を使い、`datetime.now()` を直接呼ばない。
- **`policy.py` は設定ファイル・DB・ネットワークに直接触らない。** 設定は `PolicyConfig` として引数で受け取る。`config.yaml` からの読み出しは専用のファクトリ関数1つに閉じる。
- **`policy.py` は将来の足をまとめて受け取らない。** 1営業日ぶんの観測を受け取って状態を進める逐次インターフェースにする。過去検証専用の関数にしないため。
- **固定利確線は置かない。** 実運用（`src/risk/manager.py:537-586` の `evaluate_exit()`）に存在しないため。退出は損切り線・ブレークイーブン・トレーリング・売りシグナル・最大保有期間の5つだけ。
- **既存の公開関数の挙動は、RSIの端点を除いて変えない。** `build_features()`・`build_training_set()`・`compute_indicators()` の戻り値は本計画の前後で同一であること。`src/backtest/engine.py` と `src/services/trading.py` は変更しない。

  > **例外の明示（外部レビューR18・2026-09-12）。** Task 1 は共有関数 `_rsi()` を
  > 直接書き換えるため、`compute_indicators()` と `build_features()` の戻り値は
  > **縮退系列でだけ変わる**。`engine_version=legacy` に戻しても改修前の値には
  > 戻らない。これはレビューF06（単調上昇でRSIがNaNになり行ごと落ちる）の
  > 是正であり、legacy側にも適用すべきバグ修正だと判断して**意図的な仕様差分**
  > として受け入れる。「legacyの結果は完全に同一」とは言わない。
  >
  > 実測した差分の範囲（本計画で固定する期待値）:
  >
  > | 入力系列 | 旧 `_rsi` 末尾 | 新 `_rsi` 末尾 | 旧NaN件数 | 新NaN件数 |
  > |---|---|---|---|---|
  > | 単調上昇（30本） | `NaN` | `100.0` | 30 | 14（助走のみ） |
  > | 完全横ばい（30本） | `NaN` | `50.0` | 30 | 14（助走のみ） |
  > | 単調下降（30本） | `0.0` | `0.0` | 14 | 14 |
  > | 乱数ウォーク（200本） | — | — | 14 | 14（**最大差 0.0**） |
  >
  > つまり通常の価格系列では一切変わらず、片側の変動が完全に0の縮退系列でだけ
  > NaN が実数になる。影響は `build_features()` の `dropna` を通じて
  > **件数に出る**。実測値（120本・`config.yaml` の既定窓）:
  >
  > | 入力系列 | 改修前の `build_features()` 行数 | 改修後 |
  > |---|---|---|
  > | 完全横ばい（120本） | **0行**（RSIが全行NaNで全滅） | **46行**（rsi=50.0） |
  > | 単調上昇（120本） | **0行** | **46行**（rsi=100.0） |
  > | 乱数ウォーク（200本） | 変化なし | 変化なし |
  >
  > 縮退銘柄は改修前は**学習データから完全に消えていた**。改修後は現れる。
  > `models/` の既存モデルはこの入力を見たことがないため、
  > **本番反映時は再学習が必要**。ウォッチリストに値動きの無い銘柄が
  > 含まれていない限り実害は出ないが、「無かったことになっていた」という
  > 事実そのものは完了報告に書くこと。
- 新規モジュールは追加のみ。`config.yaml` への設定追加も行わない（既存キーだけを使う）。
- ファイルは UTF-8 **BOM無し**・LF で保存する。作業ツリーは `core.autocrlf=true` で CRLF に見えるため、確認は `git show <rev>:<path>` でコミット済みblobに対して行う。
- テストは `pytest tests/<file>.py -v` で実行する。ネットワークへ出るテストを書かない。
- コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` を付ける。実装者自身のモデル名を書かない。

---

## File Structure

| ファイル | 責務 |
|---|---|
| `src/strategy/policy.py`（新規） | 退出「意図」を決める逐次状態遷移。約定価格もコストも決めない。純粋関数のみ |
| `src/backtest/execution.py`（新規） | 退出意図・エントリー意図を約定へ変換し、スリッページと手数料を一元的に控除する |
| `src/strategy/indicators.py`（改修） | RSIの端点を仕様化。日付を保持しマスクを返す新APIを追加（既存APIは無変更） |
| `tests/test_indicators.py`（新規） | RSI端点3ケース、マスク、将来行を足しても過去が変わらないこと、既存APIの回帰 |
| `tests/test_policy.py`（新規） | 基準線の算出、逐次遷移、日足内の順序、悲観／楽観の差、最大保有期間 |
| `tests/test_execution.py`（新規） | T+1寄りエントリー、ギャップダウン約定、未約定、コスト控除が一度だけ行われること |

### 段階B後半（本計画のスコープ外）

`src/strategy/dataset.py`（イベント表）と `src/strategy/labeling.py` の薄い層への縮小は、本計画の完了後に別計画で行う。後半は本計画が確定させる `ExitIntent` / `Fill` / `net_return` の型に依存するため。

---

## Task 1: RSIの端点を仕様化する

**Files:**
- Modify: `src/strategy/indicators.py:11-17`（`_rsi`）
- Test: `tests/test_indicators.py`

**Interfaces:**
- Consumes: なし
- Produces: `_rsi(series: pd.Series, length: int) -> pd.Series` — 挙動のみ変更。シグネチャは不変

**背景:** 現在の `_rsi` は `loss.replace(0, float("nan"))` により、下落幅の移動平均が0のとき（＝単調上昇）RSIをNaNにする。その後 `build_features()` が `dropna(subset=FEATURE_COLS)` でその行ごと落とすため、十分な期間があっても末尾RSIがNaNになると最新行が消える（レビューF06）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_indicators.py` を新規作成する。

```python
"""テクニカル指標の端点と欠損の扱い（src/strategy/indicators.py）

背景: RSIは下落幅の移動平均が0のときNaNになり、単調上昇の系列で
末尾RSIが欠損して行ごと落とされていた（レビューF06）。端点を仕様化する。
"""
import numpy as np
import pandas as pd
import pytest

from src.core import config as cfg
from src.strategy import indicators


@pytest.fixture(autouse=True)
def _load_config():
    """indicators は cfg.get_section("strategy") を読むため、設定を読み込んでおく"""
    cfg.load("config.yaml")


class TestRsiEdgeCases:
    def test_monotonic_rise_is_100(self):
        """下落が一度も無い（loss==0 かつ gain>0）なら RSI=100"""
        s = pd.Series([100.0 + i for i in range(40)])
        rsi = indicators._rsi(s, 14)
        assert rsi.iloc[-1] == pytest.approx(100.0)

    def test_monotonic_fall_is_0(self):
        """上昇が一度も無い（gain==0 かつ loss>0）なら RSI=0"""
        s = pd.Series([100.0 - i for i in range(40)])
        rsi = indicators._rsi(s, 14)
        assert rsi.iloc[-1] == pytest.approx(0.0)

    def test_flat_series_is_50(self):
        """まったく動かない（gain==0 かつ loss==0）なら RSI=50（中立）"""
        s = pd.Series([100.0] * 40)
        rsi = indicators._rsi(s, 14)
        assert rsi.iloc[-1] == pytest.approx(50.0)

    def test_warmup_period_stays_nan(self):
        """助走期間（min_periods未満）はNaNのまま。端点仕様で埋めない"""
        s = pd.Series([100.0 + i for i in range(40)])
        rsi = indicators._rsi(s, 14)
        assert rsi.iloc[0:13].isna().all()


class TestRsiChangeAgainstThePreviousImplementation:
    """改修前の実装との差分を、固定した期待値で見えるようにする。

    改修後どうし（旧API vs 新API）を比べても、共有関数を書き換えた影響は
    見えない。ここでは旧実装をテスト内に写して、どの入力でどう変わるかを
    数値で固定する（外部レビューR18）。

    この差分は **legacy 経路にも及ぶ意図的な仕様差分**である。
    Global Constraints の表と同じ値を置くこと。
    """

    @staticmethod
    def _rsi_before(series, length):
        """改修前の src/strategy/indicators.py:_rsi（そのまま写したもの）"""
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(com=length - 1, min_periods=length).mean()
        loss = (-delta.clip(upper=0)).ewm(com=length - 1, min_periods=length).mean()
        rs = gain / loss.replace(0, float("nan"))
        return 100 - (100 / (1 + rs))

    def test_monotonic_rise_changes_from_nan_to_100(self):
        s = pd.Series([100.0 + i for i in range(30)])
        before = self._rsi_before(s, 14)
        after = indicators._rsi(s, 14)
        assert pd.isna(before.iloc[-1])                 # 改修前: NaN
        assert after.iloc[-1] == pytest.approx(100.0)   # 改修後: 100
        assert int(before.isna().sum()) == 30           # 全行NaN
        assert int(after.isna().sum()) == 14            # 助走期間のみ

    def test_flat_series_changes_from_nan_to_50(self):
        s = pd.Series([100.0] * 30)
        before = self._rsi_before(s, 14)
        after = indicators._rsi(s, 14)
        assert pd.isna(before.iloc[-1])
        assert after.iloc[-1] == pytest.approx(50.0)
        assert int(before.isna().sum()) == 30
        assert int(after.isna().sum()) == 14

    def test_monotonic_fall_is_unchanged(self):
        """下落側は改修前から 0.0 が出ていたので変わらない"""
        s = pd.Series([200.0 - i for i in range(30)])
        before = self._rsi_before(s, 14)
        after = indicators._rsi(s, 14)
        assert before.iloc[-1] == pytest.approx(0.0)
        assert after.iloc[-1] == pytest.approx(0.0)
        assert int(before.isna().sum()) == int(after.isna().sum()) == 14

    def test_ordinary_price_series_is_bit_identical(self):
        """通常の価格系列では一切変わらない（差分は縮退系列に限られる）

        ここが崩れると legacy 経路の既存モデルの入力が広範に変わる。
        差分の範囲をこのテストで囲っておく。
        """
        rng = np.random.default_rng(0)
        s = pd.Series(1000 + np.cumsum(rng.normal(0, 10, 200)))
        before = self._rsi_before(s, 14)
        after = indicators._rsi(s, 14)
        assert int(before.isna().sum()) == int(after.isna().sum()) == 14
        pd.testing.assert_series_equal(before, after, check_names=False)

    @staticmethod
    def _degenerate(close_values):
        n = len(close_values)
        return pd.DataFrame({
            "open": close_values, "high": close_values,
            "low": close_values, "close": close_values,
            "volume": [10000] * n,
        }, index=pd.date_range("2026-01-05", periods=n, freq="D"))

    def test_legacy_build_features_now_yields_rows_for_flat_series(self):
        """完全横ばい銘柄が学習データに現れるようになる（legacyへの波及）

        改修前は RSI が全行 NaN で dropna に全滅し、**0行**だった。
        つまりこの銘柄は学習データから消えていた。改修後は現れる。
        件数（既定設定・120本で46行）は設定窓に依存するので値は固定せず、
        「0行から0行より多くなる」ことと RSI の値だけを固定する。
        """
        _load_config()
        df = self._degenerate([100.0] * 120)
        got = indicators.build_features(df)
        assert len(got) > 0
        assert got["rsi"].iloc[-1] == pytest.approx(50.0)
        assert got[indicators.FEATURE_COLS].notna().all().all()

    def test_legacy_build_features_now_yields_rows_for_monotonic_rise(self):
        _load_config()
        df = self._degenerate([100.0 + i for i in range(120)])
        got = indicators.build_features(df)
        assert len(got) > 0
        assert got["rsi"].iloc[-1] == pytest.approx(100.0)
        assert got[indicators.FEATURE_COLS].notna().all().all()

    def test_mixed_series_stays_between_0_and_100(self):
        """通常の上下動では従来どおり0〜100に収まる（既存挙動の回帰）"""
        rng = np.random.default_rng(42)
        s = pd.Series(100.0 + rng.normal(0, 1, 100).cumsum())
        rsi = indicators._rsi(s, 14).dropna()
        assert len(rsi) > 0
        assert ((rsi >= 0) & (rsi <= 100)).all()
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_indicators.py -v`
Expected: `test_monotonic_rise_is_100` と `test_flat_series_is_50` が FAIL（`nan != 100.0` / `nan != 50.0`）。他の3件はPASS。

- [ ] **Step 3: 実装を修正**

`src/strategy/indicators.py` の `_rsi` を次に置き換える。

```python
def _rsi(series: pd.Series, length: int) -> pd.Series:
    """RSI。端点（片側の変動が0の場合）を仕様として明示する。

    従来は loss を NaN に置換していたため、単調上昇の系列で末尾RSIが
    NaN になり、build_features() の dropna で行ごと落ちていた（レビューF06）。
    端点は次のとおり定義する。

      loss == 0 かつ gain > 0 … 100（下げが一度も無い）
      gain == 0 かつ loss > 0 … 0  （上げが一度も無い）
      両方 0                 … 50 （まったく動いていない＝中立）

    助走期間（ewm の min_periods 未満）は NaN のままにする。
    """
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=length - 1, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(com=length - 1, min_periods=length).mean()
    # loss==0 かつ gain>0 なら rs=inf となり 100-(100/inf)=100 に落ちる。
    # 両方0のときだけ 0/0=NaN になるので、中立の50で明示的に埋める。
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.mask((gain == 0) & (loss == 0), 50.0)
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_indicators.py -v`
Expected: PASS（5件）

- [ ] **Step 5: 全体回帰を確認**

Run: `pytest tests/ -q`
Expected: 変更前と同じ結果（失敗が増えていないこと）

- [ ] **Step 6: BOM確認**

Run: `head -c 3 src/strategy/indicators.py | xxd` と `head -c 3 tests/test_indicators.py | xxd`
Expected: `2222 22`（`"""`）。`efbb bf` が出たら次で除去する。

```python
for p in ["src/strategy/indicators.py", "tests/test_indicators.py"]:
    with open(p, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        with open(p, "wb") as f:
            f.write(data[3:])
```

- [ ] **Step 7: コミット**

```bash
git add src/strategy/indicators.py tests/test_indicators.py
git commit -m "$(cat <<'EOF'
fix(strategy): 単調上昇の系列でRSIがNaNになり行ごと落ちていた問題を修正

下落幅の移動平均が0のときlossをNaNに置換していたため、十分な期間が
あっても末尾RSIがNaNになりbuild_featuresのdropnaで最新行が消えていた。
端点を仕様化する（片側0なら100/0、両方0なら中立の50）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: 日付を保持しマスクを返す特徴量APIを追加する

**Files:**
- Modify: `src/strategy/indicators.py:63-90`（`build_features` の内部を分割し、新APIを追加）
- Test: `tests/test_indicators.py`

**Interfaces:**
- Consumes: Task 1 の `_rsi`
- Produces:
  - `build_feature_frame(df: pd.DataFrame) -> pd.DataFrame` — 行を落とさず、日付インデックスを保持し、`feature_valid: bool` 列を付けて返す（v2経路専用の新API）
  - `build_features(df: pd.DataFrame) -> pd.DataFrame` — **挙動不変**（従来どおり `dropna(subset=FEATURE_COLS)` した結果を返す）

**背景:** 現在の `build_features()` は特徴量が欠損した行を落として返すため、呼び出し側が `reset_index(drop=True)` して行番号を「N営業日」として数えると、途中欠損があった分だけ実日数とずれる（レビューF06後半）。ただし `build_features()` は `engine.py:75` と `signal.py` が依存しており、挙動を変えると `engine.py:136` の助走期間ガードが無効化されてNaN行が推論へ流れる（spec §10）。そのため**既存APIは変えず、新APIを追加する**。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_indicators.py` の末尾に追記する。冒頭の import に `from datetime import date, timedelta` を足す。

```python
def _ohlcv(n: int, start_price: float = 1000.0) -> pd.DataFrame:
    """日付インデックス・昇順・重複なしの単一銘柄OHLCVを作る"""
    start = date(2025, 1, 6)  # 月曜
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


class TestBuildFeatureFrame:
    def test_keeps_all_rows_and_dates(self):
        """行を落とさず、日付インデックスをそのまま保持する"""
        df = _ohlcv(120)
        out = indicators.build_feature_frame(df)
        assert len(out) == len(df)
        assert list(out.index) == list(df.index)

    def test_marks_warmup_rows_invalid(self):
        """助走期間（指標が揃わない先頭）は feature_valid=False"""
        df = _ohlcv(120)
        out = indicators.build_feature_frame(df)
        assert bool(out["feature_valid"].iloc[0]) is False
        assert bool(out["feature_valid"].iloc[-1]) is True

    def test_valid_mask_matches_feature_completeness(self):
        """feature_valid は FEATURE_COLS が全て揃っている行と一致する"""
        df = _ohlcv(120)
        out = indicators.build_feature_frame(df)
        expected = out[indicators.FEATURE_COLS].notna().all(axis=1)
        assert (out["feature_valid"] == expected).all()

    def test_appending_future_rows_does_not_change_past_features(self):
        """将来の行を足しても、過去の行の特徴量は変わらない（spec §14 段階B完了条件）"""
        df = _ohlcv(120)
        base = indicators.build_feature_frame(df)

        extended = _ohlcv(150)
        after = indicators.build_feature_frame(extended)

        common = base.index
        for col in indicators.FEATURE_COLS:
            pd.testing.assert_series_equal(
                base.loc[common, col], after.loc[common, col],
                check_names=False,
            )


class TestBuildFeaturesUnchanged:
    def test_legacy_api_still_drops_invalid_rows(self):
        """既存APIは従来どおり欠損行を落とす（legacy経路が依存している）"""
        df = _ohlcv(120)
        legacy = indicators.build_features(df)
        assert legacy[indicators.FEATURE_COLS].notna().all().all()
        assert len(legacy) < len(df)  # 助走期間ぶんは落ちている

    def test_legacy_api_matches_valid_rows_of_new_api(self):
        """既存APIの結果は、新APIの feature_valid=True の行と一致する"""
        df = _ohlcv(120)
        legacy = indicators.build_features(df)
        framed = indicators.build_feature_frame(df)
        valid = framed[framed["feature_valid"]]
        assert list(legacy.index) == list(valid.index)
        for col in indicators.FEATURE_COLS:
            pd.testing.assert_series_equal(
                legacy[col], valid[col], check_names=False,
            )
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_indicators.py -v`
Expected: `TestBuildFeatureFrame` の4件が FAIL（`AttributeError: module 'src.strategy.indicators' has no attribute 'build_feature_frame'`）。`TestBuildFeaturesUnchanged` の2件はPASS。

- [ ] **Step 3: 実装を修正**

`src/strategy/indicators.py` の `build_features` を、共通部の抽出と新APIの追加に置き換える。`FEATURE_COLS` の定義位置は変えない（ファイル末尾のまま）。

```python
def _add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """指標から派生する特徴量列を付加する（欠損行の扱いは呼び出し側が決める）。"""
    df = compute_indicators(df)
    conf = cfg.get_section("strategy")
    short = conf.get("ma_short", 5)
    mid = conf.get("ma_mid", 25)
    long_ = conf.get("ma_long", 75)

    df["ma_cross_sm"] = df[f"ma{short}"] - df[f"ma{mid}"]
    df["ma_cross_ml"] = df[f"ma{mid}"] - df[f"ma{long_}"]
    bb_width = (df["bb_upper"] - df["bb_lower"]).clip(lower=1e-4)
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / bb_width
    df["price_momentum_5"] = df["close"].pct_change(5, fill_method=None)
    df["price_momentum_20"] = df["close"].pct_change(20, fill_method=None)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """ML用の特徴量を作成して返す（ラベルは付与しない）。

    **挙動を変更しないこと。** legacy経路（src/backtest/engine.py・
    src/strategy/signal.py）がこの戻り値に依存しており、特に engine.py は
    「この関数が落とした行＝助走期間」という前提で
    `if dt not in featured_df.index` により推論をスキップしている。
    dropna をやめるとそのガードが無効化され、NaN行が推論へ流れる。

    日付を保持したまま欠損をマスクで扱いたい場合は build_feature_frame() を使う。
    """
    return _add_feature_columns(df).dropna(subset=FEATURE_COLS)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量を、日付インデックスを保持したまま返す（v2経路用）。

    build_features() は欠損行を落として返すため、呼び出し側が
    reset_index(drop=True) して行番号を「N営業日」として数えると、
    途中に欠損があった分だけ実日数とずれる（レビューF06後半）。
    本APIは行を落とさず、学習・判定に使ってよい行かを feature_valid 列で示す。
    時間軸の連続性は市場系列側で保ち、採否はマスクで管理する。
    """
    out = _add_feature_columns(df)
    out["feature_valid"] = out[FEATURE_COLS].notna().all(axis=1)
    return out
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_indicators.py -v`
Expected: PASS（11件）

- [ ] **Step 5: 既存経路の回帰を確認**

Run: `pytest tests/ -q`
Expected: 変更前と同じ結果。特に `tests/test_backtest_threshold_override.py` と `tests/test_ml_train_multi.py` が通ること（`build_features` の挙動不変の裏付け）

- [ ] **Step 6: BOM確認とコミット**

Run: `head -c 3 src/strategy/indicators.py | xxd`（`2222 22` を確認）

```bash
git add src/strategy/indicators.py tests/test_indicators.py
git commit -m "$(cat <<'EOF'
feat(strategy): 日付を保持しマスクを返す特徴量APIを追加

build_features()は欠損行を落として返すため、行番号を営業日として
数えると途中欠損の分だけ実日数とずれる。日付を保持しfeature_valid列で
採否を示す新APIを追加する。既存APIはlegacy経路（engine.pyの助走期間
ガード）が依存しているため挙動を変えない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 退出ポリシーの型と基準線

**Files:**
- Create: `src/strategy/policy.py`
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: `src/core/config.get_section`（ファクトリ関数の中だけ）
- Produces:
  - 定数 `STOP_LINE` / `TRAILING` / `SIGNAL_SELL` / `TIME_LIMIT` / `PEAK_BASIS_PREVIOUS` / `PEAK_BASIS_SAME_SESSION`
  - `HoldingState`（frozen dataclass）: `symbol: str`, `entry_at: date`, `avg_cost: float`, `quantity: int`, `peak_price: float`, `sessions_held: int`
  - `Observation`（frozen dataclass）: `session: date`, `open: float`, `high: float`, `low: float`, `close: float`, `score: Optional[float] = None`
  - `ExitIntent`（frozen dataclass）: `reason: str`, `trigger_price: Optional[float]`, `order_type: str`
  - `PolicyConfig`（frozen dataclass）: `stop_loss_pct: float`, `breakeven_trigger_pct: float`, `trailing_stop_pct: float`, `sell_threshold: float`, `max_holding_sessions: int`
  - `config_from_settings() -> PolicyConfig`
  - `is_armed(state: HoldingState, conf: PolicyConfig) -> bool`
  - `stop_line(state: HoldingState, conf: PolicyConfig) -> float`

**設計上の注記（spec §6 からの意図的な差分）:** spec §6 の `HoldingState` は `armed: bool` を持つとしていたが、**フィールドとしては持たせない**。実運用の `src/risk/manager.py:574` は `armed` を永続化せず、その時点の `peak_price` から毎回導出している。保存すると `peak_price` と乖離しうるため、`is_armed()` として導出関数にする。single source of truth は `peak_price` に置く。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_policy.py` を新規作成する。

```python
"""退出ポリシー（src/strategy/policy.py）のテスト

売買規則の唯一の実装。ラベル生成・バックテスト・実運用の3者が共有する。
退出条件は src/risk/manager.py:evaluate_exit() と同じ構造であること
（固定利確線は運用に存在しないため置かない）。
"""
from datetime import date

import pytest

from src.core import config as cfg
from src.strategy import policy


def _conf(stop=-0.07, breakeven=0.02, trailing=0.04,
          sell_thr=-0.25, max_holding=10) -> policy.PolicyConfig:
    return policy.PolicyConfig(
        stop_loss_pct=stop,
        breakeven_trigger_pct=breakeven,
        trailing_stop_pct=trailing,
        sell_threshold=sell_thr,
        max_holding_sessions=max_holding,
    )


def _state(avg_cost=1000.0, peak=1000.0, sessions_held=0) -> policy.HoldingState:
    return policy.HoldingState(
        symbol="7203",
        entry_at=date(2026, 9, 1),
        avg_cost=avg_cost,
        quantity=100,
        peak_price=peak,
        sessions_held=sessions_held,
    )


class TestIsArmed:
    def test_not_armed_before_trigger(self):
        """ピーク時の含み益がトリガー未満なら未発動"""
        assert policy.is_armed(_state(peak=1015.0), _conf()) is False

    def test_armed_at_trigger(self):
        """含み益+2%ちょうどで発動する"""
        assert policy.is_armed(_state(peak=1020.0), _conf()) is True

    def test_never_armed_when_trigger_disabled(self):
        """breakeven_trigger_pct=0 なら常に未発動（従来の損切りのみの挙動）"""
        assert policy.is_armed(_state(peak=2000.0), _conf(breakeven=0.0)) is False


class TestStopLine:
    def test_plain_stop_before_arming(self):
        """未発動なら基準線は取得単価×(1+stop_loss_pct)"""
        assert policy.stop_line(_state(peak=1010.0), _conf()) == pytest.approx(930.0)

    def test_raised_to_breakeven_after_arming(self):
        """発動後は取得単価まで引き上がる（元本割れリスクを取らない）"""
        # ピーク1020（+2%）でarmed。trailing線は1020*0.96=979.2で取得単価1000より下
        # なので、この時点の基準線は取得単価そのもの
        assert policy.stop_line(_state(peak=1020.0), _conf()) == pytest.approx(1000.0)

    def test_trailing_takes_over_when_higher(self):
        """ピークが伸びるとトレーリング線が取得単価を上回り、そちらが採用される"""
        # ピーク1100 → 1100*0.96=1056 > 取得単価1000
        assert policy.stop_line(_state(peak=1100.0), _conf()) == pytest.approx(1056.0)

    def test_trailing_disabled_keeps_breakeven(self):
        """trailing_stop_pct=0 なら発動後もブレークイーブン止まり"""
        line = policy.stop_line(_state(peak=1100.0), _conf(trailing=0.0))
        assert line == pytest.approx(1000.0)

    def test_matches_production_formula(self):
        """src/risk/manager.py:evaluate_exit と同じ式であること（数値で突き合わせる）"""
        avg_cost, peak = 1000.0, 1100.0
        stop_pct, trailing_pct, breakeven = -0.07, 0.04, 0.02

        # production の式をそのまま書き下したもの
        expected = avg_cost * (1 + stop_pct)
        peak_gain_pct = (peak - avg_cost) / avg_cost
        armed = breakeven > 0 and peak_gain_pct >= breakeven
        if armed:
            expected = max(expected, avg_cost)
            if trailing_pct > 0:
                expected = max(expected, peak * (1 - trailing_pct))

        actual = policy.stop_line(
            _state(avg_cost=avg_cost, peak=peak),
            _conf(stop=stop_pct, breakeven=breakeven, trailing=trailing_pct),
        )
        assert actual == pytest.approx(expected)


class TestConfigFromSettings:
    def test_reads_from_correct_sections(self):
        """trading節とstrategy節の双方から正しく読む"""
        cfg.load("config.yaml")
        conf = policy.config_from_settings()
        trading = cfg.get_section("trading")
        strategy = cfg.get_section("strategy")
        assert conf.stop_loss_pct == trading["stop_loss_pct"]
        assert conf.breakeven_trigger_pct == trading["breakeven_trigger_pct"]
        assert conf.trailing_stop_pct == trading["trailing_stop_pct"]
        assert conf.sell_threshold == strategy["sell_threshold"]
        assert conf.max_holding_sessions == strategy["tb_max_holding"]
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_policy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.strategy.policy'`

- [ ] **Step 3: 実装を書く**

`src/strategy/policy.py` を新規作成する。

```python
"""退出ポリシー — 売買規則の唯一の実装。

ラベル生成・バックテスト・実運用の3者がこのモジュールだけを使って
「いつ・なぜ退出するか」を決める。規則が1箇所にあるため、退出条件を
変えると3者が同時に追随する。従来はこの規則が labeling.py・engine.py・
risk/manager.py の3箇所に別々の実装として散らばっており、片方を直しても
他方とずれていた（レビューF04・F05）。

**約定価格はここで決めない。** 退出の「意図」（発動理由と基準価格）までを返し、
実際にいくらで約定したかは src/backtest/execution.py（過去検証）または
実運用アダプタが決める。判断と約定を同じ関数が返すと、将来の日足をまとめて
受け取る関数が過去検証専用になり、その時点までの足しか渡せない実運用の
インターフェースにならないため。

退出条件は src/risk/manager.py:evaluate_exit() と同じ構造にする。
**固定利確線は置かない**（運用に存在しないため）。
"""
from dataclasses import dataclass, replace
from datetime import date
from typing import Optional

from src.core import config as cfg

# 退出理由
STOP_LINE = "STOP_LINE"      # ブレークイーブン発動前の損切り線に到達
TRAILING = "TRAILING"        # 発動後の（引き上がった）基準線に到達
SIGNAL_SELL = "SIGNAL_SELL"  # ルール由来の売りシグナル
TIME_LIMIT = "TIME_LIMIT"    # 最大保有営業日数に到達

# 日足内でピークをいつ反映するか
PEAK_BASIS_PREVIOUS = "previous"          # 既定（悲観）: 前営業日終了時点のピークで当日の線を固定
PEAK_BASIS_SAME_SESSION = "same_session"  # 楽観: 当日の高値を即座に反映


@dataclass(frozen=True)
class HoldingState:
    """保有の状態。

    取得単価と数量だけでは、追加購入・部分決済・再起動を跨いだピーク価格や
    ストップ発動状態を表せない。実運用は src/data/database.py:162 の
    Position.peak_price としてまさにこの値を永続化している。

    `armed`（ブレークイーブン発動済みか）はフィールドとして持たない。
    実運用（risk/manager.py:574）も永続化せず peak_price から毎回導出しており、
    保存すると乖離しうるため is_armed() で導出する。
    """
    symbol: str
    entry_at: date
    avg_cost: float
    quantity: int
    peak_price: float      # 保有開始以降の最高値
    sessions_held: int     # 経過営業日数


@dataclass(frozen=True)
class Observation:
    """その時点で観測できた1営業日ぶんの情報。

    将来の足は含めない。実運用ではその日の板から、過去検証では日足から作る。
    score は combined_score（売りシグナル判定用）。ラベル生成で売りシグナルを
    考慮しない場合は None を渡す。
    """
    session: date
    open: float
    high: float
    low: float
    close: float
    score: Optional[float] = None


@dataclass(frozen=True)
class ExitIntent:
    """退出の意図。約定価格ではない。

    trigger_price は発動の基準となった価格であり、実際にいくらで約定したかは
    execution.py が決める（ギャップダウン時は基準価格では約定できない）。
    order_type は "STOP"（当日中に基準価格へ到達したとみなす）または
    "MARKET"（翌営業日の寄りで成行）。
    """
    reason: str
    trigger_price: Optional[float]
    order_type: str


@dataclass(frozen=True)
class PolicyConfig:
    """退出判定に使う設定。

    policy.py は config.yaml を直接読まない（純粋関数に保つため）。
    設定からの構築は config_from_settings() に閉じる。
    """
    stop_loss_pct: float           # trading.stop_loss_pct（負の値）
    breakeven_trigger_pct: float   # trading.breakeven_trigger_pct（0で無効）
    trailing_stop_pct: float       # trading.trailing_stop_pct（0で無効）
    sell_threshold: float          # strategy.sell_threshold
    max_holding_sessions: int      # strategy.tb_max_holding


def config_from_settings() -> PolicyConfig:
    """config.yaml から PolicyConfig を作る（唯一の読み出し口）。

    退出まわりの値は trading 節、判定閾値と保有期間は strategy 節にある。
    どちらから読んだ値かをここで固定し、呼び出し側が節を意識しないようにする。
    """
    trading = cfg.get_section("trading")
    strategy = cfg.get_section("strategy")
    return PolicyConfig(
        stop_loss_pct=trading.get("stop_loss_pct", -0.05),
        breakeven_trigger_pct=trading.get("breakeven_trigger_pct", 0.0),
        trailing_stop_pct=trading.get("trailing_stop_pct", 0.0),
        sell_threshold=strategy.get("sell_threshold", -0.25),
        max_holding_sessions=strategy.get("tb_max_holding", 10),
    )


def is_armed(state: HoldingState, conf: PolicyConfig) -> bool:
    """ブレークイーブンが発動済みか（peak_price から導出する）。"""
    if conf.breakeven_trigger_pct <= 0 or state.avg_cost <= 0:
        return False
    peak_gain_pct = (state.peak_price - state.avg_cost) / state.avg_cost
    return peak_gain_pct >= conf.breakeven_trigger_pct


def stop_line(state: HoldingState, conf: PolicyConfig) -> float:
    """この時点の基準線（損切り・ブレークイーブン・トレーリングの最も高い方）。

    src/risk/manager.py:572-578 と同じ式:
      1. 基準線 = 取得単価 × (1 + stop_loss_pct)
      2. ピーク時の含み益率が breakeven_trigger_pct 以上なら取得単価まで引き上げ
      3. さらに trailing_stop_pct > 0 なら ピーク×(1-trailing) とも比べて高い方
    """
    line = state.avg_cost * (1 + conf.stop_loss_pct)
    if is_armed(state, conf):
        line = max(line, state.avg_cost)
        if conf.trailing_stop_pct > 0:
            line = max(line, state.peak_price * (1 - conf.trailing_stop_pct))
    return line
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_policy.py -v`
Expected: PASS（9件）

- [ ] **Step 5: BOM確認とコミット**

Run: `head -c 3 src/strategy/policy.py | xxd`（`2222 22` を確認）

```bash
git add src/strategy/policy.py tests/test_policy.py
git commit -m "$(cat <<'EOF'
feat(strategy): 退出ポリシーの型と基準線の算出を追加

売買規則がlabeling/engine/risk-managerの3箇所に散らばっており、
片方を直しても他方とずれていた（レビューF04・F05）。規則を1箇所に
集約する土台として、保有状態・観測・退出意図の型と、実運用の
evaluate_exitと同じ基準線の式を置く。armedはpeak_priceから導出し
（実運用と同じ）、保存しない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: 逐次の状態遷移と悲観規約

**Files:**
- Modify: `src/strategy/policy.py`（`step` を追加）
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: Task 3 の `HoldingState` / `Observation` / `ExitIntent` / `PolicyConfig` / `is_armed` / `stop_line`
- Produces: `step(state: HoldingState, obs: Observation, conf: PolicyConfig, *, peak_basis: str = PEAK_BASIS_PREVIOUS) -> tuple[HoldingState, Optional[ExitIntent]]`

**背景（日足内の順序）:** トレーリングには日足では解けない順序問題がある。ある日の高値がピークを更新して基準線を引き上げ、同じ日の安値がその引き上がった線に触れる場合、高値と安値のどちらが先に起きたかを日足から復元できない。既定では**当日の基準線を前営業日終了時点のピークで固定**し、当日の高値によるピーク更新は当日の判定が終わってから反映する。これで未来のピークを遡ってストップに使うことがなくなる。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_policy.py` の末尾に追記する。

```python
def _obs(session=date(2026, 9, 2), o=1000.0, h=1010.0, l=990.0, c=1005.0,
         score=None) -> policy.Observation:
    return policy.Observation(session=session, open=o, high=h, low=l, close=c, score=score)


class TestStepStopTriggers:
    def test_no_exit_when_low_stays_above_line(self):
        """安値が基準線を割らなければ退出しない"""
        state = _state(avg_cost=1000.0, peak=1000.0)
        nxt, intent = policy.step(state, _obs(l=950.0), _conf())
        assert intent is None
        assert nxt.sessions_held == 1

    def test_stop_line_reason_before_arming(self):
        """未発動で基準線に到達したら STOP_LINE"""
        state = _state(avg_cost=1000.0, peak=1000.0)
        nxt, intent = policy.step(state, _obs(l=920.0), _conf())
        assert intent is not None
        assert intent.reason == policy.STOP_LINE
        assert intent.order_type == "STOP"
        assert intent.trigger_price == pytest.approx(930.0)

    def test_trailing_reason_after_arming(self):
        """発動後に基準線へ到達したら TRAILING"""
        state = _state(avg_cost=1000.0, peak=1100.0)  # armed、線は1056
        nxt, intent = policy.step(state, _obs(h=1100.0, l=1050.0), _conf())
        assert intent is not None
        assert intent.reason == policy.TRAILING
        assert intent.trigger_price == pytest.approx(1056.0)

    def test_signal_sell_is_market_order(self):
        """売りシグナルは翌営業日の寄りで成行（trigger_priceを持たない）"""
        state = _state(avg_cost=1000.0, peak=1000.0)
        nxt, intent = policy.step(state, _obs(l=990.0, score=-0.30), _conf())
        assert intent is not None
        assert intent.reason == policy.SIGNAL_SELL
        assert intent.order_type == "MARKET"
        assert intent.trigger_price is None

    def test_stop_takes_precedence_over_signal_sell(self):
        """同じ日に基準線到達と売りシグナルが揃ったら、基準線を優先する（不利側）"""
        state = _state(avg_cost=1000.0, peak=1000.0)
        nxt, intent = policy.step(state, _obs(l=920.0, score=-0.30), _conf())
        assert intent.reason == policy.STOP_LINE

    def test_time_limit_at_max_holding(self):
        """最大保有営業日数に達したら TIME_LIMIT（翌営業日の寄りで成行）"""
        state = _state(avg_cost=1000.0, peak=1000.0, sessions_held=9)
        nxt, intent = policy.step(state, _obs(l=990.0), _conf(max_holding=10))
        assert intent is not None
        assert intent.reason == policy.TIME_LIMIT
        assert intent.order_type == "MARKET"
        assert nxt.sessions_held == 10

    def test_no_time_limit_before_max_holding(self):
        state = _state(avg_cost=1000.0, peak=1000.0, sessions_held=8)
        nxt, intent = policy.step(state, _obs(l=990.0), _conf(max_holding=10))
        assert intent is None


class TestStepPeakOrdering:
    def test_same_session_high_does_not_raise_todays_line(self):
        """当日の高値でピークが更新されても、当日の基準線は前営業日ピークで固定する。

        未来（当日の高値）を遡ってストップへ使わないための規約。
        ピーク1000（未発動、線=930）の日に高値1100・安値1050が出た場合、
        同日にトレーリング線1056へ引き上げて安値1050で退出、とはしない。
        """
        state = _state(avg_cost=1000.0, peak=1000.0)
        nxt, intent = policy.step(state, _obs(h=1100.0, l=1050.0), _conf())
        assert intent is None                       # 当日は退出しない
        assert nxt.peak_price == pytest.approx(1100.0)  # ピークは当日終了後に反映

    def test_raised_line_applies_from_next_session(self):
        """引き上がった線は翌営業日から効く"""
        state = _state(avg_cost=1000.0, peak=1000.0)
        after_day1, _ = policy.step(state, _obs(h=1100.0, l=1050.0), _conf())
        # 翌日は peak=1100 に基づく線1056が有効
        _, intent = policy.step(after_day1, _obs(session=date(2026, 9, 3), h=1060.0, l=1050.0), _conf())
        assert intent is not None
        assert intent.reason == policy.TRAILING
        assert intent.trigger_price == pytest.approx(1056.0)

    def test_peak_never_decreases(self):
        """安値だけの日でもピークは下がらない"""
        state = _state(avg_cost=1000.0, peak=1100.0)
        nxt, _ = policy.step(state, _obs(h=1020.0, l=1010.0), _conf())
        assert nxt.peak_price == pytest.approx(1100.0)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_policy.py -v`
Expected: FAIL — `AttributeError: module 'src.strategy.policy' has no attribute 'step'`

- [ ] **Step 3: 実装を追加**

`src/strategy/policy.py` の末尾に追加する。

```python
def step(state: HoldingState, obs: Observation, conf: PolicyConfig, *,
         peak_basis: str = PEAK_BASIS_PREVIOUS) -> tuple[HoldingState, Optional[ExitIntent]]:
    """1営業日ぶん状態を進め、退出意図があれば返す。

    戻り値: (次の状態, ExitIntent または None)

    その時点までに観測できた情報だけを受け取り、将来の足はまとめて受け取らない。
    そのため実運用でもそのまま呼べる。

    **日足内の順序について。** トレーリングには日足では解けない順序問題がある。
    ある日の高値がピークを更新して基準線を引き上げ、同じ日の安値がその引き上がった
    線に触れる場合、どちらが先に起きたかを日足から復元できない。
    既定（peak_basis="previous"）では当日の基準線を**前営業日終了時点のピーク**で
    固定し、当日の高値によるピーク更新は判定後に反映する。これで未来のピークを
    遡ってストップへ使うことがなくなる。
    peak_basis="same_session" は当日の高値を即座に反映する楽観仮定で、
    両者の差を測って結果に記録するために用意している（差が大きい戦略は
    日中データを持つまで採用しない）。

    判定の順序は不利側を優先する。同じ日に基準線到達と売りシグナルが揃った
    場合は基準線を採る。
    """
    if peak_basis == PEAK_BASIS_PREVIOUS:
        judged = state
    elif peak_basis == PEAK_BASIS_SAME_SESSION:
        judged = replace(state, peak_price=max(state.peak_price, obs.high))
    else:
        raise ValueError(
            f"peak_basis は '{PEAK_BASIS_PREVIOUS}' か '{PEAK_BASIS_SAME_SESSION}': {peak_basis}"
        )

    next_state = replace(
        state,
        peak_price=max(state.peak_price, obs.high),
        sessions_held=state.sessions_held + 1,
    )

    line = stop_line(judged, conf)
    if obs.low <= line:
        reason = TRAILING if is_armed(judged, conf) else STOP_LINE
        return next_state, ExitIntent(reason=reason, trigger_price=line, order_type="STOP")

    if obs.score is not None and obs.score <= conf.sell_threshold:
        return next_state, ExitIntent(reason=SIGNAL_SELL, trigger_price=None, order_type="MARKET")

    if next_state.sessions_held >= conf.max_holding_sessions:
        return next_state, ExitIntent(reason=TIME_LIMIT, trigger_price=None, order_type="MARKET")

    return next_state, None
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_policy.py -v`
Expected: PASS（19件）

- [ ] **Step 5: コミット**

```bash
git add src/strategy/policy.py tests/test_policy.py
git commit -m "$(cat <<'EOF'
feat(strategy): 退出ポリシーの逐次状態遷移を追加

その時点までに観測できた情報だけで状態を進め、退出意図を返す。
将来の足をまとめて受け取らないので実運用でもそのまま呼べる。
日足内の順序は復元できないため、当日の基準線は前営業日終了時点の
ピークで固定し、当日高値によるピーク更新は判定後に反映する
（未来のピークを遡ってストップへ使わない）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 楽観仮定との差を測る

**Files:**
- Modify: `src/strategy/policy.py`（`run_session_series` を追加）
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: Task 4 の `step`
- Produces: `run_session_series(state: HoldingState, observations: list[Observation], conf: PolicyConfig, *, peak_basis: str = PEAK_BASIS_PREVIOUS) -> tuple[HoldingState, Optional[ExitIntent], Optional[Observation]]` — 退出するまで（または足が尽きるまで）進め、`(最終状態, 退出意図 or None, 意図が出た日の観測 or None)` を返す

**背景:** spec §6 は「悲観側へ寄せるだけでは順序問題は解決しないため、曖昧性を測って残す」としている。同じ観測列を2つの `peak_basis` で流して結果を比べられるようにする。差が大きい戦略は日中データを持つまで採用しない。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_policy.py` の末尾に追記する。

```python
class TestRunSessionSeries:
    def _series(self):
        """1日目に高値1100・安値1050、2日目に安値1050の観測列"""
        return [
            _obs(session=date(2026, 9, 2), o=1000.0, h=1100.0, l=1050.0, c=1090.0),
            _obs(session=date(2026, 9, 3), o=1090.0, h=1095.0, l=1050.0, c=1055.0),
        ]

    def test_pessimistic_exits_on_second_session(self):
        """既定（previous）: 1日目は線が引き上がらず退出せず、2日目に退出する"""
        final, intent, at = policy.run_session_series(
            _state(avg_cost=1000.0, peak=1000.0), self._series(), _conf())
        assert intent is not None
        assert intent.reason == policy.TRAILING
        assert at.session == date(2026, 9, 3)

    def test_optimistic_exits_on_first_session(self):
        """same_session: 1日目の高値で線が1056へ上がり、同じ日の安値1050で退出する"""
        final, intent, at = policy.run_session_series(
            _state(avg_cost=1000.0, peak=1000.0), self._series(), _conf(),
            peak_basis=policy.PEAK_BASIS_SAME_SESSION)
        assert intent is not None
        assert intent.reason == policy.TRAILING
        assert at.session == date(2026, 9, 2)

    def test_difference_between_bases_is_measurable(self):
        """2つの仮定の差（退出日）を測れる＝曖昧性を結果に記録できる"""
        pess = policy.run_session_series(
            _state(avg_cost=1000.0, peak=1000.0), self._series(), _conf())
        opt = policy.run_session_series(
            _state(avg_cost=1000.0, peak=1000.0), self._series(), _conf(),
            peak_basis=policy.PEAK_BASIS_SAME_SESSION)
        assert pess[2].session != opt[2].session

    def test_returns_none_intent_when_bars_run_out(self):
        """足が尽きても決着しない場合は意図なしで返す（未成熟として扱えるように）"""
        calm = [_obs(session=date(2026, 9, 2), h=1005.0, l=995.0)]
        final, intent, at = policy.run_session_series(
            _state(avg_cost=1000.0, peak=1000.0), calm, _conf(max_holding=10))
        assert intent is None
        assert at is None
        assert final.sessions_held == 1

    def test_stops_advancing_after_exit(self):
        """退出した時点で止まる（それ以降の足を消費しない）"""
        bars = [
            _obs(session=date(2026, 9, 2), l=920.0),   # ここで損切り
            _obs(session=date(2026, 9, 3), l=900.0),
        ]
        final, intent, at = policy.run_session_series(
            _state(avg_cost=1000.0, peak=1000.0), bars, _conf())
        assert intent.reason == policy.STOP_LINE
        assert at.session == date(2026, 9, 2)
        assert final.sessions_held == 1
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_policy.py::TestRunSessionSeries -v`
Expected: FAIL — `AttributeError: module 'src.strategy.policy' has no attribute 'run_session_series'`

- [ ] **Step 3: 実装を追加**

`src/strategy/policy.py` の末尾に追加する。

```python
def run_session_series(
    state: HoldingState,
    observations: list[Observation],
    conf: PolicyConfig,
    *,
    peak_basis: str = PEAK_BASIS_PREVIOUS,
) -> tuple[HoldingState, Optional[ExitIntent], Optional[Observation]]:
    """観測列を、退出するか足が尽きるまで進める。

    戻り値: (最終状態, 退出意図 or None, その意図が出た日の観測 or None)

    足が尽きても決着しない場合は意図なしで返す。呼び出し側はこれを
    「未成熟」として扱い、学習ラベルの対象から外す（spec §6）。

    同じ観測列を peak_basis を変えて2回流すと、日足内の順序の仮定による
    差を測れる。悲観側へ寄せるだけでは順序問題は解決しないため、
    曖昧性を測って結果に記録する（spec §6）。
    """
    current = state
    for obs in observations:
        current, intent = step(current, obs, conf, peak_basis=peak_basis)
        if intent is not None:
            return current, intent, obs
    return current, None, None
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_policy.py -v`
Expected: PASS（24件）

- [ ] **Step 5: 全体回帰を確認してコミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

```bash
git add src/strategy/policy.py tests/test_policy.py
git commit -m "$(cat <<'EOF'
feat(strategy): 観測列を流して悲観/楽観の差を測れるようにした

日足内の順序は日足のままでは解消しない。同じ観測列を2つのpeak_basisで
流して退出日の差を測り、結果に記録できるようにする。差が大きい戦略は
日中データを持つまで採用しない、という判断に使う。
足が尽きても決着しない場合は意図なしで返し、呼び出し側が未成熟として
学習対象から外せるようにする。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: 執行アダプタの型とエントリー約定

**Files:**
- Create: `src/backtest/execution.py`
- Test: `tests/test_execution.py`

**Interfaces:**
- Consumes: `src/strategy/policy.Observation`（足の型として再利用する）
- Produces:
  - `Fill`（frozen dataclass）: `at: date`, `price: float`, `quantity: int`, `reason: str`
  - `CostConfig`（frozen dataclass）: `slippage_pct: float`, `commission_pct: float`
  - `config_from_settings() -> CostConfig`
  - `buy_fill_price(price: float, costs: CostConfig) -> float`
  - `sell_fill_price(price: float, costs: CostConfig) -> float`
  - `entry_fill(next_bar: Observation, quantity: int, costs: CostConfig) -> Fill`

**背景:** spec §6 は「Tの引けで判断し、T+1の寄りで約定する」「満了日の終値で判断して同じ終値で約定する経路を作らない」としている（F04）。現行 `engine.py:111-188` は同じ `close_price` でスコア生成と約定を行っている。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_execution.py` を新規作成する。

```python
"""過去検証の執行アダプタ（src/backtest/execution.py）のテスト

退出「意図」を約定へ変換し、スリッページと手数料を一元的に控除する。
Tの引けで判断し T+1 の寄りで約定する（同じ終値で判断・約定しない）。
"""
from datetime import date

import pytest

from src.core import config as cfg
from src.backtest import execution
from src.strategy import policy


def _costs(slip=0.001, comm=0.0) -> execution.CostConfig:
    return execution.CostConfig(slippage_pct=slip, commission_pct=comm)


def _bar(session=date(2026, 9, 2), o=1000.0, h=1010.0, l=990.0, c=1005.0):
    return policy.Observation(session=session, open=o, high=h, low=l, close=c)


class TestConfigFromSettings:
    def test_reads_backtest_section(self):
        cfg.load("config.yaml")
        costs = execution.config_from_settings()
        backtest = cfg.get_section("backtest")
        assert costs.slippage_pct == backtest["slippage_pct"]
        assert costs.commission_pct == backtest["commission_pct"]


class TestFillPrices:
    def test_buy_is_unfavourable(self):
        """買いはスリッページ分だけ高く約定する"""
        assert execution.buy_fill_price(1000.0, _costs(slip=0.001)) == pytest.approx(1001.0)

    def test_sell_is_unfavourable(self):
        """売りはスリッページ分だけ安く約定する"""
        assert execution.sell_fill_price(1000.0, _costs(slip=0.001)) == pytest.approx(999.0)

    def test_matches_existing_engine_convention(self):
        """既存 engine.py の _buy_fill_price/_sell_fill_price と同じ丸め（小数2桁）"""
        assert execution.buy_fill_price(1234.567, _costs(slip=0.001)) == pytest.approx(1235.80)
        assert execution.sell_fill_price(1234.567, _costs(slip=0.001)) == pytest.approx(1233.33)


class TestEntryFill:
    def test_fills_at_next_session_open(self):
        """Tの引けで決めた買いは T+1 の寄りで約定する"""
        next_bar = _bar(session=date(2026, 9, 3), o=1020.0, c=1050.0)
        fill = execution.entry_fill(next_bar, quantity=100, costs=_costs(slip=0.001))
        assert fill.at == date(2026, 9, 3)
        assert fill.price == pytest.approx(1021.02)  # 1020 * 1.001
        assert fill.quantity == 100
        assert fill.reason == "ENTRY"

    def test_does_not_use_the_decision_session_close(self):
        """判断した日の終値では約定しない（F04の回帰防止）"""
        decision_bar = _bar(session=date(2026, 9, 2), c=1005.0)
        next_bar = _bar(session=date(2026, 9, 3), o=1020.0)
        fill = execution.entry_fill(next_bar, quantity=100, costs=_costs(slip=0.0))
        assert fill.at != decision_bar.session
        assert fill.price != pytest.approx(decision_bar.close)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_execution.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backtest.execution'`

- [ ] **Step 3: 実装を書く**

`src/backtest/execution.py` を新規作成する。

```python
"""過去検証の執行アダプタ — 意図を約定へ変換する。

policy.py は「いつ・なぜ退出するか」までを決め、約定価格には触れない。
本モジュールが、その意図と利用可能な日足から「いくらで約定したか」を
仮定として決め、スリッページと手数料を**一元的に**控除する。

Tの引けで判断し T+1 の寄りで約定する。現行 src/backtest/engine.py:111-188 は
同じ終値でスコア生成と約定を行っており、実運用（引け後スキャン→翌朝発注）と
乖離していた（レビューF04）。

足の型は policy.Observation を再利用する（session/open/high/low/close で
同じ形のため、型を二重に定義しない）。
"""
from dataclasses import dataclass
from datetime import date

from src.core import config as cfg
from src.strategy.policy import Observation


@dataclass(frozen=True)
class Fill:
    """約定。price は**コスト控除前**ではなくスリッページ込みの約定価格。

    手数料はここには含めない（数量に対する金額として net_return で控除する）。
    スリッページを二重に引かないため、控除の責務はこのモジュールに閉じる。
    """
    at: date
    price: float
    quantity: int
    reason: str


@dataclass(frozen=True)
class CostConfig:
    """約定コストの仮定。"""
    slippage_pct: float    # backtest.slippage_pct（片道）
    commission_pct: float  # backtest.commission_pct（片道）


def config_from_settings() -> CostConfig:
    """config.yaml の backtest 節から CostConfig を作る（唯一の読み出し口）。"""
    conf = cfg.get_section("backtest")
    return CostConfig(
        slippage_pct=conf.get("slippage_pct", 0.0),
        commission_pct=conf.get("commission_pct", 0.0),
    )


def buy_fill_price(price: float, costs: CostConfig) -> float:
    """買い約定価格（スリッページ分だけ不利＝高く約定する）。

    既存 src/backtest/engine.py:276-278 と同じ規約（小数2桁で丸める）。
    """
    return round(price * (1 + costs.slippage_pct), 2)


def sell_fill_price(price: float, costs: CostConfig) -> float:
    """売り約定価格（スリッページ分だけ不利＝安く約定する）。

    既存 src/backtest/engine.py:281-283 と同じ規約（小数2桁で丸める）。
    """
    return round(price * (1 - costs.slippage_pct), 2)


def entry_fill(next_bar: Observation, quantity: int, costs: CostConfig) -> Fill:
    """Tの引けで決めた買いを、T+1 の寄りで約定させる。

    判断した日の終値では約定しない。実運用は引け後にスキャンし翌朝に発注する
    ため、シグナルを作れる時点と約定できる時点が違う（レビューF04）。
    """
    return Fill(
        at=next_bar.session,
        price=buy_fill_price(next_bar.open, costs),
        quantity=quantity,
        reason="ENTRY",
    )
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_execution.py -v`
Expected: PASS（6件）

- [ ] **Step 5: BOM確認とコミット**

Run: `head -c 3 src/backtest/execution.py | xxd`（`2222 22` を確認）

```bash
git add src/backtest/execution.py tests/test_execution.py
git commit -m "$(cat <<'EOF'
feat(backtest): 執行アダプタの型とT+1寄りエントリーを追加

現行エンジンは同じ終値でスコア生成と約定を行っており、引け後スキャン→
翌朝発注という実運用と乖離していた（レビューF04）。Tの引けで判断し
T+1の寄りで約定する執行アダプタを別モジュールとして置く。
既存engine.pyは変更しない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: 退出約定とコスト控除後の純収益

**Files:**
- Modify: `src/backtest/execution.py`（`exit_fill` と `net_return` を追加）
- Test: `tests/test_execution.py`

**Interfaces:**
- Consumes: Task 6 の `Fill` / `CostConfig` / `sell_fill_price`、Task 4 の `policy.ExitIntent`
- Produces:
  - `exit_fill(intent: ExitIntent, bar: Observation, next_bar: Optional[Observation], quantity: int, costs: CostConfig) -> Optional[Fill]` — 約定できなければ `None`
  - `net_return(entry: Fill, exit_: Fill, costs: CostConfig) -> float` — コスト控除後の純収益率

**背景（spec §6 の約定仮定）:**

| 条件 | 仮定 |
|---|---|
| 基準線が寄りより上（ギャップダウン） | `min(open, trigger_price)` で約定 |
| 通常の日中到達 | `trigger_price` で約定 |
| `TIME_LIMIT` / `SIGNAL_SELL` | 翌営業日の寄りで成行約定 |
| 翌足が無い（足が尽きた） | 未約定（`None`） |

現行エンジン（`engine.py:119-123`）は損切り線ちょうどで約定できる前提になっており、ギャップダウンに楽観的である。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_execution.py` の末尾に追記する。

```python
def _intent(reason=policy.STOP_LINE, trigger=930.0, order_type="STOP"):
    return policy.ExitIntent(reason=reason, trigger_price=trigger, order_type=order_type)


class TestExitFillStop:
    def test_normal_intraday_touch_fills_at_trigger(self):
        """寄りが基準線より上なら、基準線で約定したとみなす"""
        bar = _bar(session=date(2026, 9, 2), o=1000.0, l=920.0)
        fill = execution.exit_fill(_intent(trigger=930.0), bar, None,
                                   quantity=100, costs=_costs(slip=0.0))
        assert fill is not None
        assert fill.at == date(2026, 9, 2)
        assert fill.price == pytest.approx(930.0)
        assert fill.reason == policy.STOP_LINE

    def test_gap_down_fills_at_open_not_trigger(self):
        """寄りが既に基準線を割っていたら min(open, trigger) で約定する。

        現行エンジンは基準線ちょうどで約定できる前提でギャップダウンに楽観的。
        """
        bar = _bar(session=date(2026, 9, 2), o=900.0, l=880.0)
        fill = execution.exit_fill(_intent(trigger=930.0), bar, None,
                                   quantity=100, costs=_costs(slip=0.0))
        assert fill.price == pytest.approx(900.0)

    def test_slippage_applied_unfavourably_on_exit(self):
        """退出はスリッページ分だけ安く約定する"""
        bar = _bar(session=date(2026, 9, 2), o=1000.0, l=920.0)
        fill = execution.exit_fill(_intent(trigger=930.0), bar, None,
                                   quantity=100, costs=_costs(slip=0.001))
        assert fill.price == pytest.approx(929.07)  # 930 * 0.999


class TestExitFillMarket:
    def test_signal_sell_fills_at_next_open(self):
        """売りシグナルは翌営業日の寄りで成行約定する"""
        bar = _bar(session=date(2026, 9, 2), c=1005.0)
        next_bar = _bar(session=date(2026, 9, 3), o=990.0)
        fill = execution.exit_fill(
            _intent(reason=policy.SIGNAL_SELL, trigger=None, order_type="MARKET"),
            bar, next_bar, quantity=100, costs=_costs(slip=0.0))
        assert fill.at == date(2026, 9, 3)
        assert fill.price == pytest.approx(990.0)

    def test_time_limit_fills_at_next_open(self):
        """満了も翌営業日の寄り。満了日の終値で判断して同じ終値で約定しない"""
        bar = _bar(session=date(2026, 9, 2), c=1005.0)
        next_bar = _bar(session=date(2026, 9, 3), o=990.0)
        fill = execution.exit_fill(
            _intent(reason=policy.TIME_LIMIT, trigger=None, order_type="MARKET"),
            bar, next_bar, quantity=100, costs=_costs(slip=0.0))
        assert fill.at == date(2026, 9, 3)
        assert fill.price != pytest.approx(bar.close)

    def test_unfilled_when_no_next_bar(self):
        """翌足が無ければ未約定（Noneを返す）。未成熟として扱えるように"""
        bar = _bar(session=date(2026, 9, 2))
        fill = execution.exit_fill(
            _intent(reason=policy.TIME_LIMIT, trigger=None, order_type="MARKET"),
            bar, None, quantity=100, costs=_costs())
        assert fill is None


class TestNetReturn:
    def test_positive_return_without_commission(self):
        entry = execution.Fill(at=date(2026, 9, 2), price=1000.0, quantity=100, reason="ENTRY")
        exit_ = execution.Fill(at=date(2026, 9, 5), price=1100.0, quantity=100, reason=policy.TRAILING)
        assert execution.net_return(entry, exit_, _costs(comm=0.0)) == pytest.approx(0.10)

    def test_commission_reduces_return(self):
        """手数料は売買それぞれの約定代金に掛かる"""
        entry = execution.Fill(at=date(2026, 9, 2), price=1000.0, quantity=100, reason="ENTRY")
        exit_ = execution.Fill(at=date(2026, 9, 5), price=1100.0, quantity=100, reason=policy.TRAILING)
        # 買い100,000 / 売り110,000 / 手数料0.1%ずつ = 100 + 110 = 210
        expected = (110000.0 - 100000.0 - 210.0) / 100000.0
        assert execution.net_return(entry, exit_, _costs(comm=0.001)) == pytest.approx(expected)

    def test_slippage_is_not_double_counted(self):
        """スリッページは約定価格に織り込み済み。net_returnで再度引かない"""
        costs = _costs(slip=0.001, comm=0.0)
        entry_price = execution.buy_fill_price(1000.0, costs)   # 1001.0
        exit_price = execution.sell_fill_price(1100.0, costs)   # 1098.9
        entry = execution.Fill(at=date(2026, 9, 2), price=entry_price, quantity=100, reason="ENTRY")
        exit_ = execution.Fill(at=date(2026, 9, 5), price=exit_price, quantity=100, reason=policy.TRAILING)
        expected = (exit_price - entry_price) / entry_price
        assert execution.net_return(entry, exit_, costs) == pytest.approx(expected)

    def test_negative_return_on_loss(self):
        entry = execution.Fill(at=date(2026, 9, 2), price=1000.0, quantity=100, reason="ENTRY")
        exit_ = execution.Fill(at=date(2026, 9, 5), price=930.0, quantity=100, reason=policy.STOP_LINE)
        assert execution.net_return(entry, exit_, _costs(comm=0.0)) == pytest.approx(-0.07)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_execution.py -v`
Expected: FAIL — `AttributeError: module 'src.backtest.execution' has no attribute 'exit_fill'`

- [ ] **Step 3: 実装を追加**

`src/backtest/execution.py` の末尾に追加する。import に `from typing import Optional` と `from src.strategy.policy import ExitIntent, Observation` を足す（`Observation` は既にimport済みなので `ExitIntent` を追加する形）。

```python
def exit_fill(intent: ExitIntent, bar: Observation, next_bar: Optional[Observation],
              quantity: int, costs: CostConfig) -> Optional[Fill]:
    """退出意図を約定へ変換する。約定できなければ None を返す。

    STOP（基準線への到達）:
        通常は基準線で約定したとみなす。ただし**寄りが既に基準線を割っていたら
        min(open, trigger_price) で約定する**。現行 engine.py:119-123 は
        基準線ちょうどで約定できる前提になっており、ギャップダウンに楽観的。

    MARKET（売りシグナル・満了）:
        翌営業日の寄りで成行約定する。満了日の終値で判断して同じ終値で約定する
        経路を作らない（レビューF04）。翌足が無ければ未約定。
    """
    if intent.order_type == "STOP":
        if intent.trigger_price is None:
            raise ValueError("STOP の意図には trigger_price が必要です")
        raw = min(bar.open, intent.trigger_price)
        return Fill(
            at=bar.session,
            price=sell_fill_price(raw, costs),
            quantity=quantity,
            reason=intent.reason,
        )

    if next_bar is None:
        # 足が尽きた＝この意図は約定していない。呼び出し側は未成熟として扱う
        return None
    return Fill(
        at=next_bar.session,
        price=sell_fill_price(next_bar.open, costs),
        quantity=quantity,
        reason=intent.reason,
    )


def net_return(entry: Fill, exit_: Fill, costs: CostConfig) -> float:
    """コスト控除後の純収益率。

    スリッページは entry_fill / exit_fill の約定価格に既に織り込まれているため、
    ここで二重に引かない。手数料だけを売買それぞれの約定代金に対して控除する。
    控除の責務をこのモジュールに閉じることで、期待値の式（spec §8）の末尾で
    コストを再度引く二重計上を防ぐ。
    """
    buy_amount = entry.price * entry.quantity
    if buy_amount <= 0:
        return 0.0
    sell_amount = exit_.price * exit_.quantity
    commission = (buy_amount + sell_amount) * costs.commission_pct
    return (sell_amount - buy_amount - commission) / buy_amount
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_execution.py -v`
Expected: PASS（14件）

- [ ] **Step 5: 全体回帰を確認**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

- [ ] **Step 6: BOM確認とコミット**

Run: `head -c 3 src/backtest/execution.py | xxd`（`2222 22` を確認）

```bash
git add src/backtest/execution.py tests/test_execution.py
git commit -m "$(cat <<'EOF'
feat(backtest): 退出約定とコスト控除後の純収益を追加

ギャップダウン時はmin(open, trigger)で約定させる（現行エンジンは
基準線ちょうどで約定できる前提で楽観的）。売りシグナルと満了は
翌営業日の寄りで成行。翌足が無ければ未約定として返し、呼び出し側が
未成熟として扱えるようにする。
スリッページは約定価格に織り込み済みなのでnet_returnで二重に引かない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## 段階B前半 完了条件の確認

spec §14 の段階B完了条件のうち、本計画（前半）で満たすものを検証する。

- [ ] **確認1: 将来データを追加しても過去の特徴量が変わらない**

Run: `pytest tests/test_indicators.py::TestBuildFeatureFrame::test_appending_future_rows_does_not_change_past_features -v`
Expected: PASS

- [ ] **確認2: 当日の高値を遡ってストップへ使わない**

Run: `pytest tests/test_policy.py::TestStepPeakOrdering -v`
Expected: PASS（3件）

- [ ] **確認3: Tの終値で判断してもT+1以降にのみ約定する**

Run: `pytest tests/test_execution.py::TestEntryFill -v` と `pytest tests/test_execution.py::TestExitFillMarket -v`
Expected: PASS

- [ ] **確認4: 既存の公開関数の挙動が変わっていない**

Run: `pytest tests/test_indicators.py::TestBuildFeaturesUnchanged -v`
Expected: PASS（2件）

- [ ] **確認5: legacy経路に回帰が無い**

Run: `pytest tests/ -q`
Expected: 段階B前半の着手前と同じ結果（新規テスト44件ぶんだけ増える）

**残る完了条件（後半で満たす）:** 「未成熟ラベルを学習しない」「ラベルの終了イベントが実行シミュレーションと一致する」は、イベント表を作る段階B後半で検証する。

---

## 次の段階

段階B後半は `src/strategy/dataset.py`（イベント表）を作り、`src/strategy/labeling.py` を `policy.py` + `execution.py` を呼ぶ薄い層に縮小する。本計画が確定させた `ExitIntent` / `Fill` / `net_return` / `run_session_series` をそのまま使う。`dataset.py` がコスト控除後のラベルを作るために執行アダプタへ依存するため、本計画の完了が前提となる（spec §13）。
