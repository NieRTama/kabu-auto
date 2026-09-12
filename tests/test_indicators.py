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

    def test_legacy_build_features_now_yields_rows_for_flat_series(self, _load_config):
        """完全横ばい銘柄が学習データに現れるようになる（legacyへの波及）

        改修前は RSI が全行 NaN で dropna に全滅し、**0行**だった。
        つまりこの銘柄は学習データから消えていた。改修後は現れる。
        件数（既定設定・120本で46行）は設定窓に依存するので値は固定せず、
        「0行から0行より多くなる」ことと RSI の値だけを固定する。
        """
        df = self._degenerate([100.0] * 120)
        got = indicators.build_features(df)
        assert len(got) > 0
        assert got["rsi"].iloc[-1] == pytest.approx(50.0)
        assert got[indicators.FEATURE_COLS].notna().all().all()

    def test_legacy_build_features_now_yields_rows_for_monotonic_rise(self, _load_config):
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
