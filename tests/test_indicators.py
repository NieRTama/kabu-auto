"""テクニカル指標の端点と欠損の扱い（src/strategy/indicators.py）

背景: RSIは下落幅の移動平均が0のときNaNになり、単調上昇の系列で
末尾RSIが欠損して行ごと落とされていた（レビューF06）。端点を仕様化する。
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

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
