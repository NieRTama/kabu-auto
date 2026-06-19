"""
ml_model.train_multi() / labeling._assert_single_symbol_timeseries() のテスト

経緯: ml_retrain() が複数銘柄のOHLCVを pd.concat() で単純結合して学習していたため、
移動平均/RSI/トリプルバリア法のラベルが銘柄境界をまたいで破壊されていた
（外部コードレビュー指摘）。修正: 銘柄ごとに build_training_set() で
特徴量・ラベルを作ってから連結する train_multi() を追加し、build_training_set()
側にも単一銘柄前提（日付インデックスが重複なく昇順）の検証を追加した。
"""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import src.strategy.labeling as labeling_mod
import src.strategy.ml_model as ml_mod


def _make_ohlcv_df(n: int = 50, start_price: float = 1000.0) -> pd.DataFrame:
    """日付昇順・重複なしの単一銘柄OHLCVを模したdfを作る"""
    start = date(2025, 1, 1)
    rows = []
    price = start_price
    for i in range(n):
        price *= 1 + 0.001 * ((i % 7) - 3)
        rows.append({
            "date": start + timedelta(days=i),
            "open": price, "high": price * 1.01, "low": price * 0.99,
            "close": price, "volume": 100000,
        })
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


class TestAssertSingleSymbolTimeseries:
    def test_valid_monotonic_unique_index_passes(self):
        df = _make_ohlcv_df()
        labeling_mod._assert_single_symbol_timeseries(df)  # raiseしなければOK

    def test_duplicate_index_raises(self):
        df = _make_ohlcv_df()
        broken = pd.concat([df, df])  # 同じ日付が2回ずつ出現
        with pytest.raises(ValueError, match="重複"):
            labeling_mod._assert_single_symbol_timeseries(broken)

    def test_non_monotonic_index_raises(self):
        df = _make_ohlcv_df()
        # 2銘柄を単純concatした状況を模す: 後半が先頭より古い日付に戻る
        other = _make_ohlcv_df(start_price=500.0)
        other.index = other.index - pd.Timedelta(days=3650)  # 大昔の日付にずらす
        broken = pd.concat([df, other])  # 日付が前半→後半で逆行（昇順でなくなる）
        with pytest.raises(ValueError, match="昇順"):
            labeling_mod._assert_single_symbol_timeseries(broken)

    def test_build_training_set_rejects_multi_symbol_concat(self):
        """複数銘柄を単純concatしたdfをbuild_training_set()に渡すとValueError"""
        df1 = _make_ohlcv_df()
        df2 = _make_ohlcv_df(start_price=2000.0)
        concatenated = pd.concat([df1, df2])  # 日付重複・逆順が生じる
        with pytest.raises(ValueError):
            labeling_mod.build_training_set(concatenated)


@pytest.fixture(autouse=True)
def mock_save_metrics():
    with patch("src.strategy.ml_model._save_metrics"):
        yield


def _mock_lgb_classifier():
    mock_model = MagicMock()
    mock_model.feature_importances_ = np.ones(10)
    mock_model.best_iteration_ = 50
    mock_model.predict.side_effect = lambda X: np.zeros(len(X), dtype=int)
    return mock_model


class TestTrainMulti:
    def test_concatenates_per_symbol_results_before_fit(self):
        """各dfごとにbuild_training_setを呼び、結果を連結してから1回だけ学習する"""
        sizes = [120, 150, 100]  # 銘柄ごとに異なる学習サンプル数を模す

        def fake_build_training_set(df):
            n = sizes.pop(0)
            X = pd.DataFrame(np.random.randn(n, 10))
            y = pd.Series(np.random.randint(0, 2, n))
            w = np.ones(n)
            return X, y, w

        captured = {}

        def fake_fit(X, y, weights, trigger=None, save=True):
            captured["len_X"] = len(X)
            captured["len_y"] = len(y)
            captured["len_weights"] = len(weights)
            return _mock_lgb_classifier()

        dfs = [_make_ohlcv_df(), _make_ohlcv_df(), _make_ohlcv_df()]
        with patch.object(ml_mod, "build_training_set", side_effect=fake_build_training_set), \
             patch.object(ml_mod, "_fit", side_effect=fake_fit):
            ml_mod.train_multi(dfs, trigger="weekly_schedule")

        assert captured["len_X"] == 120 + 150 + 100
        assert captured["len_y"] == 120 + 150 + 100
        assert captured["len_weights"] == 120 + 150 + 100

    def test_skips_symbol_that_raises_valueerror(self):
        """1銘柄でbuild_training_setがValueErrorを出しても、他の銘柄で学習を続行する"""
        call_count = {"n": 0}

        def fake_build_training_set(df):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("データ不足")
            X = pd.DataFrame(np.random.randn(100, 10))
            y = pd.Series(np.random.randint(0, 2, 100))
            w = np.ones(100)
            return X, y, w

        captured = {}

        def fake_fit(X, y, weights, trigger=None, save=True):
            captured["len_X"] = len(X)
            return _mock_lgb_classifier()

        dfs = [_make_ohlcv_df(), _make_ohlcv_df()]
        with patch.object(ml_mod, "build_training_set", side_effect=fake_build_training_set), \
             patch.object(ml_mod, "_fit", side_effect=fake_fit):
            ml_mod.train_multi(dfs)

        assert captured["len_X"] == 100  # 1銘柄分のみ（スキップされた分は含まれない）

    def test_all_symbols_failing_raises_valueerror(self):
        with patch.object(ml_mod, "build_training_set", side_effect=ValueError("データ不足")):
            with pytest.raises(ValueError):
                ml_mod.train_multi([_make_ohlcv_df()])

    def test_train_single_symbol_still_works(self):
        """train()（単一銘柄、既存シグネチャ）は変更なく動作する（回帰確認）"""
        X = pd.DataFrame(np.random.randn(150, 10))
        y = pd.Series(np.random.randint(0, 2, 150))
        w = np.ones(150)
        captured = {}

        def fake_fit(X_, y_, weights_, trigger=None, save=True):
            captured["len_X"] = len(X_)
            return _mock_lgb_classifier()

        with patch.object(ml_mod, "build_training_set", return_value=(X, y, w)), \
             patch.object(ml_mod, "_fit", side_effect=fake_fit):
            ml_mod.train(_make_ohlcv_df())

        assert captured["len_X"] == 150
