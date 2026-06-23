"""
LSTMアンサンブル・系列構成・シャドーA/B のテスト（torch 不要の部分）。

- build_sequence_set: 系列形状・ラベル整列（純numpy/pandas）
- _ensemble_proba: 重み付き平均・LSTM無しフォールバック（predict_proba をモック）
- lstm_model.load: torch 未導入なら静かに None（純GBMフォールバック）
- _save_signal: シャドー列の永続化（本番 action は不変）
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import src.core.config as cfg
import src.data.database as db
from src.data.database import Signal, get_session
from src.strategy.signal import Signal as TradeSignal


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    try:
        yield tmp_path
    finally:
        db._engine = None
        db._Session = None


def _ohlcv(n=140):
    idx = pd.bdate_range("2025-01-01", periods=n)
    rng = np.random.default_rng(1)
    close = 1000 + np.cumsum(rng.normal(0, 5, n))
    return pd.DataFrame({
        "open": close, "high": close + 5, "low": close - 5,
        "close": close, "volume": rng.integers(1e5, 1e6, n).astype(float),
    }, index=idx)


class TestBuildSequenceSet:
    def test_shapes_and_alignment(self):
        cfg.load("config.yaml")
        cfg.get_section("strategy")["use_news_features"] = False
        from src.strategy.labeling import build_sequence_set
        from src.strategy.indicators import active_feature_cols

        seq_len = 5
        X, y, w = build_sequence_set(_ohlcv(140), seq_len)
        n_features = len(active_feature_cols())
        assert X.ndim == 3
        assert X.shape[1] == seq_len
        assert X.shape[2] == n_features
        assert len(y) == len(X) == len(w)
        assert set(np.unique(y)).issubset({0, 1})

    def test_empty_when_too_short(self):
        cfg.load("config.yaml")
        from src.strategy.labeling import build_sequence_set
        # 指標ウォームアップに満たない短いデータ → 空
        X, y, w = build_sequence_set(_ohlcv(40), seq_len=20)
        assert len(X) == 0
        assert len(y) == 0


class TestEnsembleProba:
    def _df(self):
        # build_features をスキップさせるため、特徴量列が既にある体にする
        from src.strategy.indicators import active_feature_cols
        cols = active_feature_cols()
        data = {c: [0.0, 0.0] for c in cols}
        data["close"] = [100.0, 101.0]
        return pd.DataFrame(data)

    def test_pure_gbm_when_lstm_none(self):
        cfg.load("config.yaml")
        from src.strategy import signal as sig_mod
        with patch.object(sig_mod.ml_model, "predict_proba", return_value=0.8):
            p = sig_mod._ensemble_proba(object(), None, self._df(), None, "X")
        assert p == pytest.approx(0.8)

    def test_weighted_average_when_lstm_present(self):
        cfg.load("config.yaml")
        cfg.get_section("strategy")["ensemble_gbm_weight"] = 0.6
        cfg.get_section("strategy")["ensemble_lstm_weight"] = 0.4
        from src.strategy import signal as sig_mod
        import src.strategy.lstm_model as lstm_mod
        with patch.object(sig_mod.ml_model, "predict_proba", return_value=0.8), \
             patch.object(lstm_mod, "predict_proba", return_value=0.3):
            p = sig_mod._ensemble_proba(object(), object(), self._df(), None, "X")
        # 0.6*0.8 + 0.4*0.3 = 0.6
        assert p == pytest.approx(0.6)

    def test_falls_back_to_gbm_on_lstm_error(self):
        cfg.load("config.yaml")
        from src.strategy import signal as sig_mod
        import src.strategy.lstm_model as lstm_mod
        with patch.object(sig_mod.ml_model, "predict_proba", return_value=0.7), \
             patch.object(lstm_mod, "predict_proba", side_effect=RuntimeError("boom")):
            p = sig_mod._ensemble_proba(object(), object(), self._df(), None, "X")
        assert p == pytest.approx(0.7)


class TestLstmLoadFallback:
    def test_load_returns_none_without_torch(self):
        from src.strategy import lstm_model
        if lstm_model.available():
            pytest.skip("torch 導入済み環境では縮退を検証できない")
        assert lstm_model.load() is None
        assert lstm_model.predict_proba(None, _ohlcv(40)) == 0.5


class TestShadowPersistence:
    def test_save_signal_writes_shadow_columns(self, isolated_db):
        from src.services.trading import _save_signal
        prod = TradeSignal(symbol="7203", action="HOLD",
                           rule_score=0.1, ml_score=0.0, combined_score=0.05)
        shadow = TradeSignal(symbol="7203", action="BUY",
                             rule_score=0.1, ml_score=0.4, combined_score=0.25)
        _save_signal(prod, shadow)
        with get_session() as session:
            row = session.query(Signal).first()
        assert row.action == "HOLD"               # 本番は不変
        assert row.action_shadow == "BUY"         # シャドーは別列
        assert row.combined_score_shadow == pytest.approx(0.25)
        assert row.ml_score_shadow == pytest.approx(0.4)

    def test_save_signal_without_shadow_leaves_null(self, isolated_db):
        from src.services.trading import _save_signal
        prod = TradeSignal(symbol="7203", action="SELL",
                           rule_score=-0.1, ml_score=-0.3, combined_score=-0.2)
        _save_signal(prod)
        with get_session() as session:
            row = session.query(Signal).first()
        assert row.action == "SELL"
        assert row.action_shadow is None
        assert row.combined_score_shadow is None
