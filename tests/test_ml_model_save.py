"""
ml_model.train(save=False) のテスト

Medium #10: バックテストが本番モデルを上書きしないことを確認する
"""
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def mock_db_session():
    """モデルメトリクスの DB 保存をモック（DB 不要）"""
    from contextlib import contextmanager

    @contextmanager
    def ctx():
        yield MagicMock()

    with patch("src.strategy.ml_model._save_metrics"):
        yield


def _minimal_train_patches(mod):
    """ml_model.train() のヘビーな依存をまとめてモックする"""
    mock_model = MagicMock()
    mock_model.feature_importances_ = np.ones(10)
    mock_model.best_iteration_ = 50
    # predict() が numpy array を返さないと accuracy_score が失敗するため
    mock_model.predict.side_effect = lambda X: np.zeros(len(X), dtype=int)
    # predict_proba() は AUC/Brier 計算で [:, 1] される。2列の確率配列を返す
    def _proba(X):
        p = np.linspace(0.1, 0.9, len(X))
        return np.column_stack([1 - p, p])
    mock_model.predict_proba.side_effect = _proba

    mock_X = pd.DataFrame(np.random.randn(200, 10))
    mock_y = pd.Series(np.random.randint(0, 2, 200))
    mock_weights = np.ones(200)

    return (
        patch.object(mod, "build_training_set",
                     return_value=(mock_X, mock_y, mock_weights)),
        patch("src.strategy.ml_model.lgb.LGBMClassifier",
              return_value=mock_model),
    )


class TestTrainSaveFalse:
    def test_save_false_does_not_write_model_file(self, tmp_path):
        """train(save=False) は MODEL_PATH にファイルを書き込まない"""
        import src.strategy.ml_model as mod

        model_path = tmp_path / "lgb_model.pkl"
        original_path = mod.MODEL_PATH

        p1, p2 = _minimal_train_patches(mod)
        try:
            mod.MODEL_PATH = model_path
            with p1, p2:
                mod.train(pd.DataFrame(), save=False)
        finally:
            mod.MODEL_PATH = original_path

        assert not model_path.exists(), (
            "save=False のとき MODEL_PATH にファイルを書き込んではいけない"
        )

    def test_save_true_calls_pickle_dump(self):
        """train(save=True) は pickle.dump を呼び出す（デフォルト動作）"""
        import src.strategy.ml_model as mod

        p1, p2 = _minimal_train_patches(mod)
        with p1, p2, patch("src.strategy.ml_model.pickle.dump") as mock_dump, \
             patch("builtins.open", MagicMock()), \
             patch("src.strategy.ml_model._write_model_meta"):
            mod.train(pd.DataFrame(), save=True)

        mock_dump.assert_called_once(), "save=True のとき pickle.dump が呼ばれるべき"

    def test_save_true_writes_meta_with_hash(self, tmp_path):
        """train(save=True) はモデルと一致するSHA256のメタを書き出す"""
        import src.strategy.ml_model as mod

        original_path = mod.MODEL_PATH
        p1, p2 = _minimal_train_patches(mod)
        try:
            mod.MODEL_PATH = tmp_path / "lgb_model.pkl"
            # MagicMock はpickle化できないため、ダンプは実バイト書き込みで代用する
            with p1, p2, patch("src.strategy.ml_model.pickle.dump",
                               side_effect=lambda m, f: f.write(b"model")):
                mod.train(pd.DataFrame(), save=True)
            meta = mod._read_model_meta()
            assert meta is not None
            assert meta["sha256"] == mod._sha256_file(mod.MODEL_PATH)
            assert "trained_at" in meta
        finally:
            mod.MODEL_PATH = original_path


class TestLoadVerification:
    def test_load_refuses_on_hash_mismatch(self, tmp_path):
        """メタのSHA256と実ファイルが不一致なら load() は None を返す（fail-closed）"""
        import json
        import src.strategy.ml_model as mod

        original_path = mod.MODEL_PATH
        try:
            mod.MODEL_PATH = tmp_path / "lgb_model.pkl"
            mod.MODEL_PATH.write_bytes(b"tampered-model-bytes")
            mod._meta_path().write_text(
                json.dumps({"sha256": "0" * 64, "trained_at": "2026-01-01T00:00:00"}),
                encoding="utf-8")
            with patch("src.strategy.ml_model.pickle.load") as mock_load:
                result = mod.load()
            assert result is None
            mock_load.assert_not_called()  # 検証失敗時は pickle.load を呼ばない
        finally:
            mod.MODEL_PATH = original_path

    def test_load_ok_on_hash_match(self, tmp_path):
        """ハッシュ一致時は pickle.load してモデルを返す"""
        import json
        import src.strategy.ml_model as mod

        original_path = mod.MODEL_PATH
        try:
            mod.MODEL_PATH = tmp_path / "lgb_model.pkl"
            mod.MODEL_PATH.write_bytes(b"model-bytes")
            digest = mod._sha256_file(mod.MODEL_PATH)
            mod._meta_path().write_text(
                json.dumps({"sha256": digest, "trained_at": "2026-01-01T00:00:00"}),
                encoding="utf-8")
            sentinel = object()
            with patch("src.strategy.ml_model.pickle.load", return_value=sentinel):
                result = mod.load()
            assert result is sentinel
        finally:
            mod.MODEL_PATH = original_path

    def test_load_fails_closed_without_meta_by_default(self, tmp_path):
        """メタ無し・許可フラグ未設定（既定）なら load() はロードを拒否する（fail-closed）。

        メタファイルが消されていれば改ざんを検知できないため、既定では信用しない
        （レビュー再指摘 Critical: TOFU迂回の防止）。
        """
        import src.strategy.ml_model as mod

        original_path = mod.MODEL_PATH
        try:
            mod.MODEL_PATH = tmp_path / "lgb_model.pkl"
            mod.MODEL_PATH.write_bytes(b"legacy-model")
            with patch.object(mod.cfg, "get_section", return_value={}), \
                 patch("src.strategy.ml_model.pickle.load") as mock_load:
                result = mod.load()
            assert result is None
            mock_load.assert_not_called()
            assert mod._read_model_meta() is None  # メタも記録しない
        finally:
            mod.MODEL_PATH = original_path

    def test_load_legacy_records_hash_when_explicitly_allowed(self, tmp_path):
        """allow_unverified_model_load: true を明示した場合のみ、メタ無しモデルを
        ロードして現在のハッシュをメタに記録する（移行用の救済弁）"""
        import src.strategy.ml_model as mod

        original_path = mod.MODEL_PATH
        try:
            mod.MODEL_PATH = tmp_path / "lgb_model.pkl"
            mod.MODEL_PATH.write_bytes(b"legacy-model")
            with patch.object(mod.cfg, "get_section",
                              return_value={"allow_unverified_model_load": True}), \
                 patch("src.strategy.ml_model.pickle.load", return_value=object()):
                result = mod.load()
            assert result is not None
            meta = mod._read_model_meta()
            assert meta["sha256"] == mod._sha256_file(mod.MODEL_PATH)
        finally:
            mod.MODEL_PATH = original_path

    def test_backtest_engine_passes_save_false(self):
        """backtest/engine.py が ml_model.train(save=False) で呼び出すことをソース検証"""
        import src.backtest.engine as eng

        source = inspect.getsource(eng.run_backtest)
        assert "save=False" in source, (
            "backtest/engine.py の run_backtest は ml_model.train(save=False) を渡すべき"
        )
