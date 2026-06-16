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
             patch("builtins.open", MagicMock()):
            mod.train(pd.DataFrame(), save=True)

        mock_dump.assert_called_once(), "save=True のとき pickle.dump が呼ばれるべき"

    def test_backtest_engine_passes_save_false(self):
        """backtest/engine.py が ml_model.train(save=False) で呼び出すことをソース検証"""
        import src.backtest.engine as eng

        source = inspect.getsource(eng.run_backtest)
        assert "save=False" in source, (
            "backtest/engine.py の run_backtest は ml_model.train(save=False) を渡すべき"
        )
