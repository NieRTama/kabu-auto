"""engine_version による3経路の分岐（段階F）のテスト

段階A〜Eは新しい経路を作るだけで、どちらを使うかは誰も決めていなかった。
**legacy を選んだときの挙動が現在と1ビットも変わらない**ことを固定する。
"""
import inspect
from unittest.mock import MagicMock, patch

import pytest

from src.core import config as cfg
from src.services import trading


@pytest.fixture(autouse=True)
def _config():
    cfg.load("config.yaml")
    yield
    cfg.get_section("strategy")["engine_version"] = "legacy"


def _service():
    return trading.TradingServices(
        client=MagicMock(), risk=MagicMock(), order_mgr=MagicMock(), model=None)


class TestLegacyRetrainUnchanged:
    def test_default_is_legacy(self):
        assert cfg.get_section("strategy").get("engine_version") == "legacy"

    def test_legacy_calls_train_multi(self):
        svc = _service()
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.ml_model, "train_multi") as train, \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]):
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()
        assert train.called

    def test_legacy_assigns_the_running_model(self):
        """従来どおり self.model へ代入する（挙動不変）"""
        svc = _service()
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.ml_model, "train_multi",
                          return_value="trained-model"), \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]):
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()
        assert svc.model == "trained-model"

    def test_legacy_source_still_present(self):
        src = inspect.getsource(trading.TradingServices.ml_retrain)
        assert "ml_model.train_multi" in src
        assert "self.model =" in src


class TestV2Retrain:
    def test_v2_calls_train_v2_not_train_multi(self):
        cfg.get_section("strategy")["engine_version"] = "v2"
        svc = _service()
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.ml_model, "train_multi") as legacy_train, \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]), \
             patch("src.strategy.v2_training.train_v2") as v2_train:
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()
        assert v2_train.called
        assert not legacy_train.called

    def test_v2_does_not_replace_the_running_model(self):
        """候補を作るだけ。運用モデルは自動で入れ替わらない"""
        cfg.get_section("strategy")["engine_version"] = "v2"
        svc = _service()
        svc.model = "existing-model"
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]), \
             patch("src.strategy.v2_training.train_v2",
                   return_value=MagicMock(model_id="cand-1", skipped_reason=None)):
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()
        assert svc.model == "existing-model"

    def test_v2_failure_does_not_raise(self):
        cfg.get_section("strategy")["engine_version"] = "v2"
        svc = _service()
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]), \
             patch("src.strategy.v2_training.train_v2",
                   side_effect=RuntimeError("boom")):
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()   # 例外が外へ出ない
