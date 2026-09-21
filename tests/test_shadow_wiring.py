"""昇格済み候補モデルのshadow記録の配線（src/services/shadow_recording.py）

**観察のみ。実発注には一切影響しない。** 記録の失敗が本来の判断
（買い/売り/様子見）や paper執行を妨げないことも、このファイルで固定する。
"""
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.backtest import execution
from src.core import clock
from src.core import config as cfg
from src.data import database as db
from src.data.bar_status import BarStatus
from src.data.database import OHLCV, Signal, get_session
from src.services import shadow_recording
from src.services import trading
from src.strategy import dataset as ds
from src.strategy import model_store as ms
from src.strategy import policy
from src.strategy import shadow
from src.strategy.indicators import FEATURE_COLS, build_features
from src.strategy.signal import Signal as TradeSignal


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


class _FakeClassifier:
    """sklearn LGBMClassifier の「形」だけを持つ最小の贋物。

    predict() はクラスラベル、predict_proba() は (n, 2) の確率を返す。
    実機の LGBMClassifier で確認した挙動と同じ
    （predict -> [1] / predict_proba -> [[0.4211, 0.5789]]）。
    """

    def __init__(self, positives):
        self._p = np.asarray(positives, dtype=float)

    def predict(self, X):
        return (self._p[:len(X)] >= 0.5).astype(int)

    def predict_proba(self, X):
        p = self._p[:len(X)]
        return np.column_stack([1.0 - p, p])


class _OneColumnProbaClassifier:
    """単一クラスしか見なかった分類器（predict_proba が (n, 1) を返す）"""

    def predict_proba(self, X):
        return np.ones((len(X), 1), dtype=float)


class _FixedProbaModel:
    """候補モデル役。predict() が正例確率をそのまま返す（Booster と同じ契約）"""

    def __init__(self, probabilities):
        self._p = np.asarray(probabilities, dtype=float)

    def predict(self, X):
        return self._p[:len(X)]


class TestLegacyModelProbaAdapter:
    def test_predict_returns_probabilities_not_class_labels(self):
        """素の predict() はラベルを返す。確率として記録してはいけない"""
        model = _FakeClassifier([0.5789, 0.1])
        X = pd.DataFrame({"f": [1.0, 2.0]})

        assert list(model.predict(X)) == [1, 0]
        got = shadow_recording._LegacyModelProbaAdapter(model).predict(X)
        assert list(got) == pytest.approx([0.5789, 0.1])

    def test_rejects_a_single_class_probability_matrix(self):
        """確率が2列そろわないときは既定値で埋めず例外にする

        黙って0.5等で埋めると「現行が何を予測したか」が捏造される
        （Knowledge.md §10「欠落を既定値で埋めると未結線が正常な結果に化ける」）。
        """
        adapter = shadow_recording._LegacyModelProbaAdapter(
            _OneColumnProbaClassifier())
        with pytest.raises(ValueError):
            adapter.predict(pd.DataFrame({"f": [1.0]}))

    def test_satisfies_the_contract_shadow_compare_requires(self):
        comparisons = shadow.compare(
            ["7203:20260910"], pd.DataFrame({"f": [1.0]}),
            current=shadow_recording._LegacyModelProbaAdapter(
                _FakeClassifier([0.8])),
            candidate=_FixedProbaModel([0.2]),
            threshold=shadow_recording.SHADOW_THRESHOLD)

        assert comparisons[0].current_probability == pytest.approx(0.8)
        assert comparisons[0].current_takes is True
        assert comparisons[0].candidate_takes is False
        assert comparisons[0].agreement == shadow.AGREEMENT_ONLY_CURRENT

    def test_threshold_is_the_observation_default(self):
        assert shadow_recording.SHADOW_THRESHOLD == 0.5
