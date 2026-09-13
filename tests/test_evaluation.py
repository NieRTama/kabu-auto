"""評価の実行と記録（src/strategy/evaluation.py）のテスト

同じ入力・同じ分割で5つのモデルを比較し、予測明細を残して後から
指標も売買判断も再計算できるようにする（spec §7・§14）。
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.core import config as cfg
from src.data import database as db
from src.data.database import get_session
from src.strategy import dataset
from src.strategy import evaluation


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


# テスト用のラベル契約ID。実物は dataset.make_label_contract_id() が作る
# （退出ポリシー＋コスト＋版のハッシュ）。ここでは固定文字列で代用する。
_LC = "testcontract"


def _events(n_sessions: int = 60, symbols=("7203", "9984"),
            start=date(2026, 1, 5), holding: int = 2, seed: int = 0,
            label_contract_id: str = _LC) -> pd.DataFrame:
    """テスト用のイベント表。特徴量はラベルと弱く相関させる"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sessions):
        d = start + timedelta(days=i)
        for s in symbols:
            label = int(rng.random() < 0.45)
            rows.append({
                "event_id": f"{s}:{d:%Y%m%d}",
                "label_contract_id": label_contract_id,
                "symbol": s,
                "decision_at": d,
                "entry_at": d + timedelta(days=1),
                "label_end_at": d + timedelta(days=holding),
                "status": dataset.STATUS_RESOLVED,
                "label": label,
                "net_return": 0.02 if label else -0.015,
                "x1": label + rng.normal(0, 1.0),
                "x2": rng.normal(0, 1.0),
            })
    return pd.DataFrame(rows)


FEATURES = ["x1", "x2"]


class TestPredictionTables:
    def test_save_and_read_back_predictions(self, isolated_db):
        preds = pd.DataFrame({
            "event_id": ["7203:20260105", "9984:20260105"],
            "label_contract_id": [_LC, _LC],
            "raw_probability": [0.6, 0.3],
            "calibrated_probability": [0.55, 0.35],
            "fold_index": [0, 0],
        })
        n = evaluation.save_predictions(preds, "run1", "const")
        assert n == 2

        with get_session() as session:
            rows = list(session.scalars(select(db.Prediction)).all())
        assert len(rows) == 2
        assert {r.model_id for r in rows} == {"const"}
        assert {r.evaluation_run_id for r in rows} == {"run1"}
        assert all(r.purpose == evaluation.PURPOSE_VALIDATION for r in rows)
        assert all(r.predicted_at is not None for r in rows)

    def test_shadow_purpose_is_recorded(self, isolated_db):
        preds = pd.DataFrame({
            "event_id": ["7203:20260105"],
            "label_contract_id": [_LC],
            "raw_probability": [0.6],
            "calibrated_probability": [0.55],
            "fold_index": [-1],
        })
        evaluation.save_predictions(
            preds, "run1", "cand", purpose=evaluation.PURPOSE_SHADOW)

        with get_session() as session:
            row = session.scalar(select(db.Prediction))
        assert row.purpose == evaluation.PURPOSE_SHADOW

    def test_outcomes_are_saved_separately_from_predictions(self, isolated_db):
        """予測を先に保存し、実績は別テーブルに後から関連付ける"""
        preds = pd.DataFrame({
            "event_id": ["7203:20260105"],
            "label_contract_id": [_LC],
            "raw_probability": [0.6],
            "calibrated_probability": [0.55],
            "fold_index": [0],
        })
        evaluation.save_predictions(preds, "run1", "const")

        # この時点では実績は無い
        details = evaluation.load_prediction_details("run1")
        assert pd.isna(details["actual_label"].iloc[0])

        events = _events(n_sessions=1, symbols=("7203",))
        events.loc[0, "event_id"] = "7203:20260105"
        events.loc[0, "label"] = 1
        evaluation.save_outcomes(events)

        details = evaluation.load_prediction_details("run1")
        assert details["actual_label"].iloc[0] == 1

    def test_only_resolved_events_become_outcomes(self, isolated_db):
        events = _events(n_sessions=2, symbols=("7203",))
        events.loc[0, "status"] = dataset.STATUS_IMMATURE
        events.loc[0, "label"] = None
        n = evaluation.save_outcomes(events)
        assert n == 1

    def test_details_can_be_filtered_by_model(self, isolated_db):
        preds = pd.DataFrame({
            "event_id": ["7203:20260105"],
            "label_contract_id": [_LC],
            "raw_probability": [0.6],
            "calibrated_probability": [0.55],
            "fold_index": [0],
        })
        evaluation.save_predictions(preds, "run1", "const")
        evaluation.save_predictions(preds, "run1", "lgbm")

        assert len(evaluation.load_prediction_details("run1")) == 2
        assert len(evaluation.load_prediction_details("run1", model_id="lgbm")) == 1

    def test_save_predictions_rejects_missing_label_contract_id(self, isolated_db):
        """結合キーが欠けた予測は保存できない"""
        preds = pd.DataFrame({
            "event_id": ["7203:20260105"],
            "raw_probability": [0.6],
            "calibrated_probability": [0.55],
            "fold_index": [0],
        })
        with pytest.raises(ValueError, match="label_contract_id"):
            evaluation.save_predictions(preds, "run1", "const")


class TestOutcomeIsolationBetweenLabelContracts:
    """別のコストで評価し直しても、過去runの明細と指標が変わらないこと。

    実績を `event_id` だけで upsert していると、2回目の save_outcomes が
    1回目の実績を上書きし、保存済みの1回目の予測明細を読み直したときの
    `actual_label` まで変わってしまう（外部レビューR07）。
    """

    _CHEAP = "cheapcontract"
    _COSTLY = "costlycontrct"

    def _preds(self, contract):
        return pd.DataFrame({
            "event_id": ["7203:20260105"],
            "label_contract_id": [contract],
            "raw_probability": [0.6],
            "calibrated_probability": [0.55],
            "fold_index": [0],
        })

    def _outcome_events(self, contract, label):
        events = _events(n_sessions=1, symbols=("7203",))
        events.loc[0, "event_id"] = "7203:20260105"
        events.loc[0, "label_contract_id"] = contract
        events.loc[0, "label"] = label
        events.loc[0, "net_return"] = 0.05 if label else -0.05
        events.loc[0, "status"] = dataset.STATUS_RESOLVED
        return events

    def test_second_run_with_different_costs_does_not_rewrite_the_first(
            self, isolated_db):
        # 1回目: 手数料ゼロの契約で、このイベントは勝ち
        evaluation.save_predictions(self._preds(self._CHEAP), "run1", "const")
        evaluation.save_outcomes(self._outcome_events(self._CHEAP, 1))
        first = evaluation.load_prediction_details("run1")
        assert first["actual_label"].iloc[0] == 1

        # 2回目: 手数料を乗せた契約で評価し直すと、同じイベントが負けになる
        evaluation.save_predictions(self._preds(self._COSTLY), "run2", "const")
        evaluation.save_outcomes(self._outcome_events(self._COSTLY, 0))

        # 1回目の明細は変わらない
        again = evaluation.load_prediction_details("run1")
        assert again["actual_label"].iloc[0] == 1
        assert again["net_return"].iloc[0] == first["net_return"].iloc[0]
        # 2回目は2回目の実績を見る
        second = evaluation.load_prediction_details("run2")
        assert second["actual_label"].iloc[0] == 0

    def test_both_outcomes_coexist(self, isolated_db):
        evaluation.save_outcomes(self._outcome_events(self._CHEAP, 1))
        evaluation.save_outcomes(self._outcome_events(self._COSTLY, 0))
        with get_session() as session:
            rows = list(session.scalars(select(db.PredictionOutcome)).all())
        assert len(rows) == 2
        assert {r.label_contract_id for r in rows} == {self._CHEAP, self._COSTLY}

    def test_same_contract_still_upserts(self, isolated_db):
        """同じ契約なら従来どおり上書きする（データ改訂の反映）"""
        evaluation.save_outcomes(self._outcome_events(self._CHEAP, 1))
        evaluation.save_outcomes(self._outcome_events(self._CHEAP, 0))
        with get_session() as session:
            rows = list(session.scalars(select(db.PredictionOutcome)).all())
        assert len(rows) == 1
        assert rows[0].actual_label == 0

    def test_prediction_does_not_join_a_foreign_contract_outcome(self, isolated_db):
        """契約が違う実績は結合されない（NaN のままになる）"""
        evaluation.save_predictions(self._preds(self._CHEAP), "run1", "const")
        evaluation.save_outcomes(self._outcome_events(self._COSTLY, 0))
        details = evaluation.load_prediction_details("run1")
        assert pd.isna(details["actual_label"].iloc[0])


def _xy(events: pd.DataFrame):
    X = events[FEATURES].astype("float64")
    y = events["label"].astype(int)
    w = np.ones(len(events))
    return X, y, w


class TestConstantProbability:
    def test_predicts_the_training_positive_rate(self):
        events = _events(n_sessions=40)
        X, y, w = _xy(events)
        m = evaluation.ConstantProbability()
        m.fit(X, y, w)
        p = m.predict_proba(X)
        assert len(p) == len(X)
        assert np.allclose(p, y.mean())

    def test_does_not_look_at_features(self):
        """特徴量を変えても予測は変わらない（床としての性質）"""
        events = _events(n_sessions=40)
        X, y, w = _xy(events)
        m = evaluation.ConstantProbability()
        m.fit(X, y, w)
        shifted = X.copy()
        shifted["x1"] = shifted["x1"] + 100.0
        assert np.allclose(m.predict_proba(X), m.predict_proba(shifted))

    def test_has_a_name(self):
        assert evaluation.ConstantProbability().name == "constant_probability"


class TestMajorityClass:
    def test_predicts_one_when_positives_dominate(self):
        X = pd.DataFrame({"x1": [0.0] * 10, "x2": [0.0] * 10})
        y = pd.Series([1] * 7 + [0] * 3)
        m = evaluation.MajorityClass()
        m.fit(X, y, np.ones(10))
        assert np.allclose(m.predict_proba(X), 1.0)

    def test_predicts_zero_when_negatives_dominate(self):
        X = pd.DataFrame({"x1": [0.0] * 10, "x2": [0.0] * 10})
        y = pd.Series([1] * 3 + [0] * 7)
        m = evaluation.MajorityClass()
        m.fit(X, y, np.ones(10))
        assert np.allclose(m.predict_proba(X), 0.0)

    def test_has_a_name(self):
        assert evaluation.MajorityClass().name == "majority_class"


class TestLogisticRegressionModel:
    def test_learns_a_separable_signal(self):
        """x1がラベルと相関する合成データで、正例に高い確率を付ける"""
        rng = np.random.default_rng(3)
        y = pd.Series([0] * 100 + [1] * 100)
        X = pd.DataFrame({
            "x1": np.concatenate([rng.normal(-2, 0.5, 100), rng.normal(2, 0.5, 100)]),
            "x2": rng.normal(0, 1, 200),
        })
        m = evaluation.LogisticRegressionModel()
        m.fit(X, y, np.ones(len(X)))
        p = m.predict_proba(X)
        assert p[y == 1].mean() > p[y == 0].mean()

    def test_probabilities_are_in_range(self):
        events = _events(n_sessions=60)
        X, y, w = _xy(events)
        m = evaluation.LogisticRegressionModel()
        m.fit(X, y, w)
        p = m.predict_proba(X)
        assert ((p >= 0.0) & (p <= 1.0)).all()

    def test_standardizes_with_training_statistics_only(self):
        """学習時の統計量で変換する。推論側で fit し直さない"""
        events = _events(n_sessions=60)
        X, y, w = _xy(events)
        m = evaluation.LogisticRegressionModel()
        m.fit(X, y, w)
        p_small = m.predict_proba(X.iloc[:5])
        p_full = m.predict_proba(X)
        assert np.allclose(p_small, p_full[:5])

    def test_falls_back_to_constant_on_single_class(self):
        """学習側が片側クラスだけなら定数を返す（例外にしない）"""
        X = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 0.0, 1.0]})
        y = pd.Series([1, 1, 1])
        m = evaluation.LogisticRegressionModel()
        m.fit(X, y, np.ones(3))
        assert np.allclose(m.predict_proba(X), 1.0)

    def test_falls_back_to_constant_on_single_class_all_negative(self):
        """学習側が負例だけなら定数0.0を返す（例外にしない）"""
        X = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 0.0, 1.0]})
        y = pd.Series([0, 0, 0])
        m = evaluation.LogisticRegressionModel()
        m.fit(X, y, np.ones(3))
        assert np.allclose(m.predict_proba(X), 0.0)

    def test_has_a_name(self):
        assert evaluation.LogisticRegressionModel().name == "logistic_regression"


class TestLightGbmModels:
    def _separable(self):
        rng = np.random.default_rng(11)
        y = pd.Series([0] * 150 + [1] * 150)
        X = pd.DataFrame({
            "x1": np.concatenate([rng.normal(-1.5, 1.0, 150), rng.normal(1.5, 1.0, 150)]),
            "x2": rng.normal(0, 1, 300),
        })
        return X, y

    def test_small_lightgbm_learns_a_signal(self):
        X, y = self._separable()
        m = evaluation.SmallLightGBM()
        m.fit(X, y, np.ones(len(X)))
        p = m.predict_proba(X)
        assert p[y == 1].mean() > p[y == 0].mean()

    def test_current_lightgbm_learns_a_signal(self):
        X, y = self._separable()
        m = evaluation.CurrentLightGBM()
        m.fit(X, y, np.ones(len(X)))
        p = m.predict_proba(X)
        assert p[y == 1].mean() > p[y == 0].mean()

    def test_probabilities_are_in_range(self):
        X, y = self._separable()
        for model in (evaluation.SmallLightGBM(), evaluation.CurrentLightGBM()):
            model.fit(X, y, np.ones(len(X)))
            p = model.predict_proba(X)
            assert ((p >= 0.0) & (p <= 1.0)).all()

    def test_current_matches_existing_hyperparameters(self):
        """現行 ml_model._fit() と同じハイパーパラメータであること"""
        m = evaluation.CurrentLightGBM()
        assert m.params["n_estimators"] == 200
        assert m.params["num_leaves"] == 31
        assert m.params["learning_rate"] == pytest.approx(0.05)

    def test_small_is_smaller_than_current(self):
        small = evaluation.SmallLightGBM()
        current = evaluation.CurrentLightGBM()
        assert small.params["num_leaves"] < current.params["num_leaves"]
        assert small.params["n_estimators"] < current.params["n_estimators"]

    def test_falls_back_to_constant_on_single_class(self):
        X = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 0.0, 1.0]})
        y = pd.Series([0, 0, 0])
        m = evaluation.SmallLightGBM()
        m.fit(X, y, np.ones(3))
        assert np.allclose(m.predict_proba(X), 0.0)

    def test_names_are_distinct(self):
        assert evaluation.SmallLightGBM().name == "small_lightgbm"
        assert evaluation.CurrentLightGBM().name == "current_lightgbm"


class TestDefaultModelFactories:
    def test_provides_exactly_five_models(self):
        """比較対象は5つに固定する（深層モデルはこの段階では候補にしない）"""
        factories = evaluation.default_model_factories()
        assert set(factories) == {
            "constant_probability", "majority_class", "logistic_regression",
            "small_lightgbm", "current_lightgbm",
        }

    def test_each_factory_builds_a_fresh_model(self):
        factories = evaluation.default_model_factories()
        for name, make in factories.items():
            a, b = make(), make()
            assert a is not b
            assert a.name == name
