"""評価の実行と記録（src/strategy/evaluation.py）のテスト

同じ入力・同じ分割で5つのモデルを比較し、予測明細を残して後から
指標も売買判断も再計算できるようにする（spec §7・§14）。
"""
import json
import subprocess
import unittest.mock as mock
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.core import config as cfg
from src.data import database as db
from src.data.database import get_session
from src.strategy import dataset
from src.strategy import evaluation
from src.strategy import validation


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

    def test_train_positive_rate_and_threshold_are_persisted(self, isolated_db):
        """baseline_rate/threshold の永続化（外部レビューI-3）。保存済み明細
        だけから brier_vs_constant やその run自身の売買判断を復元できるように、
        列として保存・読み出しできること。"""
        preds = pd.DataFrame({
            "event_id": ["7203:20260105"],
            "label_contract_id": [_LC],
            "raw_probability": [0.6],
            "calibrated_probability": [0.55],
            "fold_index": [0],
            "train_positive_rate": [0.42],
            "threshold": [0.51],
        })
        evaluation.save_predictions(preds, "run1", "const")
        details = evaluation.load_prediction_details("run1")
        assert details["train_positive_rate"].iloc[0] == pytest.approx(0.42)
        assert details["threshold"].iloc[0] == pytest.approx(0.51)

    def test_missing_baseline_columns_save_as_none(self, isolated_db):
        """train_positive_rate/threshold列が無い呼び出し元との後方互換。
        既定値で埋めず None のまま保存する。"""
        preds = pd.DataFrame({
            "event_id": ["7203:20260105"],
            "label_contract_id": [_LC],
            "raw_probability": [0.6],
            "calibrated_probability": [0.55],
            "fold_index": [0],
        })
        evaluation.save_predictions(preds, "run1", "const")
        details = evaluation.load_prediction_details("run1")
        assert pd.isna(details["train_positive_rate"].iloc[0])
        assert pd.isna(details["threshold"].iloc[0])

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

    def test_save_predictions_is_idempotent_for_the_same_run_and_model(
            self, isolated_db):
        """同じ evaluation_run_id・model_id で2回呼んでも明細が二重化しない
        （外部レビューI-2。EvaluationRun/PredictionOutcomeはupsertなのに
        Predictionだけ素のinsertだったため再実行のたびに明細が倍化していた）
        """
        preds = pd.DataFrame({
            "event_id": ["7203:20260105", "9984:20260105"],
            "label_contract_id": [_LC, _LC],
            "raw_probability": [0.6, 0.3],
            "calibrated_probability": [0.55, 0.35],
            "fold_index": [0, 0],
        })
        n1 = evaluation.save_predictions(preds, "run1", "const")
        n2 = evaluation.save_predictions(preds, "run1", "const")
        assert n1 == 2
        assert n2 == 2

        with get_session() as session:
            rows = list(session.scalars(select(db.Prediction)).all())
        assert len(rows) == 2

    def test_save_predictions_idempotency_is_scoped_to_run_and_model(
            self, isolated_db):
        """置換はrun×model単位。他のrunや他のmodelの明細は消えない"""
        preds = pd.DataFrame({
            "event_id": ["7203:20260105"],
            "label_contract_id": [_LC],
            "raw_probability": [0.6],
            "calibrated_probability": [0.55],
            "fold_index": [0],
        })
        evaluation.save_predictions(preds, "run1", "const")
        evaluation.save_predictions(preds, "run1", "lgbm")
        evaluation.save_predictions(preds, "run2", "const")

        evaluation.save_predictions(preds, "run1", "const")

        with get_session() as session:
            rows = list(session.scalars(select(db.Prediction)).all())
        assert len(rows) == 3
        assert {(r.evaluation_run_id, r.model_id) for r in rows} == {
            ("run1", "const"), ("run1", "lgbm"), ("run2", "const"),
        }


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

    def test_is_constant_properties_on_single_class(self):
        """LightGBM系と同じ契約: 単一クラス学習後は is_constant/constant_probability
        で縮退を検知できること（レビュー指摘I-5）"""
        X = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 0.0, 1.0]})
        y = pd.Series([1, 1, 1])
        m = evaluation.LogisticRegressionModel()
        m.fit(X, y, np.ones(3))
        assert m.is_constant is True
        assert m.constant_probability == 1.0

    def test_is_constant_properties_on_single_class_all_negative(self):
        X = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 0.0, 1.0]})
        y = pd.Series([0, 0, 0])
        m = evaluation.LogisticRegressionModel()
        m.fit(X, y, np.ones(3))
        assert m.is_constant is True
        assert m.constant_probability == 0.0

    def test_is_constant_properties_on_normal_binary_training(self):
        events = _events(n_sessions=60)
        X, y, w = _xy(events)
        m = evaluation.LogisticRegressionModel()
        m.fit(X, y, w)
        assert m.is_constant is False
        assert m.constant_probability is None


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

    def test_falls_back_to_constant_on_all_positive(self):
        """全部positive（y が全て1）の場合もフォールバックする"""
        X = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 0.0, 1.0]})
        y = pd.Series([1, 1, 1])
        m = evaluation.SmallLightGBM()
        m.fit(X, y, np.ones(3))
        assert np.allclose(m.predict_proba(X), 1.0)

    def test_lightgbm_properties_on_all_negative_single_class(self):
        """単一クラス（全部negative）フォールバック時のプロパティ契約を検証"""
        X = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 0.0, 1.0]})
        y = pd.Series([0, 0, 0])
        m = evaluation.SmallLightGBM()
        m.fit(X, y, np.ones(3))
        # フォールバック時のプロパティ値を検証
        assert m.is_constant is True
        assert m.constant_probability == 0.0
        assert m.booster is None

    def test_lightgbm_properties_on_all_positive_single_class(self):
        """単一クラス（全部positive）フォールバック時のプロパティ契約を検証"""
        X = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 0.0, 1.0]})
        y = pd.Series([1, 1, 1])
        m = evaluation.CurrentLightGBM()
        m.fit(X, y, np.ones(3))
        # フォールバック時のプロパティ値を検証
        assert m.is_constant is True
        assert m.constant_probability == 1.0
        assert m.booster is None

    def test_lightgbm_properties_on_normal_binary_training(self):
        """通常の二値分類時のプロパティ契約を検証"""
        X, y = self._separable()
        m = evaluation.SmallLightGBM()
        m.fit(X, y, np.ones(len(X)))
        # 通常の二値分類時のプロパティ値を検証
        assert m.is_constant is False
        assert m.constant_probability is None
        assert m.booster is not None
        # booster は lgb.Booster のインスタンスであることも確認
        assert isinstance(m.booster, lgb.Booster)

    def test_lightgbm_properties_on_normal_binary_training_current(self):
        """通常の二値分類時のプロパティ契約を検証（現行版）"""
        X, y = self._separable()
        m = evaluation.CurrentLightGBM()
        m.fit(X, y, np.ones(len(X)))
        # 通常の二値分類時のプロパティ値を検証
        assert m.is_constant is False
        assert m.constant_probability is None
        assert m.booster is not None
        # booster は lgb.Booster のインスタンスであることも確認
        assert isinstance(m.booster, lgb.Booster)

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

class TestComputeMetrics:
    def test_perfect_prediction_has_auc_one(self):
        y = pd.Series([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["roc_auc"] == pytest.approx(1.0)

    def test_reversed_prediction_has_auc_zero(self):
        y = pd.Series([0, 0, 1, 1])
        p = np.array([0.9, 0.8, 0.2, 0.1])
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["roc_auc"] == pytest.approx(0.0)

    def test_auc_is_none_when_single_class(self):
        """検証foldが片側クラスのみだとAUCは未定義。例外にせずNoneにする"""
        y = pd.Series([1, 1, 1])
        p = np.array([0.2, 0.5, 0.9])
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["roc_auc"] is None

    def test_brier_matches_mean_squared_error(self):
        y = pd.Series([0, 1])
        p = np.array([0.25, 0.75])
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["brier"] == pytest.approx(0.0625)

    def test_constant_half_gives_brier_quarter(self):
        """常に0.5を予測すれば二乗誤差は0.25。単独では採用根拠にならない"""
        y = pd.Series([0, 1, 0, 1])
        p = np.full(4, 0.5)
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["brier"] == pytest.approx(0.25)
        assert m["brier_vs_constant"] == pytest.approx(0.0)

    def test_better_than_constant_is_positive(self):
        """定数モデルより良ければ差は正になる"""
        y = pd.Series([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["brier_vs_constant"] > 0
        assert m["log_loss_vs_constant"] > 0

    def test_worse_than_constant_is_negative(self):
        y = pd.Series([0, 0, 1, 1])
        p = np.array([0.9, 0.8, 0.2, 0.1])
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["brier_vs_constant"] < 0

    def test_baseline_uses_training_rate_not_validation_rate(self):
        """定数モデルの確率は学習側の正例率。検証側の正例率を使わない"""
        y = pd.Series([1, 1, 1, 0])
        p = np.full(4, 0.3)
        with_train_rate = evaluation.compute_metrics(y, p, baseline_rate=0.3)
        with_val_rate = evaluation.compute_metrics(y, p, baseline_rate=0.75)
        assert with_train_rate["brier_vs_constant"] != pytest.approx(
            with_val_rate["brier_vs_constant"])
        assert with_train_rate["brier_vs_constant"] == pytest.approx(0.0)

    def test_records_counts_and_positive_rate(self):
        y = pd.Series([0, 1, 1, 1])
        p = np.full(4, 0.5)
        m = evaluation.compute_metrics(y, p, baseline_rate=0.5)
        assert m["n"] == 4
        assert m["positive_rate"] == pytest.approx(0.75)

    def test_empty_input_returns_zero_count(self):
        m = evaluation.compute_metrics(pd.Series([], dtype=int), np.array([]),
                                       baseline_rate=0.5)
        assert m["n"] == 0
        assert m["roc_auc"] is None


class TestCalibrator:
    def test_identity_when_too_few_samples(self):
        raw = np.array([0.1, 0.9])
        cal = evaluation.fit_calibrator(raw, pd.Series([0, 1]), min_samples=50)
        assert cal.kind == "identity"
        assert np.allclose(evaluation.apply_calibrator(cal, raw), raw)

    def test_identity_when_single_class(self):
        raw = np.linspace(0.1, 0.9, 100)
        cal = evaluation.fit_calibrator(raw, pd.Series([1] * 100), min_samples=50)
        assert cal.kind == "identity"

    def test_isotonic_pulls_overconfident_probabilities_toward_truth(self):
        """実際の正例率が0.5なのに0.9を出していたら、校正後は下がる"""
        rng = np.random.default_rng(5)
        y = pd.Series(rng.integers(0, 2, 400))
        raw = np.where(y == 1, 0.95, 0.9)  # 常に自信過剰
        cal = evaluation.fit_calibrator(raw, y, min_samples=50)
        assert cal.kind == "isotonic"
        out = evaluation.apply_calibrator(cal, raw)
        assert out.mean() < raw.mean()

    def test_calibrated_values_stay_in_range(self):
        rng = np.random.default_rng(6)
        y = pd.Series(rng.integers(0, 2, 300))
        raw = rng.random(300)
        cal = evaluation.fit_calibrator(raw, y, min_samples=50)
        out = evaluation.apply_calibrator(cal, raw)
        assert ((out >= 0.0) & (out <= 1.0)).all()

    def test_is_monotonic(self):
        """校正は順序を壊さない（AUCを変えない）"""
        rng = np.random.default_rng(7)
        y = pd.Series(rng.integers(0, 2, 300))
        raw = rng.random(300)
        cal = evaluation.fit_calibrator(raw, y, min_samples=50)
        grid = np.linspace(0.0, 1.0, 50)
        out = evaluation.apply_calibrator(cal, grid)
        assert np.all(np.diff(out) >= -1e-12)


class TestSelectThreshold:
    def test_picks_threshold_that_excludes_losers(self):
        """低い確率のイベントが損失なら、それを外す閾値を選ぶ"""
        p = np.array([0.1, 0.2, 0.8, 0.9])
        ret = np.array([-0.05, -0.04, 0.03, 0.06])
        t = evaluation.select_threshold(p, ret)
        selected = p >= t
        assert selected.tolist() == [False, False, True, True]

    def test_takes_everything_when_all_profitable(self):
        p = np.array([0.1, 0.5, 0.9])
        ret = np.array([0.01, 0.02, 0.03])
        t = evaluation.select_threshold(p, ret)
        assert (p >= t).all()

    def test_boundary_is_not_always_half(self):
        """利益と損失が非対称なら境界は0.5にならない（spec §8）"""
        p = np.array([0.3, 0.45, 0.55, 0.7])
        # 0.45 の取引まで採った方が総和が大きい非対称な収益
        ret = np.array([-0.01, 0.05, 0.01, 0.02])
        t = evaluation.select_threshold(p, ret)
        assert t != pytest.approx(0.5)
        assert (p >= t).sum() == 3

    def test_returns_a_value_even_when_nothing_is_profitable(self):
        p = np.array([0.1, 0.5, 0.9])
        ret = np.array([-0.01, -0.02, -0.03])
        t = evaluation.select_threshold(p, ret)
        assert 0.0 <= t <= 1.0

    def test_empty_input_returns_half(self):
        assert evaluation.select_threshold(np.array([]), np.array([])) == pytest.approx(0.5)


class TestFitInner:
    def test_uses_only_inner_folds(self):
        """外側の検証期間を書き換えても、内側で決めた校正と閾値は変わらない"""
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]

        cal_a, thr_a = evaluation.fit_inner(
            events, fold, evaluation.LogisticRegressionModel, feature_cols=FEATURES)

        tampered = events.copy()
        in_val = tampered["decision_at"] >= fold.val_start
        tampered.loc[in_val, "label"] = 1
        tampered.loc[in_val, "net_return"] = 5.0
        tampered.loc[in_val, "x1"] = -99.0
        cal_b, thr_b = evaluation.fit_inner(
            tampered, fold, evaluation.LogisticRegressionModel, feature_cols=FEATURES)

        assert thr_a == pytest.approx(thr_b)
        assert cal_a.kind == cal_b.kind

    def test_returns_usable_threshold(self):
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]
        _, thr = evaluation.fit_inner(
            events, fold, evaluation.ConstantProbability, feature_cols=FEATURES)
        assert 0.0 <= thr <= 1.0

    def test_inner_training_and_threshold_events_are_disjoint_from_outer_validation(
            self, monkeypatch):
        """fit_inner() を実際に呼び出し、内部で validation.inner_folds() に渡された
        引数（＝外側学習側のはずのイベント集合）が外側検証のevent_idと
        完全に排他であることを検証する。

        前回のテストは fit_inner() を一度も呼ばず、テストコード自身が
        training_inputs/inner_folds/split_events を独立に再実行して
        「fit_inner が使うはずの」event_id集合を再構成するだけだった。
        これでは fit_inner() の実装が実際にその手順をなぞっているかを
        検証できず、例えば inner_folds に outer_train ではなく events
        （フルの外側検証込みデータ）を渡すリークバグが混入しても検知
        できなかった。

        ここでは src/strategy/evaluation.py が `from src.strategy import
        validation` でモジュール参照していることを利用し、
        evaluation.validation.inner_folds をスパイに差し替えて、
        fit_inner() の内部から実際に呼ばれた際の引数を捕捉する。
        """
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]

        # fit_inner の外側から見える「外側検証集合」を独立に計算
        _, outer_val = validation.split_events(events, fold)
        outer_val_ids = set(outer_val["event_id"])
        assert len(outer_val_ids) > 0

        captured_train_event_ids: set = set()
        real_inner_folds = evaluation.validation.inner_folds

        def spy_inner_folds(train_events, *args, **kwargs):
            captured_train_event_ids.update(train_events["event_id"])
            return real_inner_folds(train_events, *args, **kwargs)

        monkeypatch.setattr(evaluation.validation, "inner_folds", spy_inner_folds)

        evaluation.fit_inner(events, fold, evaluation.ConstantProbability,
                              feature_cols=FEATURES)

        # スパイが実際に呼ばれたことを保証する（空集合同士の比較は自明にPASSしてしまう）
        assert len(captured_train_event_ids) > 0
        assert captured_train_event_ids.isdisjoint(outer_val_ids)

    def test_falls_back_to_identity_when_inner_split_is_impossible(self):
        """内側分割に足りるセッション数が無いとき、ValueErrorを外へ漏らさず
        既存の退化パス（identity, 0.5）へ合流すること（レビュー指摘I-1）。

        n_sessions=30, n_splits=5 は最終ブランチレビューで実際にクラッシュを
        再現した規模（外側fold 0 の学習側が inner_folds に足りない）。
        """
        events = _events(n_sessions=30)
        fold = validation.calendar_folds(events, n_splits=5)[0]

        # 修正前は validation.inner_folds() が ValueError を送出していたことの確認
        outer = validation.training_inputs(events, fold, feature_cols=FEATURES)
        with pytest.raises(ValueError):
            list(validation.inner_folds(outer.events, n_splits=3))

        cal, thr = evaluation.fit_inner(
            events, fold, evaluation.ConstantProbability, feature_cols=FEATURES)
        assert cal.kind == "identity"
        assert cal.model is None
        assert thr == 0.5

    def test_does_not_swallow_a_genuine_fold_invariant_violation(self, monkeypatch):
        """`inner_folds()` 呼び出しは、セッション数不足という無害なケースだけを
        事前チェックで弾く。`Fold.__post_init__`（R23の不変条件強制）も同じ
        `ValueError` を送出するため、もし `except ValueError` で広く拾う実装に
        戻すと、walk-forwardの不変条件が壊れているという重大なバグまで
        「校正なしの既定値」に丸めて隠してしまう（最終ブランチレビューN-3）。

        ここでは `validation.inner_folds` 自体が（セッション数不足以外の理由で）
        `ValueError` を送出するケースを模擬し、`fit_inner` がそれを飲み込まず
        外へ伝播させることを確認する。
        """
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]

        def _broken_inner_folds(train_events, n_splits=3):
            raise ValueError("Fold不変条件違反のシミュレーション（セッション数不足ではない）")

        monkeypatch.setattr(evaluation.validation, "inner_folds", _broken_inner_folds)

        with pytest.raises(ValueError, match="不変条件違反"):
            evaluation.fit_inner(
                events, fold, evaluation.ConstantProbability, feature_cols=FEATURES)


class TestEvaluateFold:
    def test_returns_predictions_for_every_validation_event(self):
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]
        _, val = validation.split_events(events, fold)

        res = evaluation.evaluate_fold(
            events, fold, "logistic_regression",
            evaluation.LogisticRegressionModel, feature_cols=FEATURES)
        assert res is not None
        assert len(res.predictions) == len(val)
        assert set(res.predictions["event_id"]) == set(val["event_id"])

    def test_prediction_columns_are_complete(self):
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]
        res = evaluation.evaluate_fold(
            events, fold, "const", evaluation.ConstantProbability,
            feature_cols=FEATURES)
        for col in ("event_id", "raw_probability", "calibrated_probability", "fold_index",
                    "train_positive_rate", "threshold"):
            assert col in res.predictions.columns
        assert (res.predictions["fold_index"] == fold.index).all()

    def test_prediction_columns_carry_baseline_rate_and_threshold(self):
        """外部レビューI-3: predictionsのtrain_positive_rate/threshold列は
        FoldResult自身の同名フィールドと一致し、明細から復元できること"""
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]
        res = evaluation.evaluate_fold(
            events, fold, "const", evaluation.ConstantProbability,
            feature_cols=FEATURES)
        assert (res.predictions["train_positive_rate"] == res.train_positive_rate).all()
        assert (res.predictions["threshold"] == res.threshold).all()

    def test_baseline_rate_comes_from_training_side(self):
        """train_positive_rate は学習側（検証側ではない）の正例率であり、
        ConstantProbability.fit() と同じ**一意性重み付き**平均のはず
        （単純平均ではない。レビュー指摘: train_positive_rateが重み無視の
        単純平均になっていた）。
        """
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]
        inputs = validation.training_inputs(events, fold, feature_cols=FEATURES)
        res = evaluation.evaluate_fold(
            events, fold, "const", evaluation.ConstantProbability,
            feature_cols=FEATURES)
        y = inputs.events["label"].astype(int).astype(float)
        expected = float(np.average(y, weights=inputs.weights))
        assert res.train_positive_rate == pytest.approx(expected)
        # 単純平均とは異なることも確認する（一意性重みが非自明なため）
        assert res.train_positive_rate != pytest.approx(float(y.mean()))

    def test_outer_validation_values_do_not_change_training(self):
        """外側の検証側を書き換えても学習件数と学習側正例率は変わらない"""
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]
        a = evaluation.evaluate_fold(
            events, fold, "const", evaluation.ConstantProbability, feature_cols=FEATURES)

        tampered = events.copy()
        in_val = tampered["decision_at"] >= fold.val_start
        tampered.loc[in_val, "label"] = 1
        b = evaluation.evaluate_fold(
            tampered, fold, "const", evaluation.ConstantProbability, feature_cols=FEATURES)

        assert a.n_train == b.n_train
        assert a.train_positive_rate == pytest.approx(b.train_positive_rate)

    def test_returns_none_when_fold_has_no_data(self):
        events = _events(n_sessions=120)
        empty_fold = validation.Fold(
            index=9, train_start=date(2030, 1, 1), train_end=date(2030, 1, 2),
            val_start=date(2030, 1, 3), val_end=date(2030, 1, 4))
        assert evaluation.evaluate_fold(
            events, empty_fold, "const", evaluation.ConstantProbability,
            feature_cols=FEATURES) is None

    def test_train_positive_rate_is_weighted_like_constant_probability(self):
        """ConstantProbability を自己評価すると vs定数指標は厳密に0になるはず。

        `baseline_rate`（学習側正例率＝train_positive_rate）と、
        ConstantProbability.fit() がモデル自身の確率として学習する値は、
        同じ学習データ・同じ重みに対して同じ計算式で作られているべきである。
        単純平均のままだと ConstantProbability.fit() の一意性重み付き平均と
        ずれ、vs定数指標が0からずれる（レビュー実測 -0.00126）。

        内側foldの等調回帰キャリブレータは（複数の内側foldそれぞれで
        学習側正例率が異なりうるため）ConstantProbabilityであっても
        outer側の生確率を動かしうる。それ自体は別の効果なので、ここでは
        fit_inner() をidentityキャリブレータに固定して、train_positive_rate
        の重み計算だけを切り出して検証する。
        """
        events = _events(n_sessions=120)
        fold = validation.calendar_folds(events, n_splits=5)[3]

        # 一意性重みが非自明（全て1.0ではない）ことを事前確認する
        inputs = validation.training_inputs(events, fold, feature_cols=FEATURES)
        assert len(set(np.round(inputs.weights, 6).tolist())) > 1

        with mock.patch("src.strategy.evaluation.fit_inner",
                        return_value=(evaluation.Calibrator(kind="identity", model=None), 0.5)):
            res = evaluation.evaluate_fold(
                events, fold, "const", evaluation.ConstantProbability,
                feature_cols=FEATURES)

        assert res is not None
        assert abs(res.metrics["brier_vs_constant"]) < 1e-9
        assert abs(res.metrics["log_loss_vs_constant"]) < 1e-9

        # train_positive_rate 自体も ConstantProbability.fit() と同じ値のはず
        model = evaluation.ConstantProbability()
        model.fit(inputs.events[FEATURES].astype("float64"),
                  inputs.events["label"].astype(int), inputs.weights)
        assert res.train_positive_rate == pytest.approx(model._p)


class TestCaptureRunConfig:
    """RunConfig / capture_run_config() のテスト（レビュー指摘2）。

    段階Eの check_promotable() が昇格可否の根拠にする値なので、内容の
    捕捉・決定性・変化検知を検証する。
    """

    def test_captures_config_sections_and_code_version(self):
        events = _events(n_sessions=10)
        run_config = evaluation.capture_run_config(
            events, n_splits=3, window_sessions=None, feature_cols=FEATURES)

        payload = json.loads(run_config.config_json)
        assert payload["strategy"] == cfg.get_section("strategy")
        assert payload["trading"] == cfg.get_section("trading")
        assert payload["backtest"] == cfg.get_section("backtest")
        # このworktreeはgit管理下にあるので短縮SHAが取れるはず
        assert run_config.code_version is not None
        assert len(run_config.code_version) > 0

    def test_same_events_and_params_produce_a_deterministic_hash(self):
        events = _events(n_sessions=10)
        a = evaluation.capture_run_config(
            events, n_splits=3, window_sessions=None, feature_cols=FEATURES)
        b = evaluation.capture_run_config(
            events.copy(), n_splits=3, window_sessions=None, feature_cols=FEATURES)
        assert a.config_hash == b.config_hash

    def test_changing_events_changes_the_hash(self):
        """events の行数（n_events）が変われば config_hash も変わる"""
        fewer = _events(n_sessions=10)
        more = _events(n_sessions=20)
        a = evaluation.capture_run_config(
            fewer, n_splits=3, window_sessions=None, feature_cols=FEATURES)
        b = evaluation.capture_run_config(
            more, n_splits=3, window_sessions=None, feature_cols=FEATURES)
        assert a.config_hash != b.config_hash

    def test_code_version_returns_none_when_git_is_unavailable(self, monkeypatch):
        def boom(*args, **kwargs):
            raise FileNotFoundError("git not found")
        monkeypatch.setattr(subprocess, "run", boom)
        assert evaluation._code_version() is None


class TestSaveAndLoadEvaluationRun:
    """save_evaluation_run() / load_evaluation_run() のテスト（レビュー指摘2）。"""

    def test_round_trip_preserves_key_fields(self, isolated_db):
        events = _events(n_sessions=10)
        run_config = evaluation.capture_run_config(
            events, n_splits=3, window_sessions=None, feature_cols=FEATURES)

        evaluation.save_evaluation_run(
            "run-roundtrip-1", run_config,
            purpose=evaluation.PURPOSE_VALIDATION, model_id="constant_probability",
            n_folds=3, n_predictions=42, degraded_reasons=[])

        loaded = evaluation.load_evaluation_run("run-roundtrip-1")
        assert loaded is not None
        assert loaded.purpose == evaluation.PURPOSE_VALIDATION
        assert loaded.model_id == "constant_probability"
        assert loaded.dataset_id == run_config.dataset_id
        assert loaded.config_hash == run_config.config_hash
        assert loaded.n_folds == 3
        assert loaded.n_predictions == 42
        assert loaded.degraded == 0

    def test_degraded_reasons_set_the_degraded_flag(self, isolated_db):
        events = _events(n_sessions=10)
        run_config = evaluation.capture_run_config(
            events, n_splits=3, window_sessions=None, feature_cols=FEATURES)
        evaluation.save_evaluation_run(
            "run-degraded-1", run_config,
            purpose=evaluation.PURPOSE_VALIDATION, model_id="const",
            n_folds=3, n_predictions=1, degraded_reasons=["insufficient_data"])
        loaded = evaluation.load_evaluation_run("run-degraded-1")
        assert loaded.degraded == 1

    def test_same_id_overwrites_instead_of_inserting_a_new_row(self, isolated_db):
        """同じ evaluation_run_id で2回保存すると、新規行が増えず上書きされる"""
        events = _events(n_sessions=10)
        run_config = evaluation.capture_run_config(
            events, n_splits=3, window_sessions=None, feature_cols=FEATURES)

        evaluation.save_evaluation_run(
            "run-dup-1", run_config,
            purpose=evaluation.PURPOSE_VALIDATION, model_id="a",
            n_folds=3, n_predictions=10, degraded_reasons=[])
        evaluation.save_evaluation_run(
            "run-dup-1", run_config,
            purpose=evaluation.PURPOSE_SHADOW, model_id="b",
            n_folds=5, n_predictions=99, degraded_reasons=["x"])

        with get_session() as session:
            rows = list(session.scalars(
                select(db.EvaluationRun).where(
                    db.EvaluationRun.evaluation_run_id == "run-dup-1")).all())
        assert len(rows) == 1

        loaded = evaluation.load_evaluation_run("run-dup-1")
        assert loaded.purpose == evaluation.PURPOSE_SHADOW
        assert loaded.model_id == "b"
        assert loaded.n_folds == 5
        assert loaded.n_predictions == 99
        assert loaded.degraded == 1

    def test_load_returns_none_when_not_found(self, isolated_db):
        assert evaluation.load_evaluation_run("does-not-exist") is None


class TestEvaluationRunIdGeneration:
    """evaluation_run_id の既定生成のテスト（レビュー指摘3）。

    段階B2の Dataset.collection_id と同じ「秒精度タイムスタンプだけ」の
    バグを踏まないことを、タイトループで（sleepでごまかさずに）確認する。
    """

    def test_tight_loop_generates_all_unique_ids(self):
        ids = [evaluation._new_evaluation_run_id() for _ in range(500)]
        assert len(set(ids)) == len(ids)


class TestRunEvaluation:
    def test_evaluates_every_model_on_every_fold(self, isolated_db):
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events, model_factories={
                "constant_probability": evaluation.ConstantProbability,
                "logistic_regression": evaluation.LogisticRegressionModel,
            },
            n_splits=3, feature_cols=FEATURES, persist=False)
        models = {r.model_id for r in out["fold_results"]}
        assert models == {"constant_probability", "logistic_regression"}

    def test_all_models_see_the_same_folds(self, isolated_db):
        """同じ分割で比較する（モデルごとに分割を変えない）"""
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events, model_factories={
                "constant_probability": evaluation.ConstantProbability,
                "logistic_regression": evaluation.LogisticRegressionModel,
            },
            n_splits=3, feature_cols=FEATURES, persist=False)
        by_model = {}
        for r in out["fold_results"]:
            by_model.setdefault(r.model_id, []).append((r.fold_index, r.n_train, r.n_val))
        values = list(by_model.values())
        assert all(v == values[0] for v in values)

    def test_persists_predictions_and_outcomes(self, isolated_db):
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events, model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=3, feature_cols=FEATURES, persist=True)

        details = evaluation.load_prediction_details(out["evaluation_run_id"])
        assert len(details) > 0
        assert details["actual_label"].notna().all()

    def test_evaluation_run_id_is_recorded_on_every_row(self, isolated_db):
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events, model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=3, feature_cols=FEATURES, persist=True)
        details = evaluation.load_prediction_details(out["evaluation_run_id"])
        assert (details["evaluation_run_id"] == out["evaluation_run_id"]).all()

    def test_summary_frame_has_one_row_per_model_and_fold(self, isolated_db):
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events, model_factories={
                "constant_probability": evaluation.ConstantProbability,
                "small_lightgbm": evaluation.SmallLightGBM,
            },
            n_splits=3, feature_cols=FEATURES, persist=False)
        summary = out["summary"]
        assert len(summary) == len(out["fold_results"])
        for col in ("fold_index", "model_id", "n_train", "n_val",
                    "train_positive_rate", "threshold", "roc_auc", "brier",
                    "brier_vs_constant"):
            assert col in summary.columns

    def test_defaults_to_the_five_comparison_models(self, isolated_db):
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events, n_splits=3, feature_cols=FEATURES, persist=False)
        assert {r.model_id for r in out["fold_results"]} == set(
            evaluation.default_model_factories())

    def test_rerunning_the_same_run_id_does_not_duplicate_predictions(
            self, isolated_db):
        """同じ evaluation_run_id で run_evaluation() を2回呼んでも、複数fold
        が同じモデルへ保存される経路全体で明細が二重化しないこと
        （外部レビューI-2。fold単位でsave_predictions()を都度呼ぶと、
        その冪等化が直前foldぶんまで消してしまう退行を防ぐ）。"""
        events = _events(n_sessions=150)
        run_id = "run-idempotent-full"
        out1 = evaluation.run_evaluation(
            events, model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=3, feature_cols=FEATURES, evaluation_run_id=run_id, persist=True)
        first_count = len(evaluation.load_prediction_details(run_id))
        assert first_count == sum(len(r.predictions) for r in out1["fold_results"])

        evaluation.run_evaluation(
            events, model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=3, feature_cols=FEATURES, evaluation_run_id=run_id, persist=True)
        second_count = len(evaluation.load_prediction_details(run_id))
        assert second_count == first_count

    def test_does_not_crash_on_small_data_that_cannot_split_inner_folds(
            self, isolated_db):
        """n_sessions=30, n_splits=5 は内側分割を作れない規模（最終ブランチ
        レビューで実際にクラッシュを再現した規模）。例外を送出せず結果が
        返ること（レビュー指摘I-1、回帰固定M-6）。"""
        events = _events(n_sessions=30)
        out = evaluation.run_evaluation(
            events, model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=5, feature_cols=FEATURES, persist=True)
        assert len(out["fold_results"]) > 0

    def test_small_data_run_is_marked_degraded_with_reasons(self, isolated_db):
        events = _events(n_sessions=30)
        out = evaluation.run_evaluation(
            events, model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=5, feature_cols=FEATURES, persist=True)
        loaded = evaluation.load_evaluation_run(out["evaluation_run_id"])
        assert loaded.degraded == 1
        reasons = json.loads(loaded.degraded_reasons)
        assert len(reasons) > 0


class TestDegradedReasons:
    """run_evaluation() が実際に degraded_reasons へ理由を積むことの検証
    （レビュー指摘I-4）。宣言だけで結線されていない「死んだ経路」を
    それぞれの発生源ごとに固定する。
    """

    def test_records_reason_when_a_fold_is_skipped(self, isolated_db, monkeypatch):
        """(a) evaluate_fold が None を返した fold は理由付きで記録される"""
        events = _events(n_sessions=150)
        real_evaluate_fold = evaluation.evaluate_fold
        calls = {"n": 0}

        def flaky_evaluate_fold(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return None
            return real_evaluate_fold(*args, **kwargs)

        monkeypatch.setattr(evaluation, "evaluate_fold", flaky_evaluate_fold)
        out = evaluation.run_evaluation(
            events, model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=3, feature_cols=FEATURES, persist=False)
        assert any("skip" in r for r in out["degraded_reasons"])

    def test_records_reason_when_calibrator_falls_back_to_identity(
            self, isolated_db, monkeypatch):
        """(b) fit_inner が identity キャリブレータへ落ちたfoldは理由付きで記録される"""
        events = _events(n_sessions=150)
        monkeypatch.setattr(
            evaluation, "fit_inner",
            lambda *a, **kw: (evaluation.Calibrator(kind="identity", model=None), 0.5))
        out = evaluation.run_evaluation(
            events, model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=3, feature_cols=FEATURES, persist=False)
        assert any("identity" in r or "校正" in r for r in out["degraded_reasons"])

    def test_records_reason_when_model_degenerates_to_constant(self, isolated_db):
        """(c) モデルが定数に縮退したfoldは理由付きで記録される"""
        events = _events(n_sessions=60)
        events["label"] = 1  # 全件同一クラスにして LogisticRegressionModel を縮退させる
        events["net_return"] = 0.02
        out = evaluation.run_evaluation(
            events, model_factories={"logistic_regression": evaluation.LogisticRegressionModel},
            n_splits=3, feature_cols=FEATURES, persist=False)
        assert any("定数" in r for r in out["degraded_reasons"])

    def test_records_reason_when_roc_auc_is_undefined(self, isolated_db, monkeypatch):
        """(d) 検証側が片側クラスで roc_auc=None になった fold は理由付きで記録される"""
        events = _events(n_sessions=150)
        real_compute_metrics = evaluation.compute_metrics

        def one_sided_metrics(y_true, p, **kwargs):
            import numpy as _np
            forced = _np.zeros(len(y_true))
            return real_compute_metrics(forced, p, **kwargs)

        monkeypatch.setattr(evaluation, "compute_metrics", one_sided_metrics)
        out = evaluation.run_evaluation(
            events, model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=3, feature_cols=FEATURES, persist=False)
        assert any("roc_auc" in r for r in out["degraded_reasons"])


class TestRecomputeFromDetails:
    def test_metrics_match_the_original_run(self, isolated_db):
        """保存した明細**だけ**から、実行時と同じ指標が出る（spec §14 完了条件）。

        baseline_rate は明示的に渡さず、保存済みの train_positive_rate 列
        （外部レビューI-3で永続化）から導出する。以前はメモリ上の
        FoldResult から `res.train_positive_rate` を借りていたため、
        「保存データだけからは指標を再現できない」欠陥を隠していた。
        """
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events,
            model_factories={"logistic_regression": evaluation.LogisticRegressionModel},
            n_splits=3, feature_cols=FEATURES, persist=True)

        details = evaluation.load_prediction_details(
            out["evaluation_run_id"], model_id="logistic_regression")

        for res in out["fold_results"]:
            sub = details[details["fold_index"] == res.fold_index]
            again = evaluation.recompute_metrics(sub)
            assert again["n"] == res.metrics["n"]
            assert again["brier"] == pytest.approx(res.metrics["brier"])
            assert again["brier_vs_constant"] == pytest.approx(
                res.metrics["brier_vs_constant"])
            assert again["log_loss_vs_constant"] == pytest.approx(
                res.metrics["log_loss_vs_constant"])
            if res.metrics["roc_auc"] is not None:
                assert again["roc_auc"] == pytest.approx(res.metrics["roc_auc"])

    def test_baseline_rate_can_still_be_overridden_explicitly(self, isolated_db):
        """任意のbaselineで引き直したい場合のため、明示指定は従来どおり効く"""
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events,
            model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=3, feature_cols=FEATURES, persist=True)
        details = evaluation.load_prediction_details(out["evaluation_run_id"])
        sub = details[details["fold_index"] == out["fold_results"][0].fold_index]
        explicit = evaluation.recompute_metrics(sub, baseline_rate=0.5)
        auto = evaluation.recompute_metrics(sub)
        # 明示指定した baseline=0.5 が実際に効いていること（自動導出値と違う前提）
        assert sub["train_positive_rate"].iloc[0] != pytest.approx(0.5)
        assert explicit["brier_vs_constant"] != pytest.approx(auto["brier_vs_constant"])

    def test_raises_when_baseline_rate_is_ambiguous_without_explicit_value(
            self, isolated_db):
        """複数foldにまたがる明細でbaseline_rateを省略すると、train_positive_rate
        が単一値ではないため ValueError（外部レビューI-3）"""
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events,
            model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=3, feature_cols=FEATURES, persist=True)
        details = evaluation.load_prediction_details(out["evaluation_run_id"])
        assert details["train_positive_rate"].nunique() > 1
        with pytest.raises(ValueError, match="train_positive_rate"):
            evaluation.recompute_metrics(details)

    def test_trading_decisions_can_be_recomputed(self, isolated_db):
        """明細から採用群も引き直せる（閾値を変えた検討ができる）"""
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events,
            model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=3, feature_cols=FEATURES, persist=True)
        details = evaluation.load_prediction_details(out["evaluation_run_id"])
        taken = details[details["calibrated_probability"] >= 0.0]
        assert taken["net_return"].notna().all()

    def test_the_runs_own_trading_decision_can_be_recovered(self, isolated_db):
        """外部レビューI-3: 保存済みの threshold 列を使えば、その run自身が
        採用した閾値（任意の閾値ではなく）で売買判断を再現できる"""
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events,
            model_factories={"constant_probability": evaluation.ConstantProbability},
            n_splits=3, feature_cols=FEATURES, persist=True)
        details = evaluation.load_prediction_details(out["evaluation_run_id"])
        for res in out["fold_results"]:
            sub = details[details["fold_index"] == res.fold_index]
            assert sub["threshold"].iloc[0] == pytest.approx(res.threshold)
            taken = sub[sub["calibrated_probability"] >= sub["threshold"]]
            expected_taken = res.predictions[
                res.predictions["calibrated_probability"] >= res.threshold]
            assert len(taken) == len(expected_taken)

    def test_skips_rows_without_outcomes(self, isolated_db):
        preds = pd.DataFrame({
            "event_id": ["a", "b"],
            "label_contract_id": [_LC, _LC],
            "raw_probability": [0.6, 0.4],
            "calibrated_probability": [0.6, 0.4],
            "fold_index": [0, 0],
        })
        evaluation.save_predictions(preds, "run1", "m")
        details = evaluation.load_prediction_details("run1")
        got = evaluation.recompute_metrics(details, baseline_rate=0.5)
        assert got["n"] == 0


class TestSelectTrainingWindow:
    def test_returns_one_of_the_candidates(self):
        events = _events(n_sessions=150)
        fold = validation.calendar_folds(events, n_splits=3)[1]
        got = evaluation.select_training_window(
            events, fold, evaluation.LogisticRegressionModel,
            candidates=[None, 20, 40], feature_cols=FEATURES)
        assert got in (None, 20, 40)

    def test_choice_does_not_depend_on_outer_validation(self):
        """外側検証の値を書き換えても選ばれる窓は変わらない（選択は内側で）"""
        events = _events(n_sessions=150)
        fold = validation.calendar_folds(events, n_splits=3)[1]
        a = evaluation.select_training_window(
            events, fold, evaluation.LogisticRegressionModel,
            candidates=[None, 20, 40], feature_cols=FEATURES)

        tampered = events.copy()
        in_val = tampered["decision_at"] >= fold.val_start
        tampered.loc[in_val, "label"] = 1
        tampered.loc[in_val, "net_return"] = 9.9
        b = evaluation.select_training_window(
            tampered, fold, evaluation.LogisticRegressionModel,
            candidates=[None, 20, 40], feature_cols=FEATURES)
        assert a == b

    def test_empty_candidates_returns_none(self):
        events = _events(n_sessions=150)
        fold = validation.calendar_folds(events, n_splits=3)[1]
        assert evaluation.select_training_window(
            events, fold, evaluation.ConstantProbability,
            candidates=[], feature_cols=FEATURES) is None

    def test_all_candidates_share_the_same_inner_validation_set(self, monkeypatch):
        """候補窓を変えても内側の検証集合は同一である

        窓で外側集合を先に切ってから内側foldを作り直すと、短い窓と拡大窓で
        評価日も件数も変わり、「学習窓の効果」と「評価期間の差」が混ざる
        （外部レビューR19）。

        シグネチャ検査（candidates/window_sessions引数が無いこと）と戻り値の
        非空チェックだけでは、この退行を実際に注入してもPASSしたまま検知
        できないことが外部レビューの実験で確認された。ここでは
        select_training_window() に実際に複数の候補窓を渡し、
        validation.split_events() をスパイして**各候補の評価で実際に使われた
        検証event_id集合**を捕捉し、候補をまたいで完全一致することを直接
        アサートする。

        select_training_window は最初に `validation.training_inputs(events,
        fold, ...)` を1回呼んで外側学習側 `base.events` を作る（この呼び出しは
        `events` そのものを渡すので `ev is events` で区別できる）。内側の
        呼び出しはすべて `base.events`（別オブジェクト）を渡すので、
        それだけを候補間比較の対象にする。

        内側foldは3つあり（fold自体はそれぞれ検証期間が異なる）、1候補の中では
        当然それぞれ違う検証集合を持つ。R19の不変条件は「**同じ内側fold**は
        **どの候補でも**同じ検証集合を使う」ことなので、(1)全ての内側呼び出しが
        同一の events オブジェクト（`base.events`）から作られていること、
        (2) 同じ検証期間（val_start, val_end）の呼び出しは常に同じ
        event_id集合を返すこと、の両方を確かめる。学習窓を先に適用してから
        内側foldを作り直す退行が混入すると、候補ごとに**別のevents
        オブジェクト**（windowedフレーム）から内側foldが再構築されるため、
        (1)が候補の数だけ複数のオブジェクトIDに割れてFAILする
        （実際に注入して確認済み。報告書参照）。
        """
        events = _events(n_sessions=150)
        fold = validation.calendar_folds(events, n_splits=3)[1]

        captured_inner: list = []  # (id(ev), val_start, val_end, event_idの集合)
        real_split_events = evaluation.validation.split_events

        def spy_split_events(ev, f, **kwargs):
            train, val = real_split_events(ev, f, **kwargs)
            if ev is not events:
                # 外側の base.events を作るための最初の1回（ev is events）は
                # 除外し、内側fold用の呼び出しだけを候補間比較の対象にする。
                captured_inner.append(
                    (id(ev), f.val_start, f.val_end, frozenset(val["event_id"])))
            return train, val

        monkeypatch.setattr(evaluation.validation, "split_events", spy_split_events)

        got = evaluation.select_training_window(
            events, fold, evaluation.LogisticRegressionModel,
            candidates=[None, 20, 40], feature_cols=FEATURES)

        assert got in (None, 20, 40)
        # スパイが実際に呼ばれたことを保証する（空リスト同士の比較は自明にPASSしてしまう）
        assert len(captured_inner) > 0

        # (1) 内側fold用の呼び出しは、候補（窓）が変わっても常に同一の
        #     events オブジェクトから行われている＝内側foldを候補ごとに
        #     作り直していない。
        distinct_event_objects = {c[0] for c in captured_inner}
        assert len(distinct_event_objects) == 1, (
            "候補ごとに異なる events オブジェクトから内側foldが作られている"
            "（学習窓を先に適用してから内側foldを作り直す退行の兆候）"
        )

        # (2) 同じ検証期間（＝同じ内側fold）を指す呼び出しは、候補をまたいでも
        #     常に同じ検証event_id集合を返す。
        by_bounds: dict = {}
        for _, val_start, val_end, ids in captured_inner:
            by_bounds.setdefault((val_start, val_end), []).append(ids)
        assert len(by_bounds) >= 2  # 内側foldが複数あることを確認（自明なPASSを避ける）
        for (val_start, val_end), id_sets in by_bounds.items():
            assert all(s == id_sets[0] for s in id_sets), (
                f"内側fold({val_start}~{val_end})の検証集合が候補間で一致しない"
            )

        # inner_validation_event_ids() が返す値とも一致する
        all_captured_ids = set().union(*(c[3] for c in captured_inner))
        ids = evaluation.inner_validation_event_ids(
            events, fold, feature_cols=FEATURES)
        assert len(ids) > 0
        assert set(ids) == all_captured_ids

        # 内側検証集合は窓候補を引数に取らない＝窓に依存しない
        import inspect
        params = set(inspect.signature(
            evaluation.inner_validation_event_ids).parameters)
        assert "candidates" not in params
        assert "window_sessions" not in params

    def test_window_only_shrinks_the_training_side(self):
        """短い窓は学習件数を減らすが、検証件数は減らさない"""
        events = _events(n_sessions=150)
        fold = validation.calendar_folds(events, n_splits=3)[1]
        base = validation.training_inputs(events, fold, feature_cols=FEATURES)
        inner = list(validation.inner_folds(base.events, n_splits=3))[0]

        wide = validation.training_inputs(
            base.events, inner, feature_cols=FEATURES)
        narrow = validation.training_inputs(
            base.events, inner, window_sessions=10, feature_cols=FEATURES)
        _, val = validation.split_events(base.events, inner)

        assert len(narrow.events) < len(wide.events)
        # 検証側は split_events だけで決まり、窓を渡していないので変わらない
        assert len(val) > 0

    def test_comparison_is_per_event_not_a_sum(self):
        """比較値は採用イベント1件あたりの平均である（総和ではない）

        候補を1つしか渡さないテストでは実際には「比較」が起きず、
        `score = total`（総和）に書き換えても全件PASSしてしまうことが
        外部レビューの実験で確認された。ここでは意図的に

        - 候補A（窓=5・少数訓練データ）: 検証集合の中でもごく少数の
          「極上」イベント（net_return=+3.0）だけを拾う→件数は少ないが
          1件あたりの平均は非常に高い
        - 候補B（窓=None・多数訓練データ）: 「極上」に加えて多数の
          「並」イベント（net_return=+0.1）も拾う→合計netreturnは
          候補Aより大きいが、1件あたりの平均は低い

        という状況を作り、select_training_window() が候補Aを選ぶことを
        検証する。総和で比較すると候補Bの合計(10.8)が候補Aの合計(6.0)を
        上回るため誤って候補Bが選ばれ、このテストはFAILする
        （実際に `score = total` へ書き換えてFAILを確認済み。報告書参照）。

        検証集合（60件: 極上2件+並48件+不良10件）は候補間で共通（R19の
        不変条件）。モデルは学習データの件数（n_train）だけを見て、
        件数が少ない（<15件）ときは「極上」だけに高確率、多い（>=15件）
        ときは「極上+並」に高確率を返す──窓が小さいほど学習データが
        少なくなるので、これは学習窓の効果を模したふるまいになる。
        """
        start = date(2026, 1, 5)
        rows = []

        # 内側学習側（低密度・20セッション・symbol A0）
        # 窓=5 なら直近5セッション、窓=None なら20セッション全部が学習に入る。
        for i in range(20):
            d = start + timedelta(days=i)
            rows.append({
                "event_id": f"A0:{d:%Y%m%d}", "label_contract_id": _LC,
                "symbol": "A0", "decision_at": d, "entry_at": d,
                "label_end_at": d, "status": dataset.STATUS_RESOLVED,
                "label": 0, "net_return": 0.0, "x1": 0.0,
            })

        # 内側検証側（高密度・20セッション x 3銘柄=60件、候補間で共通）
        combos = []
        for i in range(20):
            d = start + timedelta(days=20 + i)
            for sym in ("V0", "V1", "V2"):
                combos.append((d, sym))
        assert len(combos) == 60
        # 先頭2件=極上、続く10件=不良、残り48件=並
        groups = ["excellent"] * 2 + ["bad"] * 10 + ["good"] * 48
        ret_map = {"excellent": 3.0, "good": 0.1, "bad": -1.0}
        x1_map = {"excellent": 2.0, "good": 1.0, "bad": 0.0}
        for (d, sym), g in zip(combos, groups):
            rows.append({
                "event_id": f"{sym}:{d:%Y%m%d}", "label_contract_id": _LC,
                "symbol": sym, "decision_at": d, "entry_at": d,
                "label_end_at": d, "status": dataset.STATUS_RESOLVED,
                "label": 1 if ret_map[g] > 0 else 0,
                "net_return": ret_map[g], "x1": x1_map[g],
            })
        events = pd.DataFrame(rows)

        fold = validation.Fold(
            index=0,
            train_start=start,
            train_end=start + timedelta(days=39),
            val_start=start + timedelta(days=40),
            val_end=start + timedelta(days=41),
        )

        class WindowSensitiveModel:
            """学習データ件数だけを見て、窓の広さに応じて拾う対象を変える
            テスト専用の模型。件数が少ないほど「極上」だけに絞り込む。"""
            name = "window_sensitive"

            def __init__(self) -> None:
                self._n_train = None

            def fit(self, X, y, sample_weight) -> None:
                self._n_train = len(X)

            def predict_proba(self, X):
                marker = X["x1"].to_numpy()
                if self._n_train is not None and self._n_train < 15:
                    return np.where(marker >= 2.0, 0.9, 0.1)
                return np.where(marker >= 1.0, 0.9, 0.1)

        got = evaluation.select_training_window(
            events, fold, WindowSensitiveModel,
            candidates=[5, None], feature_cols=["x1"], inner_splits=1)

        # 候補A（窓=5）: 極上2件のみ採用、合計=6.0、平均=3.0
        # 候補B（窓=None）: 極上+並=50件採用、合計=10.8、平均=0.216
        # 総和なら候補Bが勝つが、平均なら候補Aが勝つ。
        assert got == 5
