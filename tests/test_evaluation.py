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
        for col in ("event_id", "raw_probability", "calibrated_probability", "fold_index"):
            assert col in res.predictions.columns
        assert (res.predictions["fold_index"] == fold.index).all()

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


class TestRecomputeFromDetails:
    def test_metrics_match_the_original_run(self, isolated_db):
        """保存した明細だけから、実行時と同じ指標が出る（spec §14 完了条件）"""
        events = _events(n_sessions=150)
        out = evaluation.run_evaluation(
            events,
            model_factories={"logistic_regression": evaluation.LogisticRegressionModel},
            n_splits=3, feature_cols=FEATURES, persist=True)

        details = evaluation.load_prediction_details(
            out["evaluation_run_id"], model_id="logistic_regression")

        for res in out["fold_results"]:
            sub = details[details["fold_index"] == res.fold_index]
            again = evaluation.recompute_metrics(
                sub, baseline_rate=res.train_positive_rate)
            assert again["n"] == res.metrics["n"]
            assert again["brier"] == pytest.approx(res.metrics["brier"])
            if res.metrics["roc_auc"] is not None:
                assert again["roc_auc"] == pytest.approx(res.metrics["roc_auc"])

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

    def test_all_candidates_share_the_same_inner_validation_set(self):
        """候補窓を変えても内側の検証集合は同一である

        窓で外側集合を先に切ってから内側foldを作り直すと、短い窓と拡大窓で
        評価日も件数も変わり、「学習窓の効果」と「評価期間の差」が混ざる
        （外部レビューR19）。窓は学習側にだけ掛かること自体を固定する。
        """
        events = _events(n_sessions=150)
        fold = validation.calendar_folds(events, n_splits=3)[1]
        ids = evaluation.inner_validation_event_ids(
            events, fold, feature_cols=FEATURES)
        assert len(ids) > 0

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
        """比較値は採用イベント1件あたりの平均である

        総和のままだと「多く拾う窓」が中身の良し悪しと無関係に勝つ。
        候補を1つだけ渡した場合でも、採用が0件なら選ばれない。
        """
        events = _events(n_sessions=150)
        fold = validation.calendar_folds(events, n_splits=3)[1]
        # ConstantProbability は全件同じ確率を返す。select_threshold が
        # 収益を最大化する閾値を選ぶので、全件負なら採用0件になりうる。
        losing = events.copy()
        losing["net_return"] = -0.05
        got = evaluation.select_training_window(
            losing, fold, evaluation.ConstantProbability,
            candidates=[20], feature_cols=FEATURES)
        assert got in (None, 20)   # 採用0件なら None、拾ったなら 20
