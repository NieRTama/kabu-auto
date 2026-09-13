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
