"""shadow運用（src/strategy/shadow.py）のテスト

同じ入力に現行と候補の判断を並行記録する。**候補は発注に繋がない**。
差が何によって生じたかを、見送った候補の結果も含めて記録する（spec §9）。
"""
import inspect

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.core import config as cfg
from src.data import database as db
from src.data.database import get_session
from src.strategy import evaluation
from src.strategy import shadow


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


class _FixedModel:
    """与えた確率をそのまま返すテスト用モデル"""

    def __init__(self, probabilities):
        self._p = np.asarray(probabilities, dtype=float)

    def predict(self, X):
        return self._p[:len(X)]


def _features(n=4):
    return pd.DataFrame({"f1": np.arange(n, dtype=float),
                         "f2": np.arange(n, dtype=float)})


_LC = "lc_test"


class TestCompare:
    def test_records_both_probabilities(self):
        got = shadow.compare(
            ["a", "b"], _features(2),
            current=_FixedModel([0.2, 0.8]),
            candidate=_FixedModel([0.7, 0.3]),
            threshold=0.5)
        assert [c.current_probability for c in got] == pytest.approx([0.2, 0.8])
        assert [c.candidate_probability for c in got] == pytest.approx([0.7, 0.3])

    def test_classifies_agreement(self):
        got = shadow.compare(
            ["both", "neither", "only_current", "only_candidate"], _features(4),
            current=_FixedModel([0.9, 0.1, 0.9, 0.1]),
            candidate=_FixedModel([0.9, 0.1, 0.1, 0.9]),
            threshold=0.5)
        assert [c.agreement for c in got] == [
            "both_take", "both_skip", "only_current", "only_candidate"]

    def test_includes_events_the_candidate_skipped(self):
        """見送った候補の結果も記録に残す（採った分だけでは差が測れない）"""
        got = shadow.compare(
            ["x"], _features(1),
            current=_FixedModel([0.9]), candidate=_FixedModel([0.1]),
            threshold=0.5)
        assert len(got) == 1
        assert got[0].candidate_takes is False

    def test_works_without_a_current_model(self):
        """v2はモデル未昇格から始まる。現行が無くても候補は記録できる"""
        got = shadow.compare(
            ["x"], _features(1),
            current=None, candidate=_FixedModel([0.8]), threshold=0.5)
        assert got[0].current_probability is None
        assert got[0].current_takes is False
        assert got[0].candidate_takes is True


class TestRecordShadow:
    def test_saves_with_the_shadow_purpose(self, isolated_db):
        comparisons = shadow.compare(
            ["a"], _features(1),
            current=_FixedModel([0.2]), candidate=_FixedModel([0.7]),
            threshold=0.5)
        n = shadow.record_shadow(comparisons, evaluation_run_id="shadow1",
                                 candidate_model_id="m0002",
                                 current_model_id="m0001", threshold=0.5,
                                 label_contract_id=_LC)
        assert n == 1

        with get_session() as session:
            row = session.scalar(select(db.Prediction))
        assert row.purpose == evaluation.PURPOSE_SHADOW
        assert row.model_id == "m0002"
        assert row.evaluation_run_id == "shadow1"

    def test_uses_a_sentinel_fold_index(self, isolated_db):
        """shadowはfoldに属さない"""
        comparisons = shadow.compare(
            ["a"], _features(1),
            current=_FixedModel([0.2]), candidate=_FixedModel([0.7]),
            threshold=0.5)
        shadow.record_shadow(comparisons, evaluation_run_id="shadow1",
                             candidate_model_id="m0002",
                             current_model_id="m0001", threshold=0.5,
                             label_contract_id=_LC)
        with get_session() as session:
            row = session.scalar(select(db.Prediction))
        assert row.fold_index == -1

    def test_outcomes_are_not_written(self, isolated_db):
        """予測時点で実績は未確定。PredictionOutcomeは書かない"""
        comparisons = shadow.compare(
            ["a"], _features(1),
            current=_FixedModel([0.2]), candidate=_FixedModel([0.7]),
            threshold=0.5)
        shadow.record_shadow(comparisons, evaluation_run_id="shadow1",
                             candidate_model_id="m0002",
                             current_model_id="m0001", threshold=0.5,
                             label_contract_id=_LC)
        with get_session() as session:
            outcomes = list(session.scalars(select(db.PredictionOutcome)).all())
        assert outcomes == []


class TestShadowRecordsBothSides:
    """比較そのものが保存されること（外部レビューR21）。

    候補の確率だけを書くと、再起動後に「その時どちらを採り、なぜ見送ったか」
    を復元できない。関数名や一時的な戻り値では並行記録にならない。
    """

    def _record(self, run_id="shadow1", current=_FixedModel([0.2, 0.2]),
                current_id="m0001"):
        comparisons = shadow.compare(
            ["a", "b"], _features(2),
            current=current, candidate=_FixedModel([0.7, 0.3]),
            threshold=0.5)
        shadow.record_shadow(
            comparisons, evaluation_run_id=run_id,
            candidate_model_id="m0002", current_model_id=current_id,
            threshold=0.5, label_contract_id=_LC)
        return comparisons

    def test_stores_both_model_ids_and_both_probabilities(self, isolated_db):
        self._record(current=_FixedModel([0.2, 0.9]))
        got = shadow.load_shadow_comparisons("shadow1")
        assert len(got) == 2
        assert set(got["current_model_id"]) == {"m0001"}
        assert set(got["candidate_model_id"]) == {"m0002"}
        assert got["current_probability"].notna().all()
        assert got["candidate_probability"].notna().all()

    def test_stores_the_threshold_and_the_decisions(self, isolated_db):
        self._record(current=_FixedModel([0.2, 0.9]))
        got = shadow.load_shadow_comparisons("shadow1").set_index("event_id")
        assert set(got["threshold"]) == {0.5}
        assert got.loc["a", "current_takes"] is False or \
            got.loc["a", "current_takes"] == False  # noqa: E712
        assert got.loc["a", "candidate_takes"] == True  # noqa: E712
        assert got.loc["b", "current_takes"] == True  # noqa: E712
        assert got.loc["b", "candidate_takes"] == False  # noqa: E712

    def test_the_summary_can_be_recomputed_from_the_database(self, isolated_db):
        """再起動後に当時の比較内訳を復元できること（合格条件）"""
        comparisons = self._record(current=_FixedModel([0.2, 0.9]))
        expected = shadow.disagreement_summary(comparisons)

        stored = shadow.load_shadow_comparisons("shadow1")
        rebuilt = shadow.disagreement_summary([
            shadow.ShadowComparison(
                event_id=r["event_id"],
                current_probability=r["current_probability"],
                candidate_probability=r["candidate_probability"],
                current_takes=bool(r["current_takes"]),
                candidate_takes=bool(r["candidate_takes"]),
                agreement=r["agreement"])
            for _, r in stored.iterrows()])
        assert rebuilt == expected

    def test_records_the_unpromoted_current_as_its_own_state(self, isolated_db):
        """現行が未昇格でも行は残す

        行を作らないと「比較しなかった」のか「現行が無かった」のかを
        後から区別できない。
        """
        self._record(current=None, current_id=None)
        got = shadow.load_shadow_comparisons("shadow1")
        assert len(got) == 2
        assert got["current_model_id"].isna().all()
        assert got["current_probability"].isna().all()
        assert (~got["current_takes"]).all()

    def test_stores_the_label_contract(self, isolated_db):
        """どのラベル契約に対する比較かを残す（実績と結合するため）"""
        self._record()
        got = shadow.load_shadow_comparisons("shadow1")
        assert set(got["label_contract_id"]) == {_LC}

    def test_runs_are_isolated(self, isolated_db):
        self._record(run_id="shadow1")
        self._record(run_id="shadow2")
        assert len(shadow.load_shadow_comparisons("shadow1")) == 2
        assert len(shadow.load_shadow_comparisons("shadow2")) == 2

    def test_rerunning_the_same_run_replaces_the_comparison_rows(self, isolated_db):
        """同じrunを2回記録しても例外にならず、2回目の内容で上書きされる（M1）。

        save_predictions() と同じ「run単位の置換」に揃える前は、
        shadow_comparisons のUNIQUE索引にひっかかり2回目がIntegrityErrorに
        なった上、先にcommit済みのPredictionだけが新しい内容になり
        比較行だけ古いまま残る恒久的な食い違いが起きていた。
        """
        comparisons1 = shadow.compare(
            ["a", "b"], _features(2),
            current=_FixedModel([0.2, 0.2]), candidate=_FixedModel([0.7, 0.7]),
            threshold=0.5)
        n1 = shadow.record_shadow(
            comparisons1, evaluation_run_id="shadow1",
            candidate_model_id="m0002", current_model_id="m0001",
            threshold=0.5, label_contract_id=_LC)
        assert n1 == 2

        comparisons2 = shadow.compare(
            ["a", "b", "c"], _features(3),
            current=_FixedModel([0.2, 0.2, 0.2]),
            candidate=_FixedModel([0.2, 0.2, 0.2]),
            threshold=0.5)
        n2 = shadow.record_shadow(
            comparisons2, evaluation_run_id="shadow1",
            candidate_model_id="m0002", current_model_id="m0001",
            threshold=0.5, label_contract_id=_LC)
        assert n2 == 3

        got = shadow.load_shadow_comparisons("shadow1")
        assert len(got) == 3
        assert set(got["event_id"]) == {"a", "b", "c"}
        assert (got["candidate_probability"] == 0.2).all()

        with get_session() as session:
            preds = list(session.scalars(select(db.Prediction)).all())
        assert len(preds) == 3
        assert all(p.calibrated_probability == 0.2 for p in preds)


class TestDisagreementSummary:
    def test_counts_each_category(self):
        comparisons = shadow.compare(
            ["a", "b", "c", "d"], _features(4),
            current=_FixedModel([0.9, 0.1, 0.9, 0.1]),
            candidate=_FixedModel([0.9, 0.1, 0.1, 0.9]),
            threshold=0.5)
        got = shadow.disagreement_summary(comparisons)
        assert got["both_take"] == 1
        assert got["both_skip"] == 1
        assert got["only_current"] == 1
        assert got["only_candidate"] == 1
        assert got["n"] == 4

    def test_agreement_rate_is_reported(self):
        comparisons = shadow.compare(
            ["a", "b", "c", "d"], _features(4),
            current=_FixedModel([0.9, 0.1, 0.9, 0.1]),
            candidate=_FixedModel([0.9, 0.1, 0.1, 0.9]),
            threshold=0.5)
        assert shadow.disagreement_summary(comparisons)["agreement_rate"] == pytest.approx(0.5)

    def test_empty_input_is_safe(self):
        got = shadow.disagreement_summary([])
        assert got["n"] == 0
        assert got["agreement_rate"] is None


class TestNotWiredToOrdering:
    """候補は発注に繋がらない（spec §9）"""

    def _import_lines(self):
        """import文だけを取り出す。

        ソース全体を対象にすると、docstringが発注系に言及しただけで落ちる
        脆いテストになる。依存の有無だけを見る。
        """
        src = inspect.getsource(shadow)
        return "\n".join(
            line for line in src.splitlines()
            if line.startswith("import ") or line.startswith("from ")
        )

    def test_module_does_not_import_execution_modules(self):
        joined = self._import_lines()
        assert "execution" not in joined
        assert "order" not in joined
        assert "kabu_client" not in joined

    def test_module_does_not_import_the_trading_service(self):
        assert "trading" not in self._import_lines()

    def test_public_functions_return_records_not_orders(self):
        """公開関数はどれも記録用の値だけを返す"""
        for name in ("compare", "record_shadow", "disagreement_summary"):
            assert callable(getattr(shadow, name))
        assert not hasattr(shadow, "place_order")
        assert not hasattr(shadow, "execute")
