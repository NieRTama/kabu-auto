"""昇格の契約（src/strategy/promotion.py）のテスト

自動昇格は実装しない。昇格は明示的な呼び出しでのみ起き、
判断者と理由の記録を必須にする（spec §9）。
"""
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.core import config as cfg
from src.data import database as db
from src.data.database import get_session
from src.strategy import evaluation
from src.strategy import model_store as ms
from src.strategy import promotion


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


_LC = "testcontract"


def _saved_model(tmp_path, model_id="m0001", feature_cols=("f1", "f2"),
                  label_contract_id=_LC):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({c: rng.normal(0, 1, 200) for c in feature_cols})
    y = pd.Series((X[feature_cols[0]] > 0).astype(int))
    m = lgb.LGBMClassifier(n_estimators=10, num_leaves=4, random_state=42, verbose=-1)
    m.fit(X, y)
    meta = ms.ModelMeta(
        model_id=model_id, trained_at=datetime(2026, 9, 12, 10, 0, 0),
        feature_cols=list(feature_cols), dataset_id="ds0001",
        lightgbm_version=lgb.__version__, label_contract_id=label_contract_id)
    ms.save_candidate(m, meta, base_dir=str(tmp_path))
    return model_id


def _recorded_evaluation(run_id="run1", model_id="m0001", *,
                         resolved=True, purpose=None, degraded=False):
    """評価実行を1件ぶん保存する。

    **予測・実績・実行記録の3つを揃える。** 予測だけを保存した状態を
    「評価済み」と呼ばないため（外部レビューR13）。
    `resolved=False` で実績を保存しない状態を作れる。
    """
    from src.strategy.evaluation import (
        PURPOSE_SHADOW, PURPOSE_VALIDATION, RunConfig, save_evaluation_run,
        save_outcomes, save_predictions)

    purpose = purpose or PURPOSE_VALIDATION
    preds = pd.DataFrame({
        "event_id": ["7203:20260105"],
        "label_contract_id": [_LC],
        "raw_probability": [0.6],
        "calibrated_probability": [0.55],
        "fold_index": [0 if purpose == PURPOSE_VALIDATION else -1],
    })
    save_predictions(preds, run_id, model_id, purpose=purpose)

    if resolved:
        events = pd.DataFrame({
            "event_id": ["7203:20260105"],
            "label_contract_id": [_LC],
            "status": ["resolved"],
            "label": [1],
            "net_return": [0.03],
        })
        save_outcomes(events)

    save_evaluation_run(
        run_id,
        RunConfig(dataset_id="ds0001", label_contract_id=_LC,
                  feature_version="f1", execution_model_version="t1_open_v1",
                  code_version="abc1234", config_json="{}", config_hash="cfg1"),
        purpose=purpose, model_id=model_id, n_folds=1, n_predictions=1,
        degraded_reasons=(["推論に失敗しました"] if degraded else []))


class TestPromotionBlockers:
    def test_degraded_run_blocks_promotion(self, isolated_db, tmp_path):
        _saved_model(tmp_path)
        _recorded_evaluation()
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=True,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("degraded" in b for b in got.blockers)

    def test_missing_evaluation_blocks_promotion(self, isolated_db, tmp_path):
        _saved_model(tmp_path)
        got = promotion.check_promotable(
            "m0001", evaluation_run_id=None, degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("未評価" in b for b in got.blockers)

    def test_evaluation_without_predictions_blocks_promotion(self, isolated_db, tmp_path):
        """評価IDがあっても予測明細が無ければ根拠にならない"""
        _saved_model(tmp_path)
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run-with-no-rows", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("予測明細" in b for b in got.blockers)

    def test_feature_mismatch_blocks_promotion(self, isolated_db, tmp_path):
        _saved_model(tmp_path, feature_cols=("f1", "f2"))
        _recorded_evaluation()
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2", "f3"])
        assert got.ok is False
        assert any("特徴量" in b for b in got.blockers)

    def test_unsaved_model_blocks_promotion(self, isolated_db, tmp_path):
        got = promotion.check_promotable(
            "ghost", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1"])
        assert got.ok is False

    def test_all_blockers_are_reported_together(self, isolated_db, tmp_path):
        """1つ直せば通る、を繰り返さずに済むよう全件返す"""
        _saved_model(tmp_path, feature_cols=("f1",))
        got = promotion.check_promotable(
            "m0001", evaluation_run_id=None, degraded=True,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert len(got.blockers) >= 3

    def test_clean_candidate_passes(self, isolated_db, tmp_path):
        _saved_model(tmp_path)
        _recorded_evaluation()
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is True
        assert got.blockers == []


class TestPromote:
    def _clean(self, tmp_path, model_id="m0001"):
        _saved_model(tmp_path, model_id=model_id)
        _recorded_evaluation(model_id=model_id)
        return model_id

    def test_records_who_decided_and_why(self, isolated_db, tmp_path):
        self._clean(tmp_path)
        pid = promotion.promote(
            "m0001", evaluation_run_id="run1", decided_by="garnet",
            reason="内側foldで定数モデルを上回ったため", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])

        with get_session() as session:
            row = session.scalar(select(db.ModelPromotion))
        assert row.id == pid
        assert row.model_id == "m0001"
        assert row.evaluation_run_id == "run1"
        assert row.decided_by == "garnet"
        assert "定数モデル" in row.reason
        assert row.switched_at is not None

    def test_switches_the_current_reference(self, isolated_db, tmp_path):
        self._clean(tmp_path)
        promotion.promote(
            "m0001", evaluation_run_id="run1", decided_by="garnet",
            reason="ok", degraded=False, base_dir=str(tmp_path),
            expected_feature_cols=["f1", "f2"])
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"

    def test_records_the_previous_model(self, isolated_db, tmp_path):
        self._clean(tmp_path, "m0001")
        self._clean(tmp_path, "m0002")
        promotion.promote("m0001", evaluation_run_id="run1", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])
        promotion.promote("m0002", evaluation_run_id="run1", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])
        with get_session() as session:
            rows = list(session.scalars(select(db.ModelPromotion)).all())
        assert rows[-1].previous_model_id == "m0001"

    def test_refuses_a_blocked_candidate(self, isolated_db, tmp_path):
        _saved_model(tmp_path)
        with pytest.raises(ValueError, match="昇格できません"):
            promotion.promote(
                "m0001", evaluation_run_id="run1", decided_by="garnet",
                reason="ok", degraded=True, base_dir=str(tmp_path),
                expected_feature_cols=["f1", "f2"])

    def test_refused_promotion_leaves_the_current_untouched(self, isolated_db, tmp_path):
        """昇格が拒否されても現行は動かない（途中状態を残さない）"""
        self._clean(tmp_path, "m0001")
        promotion.promote("m0001", evaluation_run_id="run1", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])
        _saved_model(tmp_path, model_id="m0002")

        with pytest.raises(ValueError):
            promotion.promote("m0002", evaluation_run_id=None, decided_by="g",
                              reason="ng", degraded=False, base_dir=str(tmp_path),
                              expected_feature_cols=["f1", "f2"])
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"

    def test_requires_a_decider_and_a_reason(self, isolated_db, tmp_path):
        """自動昇格を実装しない。判断者と理由は必須"""
        self._clean(tmp_path)
        with pytest.raises(ValueError, match="判断者"):
            promotion.promote("m0001", evaluation_run_id="run1", decided_by="",
                              reason="ok", degraded=False, base_dir=str(tmp_path),
                              expected_feature_cols=["f1", "f2"])
        with pytest.raises(ValueError, match="理由"):
            promotion.promote("m0001", evaluation_run_id="run1", decided_by="g",
                              reason="", degraded=False, base_dir=str(tmp_path),
                              expected_feature_cols=["f1", "f2"])


class TestAlreadyCurrentBlocksPromotion:
    """既に現行のモデルを再昇格すると previous_model_id が自分自身になり、
    rollback() が過去のモデルへ永久に戻れなくなる（外部レビュー最終
    ブランチレビュー M4）。自動昇格を実装しない契約に沿い、明示的に拒否する。
    """

    def test_check_promotable_blocks_the_current_model(self, isolated_db, tmp_path):
        self_id = "m0001"
        _saved_model(tmp_path, model_id=self_id)
        _recorded_evaluation(model_id=self_id)
        ms.set_current(self_id, base_dir=str(tmp_path))

        got = promotion.check_promotable(
            self_id, evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("既に現行" in b for b in got.blockers)

    def test_promote_refuses_the_current_model(self, isolated_db, tmp_path):
        _saved_model(tmp_path, model_id="m0001")
        _saved_model(tmp_path, model_id="m0002")
        _recorded_evaluation(model_id="m0001")
        _recorded_evaluation(run_id="run2", model_id="m0002")
        promotion.promote("m0001", evaluation_run_id="run1", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])
        promotion.promote("m0002", evaluation_run_id="run2", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])

        with pytest.raises(ValueError, match="昇格できません"):
            promotion.promote("m0002", evaluation_run_id="run2", decided_by="g",
                              reason="再昇格", degraded=False,
                              base_dir=str(tmp_path),
                              expected_feature_cols=["f1", "f2"])

        # 拒否後も rollback() は正しく過去のモデルへ戻れる
        ref = ms.rollback(base_dir=str(tmp_path))
        assert ref.model_id == "m0001"


class TestMissingLabelContractIdBlocksPromotion:
    """meta.label_contract_id が None（既定値のまま）だと、ラベル契約の
    食い違い検査そのものが黙ってスキップされていた（外部レビュー最終
    ブランチレビュー 保留Ruling m5 の再確認）。欠落は沈黙ではなく拒否にする。
    """

    def test_missing_meta_label_contract_id_blocks_promotion(
            self, isolated_db, tmp_path):
        _saved_model(tmp_path, label_contract_id=None)
        _recorded_evaluation()
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("ラベル契約ID" in b for b in got.blockers)

    def test_matching_label_contract_id_still_passes(self, isolated_db, tmp_path):
        """既存の一致ケースは従来どおり通る（回帰確認）"""
        _saved_model(tmp_path, label_contract_id=_LC)
        _recorded_evaluation()
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is True, got.blockers


class TestUnresolvedEvaluationBlocksPromotion:
    """予測だけでは「評価済み」にしない（外部レビューR13）。

    `load_prediction_details()` は実績が無くても行を返す。行数だけを
    見る条件では、ラベルが全て未確定の shadow 予測でも昇格できてしまう。
    """

    def test_predictions_without_outcomes_block_promotion(self, isolated_db, tmp_path):
        _saved_model(tmp_path)
        _recorded_evaluation(resolved=False)
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("実績が1件も確定していません" in b for b in got.blockers)

    def test_shadow_only_blocks_promotion(self, isolated_db, tmp_path):
        from src.strategy.evaluation import PURPOSE_SHADOW
        _saved_model(tmp_path)
        _recorded_evaluation(purpose=PURPOSE_SHADOW)
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("shadow" in b for b in got.blockers)

    def test_stored_degraded_blocks_even_when_the_caller_says_otherwise(
            self, isolated_db, tmp_path):
        """呼び出し側の bool ではなく保存済みの実行記録を根拠にする"""
        _saved_model(tmp_path)
        _recorded_evaluation(degraded=True)
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,   # 嘘の申告
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("保存済みの実行記録が degraded" in b for b in got.blockers)

    def test_missing_run_record_blocks_promotion(self, isolated_db, tmp_path):
        """予測明細はあるが実行記録が無い → 何を測ったのか復元できない"""
        from src.strategy.evaluation import save_predictions
        _saved_model(tmp_path)
        save_predictions(pd.DataFrame({
            "event_id": ["7203:20260105"], "label_contract_id": [_LC],
            "raw_probability": [0.6], "calibrated_probability": [0.55],
            "fold_index": [0]}), "run1", "m0001")
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("評価実行の記録がありません" in b for b in got.blockers)

    def test_run_for_a_different_model_blocks_promotion(self, isolated_db, tmp_path):
        _saved_model(tmp_path, model_id="m0002")
        _recorded_evaluation(model_id="m0001")
        got = promotion.check_promotable(
            "m0002", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is False
        assert any("別のモデル" in b for b in got.blockers)

    def test_a_fully_resolved_validation_run_is_promotable(self, isolated_db, tmp_path):
        _saved_model(tmp_path)
        _recorded_evaluation()
        got = promotion.check_promotable(
            "m0001", evaluation_run_id="run1", degraded=False,
            base_dir=str(tmp_path), expected_feature_cols=["f1", "f2"])
        assert got.ok is True, got.blockers


class TestPromotionIsRecoverable:
    """切替の途中で落ちても決着できること（外部レビューR11）。

    参照ファイルの原子的置換は、DBを含む取引の原子性ではない。
    「参照は新モデルなのに昇格記録が無い」状態を作らないため、
    昇格の意図を先に永続化し、切替後に確定させる。
    """

    def _ready(self, tmp_path, model_id="m0001"):
        _saved_model(tmp_path, model_id=model_id)
        _recorded_evaluation(model_id=model_id)

    def test_successful_promotion_is_committed(self, isolated_db, tmp_path):
        self._ready(tmp_path)
        pid = promotion.promote(
            "m0001", evaluation_run_id="run1", decided_by="g", reason="ok",
            degraded=False, base_dir=str(tmp_path),
            expected_feature_cols=["f1", "f2"])
        with get_session() as session:
            row = session.get(db.ModelPromotion, pid)
            assert row.state == promotion.PROMOTION_COMMITTED
            assert row.switched_at is not None

    def test_a_failed_switch_is_recorded_as_failed(self, isolated_db, tmp_path):
        """参照の切替に失敗しても、記録だけが committed で残らない"""
        self._ready(tmp_path)

        def boom(*args, **kwargs):
            raise OSError("参照ファイルを書けません")

        original = ms.set_current
        ms.set_current = boom
        try:
            with pytest.raises(OSError):
                promotion.promote(
                    "m0001", evaluation_run_id="run1", decided_by="g",
                    reason="ok", degraded=False, base_dir=str(tmp_path),
                    expected_feature_cols=["f1", "f2"])
        finally:
            ms.set_current = original

        with get_session() as session:
            row = session.scalar(select(db.ModelPromotion))
        assert row.state == promotion.PROMOTION_FAILED
        assert row.switched_at is None
        # 現行は切り替わっていない
        assert ms.read_current(base_dir=str(tmp_path)) is None

    def test_recovery_commits_a_pending_row_whose_switch_actually_happened(
            self, isolated_db, tmp_path):
        """切替後・確定前に落ちた場合 → 実体に合わせて committed にする"""
        self._ready(tmp_path)
        with get_session() as session:
            row = db.ModelPromotion(
                model_id="m0001", evaluation_run_id="run1", decided_by="g",
                reason="ok", previous_model_id=None,
                state=promotion.PROMOTION_PENDING, switched_at=None)
            session.add(row)
            session.commit()
            pid = row.id
        ms.set_current("m0001", base_dir=str(tmp_path))   # 切替は完了していた

        resolved = promotion.recover_promotions(base_dir=str(tmp_path))
        assert len(resolved) == 1
        assert resolved[0]["resolved_to"] == promotion.PROMOTION_COMMITTED
        with get_session() as session:
            assert session.get(db.ModelPromotion, pid).state == \
                promotion.PROMOTION_COMMITTED

    def test_recovery_fails_a_pending_row_whose_switch_never_happened(
            self, isolated_db, tmp_path):
        """切替前に落ちた場合 → failed にする。参照は触らない"""
        self._ready(tmp_path)
        with get_session() as session:
            row = db.ModelPromotion(
                model_id="m0001", evaluation_run_id="run1", decided_by="g",
                reason="ok", previous_model_id=None,
                state=promotion.PROMOTION_PENDING, switched_at=None)
            session.add(row)
            session.commit()
            pid = row.id

        resolved = promotion.recover_promotions(base_dir=str(tmp_path))
        assert resolved[0]["resolved_to"] == promotion.PROMOTION_FAILED
        with get_session() as session:
            assert session.get(db.ModelPromotion, pid).state == \
                promotion.PROMOTION_FAILED
        # 参照は書き換えない（自動でやり直さない）
        assert ms.read_current(base_dir=str(tmp_path)) is None

    def test_recovery_is_idempotent(self, isolated_db, tmp_path):
        self._ready(tmp_path)
        promotion.promote("m0001", evaluation_run_id="run1", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])
        assert promotion.recover_promotions(base_dir=str(tmp_path)) == []
        assert promotion.recover_promotions(base_dir=str(tmp_path)) == []

    def test_no_committed_row_without_a_switched_at(self, isolated_db, tmp_path):
        """不変条件: committed なら切替時刻が必ずある"""
        self._ready(tmp_path)
        promotion.promote("m0001", evaluation_run_id="run1", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])
        for row in promotion.promotion_history():
            if row.state == promotion.PROMOTION_COMMITTED:
                assert row.switched_at is not None


class TestPromotionRollback:
    """`promotion.rollback()` — ロールバックも ModelPromotion に記録を
    残す（外部レビュー最終ブランチレビュー M3）。`model_store.rollback()`
    は参照の切替だけで、DBには何も残さないため監査証跡と実際の現行が
    食い違っていた。
    """

    def _promoted_twice(self, tmp_path):
        _saved_model(tmp_path, model_id="m0001")
        _saved_model(tmp_path, model_id="m0002")
        _recorded_evaluation(run_id="run1", model_id="m0001")
        _recorded_evaluation(run_id="run2", model_id="m0002")
        promotion.promote("m0001", evaluation_run_id="run1", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])
        promotion.promote("m0002", evaluation_run_id="run2", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])

    def test_rollback_switches_the_reference(self, isolated_db, tmp_path):
        self._promoted_twice(tmp_path)
        pid = promotion.rollback(
            decided_by="garnet", reason="候補で異常検知のため",
            base_dir=str(tmp_path))
        assert pid is not None
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"

    def test_rollback_records_who_and_why(self, isolated_db, tmp_path):
        self._promoted_twice(tmp_path)
        pid = promotion.rollback(
            decided_by="garnet", reason="候補で異常検知のため",
            base_dir=str(tmp_path))

        with get_session() as session:
            row = session.get(db.ModelPromotion, pid)
        assert row.model_id == "m0001"
        assert row.previous_model_id == "m0002"
        assert row.decided_by == "garnet"
        assert row.reason == "候補で異常検知のため"
        assert row.state == promotion.PROMOTION_COMMITTED
        assert row.switched_at is not None

    def test_rollback_requires_a_decider_and_a_reason(self, isolated_db, tmp_path):
        self._promoted_twice(tmp_path)
        with pytest.raises(ValueError, match="判断者"):
            promotion.rollback(decided_by="", reason="ok", base_dir=str(tmp_path))
        with pytest.raises(ValueError, match="理由"):
            promotion.rollback(decided_by="g", reason="", base_dir=str(tmp_path))

    def test_rollback_with_nothing_to_roll_back_to_records_nothing(
            self, isolated_db, tmp_path):
        _saved_model(tmp_path, model_id="m0001")
        _recorded_evaluation(model_id="m0001")
        promotion.promote("m0001", evaluation_run_id="run1", decided_by="g",
                          reason="ok", degraded=False, base_dir=str(tmp_path),
                          expected_feature_cols=["f1", "f2"])

        result = promotion.rollback(
            decided_by="g", reason="ok", base_dir=str(tmp_path))
        assert result is None
        with get_session() as session:
            rows = list(session.scalars(select(db.ModelPromotion)).all())
        assert len(rows) == 1  # promote() の1件のみ。rollback()は何も残さない
