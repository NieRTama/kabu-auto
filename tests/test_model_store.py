"""モデルの保存と現行の管理（src/strategy/model_store.py）のテスト

学習成功はモデル更新ではなく候補の生成である（spec §9）。実体は
バージョン別の不変ディレクトリへ置き、現行を指す小さな参照だけを
原子的に更新する。pickleは任意コード実行の経路なので使わない。
"""
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.strategy import model_store as ms


def _trained_model():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 200), "f2": rng.normal(0, 1, 200)})
    y = pd.Series((X["f1"] > 0).astype(int))
    m = lgb.LGBMClassifier(n_estimators=10, num_leaves=4,
                           random_state=42, verbose=-1)
    m.fit(X, y)
    return m, X


def _meta(model_id="m0001"):
    return ms.ModelMeta(
        model_id=model_id,
        trained_at=datetime(2026, 9, 12, 10, 0, 0),
        training_window_sessions=500,
        symbols=["7203", "9984"],
        label_definition="net_return>0",
        feature_cols=["f1", "f2"],
        positive_rate=0.48,
        fold_results=[{"fold": 0, "roc_auc": 0.52}],
        code_version="abc1234",
        lightgbm_version=lgb.__version__,
        dataset_id="ds0001",
    )


class TestSaveCandidate:
    def test_writes_native_format_not_pickle(self, tmp_path):
        model, _ = _trained_model()
        path = ms.save_candidate(model, _meta(), base_dir=str(tmp_path))
        assert (path / "model.txt").exists()
        assert (path / "meta.json").exists()
        assert not list(path.glob("*.pkl"))

    def test_model_file_is_text_not_a_pickle_stream(self, tmp_path):
        """LightGBMネイティブ形式はテキスト。pickleのマジックバイトが無い"""
        model, _ = _trained_model()
        path = ms.save_candidate(model, _meta(), base_dir=str(tmp_path))
        head = (path / "model.txt").read_bytes()[:16]
        assert b"\x80" not in head[:2]      # pickle protocol marker
        assert b"tree" in (path / "model.txt").read_bytes()[:200].lower()

    def test_goes_under_candidates_not_current(self, tmp_path):
        model, _ = _trained_model()
        path = ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        assert path == Path(tmp_path) / "candidates" / "m0001"
        assert not (Path(tmp_path) / "current.json").exists()

    def test_meta_records_the_full_contract(self, tmp_path):
        model, _ = _trained_model()
        path = ms.save_candidate(model, _meta(), base_dir=str(tmp_path))
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        for key in ("model_id", "trained_at", "training_window_sessions",
                    "symbols", "label_definition", "feature_cols",
                    "positive_rate", "fold_results", "code_version",
                    "lightgbm_version", "dataset_id"):
            assert key in meta

    def test_each_model_id_gets_its_own_directory(self, tmp_path):
        model, _ = _trained_model()
        a = ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        b = ms.save_candidate(model, _meta("m0002"), base_dir=str(tmp_path))
        assert a != b
        assert a.exists() and b.exists()

    def test_does_not_touch_the_legacy_pickle(self, tmp_path):
        """legacy の models/lgb_model.pkl を壊さない"""
        legacy = Path(tmp_path) / "lgb_model.pkl"
        legacy.write_bytes(b"legacy-model-bytes")
        model, _ = _trained_model()
        ms.save_candidate(model, _meta(), base_dir=str(tmp_path))
        assert legacy.read_bytes() == b"legacy-model-bytes"


class TestLoadModel:
    def test_round_trip_preserves_predictions(self, tmp_path):
        model, X = _trained_model()
        before = model.predict_proba(X)[:, 1]
        ms.save_candidate(model, _meta(), base_dir=str(tmp_path))
        loaded, _ = ms.load_model("m0001", base_dir=str(tmp_path))
        after = loaded.predict(X)
        assert np.allclose(before, after, atol=1e-9)

    def test_reads_back_the_meta(self, tmp_path):
        model, _ = _trained_model()
        ms.save_candidate(model, _meta(), base_dir=str(tmp_path))
        meta = ms.read_meta("m0001", base_dir=str(tmp_path))
        assert meta.model_id == "m0001"
        assert meta.feature_cols == ["f1", "f2"]
        assert meta.dataset_id == "ds0001"

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ms.load_model("nope", base_dir=str(tmp_path))


class TestSaveWrapperModels:
    """段階Cのラッパー型がそのまま保存できること（外部レビューR02）。

    `CurrentLightGBM` は `_model` と `predict_proba()` しか持たない。
    `getattr(model, "booster_", model).save_model(...)` という書き方だと
    AttributeError になり、それを握り潰した呼び出し側が
    「保存できていないのに学習成功」と扱ってしまう。
    """

    def _wrapper(self, single_class=False):
        from src.strategy.evaluation import CurrentLightGBM
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"f1": rng.normal(0, 1, 200), "f2": rng.normal(0, 1, 200)})
        y = pd.Series(np.ones(200, dtype=int) if single_class
                      else (X["f1"] > 0).astype(int))
        m = CurrentLightGBM()
        m.fit(X, y, np.ones(len(X)))
        return m, X

    def test_saves_a_two_class_wrapper(self, tmp_path):
        model, X = self._wrapper()
        path = ms.save_candidate(model, _meta("w0001"), base_dir=str(tmp_path))
        assert (path / "model.txt").exists()
        meta = ms.read_meta("w0001", base_dir=str(tmp_path))
        assert meta.model_kind == ms.KIND_BOOSTER

    def test_wrapper_round_trip_preserves_predictions(self, tmp_path):
        model, X = self._wrapper()
        before = model.predict_proba(X)
        ms.save_candidate(model, _meta("w0001"), base_dir=str(tmp_path))
        loaded, _ = ms.load_model("w0001", base_dir=str(tmp_path))
        assert np.allclose(before, loaded.predict(X), atol=1e-9)

    def test_saves_a_constant_model_without_a_booster(self, tmp_path):
        """単一クラスの学習結果には Booster が無い。定数として保存する"""
        model, X = self._wrapper(single_class=True)
        assert model.is_constant is True
        path = ms.save_candidate(model, _meta("c0001"), base_dir=str(tmp_path))
        assert (path / "constant.json").exists()
        assert not (path / "model.txt").exists()
        meta = ms.read_meta("c0001", base_dir=str(tmp_path))
        assert meta.model_kind == ms.KIND_CONSTANT

    def test_constant_model_round_trip_preserves_predictions(self, tmp_path):
        model, X = self._wrapper(single_class=True)
        before = model.predict_proba(X)
        ms.save_candidate(model, _meta("c0001"), base_dir=str(tmp_path))
        loaded, _ = ms.load_model("c0001", base_dir=str(tmp_path))
        after = loaded.predict(X)
        assert np.allclose(before, after, atol=1e-12)
        assert len(after) == len(X)

    def test_unsupported_type_raises_instead_of_silently_failing(self, tmp_path):
        class NotAModel:
            def predict_proba(self, X):
                return None

        with pytest.raises(TypeError, match="保存できないモデル型"):
            ms.save_candidate(NotAModel(), _meta("x0001"), base_dir=str(tmp_path))
        # 失敗したときにディレクトリを残さない
        assert not (Path(tmp_path) / "candidates" / "x0001").exists()


class TestCandidateDirectoriesAreImmutable:
    """候補ディレクトリは不変であること（外部レビューR12）。

    `exist_ok=True` で既存を受け入れて中身を上書きすると、そのIDを
    current や rollback 先が指していた場合に**昇格操作なしで実体が
    入れ替わる**。
    """

    def test_rejects_a_second_save_with_the_same_id(self, tmp_path):
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        with pytest.raises(ms.CandidateExists):
            ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))

    def test_the_original_files_are_untouched_after_a_rejected_save(self, tmp_path):
        model, X = _trained_model()
        path = ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        original = (path / "model.txt").read_bytes()

        other, _ = _trained_model()
        with pytest.raises(ms.CandidateExists):
            ms.save_candidate(other, _meta("m0001"), base_dir=str(tmp_path))
        assert (path / "model.txt").read_bytes() == original

    def test_the_model_current_points_at_cannot_be_replaced(self, tmp_path):
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        before = ms.load_model("m0001", base_dir=str(tmp_path))[0].model_to_string()

        with pytest.raises(ms.CandidateExists):
            ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        after = ms.load_model("m0001", base_dir=str(tmp_path))[0].model_to_string()
        assert before == after

    def test_a_failure_after_writing_the_model_leaves_nothing_behind(self, tmp_path):
        """モデル保存後にメタの書き込みが落ちても、公開されない"""
        model, _ = _trained_model()
        import src.strategy.model_store as mod

        original = mod._meta_to_json

        def boom(meta):
            raise OSError("ディスクが一杯です")

        mod._meta_to_json = boom
        try:
            with pytest.raises(OSError):
                ms.save_candidate(model, _meta("m0009"), base_dir=str(tmp_path))
        finally:
            mod._meta_to_json = original

        assert not (Path(tmp_path) / "candidates" / "m0009").exists()
        # 一時ディレクトリも残さない
        leftovers = list((Path(tmp_path) / "candidates").glob(".m0009.*"))
        assert leftovers == []

    def test_publishing_is_all_or_nothing(self, tmp_path):
        """公開されたディレクトリには必ずメタと本体が揃っている"""
        model, _ = _trained_model()
        path = ms.save_candidate(model, _meta("m0010"), base_dir=str(tmp_path))
        assert (path / "meta.json").exists()
        assert (path / "model.txt").exists()
        # 読み直せる（保存時に検証済み）
        loaded, meta = ms.load_model("m0010", base_dir=str(tmp_path))
        assert meta.model_id == "m0010"
        assert loaded is not None


class TestCurrentRef:
    def test_starts_unpromoted(self, tmp_path):
        """v2はモデル未昇格の状態から開始する（spec §9）"""
        assert ms.read_current(base_dir=str(tmp_path)) is None
        assert ms.load_current(base_dir=str(tmp_path)) is None

    def test_set_current_points_at_the_model(self, tmp_path):
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ref = ms.set_current("m0001", base_dir=str(tmp_path))
        assert ref.model_id == "m0001"
        assert ref.previous_model_id is None
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"

    def test_load_current_returns_the_model_and_meta(self, tmp_path):
        model, X = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        booster, meta = ms.load_current(base_dir=str(tmp_path))
        assert meta.model_id == "m0001"
        assert len(booster.predict(X)) == len(X)

    def test_switching_records_the_previous(self, tmp_path):
        model, _ = _trained_model()
        for mid in ("m0001", "m0002"):
            ms.save_candidate(model, _meta(mid), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        ref = ms.set_current("m0002", base_dir=str(tmp_path))
        assert ref.model_id == "m0002"
        assert ref.previous_model_id == "m0001"

    def test_rejects_a_model_that_was_never_saved(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ms.set_current("ghost", base_dir=str(tmp_path))

    def test_the_reference_is_small_and_the_model_stays_put(self, tmp_path):
        """切替は参照だけを書き換える。モデルの実体はコピーも移動もしない"""
        model, _ = _trained_model()
        path = ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        before = (path / "model.txt").read_bytes()
        ms.set_current("m0001", base_dir=str(tmp_path))
        assert (path / "model.txt").read_bytes() == before
        assert ms.current_ref_path(str(tmp_path)).stat().st_size < 1000


class TestAtomicSwitch:
    def test_no_partial_reference_is_left_behind(self, tmp_path):
        """一時ファイルが残らない（原子的な置換の副作用が無い）"""
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        leftovers = [p for p in Path(tmp_path).iterdir()
                     if p.name != ms.CURRENT_REF and p.is_file()]
        assert leftovers == []

    def test_previous_reference_survives_a_failed_switch(self, tmp_path):
        """存在しないモデルへの切替が失敗しても、現行は前のまま残る"""
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            ms.set_current("ghost", base_dir=str(tmp_path))
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"

    def test_reference_survives_a_reread(self, tmp_path):
        """再起動後に同じモデルへ復帰する（参照を読み直しても同じ）"""
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        first = ms.read_current(base_dir=str(tmp_path))
        second = ms.read_current(base_dir=str(tmp_path))
        assert first.model_id == second.model_id
        assert first.switched_at == second.switched_at


class TestRollback:
    def test_returns_to_the_previous_model(self, tmp_path):
        model, _ = _trained_model()
        for mid in ("m0001", "m0002"):
            ms.save_candidate(model, _meta(mid), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        ms.set_current("m0002", base_dir=str(tmp_path))

        ref = ms.rollback(base_dir=str(tmp_path))
        assert ref.model_id == "m0001"
        assert ms.load_current(base_dir=str(tmp_path))[1].model_id == "m0001"

    def test_records_the_rolled_back_model_as_previous(self, tmp_path):
        model, _ = _trained_model()
        for mid in ("m0001", "m0002"):
            ms.save_candidate(model, _meta(mid), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        ms.set_current("m0002", base_dir=str(tmp_path))
        ref = ms.rollback(base_dir=str(tmp_path))
        assert ref.previous_model_id == "m0002"

    def test_returns_none_when_there_is_nothing_to_roll_back_to(self, tmp_path):
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))
        assert ms.rollback(base_dir=str(tmp_path)) is None
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"

    def test_returns_none_when_unpromoted(self, tmp_path):
        assert ms.rollback(base_dir=str(tmp_path)) is None


class TestTrainAsCandidate:
    def _promoted(self, tmp_path):
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        ms.set_current("m0001", base_dir=str(tmp_path))

    def test_successful_training_creates_a_candidate_not_a_switch(self, tmp_path):
        """学習成功はモデル更新ではなく候補の生成（spec §9）"""
        self._promoted(tmp_path)
        model, _ = _trained_model()

        model_id = ms.train_as_candidate(
            lambda: model, lambda: _meta("m0002"), base_dir=str(tmp_path))

        assert model_id == "m0002"
        assert ms.candidate_dir("m0002", str(tmp_path)).exists()
        # 現行は変わらない
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"

    def test_training_failure_leaves_the_current_intact(self, tmp_path):
        """学習失敗で現行モデルが失われない（spec §14 段階E完了条件）"""
        self._promoted(tmp_path)

        def broken():
            raise RuntimeError("学習に失敗しました")

        assert ms.train_as_candidate(
            broken, lambda: _meta("m0002"), base_dir=str(tmp_path)) is None
        assert ms.read_current(base_dir=str(tmp_path)).model_id == "m0001"
        assert ms.load_current(base_dir=str(tmp_path))[1].model_id == "m0001"

    def test_metadata_failure_leaves_no_half_written_candidate(self, tmp_path):
        """メタの生成で落ちたら候補ディレクトリを残さない"""
        self._promoted(tmp_path)
        model, _ = _trained_model()

        def broken_meta():
            raise RuntimeError("メタの生成に失敗しました")

        assert ms.train_as_candidate(
            lambda: model, broken_meta, base_dir=str(tmp_path)) is None
        assert not ms.candidate_dir("m0002", str(tmp_path)).exists()

    def test_failure_when_unpromoted_stays_unpromoted(self, tmp_path):
        def broken():
            raise RuntimeError("boom")

        assert ms.train_as_candidate(
            broken, lambda: _meta("m0002"), base_dir=str(tmp_path)) is None
        assert ms.read_current(base_dir=str(tmp_path)) is None

    def test_previous_candidate_survives_a_new_failure(self, tmp_path):
        """過去の候補も壊さない"""
        model, _ = _trained_model()
        ms.save_candidate(model, _meta("m0001"), base_dir=str(tmp_path))
        before = (ms.candidate_dir("m0001", str(tmp_path)) / "model.txt").read_bytes()

        def broken():
            raise RuntimeError("boom")

        ms.train_as_candidate(broken, lambda: _meta("m0002"), base_dir=str(tmp_path))
        after = (ms.candidate_dir("m0001", str(tmp_path)) / "model.txt").read_bytes()
        assert after == before
