"""v2の学習経路（src/strategy/v2_training.py）のテスト

段階A〜Eは新しい経路を作るだけで、どちらを使うかは誰も決めていなかった。
本計画が分岐を実装する。legacyを選んだときの経路は1行も変えない。
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.core import config as cfg
from src.data import database as db
from src.data.database import get_session


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    # dataset.save_events() は base_dir 既定値("data/datasets"、相対パス)で
    # 呼ばれる（v2_training.train_v2() は base_dir をモデル保存にしか渡さない）。
    # chdir せずに実行すると、このテストファイルを走らせるたびに実際の
    # リポジトリ直下の data/datasets/ へ .csv.gz が生成され続けてしまう
    # （実際に発生・確認済み）。cfg.load/db.init の後で tmp_path へ chdir し、
    # 相対パス書き込みをテスト用ディレクトリに閉じ込める。
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestModelMetricsColumns:
    def test_new_columns_exist(self, isolated_db):
        from src.data.database import ModelMetrics

        with get_session() as session:
            session.add(ModelMetrics(
                cv_mean_accuracy=0.55, n_samples=100, trigger="test",
                model_id="m0001", positive_rate=0.48,
                training_window_sessions=500, engine_version="v2"))
            session.commit()
            row = session.scalar(select(ModelMetrics))
        assert row.model_id == "m0001"
        assert row.positive_rate == pytest.approx(0.48)
        assert row.training_window_sessions == 500
        assert row.engine_version == "v2"

    def test_columns_are_nullable(self, isolated_db):
        """legacy の _save_metrics は新列を書かない。書かなくても通ること"""
        from src.data.database import ModelMetrics

        with get_session() as session:
            session.add(ModelMetrics(
                cv_mean_accuracy=0.55, n_samples=100, trigger="weekly_schedule"))
            session.commit()
            row = session.scalar(select(ModelMetrics))
        assert row.model_id is None
        assert row.engine_version is None

    def test_legacy_and_v2_records_are_distinguishable(self, isolated_db):
        """同じテーブルに混ざるので、どちらの方式かが分かること"""
        from src.data.database import ModelMetrics

        with get_session() as session:
            session.add(ModelMetrics(cv_mean_accuracy=0.55, n_samples=100,
                                     trigger="weekly_schedule"))
            session.add(ModelMetrics(cv_mean_accuracy=0.52, n_samples=90,
                                     trigger="weekly_schedule",
                                     engine_version="v2", model_id="m0001"))
            session.commit()
            rows = list(session.scalars(select(ModelMetrics)).all())
        versions = {r.engine_version for r in rows}
        assert versions == {None, "v2"}


def _ohlcv(n=300, start_price=1000.0, seed=0):
    rng = np.random.default_rng(seed)
    start = date(2025, 1, 6)
    rows, p = [], start_price
    for i in range(n):
        p *= 1 + rng.normal(0, 0.01)
        rows.append({"date": start + timedelta(days=i), "open": p,
                     "high": p * 1.01, "low": p * 0.99, "close": p,
                     "volume": 1_000_000})
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def _policy_conf():
    from src.strategy import policy
    return policy.PolicyConfig(
        stop_loss_pct=-0.07, breakeven_trigger_pct=0.02, trailing_stop_pct=0.04,
        sell_threshold=-0.25, max_holding_sessions=10)


def _costs():
    from src.backtest import execution
    return execution.CostConfig(slippage_pct=0.001, commission_pct=0.0)


# dataset.simulate_event() は候補ごとに観測列を series の末尾まで作る
# （max_holding_sessions で早期に打ち切らない・実測でO(n^2)）。ブリーフ記載の
# n=300 では合成データの決着イベントが51件しかできず本番既定の
# MIN_RESOLVED_EVENTS=200に届かないが、届かせるだけの本数（n>=1500）に
# 増やすと単発の train_v2() 呼び出しだけで4〜5分かかる（実測）。
# 実運用のしきい値(200)自体は変えず、成功経路まで到達させる必要がある
# テストだけ v2_training.MIN_RESOLVED_EVENTS を一時的に下げる
# （task-2-report.md参照）。
_TEST_MIN_RESOLVED_EVENTS = 10


class TestTrainV2:
    def _bars(self):
        return {"7203": _ohlcv(seed=1), "9984": _ohlcv(seed=2, start_price=500.0)}

    def test_produces_a_candidate_not_a_promotion(
            self, isolated_db, tmp_path, monkeypatch):
        """学習成功は候補の生成であって運用モデルの差し替えではない

        **`if res.model_id is not None:` で包まない。** 包むと、保存が
        AttributeError で失敗して `train_as_candidate` が None を返した
        場合でもこのテストが通ってしまう（外部レビューR02）。
        成功ケースは成功したことを断言する。
        """
        from src.strategy import model_store as ms
        from src.strategy import v2_training

        monkeypatch.setattr(
            v2_training, "MIN_RESOLVED_EVENTS", _TEST_MIN_RESOLVED_EVENTS)
        res = v2_training.train_v2(
            self._bars(), policy_conf=_policy_conf(), costs=_costs(),
            base_dir=str(tmp_path / "models"))

        assert res.skipped_reason is None, res.skipped_reason
        assert res.model_id is not None
        assert ms.candidate_dir(res.model_id, str(tmp_path / "models")).exists()
        # 現行は未昇格のまま
        assert ms.read_current(base_dir=str(tmp_path / "models")) is None

    def test_the_saved_candidate_can_be_loaded_and_predicts_the_same(
            self, isolated_db, tmp_path, monkeypatch):
        """保存物を読み直して、学習直後と同じ予測が出ること

        段階Cのラッパーは `booster_` も `save_model()` も持たない。
        保存側と学習側の型契約が合っていないと、ここで落ちる
        （外部レビューR02）。
        """
        import numpy as np
        import pandas as pd

        from src.strategy import model_store as ms
        from src.strategy import v2_training
        from src.strategy.indicators import FEATURE_COLS

        monkeypatch.setattr(
            v2_training, "MIN_RESOLVED_EVENTS", _TEST_MIN_RESOLVED_EVENTS)
        base = str(tmp_path / "models")
        res = v2_training.train_v2(
            self._bars(), policy_conf=_policy_conf(), costs=_costs(),
            base_dir=base)
        assert res.model_id is not None

        loaded, meta = ms.load_model(res.model_id, base_dir=base)
        assert list(meta.feature_cols) == list(FEATURE_COLS)
        assert meta.label_contract_id is not None

        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            rng.normal(0, 1, (5, len(FEATURE_COLS))), columns=list(FEATURE_COLS))
        proba = loaded.predict(X)
        assert len(proba) == 5
        assert np.all((proba >= 0.0) & (proba <= 1.0))

    def test_a_save_failure_is_reported_not_swallowed(
            self, isolated_db, tmp_path, monkeypatch):
        """保存に失敗したら model_id は None で理由が残る（成功と紛れない）

        `MIN_RESOLVED_EVENTS` を下げずに合成データ本来の51件のまま流すと、
        `_fit_candidate` に達する前の「イベント不足」で早期returnし、
        `_fit_candidate` を差し替えた意味が無いまま同じ assert が
        たまたま通ってしまう（保存失敗経路を検証したことにならない）。
        """
        from src.strategy import v2_training

        monkeypatch.setattr(
            v2_training, "MIN_RESOLVED_EVENTS", _TEST_MIN_RESOLVED_EVENTS)
        original = v2_training._fit_candidate

        class Unsavable:
            def predict_proba(self, X):
                return None

        v2_training._fit_candidate = lambda events, weights: Unsavable()
        try:
            res = v2_training.train_v2(
                self._bars(), policy_conf=_policy_conf(), costs=_costs(),
                base_dir=str(tmp_path / "models"))
        finally:
            v2_training._fit_candidate = original

        assert res.model_id is None
        assert res.skipped_reason is not None

    def test_records_the_dataset(self, isolated_db, tmp_path):
        from src.data.database import Dataset
        from src.strategy import v2_training

        res = v2_training.train_v2(
            self._bars(), policy_conf=_policy_conf(), costs=_costs(),
            base_dir=str(tmp_path / "models"))

        with get_session() as session:
            row = session.scalar(select(Dataset))
        assert row is not None
        assert row.dataset_id == res.dataset_id

    def test_records_metrics_with_the_engine_version(self, isolated_db, tmp_path, monkeypatch):
        from src.data.database import ModelMetrics
        from src.strategy import v2_training

        monkeypatch.setattr(
            v2_training, "MIN_RESOLVED_EVENTS", _TEST_MIN_RESOLVED_EVENTS)
        v2_training.train_v2(
            self._bars(), policy_conf=_policy_conf(), costs=_costs(),
            base_dir=str(tmp_path / "models"))

        with get_session() as session:
            rows = list(session.scalars(select(ModelMetrics)).all())
        assert rows
        assert rows[-1].engine_version == "v2"

    def test_window_sessions_is_recorded_in_metrics(
            self, isolated_db, tmp_path, monkeypatch):
        """`window_sessions` を渡すと `ModelMetrics.training_window_sessions`
        へそのまま記録される（段階F残課題6）。
        """
        from src.data.database import ModelMetrics
        from src.strategy import v2_training

        monkeypatch.setattr(
            v2_training, "MIN_RESOLVED_EVENTS", _TEST_MIN_RESOLVED_EVENTS)
        v2_training.train_v2(
            self._bars(), policy_conf=_policy_conf(), costs=_costs(),
            base_dir=str(tmp_path / "models"), window_sessions=250)

        with get_session() as session:
            rows = list(session.scalars(select(ModelMetrics)).all())
        assert rows
        assert rows[-1].training_window_sessions == 250

    def test_skips_when_there_are_too_few_events(self, isolated_db, tmp_path):
        """イベントが足りなければ学習せず理由を返す（例外にしない）"""
        from src.strategy import v2_training

        res = v2_training.train_v2(
            {"7203": _ohlcv(n=30)}, policy_conf=_policy_conf(), costs=_costs(),
            base_dir=str(tmp_path / "models"))
        assert res.model_id is None
        assert res.skipped_reason is not None

    def test_skip_due_to_too_few_events_still_records_a_metrics_row(
            self, isolated_db, tmp_path):
        """イベント不足でスキップしても、見送った履歴として`ModelMetrics`に
        1行だけ残す（段階F残課題6）。model_id/positive_rateはNoneのまま、
        n_samplesは決着イベント数、engine_versionは"v2"、triggerは
        呼び出し元が渡した値になる。
        """
        from src.data.database import ModelMetrics
        from src.strategy import v2_training

        res = v2_training.train_v2(
            {"7203": _ohlcv(n=30)}, policy_conf=_policy_conf(), costs=_costs(),
            base_dir=str(tmp_path / "models"), trigger="manual_test")
        assert res.model_id is None
        assert res.skipped_reason is not None

        with get_session() as session:
            rows = list(session.scalars(select(ModelMetrics)).all())
        assert len(rows) == 1
        row = rows[0]
        assert row.model_id is None
        assert row.positive_rate is None
        assert row.n_samples == res.n_resolved
        assert row.engine_version == "v2"
        assert row.trigger == "manual_test"

    def test_does_not_touch_the_legacy_model_file(self, isolated_db, tmp_path):
        """models/lgb_model.pkl を壊さない"""
        legacy = tmp_path / "models" / "lgb_model.pkl"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(b"legacy-bytes")

        from src.strategy import v2_training

        v2_training.train_v2(
            self._bars(), policy_conf=_policy_conf(), costs=_costs(),
            base_dir=str(tmp_path / "models"))
        assert legacy.read_bytes() == b"legacy-bytes"

    def test_training_failure_returns_a_reason(self, isolated_db, tmp_path, monkeypatch):
        """`_fit_candidate` の例外は握り潰されず理由が残ること

        （同じ理由で `MIN_RESOLVED_EVENTS` を下げないと `_fit_candidate` に
        到達せず、例外を差し替えた意味が無いまま通ってしまう）
        """
        from src.strategy import v2_training

        monkeypatch.setattr(
            v2_training, "MIN_RESOLVED_EVENTS", _TEST_MIN_RESOLVED_EVENTS)
        monkeypatch.setattr(
            v2_training, "_fit_candidate",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        res = v2_training.train_v2(
            self._bars(), policy_conf=_policy_conf(), costs=_costs(),
            base_dir=str(tmp_path / "models"))
        assert res.model_id is None
        assert res.skipped_reason is not None
