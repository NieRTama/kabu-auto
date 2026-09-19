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
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
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
