"""
/api/model/latest がCV指標を持たない行（v2候補学習など）を飛ばして、
実際に発注経路で使われているlegacyモデルの最新学習結果を返すことの確認。

経緯: v2候補学習(_save_metrics)はcv_mean_accuracy等をNULLのまま保存する。
最新id優先で拾うと、候補学習の方が新しくても中身の無い行を返してしまい、
ダッシュボードのCV精度推移グラフがnull*100=0として誤表示されていた。
"""
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import src.dashboard.app as dash
from src.data.database import ModelMetrics, get_session


@pytest.fixture
def isolated_db(isolated_db):
    dash._auth_required = False
    return isolated_db


def test_latest_skips_null_cv_row(isolated_db):
    now = datetime.now()
    with get_session() as session:
        session.add(ModelMetrics(
            trained_at=now - timedelta(days=1),
            cv_mean_accuracy=0.55, cv_std_accuracy=0.03,
            n_samples=100, n_estimators=4, trigger="weekly_schedule",
        ))
        # v2候補学習: CV指標がNULLだが、こちらの方がidが新しい
        session.add(ModelMetrics(
            trained_at=now,
            n_samples=50, trigger="manual_workflow", engine_version="v2",
        ))
        session.commit()

    client = TestClient(dash.app)
    body = client.get("/api/model/latest").json()
    assert body["cv_mean_accuracy"] == 0.55
