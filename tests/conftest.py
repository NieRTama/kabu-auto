"""pytest 設定・共通フィクスチャ"""
import pytest

import src.core.config as cfg
import src.data.database as db


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    try:
        yield tmp_path
    finally:
        db._engine = None
        db._Session = None
