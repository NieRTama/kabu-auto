"""
スキーマバージョン管理のテスト（Phase 5 / P2-4）
"""
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


class TestSchemaVersion:
    def test_fresh_db_stamped_to_current(self, isolated_db):
        assert db.get_schema_version() == db.SCHEMA_VERSION

    def test_reinit_is_idempotent(self, isolated_db):
        v1 = db.get_schema_version()
        db.init()  # 再初期化しても壊れない・上がりすぎない
        assert db.get_schema_version() == v1

    def test_ordered_migration_runs_once(self, isolated_db, monkeypatch):
        """登録された順序付きマイグレーションが番号順に1度だけ適用されること"""
        calls = []
        monkeypatch.setattr(db, "SCHEMA_VERSION", db.SCHEMA_VERSION + 1)
        target = db.SCHEMA_VERSION  # = 元のSCHEMA_VERSION+1
        monkeypatch.setitem(db._MIGRATIONS, target, lambda conn: calls.append(target))
        # 旧バージョンのDBを模すため version を1つ戻す
        with db._engine.begin() as conn:
            from sqlalchemy import text
            conn.execute(text("UPDATE schema_version SET version=:v WHERE id=1"),
                         {"v": target - 1})
        db._run_migrations(db._engine)
        assert calls == [target]
        assert db.get_schema_version() == target
        # もう一度走らせても再適用されない
        db._run_migrations(db._engine)
        assert calls == [target]

    def test_get_version_without_engine_returns_zero(self):
        saved = db._engine
        db._engine = None
        try:
            assert db.get_schema_version() == 0
        finally:
            db._engine = saved
