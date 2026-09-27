"""pytest 設定・共通フィクスチャ"""
import pytest

import src.core.auth as auth
import src.core.config as cfg
import src.data.database as db


@pytest.fixture(autouse=True, scope="session")
def _fast_password_hashing():
    """テストではPBKDF2の反復回数を1回に落とす（本番は20万回のまま）。

    本番同等の反復回数だとパスワード関連のテストだけで1件あたり1秒前後かかり、
    全体で有意な時間になる（2026-09-27、16件で約20秒と判明）。反復回数は
    auth.verify() がハッシュごとに保存済みの値を読むため、create_user() 実行時点の
    _ITERATIONS だけ差し替えれば足りる（本番のauth.jsonやverify()の挙動には影響しない）。
    """
    auth._ITERATIONS = 1


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
