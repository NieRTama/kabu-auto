"""pytest 設定・共通フィクスチャ"""
import tempfile
from pathlib import Path

import pytest

import src.core.auth as auth
import src.core.config as cfg
import src.data.database as db


# db.get_session() は _Session が None のとき db.init() を遅延呼び出しする。
# isolated_db フィクスチャは teardown で db._engine/_Session を None に戻すため、
# その後に isolated_db を使わないテストや、テストから漏れた threading.Timer の
# スレッド（order_manager._timeout_cancel 等）がDBへ触ると、config.yaml の実値
# である本番DB (data/kabu_auto.db) をそのまま開いてしまう。2026-09-28、フルテスト
# 実行中にこの経路で本番 positions に架空の建玉（id=7, 7203, 100株）が書き込まれる
# 実害が発生した。get_session() はモジュールグローバルの db.init を名前で引くため、
# フィクスチャ（monkeypatch）ではなく、ここ（conftest のトップレベル・import時）で
# db.init 自体を永続的に差し替える。フィクスチャだと teardown 後に漏れたスレッドの
# 呼び出しには効かない。
_PROD_DB_PATH = Path("data/kabu_auto.db").resolve()
_real_db_init = db.init


def _guarded_db_init() -> None:
    conf = cfg.get_section("data")
    configured = Path(conf.get("db_path", "data/kabu_auto.db"))
    if configured.resolve() == _PROD_DB_PATH:
        conf["db_path"] = str(Path(tempfile.mkdtemp(prefix="kabu-test-")) / "test.db")
    _real_db_init()


db.init = _guarded_db_init


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
