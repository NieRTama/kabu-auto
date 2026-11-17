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

# 2026-11-18: テスト収集中に scripts/ のモジュールが import 時に load_dotenv() しており、
# 本物の DISCORD_WEBHOOK_URL が環境変数に入って、テストの「朝買い発注 7203 @1,000円」等が
# 本番の Discord へ投稿された。どのモジュールが .env を読んでも外へ出ないよう、
# 各テストの開始時に .env 由来の秘密・通知先を環境変数から外す（テスト側の setenv はこの後なので有効）。
_SECRET_ENV = (
    "KABU_API_PASSWORD", "KABU_API_PASSWORD_LIVE", "KABU_API_PASSWORD_VERIFY",
    "KABU_DASHBOARD_TOKEN", "DISCORD_WEBHOOK_URL", "DISCORD_REPORT_WEBHOOK_URL",
    "DISCORD_BOT_TOKEN", "DISCORD_BOT_CHANNEL_ID", "DISCORD_ALLOWED_USER_ID",
    "X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET", "JQUANTS_API_KEY",
)


@pytest.fixture(autouse=True)
def _no_real_secrets(monkeypatch):
    for name in _SECRET_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True, scope="session")
def _fast_password_hashing():
    """テストではPBKDF2の反復回数を1回に落とす（本番は20万回のまま）。

    本番同等の反復回数だとパスワード関連のテストだけで1件あたり1秒前後かかり、
    全体で有意な時間になる（2026-09-27、16件で約20秒と判明）。反復回数は
    auth.verify() がハッシュごとに保存済みの値を読むため、create_user() 実行時点の
    _ITERATIONS だけ差し替えれば足りる（本番のauth.jsonやverify()の挙動には影響しない）。
    """
    auth._ITERATIONS = 1


@pytest.fixture(autouse=True)
def _no_stop_loss_pause(tmp_path, monkeypatch):
    """本番の data/stop_loss.pause が置かれていても、テストの損切り判定に影響させない。"""
    import src.risk.manager as risk_mod
    monkeypatch.setattr(risk_mod, "STOP_LOSS_PAUSE", tmp_path / "stop_loss.pause")


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
