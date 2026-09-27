"""conftest.py の db.init ガードを検証する（2026-09-28 本番DB汚染事故の再発防止）。

db_path を本番のままにして db.init() を呼んでも、実際に開かれるDBが
本番パス data/kabu_auto.db ではないことを確認する。"""
from pathlib import Path

import src.core.config as cfg
import src.data.database as db


def test_db_init_never_opens_prod_db_path():
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = "data/kabu_auto.db"
    try:
        db.init()
        opened = Path(str(db._engine.url.database)).resolve()
        assert opened != Path("data/kabu_auto.db").resolve()
    finally:
        db._engine = None
        db._Session = None
