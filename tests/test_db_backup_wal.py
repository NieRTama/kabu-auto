"""日次DBバックアップの回帰テスト（WAL 未反映分の取りこぼし）

背景: backup() が shutil.copy2 で .db 本体だけをコピーしていたため、
      -wal に残った「コミット済みだが未チェックポイント」のデータが欠けていた。
      本番DBで実測したところ signals 43件 / ohlcv 5件が欠落していた。

本体プロセス(python main.py)はDB接続を開いたまま動き続けるので、
バックアップ時点で WAL に未反映分が残っているのが平常運転。
「接続を閉じればチェックポイントされる」ため、テストでも接続を開いたまま検証する。
"""
import sqlite3

import pytest

from src.data import database as db


def _count_rows(path) -> int:
    """バックアップ単体を開いて件数を数える（-wal を連れて行かない状態で読む）。"""
    con = sqlite3.connect(str(path))
    try:
        return con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    finally:
        con.close()


@pytest.fixture
def live_wal_db(tmp_path):
    """コミット済みだが WAL にしか無い行を持つDBを、接続を開いたまま渡す。"""
    db_path = tmp_path / "kabu_auto.db"
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE signals (id INTEGER PRIMARY KEY, symbol TEXT)")
    con.commit()
    # ここまでを本体ファイルへ反映し、以降の書き込みが WAL だけに残るようにする
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.executemany(
        "INSERT INTO signals (symbol) VALUES (?)", [(f"S{i}",) for i in range(43)]
    )
    con.commit()
    try:
        yield db_path, con
    finally:
        con.close()


def test_backup_includes_rows_that_live_only_in_wal(live_wal_db, tmp_path, monkeypatch):
    """バックアップに WAL 未反映分の43件が含まれること。"""
    db_path, _con = live_wal_db
    backup_dir = tmp_path / "backups"

    # 前提の確認: 本体ファイルだけを見ると、この43件はまだ見えない。
    # これが崩れるとテストが何も検証しなくなるので明示的に確かめる。
    import shutil

    plain = tmp_path / "plain_copy.db"
    shutil.copy2(db_path, plain)
    assert _count_rows(plain) == 0, "前提が崩れている: 43件が既に本体ファイルへ反映済み"

    monkeypatch.setattr(
        db.cfg,
        "get_section",
        lambda section: (
            {"db_path": str(db_path), "backup_dir": str(backup_dir)}
            if section == "data"
            else {}
        ),
    )

    db.backup()

    backups = list(backup_dir.glob("kabu_auto_*.db"))
    assert len(backups) == 1, f"バックアップが1本作られるはず: {backups}"
    assert _count_rows(backups[0]) == 43
