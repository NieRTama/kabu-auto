"""SQLiteデータベース管理（WALモード有効）"""
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from sqlalchemy import (
    Column, Date, DateTime, Float, Index, Integer, String, Text,
    create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.core import config as cfg


class Base(DeclarativeBase):
    pass


class OHLCV(Base):
    __tablename__ = "ohlcv"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    adjusted_close = Column(Float)  # 権利修正後終値

    __table_args__ = (Index("ix_ohlcv_symbol_date", "symbol", "date", unique=True),)


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), unique=True)
    symbol = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)  # "BUY" or "SELL"
    quantity = Column(Integer, nullable=False)
    price = Column(Float)
    filled_at = Column(DateTime)
    status = Column(String(20), default="PENDING")
    pnl = Column(Float)
    note = Column(Text)


class Position(Base):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, unique=True)
    quantity = Column(Integer, nullable=False, default=0)
    avg_cost = Column(Float, nullable=False, default=0.0)
    sector = Column(String(50))
    opened_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)
    rule_score = Column(Float)
    ml_score = Column(Float)
    combined_score = Column(Float)
    action = Column(String(10))  # "BUY", "SELL", "HOLD"


_engine = None
_Session: Optional[sessionmaker] = None


def init() -> None:
    global _engine, _Session
    conf = cfg.get_section("data")
    db_path = Path(conf.get("db_path", "data/kabu_auto.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    with _engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.commit()

    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    logger.info(f"DB初期化完了: {db_path}")


def get_session() -> Session:
    if _Session is None:
        init()
    return _Session()


def backup() -> None:
    """日次バックアップ"""
    conf = cfg.get_section("data")
    db_path = Path(conf.get("db_path", "data/kabu_auto.db"))
    backup_dir = Path(conf.get("backup_dir", "data/backups"))
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / f"kabu_auto_{date.today().isoformat()}.db"
    shutil.copy2(db_path, dst)
    logger.info(f"DB バックアップ完了: {dst}")
    _cleanup_old_backups(backup_dir, keep_days=30)


def _cleanup_old_backups(backup_dir: Path, keep_days: int = 30) -> None:
    files = sorted(backup_dir.glob("kabu_auto_*.db"))
    if len(files) > keep_days:
        for f in files[:-keep_days]:
            f.unlink()
