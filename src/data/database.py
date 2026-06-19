"""SQLiteデータベース管理（WALモード有効）"""
import shutil
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger
from sqlalchemy import (
    Column, Date, DateTime, Float, Index, Integer, String, Text,
    create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.core import clock
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
    sector = Column(String(50))  # 発注時点のセクター（新規銘柄はPosition未作成のため、
    # 未約定BUYのセクター引当をPosition経由でなくここから直接解決できるようにする）
    quantity = Column(Integer, nullable=False)  # 発注数量
    price = Column(Float)  # 発注時の指値（成行は0）。実約定価格は filled_price を使う
    filled_price = Column(Float)  # 実約定単価（約定イベント/注文照会から取得）
    filled_quantity = Column(Integer)  # 実約定数量（部分約定時は quantity 未満）
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
    # 日時はプロジェクト全体で JST naive に統一する（scheduler・order・ml も datetime.now()=JST）。
    # 旧 datetime.utcnow と混在すると morning_execution の cutoff 比較が9時間ずれて
    # 前日シグナルを取りこぼすため、必ず datetime.now を使うこと。
    opened_at = Column(DateTime, default=clock.now)
    updated_at = Column(DateTime, default=clock.now, onupdate=clock.now)


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    generated_at = Column(DateTime, default=clock.now)  # JST naive（morning_executionのcutoffと統一）
    rule_score = Column(Float)
    ml_score = Column(Float)
    combined_score = Column(Float)
    action = Column(String(10))  # "BUY", "SELL", "HOLD"


class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    id = Column(Integer, primary_key=True)
    trained_at = Column(DateTime)
    cv_mean_accuracy = Column(Float)
    cv_std_accuracy = Column(Float)
    n_samples = Column(Integer)
    n_estimators = Column(Integer)
    feature_importances_json = Column(Text)  # JSON: {"feature": importance_score}
    trigger = Column(String(20), default="manual")  # "weekly_schedule" / "manual"


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    initial_capital = Column(Float)
    final_capital = Column(Float)
    total_return = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    win_rate = Column(Float)
    trade_count = Column(Integer)
    use_ml = Column(Integer, default=0)
    created_at = Column(DateTime)
    equity_curve_json = Column(Text)  # JSON: [{"date": "YYYY-MM-DD", "equity": float}]
    buy_threshold = Column(Float)   # この実行で実際に使われた買い閾値（探索的に上書きされた場合も記録）
    sell_threshold = Column(Float)  # この実行で実際に使われた売り閾値


class BacktestTradeRecord(Base):
    __tablename__ = "backtest_trades"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, nullable=False)
    symbol = Column(String(10))
    entry_date = Column(Date)
    entry_price = Column(Float)
    exit_date = Column(Date)
    exit_price = Column(Float)
    quantity = Column(Integer)
    pnl = Column(Float)
    exit_reason = Column(String(30))  # STOP_LOSS / SIGNAL_SELL / END_OF_PERIOD


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
    _migrate_add_missing_columns(_engine)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    logger.info(f"DB初期化完了: {db_path}")


def _migrate_add_missing_columns(engine) -> None:
    """create_all はテーブル新規作成のみ行うため、既存DBに後から追加したカラムを
    SQLiteの ALTER TABLE ADD COLUMN で補う簡易マイグレーション。"""
    with engine.connect() as conn:
        for table in Base.metadata.tables.values():
            existing = {row[1] for row in conn.execute(text(f'PRAGMA table_info("{table.name}")'))}
            for col in table.columns:
                if col.name not in existing:
                    col_type = col.type.compile(engine.dialect)
                    conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col_type}'))
                    logger.info(f"DBマイグレーション: {table.name}.{col.name} ({col_type}) を追加")
        conn.commit()


@contextmanager
def get_session() -> Iterator[Session]:
    if _Session is None:
        init()
    session = _Session()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


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
