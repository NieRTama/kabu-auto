"""SQLiteデータベース管理（WALモード有効）"""
import shutil
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger
from sqlalchemy import (
    Column, Date, DateTime, Float, Index, Integer, String, Text,
    create_engine, select, text,
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


class OrderIntent(Base):
    """発注の「意図」（Phase 5 / 4.2）。

    戦略が「なぜ・何を」やりたいかを表す層。1つの意図から複数のBrokerOrder（trades行）が
    生まれうる（タイムアウトキャンセル後の再発注・部分約定後の追撃等）。`Trade.intent_id` で
    紐づく。意図そのものは取消・拒否されても消えない（再現性のため）。
    """
    __tablename__ = "order_intents"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=clock.now)
    symbol = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)         # "BUY" / "SELL"
    target_quantity = Column(Integer, nullable=False)
    order_type = Column(String(10))                    # "LIMIT" / "MARKET"
    limit_price = Column(Float)                        # 指値（成行はNone/0）
    sector = Column(String(50))
    rationale = Column(Text)    # シグナルスコア・損切り理由等（7.6 と同内容をここが正本として持つ）
    source = Column(String(20))  # signal_scan / morning_execution / stop_loss / emergency / manual / approval / backfill
    mode = Column(String(10))    # paper / live / dry_run / semi_live
    status = Column(String(12), default="PENDING")  # PENDING/SUBMITTED/PARTIAL/COMPLETED/CANCELLED/REJECTED


class Trade(Base):
    """ブローカーへ送った1注文（= BrokerOrder。Phase 5 / 4.2 で意図(OrderIntent)・
    約定明細(Fill)から分離した）。"""
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    intent_id = Column(Integer)  # OrderIntent.id（紐づく意図。backfill対象の旧データはマイグレーションで補完）
    order_id = Column(String(50), unique=True)
    symbol = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)  # "BUY" or "SELL"
    sector = Column(String(50))  # 発注時点のセクター（新規銘柄はPosition未作成のため、
    # 未約定BUYのセクター引当をPosition経由でなくここから直接解決できるようにする）
    quantity = Column(Integer, nullable=False)  # 発注数量
    price = Column(Float)  # 発注時の指値（成行は0）。実約定価格は filled_price を使う
    # filled_price/filled_quantity/pnl は Fill から積み上げる派生（非正規化）列。
    # 既存の読み手（ダッシュボード・リスク管理・ジャーナル等）はこの列を読むだけで動くよう、
    # _apply_fill() がFill確定ごとにここへロールアップする。
    filled_price = Column(Float)  # 実約定単価（Fillの出来高加重平均=VWAP）
    filled_quantity = Column(Integer)  # 実約定数量（部分約定時は quantity 未満）
    filled_at = Column(DateTime)
    status = Column(String(20), default="PENDING")
    pnl = Column(Float)  # SELL: FIFOで確定した実現損益の合計（複数Fillに分かれた場合は加算）
    note = Column(Text)        # 利用者が後から書く自由記述メモ（取引ジャーナル）
    rationale = Column(Text)   # 発注時に自動記録する売買根拠（intentからの複製。既存リーダー互換のため維持）


class OrderApproval(Base):
    """semi_live モードの発注承認キュー（Phase 3 / 7.3）。

    計画注文をいったんここに PENDING で積み、ダッシュボードで人が承認すると
    実APIへ発注して APPROVED（resulting_order_id に発注ID）、却下すると REJECTED にする。
    """
    __tablename__ = "order_approvals"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=clock.now)
    symbol = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)        # "BUY" / "SELL"
    order_type = Column(String(10), nullable=False)  # "LIMIT" / "MARKET"
    price = Column(Float)                             # 指値（成行は0/None）
    quantity = Column(Integer, nullable=False)
    sector = Column(String(50))
    status = Column(String(12), default="PENDING")   # PENDING / APPROVED / REJECTED
    decided_at = Column(DateTime)
    resulting_order_id = Column(String(50))          # 承認実行で発注した注文ID
    intent_id = Column(Integer)                       # 紐づくOrderIntent.id（承認時にTradeへ引き継ぐ）
    rationale = Column(Text)                          # 発注根拠（承認後にTradeへ引き継ぐ。7.6）
    note = Column(Text)


class Fill(Base):
    """1回の約定明細（Phase 5 / 4.2）。BUYの約定はFIFOロット台帳の単位にもなる。

    BUY: `remaining_qty` がそのロットの未消費株数（0になったら売り切り済み）。
    SELL: `realized_pnl` がこの約定でFIFO消費した分の確定損益。
    `src/execution/lots.py` がこのテーブルを使ってFIFO消費・Position再構成を行う。
    """
    __tablename__ = "fills"
    id = Column(Integer, primary_key=True)
    broker_order_id = Column(Integer, nullable=False)  # Trade.id
    symbol = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)  # "BUY" / "SELL"
    fill_qty = Column(Integer, nullable=False)
    fill_price = Column(Float, nullable=False)
    filled_at = Column(DateTime, nullable=False)
    source = Column(String(20))  # ws / reconcile / paper / backfill
    remaining_qty = Column(Integer)   # BUYのみ: FIFO未消費株数
    realized_pnl = Column(Float)      # SELLのみ: この約定での確定損益

    __table_args__ = (
        Index("ix_fills_symbol_side_filled_at", "symbol", "side", "filled_at"),
        Index("ix_fills_broker_order_id", "broker_order_id"),
    )


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
    archived = Column(Integer, default=0)  # 1=アーカイブ済み（一覧から除外。履歴は保持）


class SchemaVersion(Base):
    """DBスキーマのバージョンを記録する1行テーブル（P2-4）。

    既存の `_migrate_add_missing_columns()` は「不足カラムの追加」という冪等な操作のみで、
    既存データの変換を伴う順序付きマイグレーションは表現できない。将来そうした移行が必要に
    なったとき（列の意味変更・データ移送・テーブル再編など）に、適用済みバージョンを基準に
    一度だけ順番に流せるよう、現在のスキーマバージョンを永続化しておく。
    """
    __tablename__ = "schema_version"
    id = Column(Integer, primary_key=True)  # 常に 1（単一行）
    version = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, default=clock.now, onupdate=clock.now)


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


# 現在のスキーマバージョン。順序付きマイグレーションを追加するたびに +1 する。
# v1 = schema_version 導入時点のベースライン（既存テーブル群＋additive列追加で表現できる範囲）。
# v2 = OrderIntent/Fill分離のバックフィル（Phase 5 / 4.2。下記 _migrate_v2_order_model_backfill）。
SCHEMA_VERSION = 2


def _backfill_intent_status(trade_status: str) -> str:
    """旧Trade.statusから、バックフィルで生成するOrderIntent.statusを決める。"""
    if trade_status == "REJECTED":
        return "REJECTED"
    if trade_status in ("FILLED", "DRY_RUN"):
        return "COMPLETED"
    if trade_status == "PARTIALLY_FILLED":
        return "PARTIAL"
    if trade_status in ("CANCELLED", "CANCEL_FAILED"):
        return "CANCELLED"
    return "SUBMITTED"


def _migrate_v2_order_model_backfill(conn) -> None:
    """v2: 既存 trades から OrderIntent・Fill をバックフィルし、FIFOで実現損益・Positionを
    遡及再計算する（Phase 5 / 4.2: OrderIntent/BrokerOrder/Fill分離）。

    1. 全TradeにOrderIntentを生成して紐付ける（intent_idが未設定のもののみ。履歴の追跡性確保）。
    2. 約定済み(FILLED/PARTIALLY_FILLED)のTradeからFillを生成する。BUYは新規ロットとして
       積み、SELLは `src/execution/lots.py` のFIFOロジックで既存ロットを古い順に消費する
       （`fn(conn)` で受け取る Connection は `engine.begin()` の進行中トランザクションに
       紐づくため、`Session(bind=conn)` でこれに合流させ、flush のみ行い commit はしない
       ＝ 外側の `_run_migrations` のトランザクション境界に委ねる）。
    3. SELLの実現損益(Trade.pnl)はFIFO消費結果で上書きする（平均単価会計→FIFOロット会計へ
       遡及再計算。ユーザー確認済みの方針）。
    4. Fillが生成された全symbolについて、最終的な残存ロットからPositionを再構成する。

    冪等性: 既にintent_idが設定済みのTrade、既にFillが存在するTradeはスキップするため、
    複数回実行しても安全（起動時に毎回 schema_version を見て1回だけ走るが、保険として）。
    """
    from src.execution import lots

    session = Session(bind=conn)

    # ─── 1. 全TradeへOrderIntentを生成 ─────────────────────────────────
    trades = session.scalars(select(Trade)).all()
    for t in trades:
        if t.intent_id is not None:
            continue
        order_type = "MARKET" if (t.side == "SELL" and not t.price) else "LIMIT"
        mode = (
            "paper" if (t.order_id or "").startswith("PAPER-") else
            "dry_run" if (t.order_id or "").startswith("DRYRUN-") else
            "live"
        )
        intent = OrderIntent(
            symbol=t.symbol, side=t.side, target_quantity=t.quantity,
            order_type=order_type, limit_price=t.price if order_type == "LIMIT" else None,
            sector=t.sector, rationale=t.rationale, source="backfill", mode=mode,
            status=_backfill_intent_status(t.status),
            created_at=t.filled_at or clock.now(),
        )
        session.add(intent)
        session.flush()
        t.intent_id = intent.id

    # ─── 2-3. 約定済みTradeからFillを生成しFIFOで遡及再計算 ──────────────
    filled_trades = [
        t for t in trades
        if t.status in ("FILLED", "PARTIALLY_FILLED") and (t.filled_quantity or 0) > 0
    ]
    has_fill = {
        row[0] for row in session.execute(select(Fill.broker_order_id)).all()
    }
    touched_symbols: set = set()

    # BUYを先にすべて積む（SELLのFIFO消費がこれらのロットを参照するため）
    buys = sorted(
        (t for t in filled_trades if t.side == "BUY" and t.id not in has_fill),
        key=lambda t: (t.filled_at or clock.now(), t.id),
    )
    for t in buys:
        price = t.filled_price if t.filled_price else t.price
        lots.record_buy_fill(session, t.id, t.symbol, t.filled_quantity, price,
                             t.filled_at or clock.now(), source="backfill")
        touched_symbols.add(t.symbol)

    sells = sorted(
        (t for t in filled_trades if t.side == "SELL" and t.id not in has_fill),
        key=lambda t: (t.filled_at or clock.now(), t.id),
    )
    for t in sells:
        price = t.filled_price if t.filled_price else t.price
        _, realized, _consumed = lots.consume_fifo(
            session, t.id, t.symbol, t.filled_quantity, price,
            t.filled_at or clock.now(), source="backfill",
        )
        t.pnl = realized  # 旧・平均単価会計の値をFIFO再計算で上書き
        touched_symbols.add(t.symbol)

    # ─── 4. Positionを最終残存ロットから再構成 ─────────────────────────
    for symbol in touched_symbols:
        lots.rebuild_position(session, symbol)

    session.flush()


# version -> 適用関数 fn(conn) の登録簿。additive な列追加では表現できない
# 順序付きマイグレーション（データ変換等）をここへ登録する。
_MIGRATIONS: dict = {
    2: _migrate_v2_order_model_backfill,
}

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
    _run_migrations(_engine)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    logger.info(f"DB初期化完了: {db_path}")


def _run_migrations(engine) -> None:
    """スキーマ移行を実行する。

    1. 不足カラムの追加（冪等。既存の additive マイグレーション）
    2. schema_version を基準に、未適用の順序付きマイグレーションを番号順に1度だけ適用
    3. schema_version を最新へ更新
    """
    _migrate_add_missing_columns(engine)
    with engine.begin() as conn:
        current = conn.execute(text("SELECT version FROM schema_version WHERE id=1")).scalar()
        if current is None:
            conn.execute(text("INSERT INTO schema_version (id, version) VALUES (1, 0)"))
            current = 0
        for v in range(current + 1, SCHEMA_VERSION + 1):
            fn = _MIGRATIONS.get(v)
            if fn is not None:
                logger.info(f"スキーマ移行 v{v} を適用します")
                fn(conn)
        if current < SCHEMA_VERSION:
            conn.execute(
                text("UPDATE schema_version SET version=:v, updated_at=:t WHERE id=1"),
                {"v": SCHEMA_VERSION, "t": clock.now()},
            )
            logger.info(f"スキーマバージョン: {current} → {SCHEMA_VERSION}")


def get_schema_version() -> int:
    """現在記録されているスキーマバージョンを返す（未初期化なら0）。"""
    if _engine is None:
        return 0
    with _engine.connect() as conn:
        return conn.execute(text("SELECT version FROM schema_version WHERE id=1")).scalar() or 0


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
