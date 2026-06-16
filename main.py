"""
kabu-auto メインエントリポイント
起動: python main.py
ダッシュボード: http://localhost:8080
"""
import threading
import time

import pandas as pd
import uvicorn
from loguru import logger
from sqlalchemy import select

from src.core import config as cfg, logger as log_setup
from src.core.alerts import alert
from src.core.scheduler import TradingScheduler
from src.api.kabu_client import KabuClient
from src.data import database as db
from src.data.database import Position, Signal, get_session
from src.data.market_data import load_ohlcv, update_symbol
from src.execution.order_manager import OrderManager
from src.risk.manager import RiskManager
from src.strategy import ml_model
from src.strategy.signal import Signal as TradeSignal, generate as gen_signal
from src.dashboard.app import app as dashboard_app, set_order_manager, update_status


def main() -> None:
    cfg.load("config.yaml")
    log_setup.setup()
    db.init()

    trading_conf = cfg.get_section("trading")
    data_conf = cfg.get_section("data")
    dash_conf = cfg.get_section("dashboard")

    client = KabuClient()
    risk = RiskManager()
    order_mgr = OrderManager(client, risk)
    scheduler = TradingScheduler()

    set_order_manager(order_mgr)

    # ─── 初回トークン取得 ─────────────────────────────────
    try:
        client.refresh_token()
    except Exception as e:
        logger.warning(f"kabuステーション接続失敗（ペーパーモードで継続）: {e}")

    # ─── MLモデルロード ────────────────────────────────────
    model = ml_model.load()
    if model is None:
        logger.info("学習済みモデルなし。十分なデータが揃い次第 /retrain を実行してください。")

    # ─── ウォッチリスト（設定から読み込み）────────────────
    watchlist: list[str] = trading_conf.get("watchlist", [])

    # ─── スケジューラのコールバック登録 ──────────────────────

    def token_refresh():
        try:
            client.refresh_token()
        except Exception as e:
            alert("APIトークン更新失敗", str(e))

    def data_update():
        years = data_conf.get("history_years", 3)
        for sym in watchlist:
            try:
                update_symbol(sym, years=years)
            except Exception as e:
                logger.error(f"データ更新失敗: {sym} {e}")

    def db_backup():
        try:
            db.backup()
        except Exception as e:
            logger.error(f"バックアップ失敗: {e}")

    def ml_retrain():
        nonlocal model
        logger.info("MLモデル週次再学習を開始...")
        combined_df = None
        for sym in watchlist:
            try:
                df = load_ohlcv(sym)
                if len(df) < 200:
                    continue
                combined_df = df if combined_df is None else pd.concat([combined_df, df])
            except Exception as e:
                logger.error(f"データ読み込み失敗: {sym} {e}")
        if combined_df is not None and len(combined_df) >= 200:
            try:
                model = ml_model.train(combined_df)
            except Exception as e:
                logger.error(f"再学習失敗: {e}")

    def stop_loss_check():
        if not TradingScheduler.is_market_open():
            return
        for sym in watchlist:
            try:
                board = client.get_board(sym)
                price = board.get("CurrentPrice", 0)
                if price and risk.should_stop_loss(sym, price):
                    logger.warning(f"損切り発動: {sym}")
                    order_mgr.sell(sym, price, _get_position_qty(sym))
                    alert("損切り実行", f"{sym} @{price:.0f}円")
            except Exception as e:
                logger.error(f"損切りチェックエラー: {sym} {e}")

    def signal_scan():
        """15:35に翌営業日の売買候補をスキャン"""
        if TradingScheduler.is_maintenance_window():
            return
        logger.info("シグナルスキャン開始...")
        for sym in watchlist:
            try:
                df = load_ohlcv(sym)
                if len(df) < 30:
                    continue
                sig = gen_signal(sym, df, model)
                _save_signal(sig)
                if sig.action in ("BUY", "SELL"):
                    logger.info(f"シグナル: {sym} → {sig.action} (score={sig.combined_score:.2f})")
            except Exception as e:
                logger.error(f"シグナルスキャンエラー: {sym} {e}")

    scheduler.register("token_refresh", token_refresh)
    scheduler.register("data_update", data_update)
    scheduler.register("db_backup", db_backup)
    scheduler.register("ml_retrain", ml_retrain)
    scheduler.register("stop_loss_check", stop_loss_check)
    scheduler.register("signal_scan", signal_scan)

    # ─── WebSocket 開始 ────────────────────────────────────
    client.start_websocket(on_order_event=order_mgr.on_order_event)

    # ─── スケジューラ起動 ───────────────────────────────────
    scheduler.start()
    update_status(running=True, ws_connected=True, mode=trading_conf.get("mode", "paper"))

    # ─── ダッシュボード起動（別スレッド）────────────────────
    dash_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": dashboard_app,
            "host": dash_conf.get("host", "0.0.0.0"),
            "port": dash_conf.get("port", 8080),
            "log_level": "warning",
        },
        daemon=True,
    )
    dash_thread.start()
    logger.info(f"ダッシュボード起動: http://localhost:{dash_conf.get('port', 8080)}")

    # ─── メインループ ────────────────────────────────────────
    logger.info("kabu-auto 起動完了。Ctrl+C で終了。")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("シャットダウン中...")
        scheduler.stop()
        client.stop_websocket()
        update_status(running=False, ws_connected=False, mode=trading_conf.get("mode", "paper"))
        logger.info("終了しました")


def _get_position_qty(symbol: str) -> int:
    with get_session() as session:
        pos = session.scalar(select(Position).where(Position.symbol == symbol))
    return pos.quantity if pos else 0


def _save_signal(sig: TradeSignal) -> None:
    with get_session() as session:
        session.add(Signal(
            symbol=sig.symbol,
            rule_score=sig.rule_score,
            ml_score=sig.ml_score,
            combined_score=sig.combined_score,
            action=sig.action,
        ))
        session.commit()


if __name__ == "__main__":
    main()
