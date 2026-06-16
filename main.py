"""
kabu-auto メインエントリポイント
起動: python main.py
ダッシュボード: http://localhost:8080
"""
import os
import sys
import threading
import time
from datetime import datetime, timedelta

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
from src.dashboard.app import (
    app as dashboard_app, set_order_manager, set_ml_retrain_fn, set_data_update_fn, update_status,
)


def main() -> None:
    cfg.load("config.yaml")
    log_setup.setup()
    db.init()

    trading_conf = cfg.get_section("trading")
    data_conf = cfg.get_section("data")
    dash_conf = cfg.get_section("dashboard")

    # ─── ライブモード起動確認 ──────────────────────────────────
    if trading_conf.get("mode", "paper") == "live":
        if os.environ.get("CONFIRM_LIVE_TRADING", "").lower() != "true":
            logger.error(
                "ライブモードを起動するには環境変数 CONFIRM_LIVE_TRADING=true が必要です。"
                "  例: CONFIRM_LIVE_TRADING=true python main.py"
            )
            sys.exit(1)
        logger.warning("【ライブモード】実際の資金を使用して取引します。")

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

    # ─── 起動時注文同期（ライブモードのみ）────────────────
    order_mgr.sync_on_startup()

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
                model = ml_model.train(combined_df, trigger="weekly_schedule")
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
        """15:35 に翌営業日の売買候補をスキャン。ペーパーモードは終値で即時シミュレート"""
        if TradingScheduler.is_maintenance_window():
            return
        logger.info("シグナルスキャン開始...")
        is_paper = trading_conf.get("mode", "paper") == "paper"
        for sym in watchlist:
            try:
                df = load_ohlcv(sym)
                if len(df) < 30:
                    continue
                sig = gen_signal(sym, df, model)
                _save_signal(sig)
                if sig.action not in ("BUY", "SELL"):
                    continue
                logger.info(f"シグナル: {sym} → {sig.action} (score={sig.combined_score:.2f})")
                if is_paper:
                    # ペーパーモード: 当日終値でシミュレート
                    close_price = float(df["close"].iloc[-1])
                    if sig.action == "BUY":
                        qty = risk.calc_position_size(sym, close_price, 500_000)
                        if qty > 0:
                            order_mgr.buy(sym, close_price, qty)
                    elif sig.action == "SELL":
                        qty = _get_position_qty(sym)
                        if qty > 0:
                            order_mgr.sell(sym, close_price, qty)
            except Exception as e:
                logger.error(f"シグナルスキャンエラー: {sym} {e}")

    def morning_execution():
        """9:05 に前日のBUY/SELLシグナルを元に発注（ライブモードのみ）"""
        if trading_conf.get("mode", "paper") != "live":
            return
        if not TradingScheduler.is_market_open():
            return
        cutoff = datetime.now() - timedelta(hours=20)
        with get_session() as session:
            signals = session.scalars(
                select(Signal)
                .where(Signal.action.in_(["BUY", "SELL"]), Signal.generated_at >= cutoff)
                .order_by(Signal.generated_at.desc())
            ).all()
            seen: set = set()
            pending: list = []
            for s in signals:
                if s.symbol not in seen:
                    seen.add(s.symbol)
                    pending.append(s)

        if not pending:
            return

        buy_signals = [s for s in pending if s.action == "BUY"]
        sell_signals = [s for s in pending if s.action == "SELL"]

        # ── SELL シグナル: 保有ポジションがあれば売る ─────────────
        for sig in sell_signals:
            try:
                qty = _get_position_qty(sig.symbol)
                if qty <= 0:
                    continue
                board = client.get_board(sig.symbol)
                price = board.get("CurrentPrice") or board.get("Buy1", {}).get("Price", 0)
                if not price:
                    continue
                order_mgr.sell(sig.symbol, float(price), qty)
                logger.info(f"朝売り発注: {sig.symbol} {qty}株 @{price:.0f}円")
            except Exception as e:
                logger.error(f"朝売り発注失敗: {sig.symbol} {e}")

        # ── BUY シグナル: 余力を確認して買う ──────────────────────
        if not buy_signals:
            return
        try:
            wallet = client.get_wallet()
            cash = float(wallet.get("StockAccountWallet", 0))
        except Exception as e:
            logger.error(f"余力取得失敗: {e}")
            return
        for sig in buy_signals:
            try:
                board = client.get_board(sig.symbol)
                price = board.get("CurrentPrice") or board.get("Sell1", {}).get("Price", 0)
                if not price:
                    continue
                qty = risk.calc_position_size(sig.symbol, float(price), cash)
                if qty > 0:
                    order_mgr.buy(sig.symbol, float(price), qty)
                    logger.info(f"朝買い発注: {sig.symbol} {qty}株 @{price:.0f}円")
            except Exception as e:
                logger.error(f"朝買い発注失敗: {sig.symbol} {e}")

    set_ml_retrain_fn(ml_retrain)
    set_data_update_fn(data_update)

    scheduler.register("token_refresh", token_refresh)
    scheduler.register("data_update", data_update)
    scheduler.register("db_backup", db_backup)
    scheduler.register("ml_retrain", ml_retrain)
    scheduler.register("stop_loss_check", stop_loss_check)
    scheduler.register("signal_scan", signal_scan)
    scheduler.register("morning_execution", morning_execution)

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
            "host": dash_conf.get("host", "127.0.0.1"),
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
