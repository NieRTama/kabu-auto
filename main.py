"""
kabu-auto メインエントリポイント
起動: python main.py
ダッシュボード: http://localhost:8080

このファイルは依存を結線してスケジューラにジョブを登録するだけの薄い composition root。
データ更新・ML再学習・損切り監視・シグナルスキャン・朝の発注の各ロジックは
src/services/trading.py の TradingServices に切り出している。
"""
import os
import sys
import threading
import time

import uvicorn
from dotenv import load_dotenv
from loguru import logger

from src.core import (
    config as cfg, logger as log_setup, watchlist as watchlist_store,
    risk_profile as risk_profile_store,
)
from src.core.alerts import alert
from src.core.netutil import is_port_available
from src.core.scheduler import TradingScheduler
from src.api.kabu_client import KabuClient
from src.data import database as db
from src.execution.order_manager import OrderManager
from src.risk.manager import RiskManager
from src.strategy import ml_model
from src.services.trading import TradingServices, _select_latest_signals  # noqa: F401  (テスト互換のため再エクスポート)
from src.dashboard.app import (
    app as dashboard_app, set_order_manager, set_ml_retrain_fn, set_data_update_fn, update_status,
    _get_lan_ip,
)


def main() -> None:
    load_dotenv()  # .env の KABU_API_PASSWORD などを読み込む（任意・存在しなければ無視）
    cfg.load("config.yaml")
    watchlist_store.load("watchlists.json")  # 旧 watchlist.json があれば自動移行される
    risk_profile_store.load("risk_profile.json")  # アクティブなリスクプロファイルを config に適用
    log_setup.setup()
    db.init()

    trading_conf = cfg.get_section("trading")
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
    risk.restore_daily_state()  # 再起動しても当日の損失上限・注文数カウンタを引き継ぐ
    order_mgr = OrderManager(client, risk)
    scheduler = TradingScheduler()

    set_order_manager(order_mgr)

    # ─── 初回トークン取得 ─────────────────────────────────
    try:
        client.refresh_token()
    except Exception as e:
        if trading_conf.get("mode", "paper") == "live":
            # ライブモードでAPI接続できないまま起動を続けると、口座状態を把握できない
            # まま発注ロジックだけが動く危険な状態になる（fail-closed）
            logger.critical(f"ライブモードでkabuステーション接続に失敗。起動を中断します: {e}")
            sys.exit(1)
        logger.warning(f"kabuステーション接続失敗（ペーパーモードで継続）: {e}")

    # ─── 起動時注文同期（ライブモードのみ）────────────────
    order_mgr.sync_on_startup()

    # ─── MLモデルロード ────────────────────────────────────
    model = ml_model.load()
    if model is None:
        logger.info("学習済みモデルなし。十分なデータが揃い次第 /retrain を実行してください。")

    # ─── 取引サービス（スケジューラジョブの実体）────────────
    # ウォッチリストは各ジョブ内で watchlist_store から毎回最新を取得するため、
    # GUIでの追加・削除がプロセス再起動なしで次回実行から反映される。
    services = TradingServices(client, risk, order_mgr, model)

    # ─── インフラ系の小ジョブ（composition root に置く）──────
    def token_refresh():
        try:
            client.refresh_token()
        except Exception as e:
            alert("APIトークン更新失敗", str(e))

    def db_backup():
        try:
            db.backup()
        except Exception as e:
            logger.error(f"バックアップ失敗: {e}")

    set_ml_retrain_fn(services.ml_retrain)
    set_data_update_fn(services.data_update)

    # ─── スケジューラのコールバック登録 ──────────────────────
    scheduler.register("risk_reset", risk.reset_daily_counters)
    scheduler.register("token_refresh", token_refresh)
    scheduler.register("data_update", services.data_update)
    scheduler.register("db_backup", db_backup)
    scheduler.register("ml_retrain", services.ml_retrain)
    scheduler.register("stop_loss_check", services.stop_loss_check)
    scheduler.register("signal_scan", services.signal_scan)
    scheduler.register("morning_execution", services.morning_execution)
    scheduler.register("reconcile_orders", services.reconcile_orders)

    # ─── WebSocket 開始 ────────────────────────────────────
    client.start_websocket(on_order_event=order_mgr.on_order_event)

    # ─── スケジューラ起動 ───────────────────────────────────
    scheduler.start()
    update_status(running=True, ws_connected=True, mode=trading_conf.get("mode", "paper"))

    # ─── ダッシュボード起動（別スレッド）────────────────────
    dash_host = dash_conf.get("host", "127.0.0.1")
    dash_port = dash_conf.get("port", 8080)
    if not is_port_available(dash_host, dash_port):
        msg = f"ダッシュボードのポート {dash_host}:{dash_port} は既に使用中です"
        if trading_conf.get("mode", "paper") == "live":
            # ライブモードでダッシュボードが起動できないと、発注ロジックは動くのに
            # 状態の監視・緊急決済操作ができない危険な状態になるため起動を中断する
            logger.critical(f"{msg}。ライブモードのため起動を中断します。")
            sys.exit(1)
        logger.warning(f"{msg}。ダッシュボードが起動できない可能性があります。")

    dash_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": dashboard_app,
            "host": dash_host,
            "port": dash_port,
            "log_level": "warning",
        },
        daemon=True,
    )
    dash_thread.start()
    logger.info(f"ダッシュボード起動: http://localhost:{dash_port}")
    if dash_host == "0.0.0.0":
        logger.info(f"LANからのアクセス: http://{_get_lan_ip()}:{dash_port}")

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


if __name__ == "__main__":
    main()
