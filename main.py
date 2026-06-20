"""
kabu-auto メインエントリポイント
起動: python main.py
ダッシュボード: http://localhost:8080

このファイルは依存を結線してスケジューラにジョブを登録するだけの薄い composition root。
データ更新・ML再学習・損切り監視・シグナルスキャン・朝の発注の各ロジックは
src/services/trading.py の TradingServices に切り出している。
"""
import atexit
import os
import sys
import threading
import time

import uvicorn
from dotenv import load_dotenv
from loguru import logger

from src.core import (
    config as cfg, logger as log_setup, watchlist as watchlist_store,
    risk_profile as risk_profile_store, halt as halt_store, trading_mode as tm,
    process_lock,
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
    halt_store.load("data/trading_halt.json")  # 取引停止スイッチの状態を復元（停止中なら起動後も維持）
    log_setup.setup()
    db.init()

    trading_conf = cfg.get_section("trading")
    dash_conf = cfg.get_section("dashboard")
    mode = trading_conf.get("mode", "paper")

    # ─── モード妥当性チェック ──────────────────────────────────
    if not tm.is_valid(mode):
        logger.critical(
            f"不正な trading.mode です: {mode!r}。"
            f"次のいずれかを指定してください: {', '.join(tm.VALID_MODES)}"
        )
        sys.exit(1)
    logger.info(f"取引モード: {tm.description(mode)}")

    # ─── 多重起動防止（再レビュー P1-1）────────────────────────
    # 同一PCでの誤った二重起動はスケジューラジョブの重複実行・二重発注・WebSocket
    # 接続競合・DB破壊につながるため、実発注の可能性があるモードでは起動を中断する。
    # paper はDB/ログ/バックテスト結果の汚染リスクはあるが実資金リスクは無いため、
    # config の runtime.allow_multiple_paper_instances=true のときのみ警告のみで続行する
    # （既定 false = paper でも多重起動は中断。テストデータ整合性を優先した安全側の既定）。
    allow_multi_paper = cfg.get_section("runtime").get("allow_multiple_paper_instances", False)
    lock_ok, lock_detail = process_lock.acquire("data/kabu_auto.lock")
    if lock_ok:
        # 正常取得時のみ解放を atexit に登録する。sys.exit(1) や予期せぬ例外で
        # 起動が中断されてもロックファイルが残らないようにする（KeyboardInterrupt
        # だけでなくあらゆる正常終了経路をカバー）。release() は自分が書いたロックの
        # ときのみ削除するため、多重呼び出し・他プロセスのロック残存に対して安全。
        atexit.register(process_lock.release)
    else:
        if mode == "paper" and allow_multi_paper:
            logger.warning(
                f"多重起動を検知しましたが paper モード（allow_multiple_paper_instances=true）"
                f"のため続行します: {lock_detail}"
            )
        else:
            logger.critical(
                f"多重起動を検知したため起動を中断します: {lock_detail}。"
                "別の kabu-auto プロセスを終了してから再起動してください"
                + ("（paper でも多重起動を許可するには config の "
                   "runtime.allow_multiple_paper_instances を true にしてください）"
                   if mode == "paper" else "")
            )
            sys.exit(1)

    # ─── 実発注モード（live / semi_live）の起動確認 ─────────────
    # 実際の資金で発注しうるモードは二重確認を要求する（dry_run は読み取りのみなので不要）。
    if tm.places_real_orders(mode):
        if os.environ.get("CONFIRM_LIVE_TRADING", "").lower() != "true":
            logger.error(
                f"{tm.description(mode)}を起動するには環境変数 CONFIRM_LIVE_TRADING=true が必要です。"
                "  例: CONFIRM_LIVE_TRADING=true python main.py"
            )
            sys.exit(1)
        logger.warning(f"【{tm.description(mode)}】実際の資金を使用して取引します。")

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
        # 実発注モード（live / semi_live）でAPI接続できないまま起動を続けると、口座状態を
        # 把握できないまま発注ロジックだけが動く危険な状態になる（fail-closed）。
        if tm.places_real_orders(mode):
            logger.critical(
                f"{tm.description(mode)}でkabuステーション接続に失敗。起動を中断します: {e}"
            )
            sys.exit(1)
        logger.warning(f"kabuステーション接続失敗（{tm.description(mode)}で継続）: {e}")

    # ─── 起動時注文同期（ライブモードのみ）────────────────
    order_mgr.sync_on_startup()

    # ─── プリフライトチェック（paper 以外）────────────────
    # 実発注モードは致命的失敗があれば起動中断（fail-closed）。dry_run は記録のみ。
    if mode != "paper":
        from src.core import preflight
        result = preflight.run_preflight(
            client, mode,
            base_url=cfg.get_section("kabu_station").get("base_url", ""),
            dash_host=dash_conf.get("host", "127.0.0.1"),
            dash_port=dash_conf.get("port", 8080),
        )
        preflight.log_results(result)
        if not result["ok"] and tm.places_real_orders(mode):
            logger.critical(
                f"{tm.description(mode)}のプリフライトチェックに失敗しました。起動を中断します。"
            )
            sys.exit(1)

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
    scheduler.register("health_check", services.health_check)

    # ─── WebSocket 開始 ────────────────────────────────────
    client.start_websocket(on_order_event=order_mgr.on_order_event)

    # ─── スケジューラ起動 ───────────────────────────────────
    scheduler.start()
    update_status(running=True, ws_connected=True, mode=mode)

    # ─── 運用状態バナー（11.1: mode/endpoint/発注可否/LAN公開を起動時に明示）──
    snap = order_mgr.status_snapshot()
    can_order = "可" if snap["can_place_order"] else f"不可（{snap['block_reason']}）"
    lan_exposed = dash_conf.get("host", "127.0.0.1") == "0.0.0.0"
    logger.info(
        "─── 運用状態 ───\n"
        f"  モード      : {tm.description(mode)}\n"
        f"  エンドポイント: {cfg.get_section('kabu_station').get('base_url', '')}\n"
        f"  資金ソース  : {'口座余力(/wallet)' if tm.reads_broker_api(mode) else 'ペーパー初期資金'}\n"
        f"  発注        : {can_order}\n"
        f"  未解決注文  : {snap['unresolved_orders']}件 / 承認待ち: {snap['pending_approvals']}件\n"
        f"  LAN公開     : {'はい(0.0.0.0)' if lan_exposed else 'いいえ(localhost)'}"
    )
    if lan_exposed and tm.places_real_orders(mode):
        logger.warning(
            f"【注意】{tm.description(mode)}でダッシュボードをLAN公開しています。"
            "アクセストークン・ログイン認証が有効であることを確認してください。"
        )
    # dry_run が本番用エンドポイント（検証ポート18081以外）から実際の口座データを
    # 読んでいることを WARNING で明示する（発注はしないが本番口座の実情報を扱っている。
    # プリフライトでも検出するが、運用バナー直後にも目立つ形で出して誤認を防ぐ）。
    base_url = cfg.get_section("kabu_station").get("base_url", "")
    if mode == tm.DRY_RUN and "18081" not in base_url:
        logger.warning(
            f"【注意】dry_run が本番口座データを読み取っています（{base_url}）。"
            "発注は行いませんが、接続先は本番環境です。"
        )

    # ─── ダッシュボード起動（別スレッド）────────────────────
    dash_host = dash_conf.get("host", "127.0.0.1")
    dash_port = dash_conf.get("port", 8080)
    if not is_port_available(dash_host, dash_port):
        msg = f"ダッシュボードのポート {dash_host}:{dash_port} は既に使用中です"
        # 実発注モード（live / semi_live）でダッシュボードが起動できないと、発注ロジックは
        # 動くのに状態の監視・緊急決済操作ができない危険な状態になるため起動を中断する
        if tm.places_real_orders(mode):
            logger.critical(f"{msg}。{tm.description(mode)}のため起動を中断します。")
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
        # ロックファイルの解放は atexit.register(process_lock.release) が担う
        # （KeyboardInterrupt 以外の終了経路もカバーするため一元化している）
        logger.info("終了しました")


if __name__ == "__main__":
    main()
