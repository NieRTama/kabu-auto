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
    process_lock, reference_capital as reference_capital_store, broker_wait, broker_auth,
    discord_bot, auth_recovery, broker_launcher, discord_queries, discord_slash,
)
from src.core import alerts as alerts_mod
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
    reference_capital_store.load("reference_capital.json")  # X日次レポートの%算出用基準資金
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

    # ─── 初回トークン取得（kabuステーションのログイン待ちを含む）──────
    # OS起動時に自動実行すると kabuステーションがまだ起動・ログインされておらず
    # 必ず接続に失敗する。ログイン自体は2段階認証があり人手が要るため、
    # 「ログインされるまで静かに待つ」ことで自動起動を成立させる（0で従来動作）。
    # 0 = 無制限に待つ（既定）。時間制限を設けると「その時間内にログインしないと
    # 起動しない」ことになり、朝寝坊や通知の見落としで丸一日動かなくなる。
    wait_minutes = float(cfg.get_section("runtime").get("wait_for_broker_minutes", 0) or 0)
    wait_unlimited = wait_minutes <= 0
    limit_text = "接続できるまで待機します" if wait_unlimited else f"最大{wait_minutes:.0f}分待機します"
    connected = broker_wait.wait_for_broker(
        client.refresh_token,
        timeout_seconds=wait_minutes * 60,
        unlimited=wait_unlimited,
        on_wait_start=lambda: alert(
            "kabuステーションのログイン待ち",
            f"{tm.description(mode)}で起動しましたが、kabuステーションへ接続できません。"
            f"kabuステーションを起動してログインしてください（{limit_text}）。",
        ),
        on_connected=lambda waited: alert(
            "kabuステーションへ接続しました",
            f"{waited / 60:.1f}分の待機後に接続し、{tm.description(mode)}で稼働を開始します。",
        ),
    )
    if connected:
        broker_auth.mark_valid()
    else:
        # 接続できないまま続行する場合（paper/dry_run）も認証切れとして記録し、
        # ダッシュボード・発注ゲートが実態を反映するようにする。
        broker_auth.mark_expired("起動時にkabuステーションへ接続できませんでした")
        # 実発注モード（live / semi_live）でAPI接続できないまま起動を続けると、口座状態を
        # 把握できないまま発注ロジックだけが動く危険な状態になる（fail-closed）。
        if tm.places_real_orders(mode):
            logger.critical(
                f"{tm.description(mode)}でkabuステーション接続に失敗。起動を中断します。"
            )
            sys.exit(1)
        logger.warning(f"kabuステーション接続失敗（{tm.description(mode)}で継続）")

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
    def _launch_broker() -> tuple[bool, str]:
        """kabuステーションを起動する（設定を読んで launcher へ委譲）。"""
        rt = cfg.get_section("runtime")
        return broker_launcher.launch(
            rt.get("broker_exe_path", "") or "",
            max_attempts_per_day=int(
                rt.get("max_launch_attempts_per_day",
                       broker_launcher.DEFAULT_MAX_ATTEMPTS_PER_DAY)
            ),
        )

    def token_refresh():
        """毎朝のトークン更新。失敗＝ログイン認証切れとみなし、再ログインを待って復帰する。

        kabuステーションの認証はPCを起動したままでも日をまたぐと切れるため、
        毎朝ここで401になるのが通常運転（2026-08-26/27 は失敗後リトライが無く、
        終日401のまま「発注可」を表示し続ける抜け殻状態になっていた）。
        失敗したら発注を止めたうえで通知し、ログインされ次第そのまま自動復帰する。
        """
        try:
            client.refresh_token()
            broker_auth.mark_valid()
        except Exception as e:
            broker_auth.mark_expired(str(e))
            alert(
                "kabuステーションの再ログインが必要です",
                "APIトークン更新に失敗しました（ログイン認証切れ）。kabuステーションに"
                "ログインしてください。ログインが済み次第、自動で取引を再開します"
                f"（時間制限なく待ち続けます）。詳細: {e}",
            )

    def auth_recovery_check():
        """認証切れの間、定期的に再接続を試みて自動復帰する。

        以前は token_refresh 内で30分だけブロッキング待機していたが、
        タイムアウトすると翌朝まで復帰しない穴があった（2026-08-29 に発生。
        朝寝坊や通知の見落としで丸一日の取引機会を失う）。時間制限を撤廃し、
        認証が戻るまで何度でも試し続ける方式に変えた。
        """
        def _on_recovered():
            alert(
                "kabuステーションの認証が回復しました",
                "再ログインを検知しました。取引を再開します。",
                level=alerts_mod.LEVEL_INFO,
            )
            # 朝のログインが遅れて 9:05 の発注を逃した場合に備え、
            # 場中かつ締切前なら逃した分をその場で取り返す
            # （2026-08-31 は10:58復帰で朝の発注機会を失い当日の約定が0件だった）。
            services.catchup_execution()

        recovered = auth_recovery.attempt_recovery(
            client.refresh_token,
            on_recovered=_on_recovered,
        )
        if recovered or not broker_auth.is_expired():
            return

        # 認証切れが続いている。kabuステーションのプロセス自体が落ちていれば
        # 起動する（2026-08-31 にアプリがクラッシュし、翌朝まで待機のままだった）。
        # 認証（2段階認証）は自動化せず、承認は認証アプリで人が行う。
        rt = cfg.get_section("runtime")
        if not rt.get("auto_launch_broker", True):
            return
        if broker_launcher.is_running():
            return
        ok, detail = _launch_broker()
        if ok:
            alert(
                "kabuステーションを起動しました",
                f"{detail}\n認証アプリで承認してください。承認後は自動で取引を再開します。",
                level=alerts_mod.LEVEL_INFO,
            )
        else:
            logger.info(f"kabuステーションの自動起動は行いませんでした: {detail}")

    def db_backup():
        try:
            db.backup()
        except Exception as e:
            logger.error(f"バックアップ失敗: {e}")

    set_ml_retrain_fn(services.ml_retrain)
    set_data_update_fn(services.data_update)

    # ─── Discordリモコン（任意。未設定なら無効）──────────────
    # 外出先からの状態確認・緊急停止用。秘密情報は運ばせない方針のため、
    # .env の書き込みや認証情報の送受信、緊急全決済は実装しない
    # （全決済は誤爆時の損害が大きいのでダッシュボードのトークン必須経路に限定）。
    def _cmd_status(_args: str) -> str:
        snap = order_mgr.status_snapshot()
        auth = "有効" if not broker_auth.is_expired() else "切れ（要ログイン）"
        order_state = "可" if snap["can_place_order"] else f"不可（{snap['block_reason']}）"
        return (f"モード: {tm.description(mode)}\n"
                f"発注: {order_state}\n"
                f"kabuステーション認証: {auth}\n"
                f"未解決注文: {snap['unresolved_orders']}件 / "
                f"承認待ち: {snap['pending_approvals']}件")

    def _reference_capital() -> float:
        """損益率の算出基準（日次レポートと同じ基準を使い、数字を食い違わせない）。"""
        from src.core import reference_capital as ref_capital_store
        paper_base = float(trading_conf.get("paper_initial_capital", 500_000))
        return ref_capital_store.percent_basis(mode, paper_initial_capital=paper_base)

    def _cmd_positions(_args: str) -> str:
        return discord_queries.format_positions(_reference_capital())

    def _cmd_today(_args: str) -> str:
        return discord_queries.format_today_trades()

    def _cmd_pnl(_args: str) -> str:
        return discord_queries.format_pnl(_reference_capital())

    def _cmd_orders(_args: str) -> str:
        return discord_queries.format_open_orders()

    def _cmd_launch(_args: str) -> str:
        """kabuステーションを起動する（認証は認証アプリで人が行う）。"""
        ok, detail = _launch_broker()
        if ok:
            return f"{detail}\n認証アプリで承認してください。承認後は自動で取引を再開します"
        return detail

    def _cmd_reconnect(_args: str) -> str:
        """ログイン直後に「今すぐ繋いで」と指示するためのコマンド。

        定期チェック（5分間隔）でも復帰するが、外出先からログインした直後に
        待たずに反映させたい場合に使う。
        """
        if not broker_auth.is_expired():
            return "認証は有効です（再接続は不要）"
        try:
            client.refresh_token()
        except Exception as e:
            return f"再接続に失敗しました。kabuステーションにログインしてから再度お試しください: {e}"
        broker_auth.mark_valid()
        return "再接続しました。取引を再開します"

    def _cmd_halt(args: str) -> str:
        halt_store.engage(args.strip() or "Discordから手動停止")
        return "取引を停止しました（新規発注を抑止。損切り・緊急決済は引き続き動作します）"

    def _cmd_resume(_args: str) -> str:
        halt_store.release()
        return "取引を再開しました"

    discord_conf = cfg.get_section("alerts")
    _bot_token = (os.environ.get("DISCORD_BOT_TOKEN", "")
                  or discord_conf.get("discord_bot_token", ""))
    _bot_channel = (os.environ.get("DISCORD_BOT_CHANNEL_ID", "")
                    or str(discord_conf.get("discord_bot_channel_id", "")))
    _bot_users = {os.environ.get("DISCORD_ALLOWED_USER_ID", "")}
    # (関数, 説明)。説明は help とスラッシュコマンドの候補表示に使う。
    # メンション方式とスラッシュ方式で**同じ定義を共有**する（実装を二重に持たない）。
    _discord_handlers = {
        "status": (_cmd_status, "稼働状況（モード・発注可否・認証・未解決注文）"),
        "positions": (_cmd_positions, "保有建玉（取得単価・現在値・含み損益）"),
        "today": (_cmd_today, "本日の約定履歴と確定損益"),
        "pnl": (_cmd_pnl, "損益サマリ（当日/週次/月次/総合）"),
        "orders": (_cmd_orders, "未約定注文の一覧"),
        "reconnect": (_cmd_reconnect, "認証切れのとき即座に再接続を試みる"),
        "launch": (_cmd_launch, "kabuステーションを起動する（認証は手動）"),
        "halt": (_cmd_halt, "取引を停止する（例: halt 様子見。退出は継続）"),
        "resume": (_cmd_resume, "取引を再開する"),
    }
    # メンション方式（30秒ポーリング）。Gateway が切れていても操作できる経路として残す。
    remote = discord_bot.build(
        token=_bot_token, channel_id=_bot_channel,
        allowed_user_ids=_bot_users, handlers=_discord_handlers,
    )
    # スラッシュコマンド（Gateway 常時接続・即時応答）。discord.py 未導入なら無効。
    slash = discord_slash.build(
        token=_bot_token, channel_id=_bot_channel,
        allowed_user_ids=_bot_users, handlers=_discord_handlers,
    )
    if slash is not None:
        atexit.register(slash.stop)

    # ─── スケジューラのコールバック登録 ──────────────────────
    scheduler.register("risk_reset", risk.reset_daily_counters)
    scheduler.register("token_refresh", token_refresh)
    scheduler.register("data_update", services.data_update)
    scheduler.register("db_backup", db_backup)
    scheduler.register("ml_retrain", services.ml_retrain)
    scheduler.register("stop_loss_check", services.stop_loss_check)
    scheduler.register("signal_scan", services.signal_scan)
    scheduler.register("morning_execution", services.morning_execution)
    scheduler.register("afternoon_execution", services.afternoon_execution)
    scheduler.register("reconcile_orders", services.reconcile_orders)
    scheduler.register("health_check", services.health_check)
    scheduler.register("heartbeat", services.heartbeat)
    scheduler.register("auth_recovery_check", auth_recovery_check)
    scheduler.register("discord_weekly_report", services.post_weekly_summary_to_discord)
    if remote is not None:
        scheduler.register("discord_poll", remote.poll_once)
    scheduler.register("x_daily_report", services.post_daily_summary_to_x)
    scheduler.register("discord_daily_report", services.post_daily_summary_to_discord)

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
