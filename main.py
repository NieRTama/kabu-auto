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

import uvicorn
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import func, select

from src.core import (
    config as cfg, logger as log_setup, watchlist as watchlist_store,
    risk_profile as risk_profile_store,
)
from src.core.alerts import alert
from src.core.scheduler import TradingScheduler
from src.api.kabu_client import KabuClient
from src.data import database as db
from src.data.database import Position, Signal, Trade, get_session
from src.data.market_data import load_ohlcv, update_symbol
from src.execution.order_manager import OrderManager
from src.risk.manager import RiskManager
from src.strategy import ml_model
from src.strategy.signal import Signal as TradeSignal, generate as gen_signal
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
    risk.restore_daily_state()  # 再起動しても当日の損失上限・注文数カウンタを引き継ぐ
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

    # ─── スケジューラのコールバック登録 ──────────────────────
    # ウォッチリストは watchlist_store.get_codes() で毎回最新を取得する。
    # GUIでの追加・削除がプロセス再起動なしで次回実行から反映される。

    def token_refresh():
        try:
            client.refresh_token()
        except Exception as e:
            alert("APIトークン更新失敗", str(e))

    def data_update():
        years = data_conf.get("history_years", 3)
        # 全リストの銘柄を更新する（非アクティブリストもMLモデル学習データとして使うため）
        for sym in watchlist_store.get_all_codes():
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
        # 学習データはアクティブリストに限定せず全リストの銘柄を対象にする（サンプル数確保のため）。
        # 各銘柄のOHLCVは単純結合せず、銘柄ごとに train_multi() 内で特徴量・ラベルを
        # 作ってから連結する（移動平均/RSI/トリプルバリア法が銘柄境界をまたいで
        # 壊れるのを防ぐため。詳細は ml_model.train_multi() のdocstring参照）。
        dfs = []
        for sym in watchlist_store.get_all_codes():
            try:
                df = load_ohlcv(sym)
                if len(df) < 200:
                    continue
                dfs.append(df)
            except Exception as e:
                logger.error(f"データ読み込み失敗: {sym} {e}")
        if dfs:
            try:
                model = ml_model.train_multi(dfs, trigger="weekly_schedule")
            except Exception as e:
                logger.error(f"再学習失敗: {e}")

    def stop_loss_check():
        if not TradingScheduler.is_market_open():
            return
        is_paper = trading_conf.get("mode", "paper") == "paper"
        for sym in watchlist_store.get_codes():
            try:
                qty = _get_position_qty(sym)
                if qty <= 0:
                    continue  # 保有していない銘柄の板取得は無駄なのでスキップ
                if is_paper:
                    # ペーパーモードはリアルタイム板が無いため日足終値で損切り判定する
                    df = load_ohlcv(sym)
                    price = float(df["close"].iloc[-1]) if len(df) else 0
                else:
                    board = client.get_board(sym)
                    price = board.get("CurrentPrice", 0)
                if price and risk.should_stop_loss(sym, price):
                    logger.warning(f"損切り発動: {sym}")
                    order_mgr.sell(sym, float(price), qty)
                    alert("損切り実行", f"{sym} @{price:.0f}円")
            except Exception as e:
                logger.error(f"損切りチェックエラー: {sym} {e}")

    def signal_scan():
        """16:20（data_update完了後）に翌営業日の売買候補をスキャン。
        ペーパーモードは終値で即時シミュレート"""
        if TradingScheduler.is_maintenance_window():
            return
        logger.info("シグナルスキャン開始...")
        is_paper = trading_conf.get("mode", "paper") == "paper"
        sectors = watchlist_store.get_sectors()
        paper_base = float(trading_conf.get("paper_initial_capital", 500_000))
        for sym in watchlist_store.get_codes():
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
                        sector = sectors.get(sym, "")
                        # 固定額ではなく仮想ウォレット残高で発注サイズを決める
                        cash = _paper_available_cash(paper_base)
                        ok, reason = risk.validate_buy(sym, close_price, cash, sector)
                        if not ok:
                            logger.info(f"買い見送り: {sym} - {reason}")
                            continue
                        qty = risk.calc_position_size(sym, close_price, cash)
                        if qty > 0:
                            order_mgr.buy(sym, close_price, qty, sector=sector)
                    elif sig.action == "SELL":
                        qty = _get_position_qty(sym)
                        if qty > 0:
                            order_mgr.sell(sym, close_price, qty)
            except Exception as e:
                logger.error(f"シグナルスキャンエラー: {sym} {e}")

    def morning_execution():
        """9:05 に前営業日のBUY/SELLシグナルを元に発注（ライブモードのみ）"""
        if trading_conf.get("mode", "paper") != "live":
            return
        if not TradingScheduler.is_market_open():
            return
        with get_session() as session:
            pending = _select_latest_signals(session)

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
        sectors = watchlist_store.get_sectors()
        for sig in buy_signals:
            try:
                board = client.get_board(sig.symbol)
                price = board.get("CurrentPrice") or board.get("Sell1", {}).get("Price", 0)
                if not price:
                    continue
                sector = sectors.get(sig.symbol, "")
                ok, reason = risk.validate_buy(sig.symbol, float(price), cash, sector)
                if not ok:
                    logger.info(f"朝買い見送り: {sig.symbol} - {reason}")
                    continue
                qty = risk.calc_position_size(sig.symbol, float(price), cash)
                if qty <= 0:
                    continue
                order_id = order_mgr.buy(sig.symbol, float(price), qty, sector=sector)
                if order_id:
                    # 同一スキャン内の以降の銘柄が同じ余力を前提に判定しないよう、
                    # 発注成功分をその場で減算する（複数銘柄の資金二重計上を防ぐ）
                    cash -= float(price) * qty
                    logger.info(f"朝買い発注: {sig.symbol} {qty}株 @{price:.0f}円")
                else:
                    logger.warning(f"朝買い発注失敗（注文拒否）: {sig.symbol}")
            except Exception as e:
                logger.error(f"朝買い発注失敗: {sig.symbol} {e}")

    set_ml_retrain_fn(ml_retrain)
    set_data_update_fn(data_update)

    scheduler.register("risk_reset", risk.reset_daily_counters)
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
    dash_port = dash_conf.get("port", 8080)
    logger.info(f"ダッシュボード起動: http://localhost:{dash_port}")
    if dash_conf.get("host", "127.0.0.1") == "0.0.0.0":
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


def _select_latest_signals(session, max_age_days: int = 5) -> list:
    """直近のシグナル生成日（最新の signal_scan バッチの日付）のBUY/SELLシグナルを、
    銘柄ごと最新1件にdedupして返す。

    「now - 20時間」のような固定時間窓では、土日・祝日を挟むと前営業日（例: 金曜16:20）の
    シグナルを月曜9:05の発注時に取りこぼす（20時間を超えるため）。そのため「最新のシグナル
    生成日そのもの」を基準にすることで、休場日数に関わらず前営業日分を正しく拾う。
    生成日が max_age_days を超えて古い場合は陳腐化したシグナルとみなし空リストを返す
    （長期間ジョブが止まっていた場合の誤発注を防ぐ）。
    """
    latest_at = session.scalar(
        select(func.max(Signal.generated_at)).where(Signal.action.in_(["BUY", "SELL"]))
    )
    if latest_at is None:
        return []
    if (datetime.now() - latest_at).days > max_age_days:
        return []
    day_start = datetime.combine(latest_at.date(), datetime.min.time())
    day_end = day_start + timedelta(days=1)
    signals = session.scalars(
        select(Signal)
        .where(
            Signal.action.in_(["BUY", "SELL"]),
            Signal.generated_at >= day_start,
            Signal.generated_at < day_end,
        )
        .order_by(Signal.generated_at.desc())
    ).all()
    seen: set = set()
    pending: list = []
    for s in signals:
        if s.symbol not in seen:
            seen.add(s.symbol)
            pending.append(s)
    return pending


def _get_position_qty(symbol: str) -> int:
    with get_session() as session:
        pos = session.scalar(select(Position).where(Position.symbol == symbol))
        qty = pos.quantity if pos else 0
    return qty


def _paper_available_cash(base_capital: float) -> float:
    """ペーパーモードの利用可能資金を算出する。

    利用可能資金 = 初期資金 + 累積実現損益 − 現在の建玉簿価（avg_cost × 数量）。
    固定額50万だと複数銘柄で資金制約が効かず、実現損益も反映されないため、
    実際の口座挙動に近づける。
    """
    with get_session() as session:
        realized = session.scalar(
            select(func.sum(Trade.pnl)).where(Trade.pnl.isnot(None))
        ) or 0.0
        positions = session.scalars(
            select(Position).where(Position.quantity > 0)
        ).all()
        invested = sum(p.avg_cost * p.quantity for p in positions)
    return max(0.0, base_capital + float(realized) - invested)


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
