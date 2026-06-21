"""取引サービス層

main.py に集中していたスケジューラジョブのロジック（データ更新・ML再学習・損切り監視・
シグナルスキャン・朝の発注）を責務ごとに切り出したもの。main.py は依存を結線して
これらを登録するだけの薄い composition root になる。

各ジョブは client / risk / order_mgr / 設定、および ml_retrain が書き換え signal_scan が
読む可変の `model` を共有するため、状態を1つの `TradingServices` に集約する
（5モジュールに分割すると同じ依存と可変modelを相互参照する結合が増えるため、
凝集した単一クラスとした）。純粋な抽出であり挙動は main.py の旧クロージャと同一。
"""
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger
from sqlalchemy import func, select

from src.core import clock
from src.core import config as cfg
from src.core import trading_mode as tm
from src.core import watchlist as watchlist_store
from src.core.alerts import alert
from src.core.scheduler import TradingScheduler
from src.data.database import Position, Signal, Trade, get_session
from src.data.market_data import load_ohlcv, update_symbol
from src.risk import liquidity
from src.strategy import ml_model
from src.strategy.signal import Signal as TradeSignal, generate as gen_signal


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
    if (clock.now() - latest_at).days > max_age_days:
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


def _signal_rationale(sig) -> str:
    """シグナル（TradeSignal / DB Signal）から発注根拠の説明文を作る（7.6）。

    combined / rule / ml の各スコアを記録し、後から「なぜこの取引をしたか」を辿れるようにする。
    """
    def _f(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"
    return (f"{sig.action} score={_f(sig.combined_score)} "
            f"(rule={_f(sig.rule_score)}, ml={_f(sig.ml_score)})")


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


class TradingServices:
    """スケジューラに登録される取引ジョブ群。依存と可変modelを保持する。"""

    def __init__(self, client, risk, order_mgr, model=None):
        self.client = client
        self.risk = risk
        self.order_mgr = order_mgr
        self.model = model
        self.trading_conf = cfg.get_section("trading")
        self.data_conf = cfg.get_section("data")
        self.liquidity_conf = cfg.get_section("liquidity")

    # ─── データ更新 ─────────────────────────────────────
    def data_update(self) -> None:
        years = self.data_conf.get("history_years", 3)
        # 全リストの銘柄を更新する（非アクティブリストもMLモデル学習データとして使うため）
        for sym in watchlist_store.get_all_codes():
            try:
                update_symbol(sym, years=years)
            except Exception as e:
                logger.error(f"データ更新失敗: {sym} {e}")

    # ─── ML週次再学習 ───────────────────────────────────
    def ml_retrain(self) -> None:
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
                self.model = ml_model.train_multi(dfs, trigger="weekly_schedule")
            except Exception as e:
                logger.error(f"再学習失敗: {e}")

    # ─── 注文状態の定期照合 ─────────────────────────────
    def reconcile_orders(self) -> None:
        """WebSocketイベントの取り逃し・切断・再起動に備え、未約定注文をブローカーの
        /orders 照会結果へ定期的に収束させる（市場時間外は何もしない）。

        合わせて建玉(/positions)もブローカー実態と照合する（P0-3）。注文照合が
        失敗してもポジション照合は独立して実行する（どちらかの失敗が他方を隠さないため）。
        """
        if not TradingScheduler.is_market_open():
            return
        try:
            self.order_mgr.reconcile_open_orders()
        except Exception as e:
            logger.error(f"注文照合エラー: {e}")
        try:
            self.order_mgr.reconcile_positions_with_broker()
        except Exception as e:
            logger.error(f"建玉照合エラー: {e}")

    # ─── 異常検知・アラート ─────────────────────────────
    def health_check(self) -> None:
        """運用上の異常（未解決注文・損失上限接近・kill switch等）を検知して通知する（7.5）。

        市場時間に限定せず動かす（場が引けた後でも未解決注文は要対応のため）。
        """
        try:
            from src.core import health
            health.run_and_alert(self.risk)
        except Exception as e:
            logger.error(f"異常検知ジョブエラー: {e}")

    # ─── 損切り監視 ─────────────────────────────────────
    def stop_loss_check(self) -> None:
        if not TradingScheduler.is_market_open():
            return
        is_paper = self.trading_conf.get("mode", "paper") == "paper"
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
                    board = self.client.get_board(sym)
                    price = board.get("CurrentPrice", 0)
                if price and self.risk.should_stop_loss(sym, price):
                    logger.warning(f"損切り発動: {sym}")
                    # 損切りは確実な約定を優先し成行で発注する（指値だと急変時に約定しない）。
                    # reason="stop_loss" により日次上限・損失上限等の新規発注ゲートを
                    # バイパスする（損切りは既存リスクを減らす退出操作のため止めてはいけない）
                    self.order_mgr.sell_market(sym, qty, reason="stop_loss")
                    alert("損切り実行", f"{sym} @{price:.0f}円")
            except Exception as e:
                logger.error(f"損切りチェックエラー: {sym} {e}")

    # ─── シグナルスキャン ───────────────────────────────
    def signal_scan(self) -> None:
        """16:20（data_update完了後）に翌営業日の売買候補をスキャン。
        ペーパーモードは終値で即時シミュレート"""
        if TradingScheduler.is_maintenance_window():
            return
        logger.info("シグナルスキャン開始...")
        is_paper = self.trading_conf.get("mode", "paper") == "paper"
        sectors = watchlist_store.get_sectors()
        paper_base = float(self.trading_conf.get("paper_initial_capital", 500_000))
        for sym in watchlist_store.get_codes():
            try:
                df = load_ohlcv(sym)
                if len(df) < 30:
                    continue
                sig = gen_signal(sym, df, self.model)
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
                        ok, reason = self.risk.validate_buy(sym, close_price, cash, sector)
                        if not ok:
                            logger.info(f"買い見送り: {sym} - {reason}")
                            continue
                        ok_liq, liq_reason = liquidity.check_liquidity(
                            sym, df, self.liquidity_conf)
                        if not ok_liq:
                            logger.info(f"買い見送り: {liq_reason}")
                            continue
                        qty = self.risk.calc_position_size(sym, close_price, cash)
                        if qty > 0:
                            self.order_mgr.buy(sym, close_price, qty, sector=sector,
                                               rationale=_signal_rationale(sig),
                                               source="signal_scan")
                    elif sig.action == "SELL":
                        qty = _get_position_qty(sym)
                        if qty > 0:
                            self.order_mgr.sell(sym, close_price, qty,
                                                rationale=_signal_rationale(sig),
                                                source="signal_scan")
            except Exception as e:
                logger.error(f"シグナルスキャンエラー: {sym} {e}")

    # ─── 朝の発注（ライブモードのみ）────────────────────
    def morning_execution(self) -> None:
        """9:05 に前営業日のBUY/SELLシグナルを元に発注する。

        実行対象は paper 以外（live / dry_run / semi_live）。発注の実体は OrderManager が
        モードに応じて分岐する（live=実発注 / dry_run=実発注せず記録のみ / semi_live=承認キュー）。
        paper は signal_scan 内で当日終値で即時シミュレートするため morning は不要。
        """
        mode = self.trading_conf.get("mode", "paper")
        if not tm.uses_morning_execution(mode):
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
                board = self.client.get_board(sig.symbol)
                price = board.get("CurrentPrice") or board.get("Buy1", {}).get("Price", 0)
                if not price:
                    continue
                self.order_mgr.sell(sig.symbol, float(price), qty,
                                    rationale=_signal_rationale(sig),
                                    source="morning_execution")
                logger.info(f"朝売り発注: {sig.symbol} {qty}株 @{price:.0f}円")
            except Exception as e:
                logger.error(f"朝売り発注失敗: {sig.symbol} {e}")

        # ── BUY シグナル: 余力を確認して買う ──────────────────────
        if not buy_signals:
            return
        # 大引け間際の新規BUYは薄商い・不利約定を招きやすいので見送る（P0-6。0で無効）
        near_close_min = int(self.liquidity_conf.get("no_new_buy_minutes_before_close", 0) or 0)
        if TradingScheduler.is_near_close(near_close_min):
            logger.info(f"朝買い見送り: 大引け{near_close_min}分前以降のため新規BUYを抑止")
            return
        try:
            wallet = self.client.get_wallet()
            cash = float(wallet.get("StockAccountWallet", 0))
        except Exception as e:
            logger.error(f"余力取得失敗: {e}")
            return
        sectors = watchlist_store.get_sectors()
        for sig in buy_signals:
            try:
                board = self.client.get_board(sig.symbol)
                price = board.get("CurrentPrice") or board.get("Sell1", {}).get("Price", 0)
                if not price:
                    continue
                sector = sectors.get(sig.symbol, "")
                ok, reason = self.risk.validate_buy(sig.symbol, float(price), cash, sector)
                if not ok:
                    logger.info(f"朝買い見送り: {sig.symbol} - {reason}")
                    continue
                ok_liq, liq_reason = liquidity.check_liquidity(
                    sig.symbol, load_ohlcv(sig.symbol), self.liquidity_conf)
                if not ok_liq:
                    logger.info(f"朝買い見送り: {liq_reason}")
                    continue
                ok_sp, sp_reason = liquidity.check_spread(board, self.liquidity_conf)
                if not ok_sp:
                    logger.info(f"朝買い見送り: {sig.symbol} - {sp_reason}")
                    continue
                qty = self.risk.calc_position_size(sig.symbol, float(price), cash)
                if qty <= 0:
                    continue
                order_id = self.order_mgr.buy(sig.symbol, float(price), qty, sector=sector,
                                              rationale=_signal_rationale(sig),
                                              source="morning_execution")
                if order_id:
                    # 同一スキャン内の以降の銘柄が同じ余力を前提に判定しないよう、
                    # 発注成功分をその場で減算する（複数銘柄の資金二重計上を防ぐ。
                    # RiskManager の未約定引当と二重で守る）
                    cash -= float(price) * qty
                    logger.info(f"朝買い発注: {sig.symbol} {qty}株 @{price:.0f}円")
                else:
                    logger.warning(f"朝買い発注失敗（注文拒否）: {sig.symbol}")
            except Exception as e:
                logger.error(f"朝買い発注失敗: {sig.symbol} {e}")
