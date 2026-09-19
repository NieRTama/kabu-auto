"""取引サービス層

main.py に集中していたスケジューラジョブのロジック（データ更新・ML再学習・損切り監視・
シグナルスキャン・朝の発注）を責務ごとに切り出したもの。main.py は依存を結線して
これらを登録するだけの薄い composition root になる。

各ジョブは client / risk / order_mgr / 設定、および ml_retrain が書き換え signal_scan が
読む可変の `model` を共有するため、状態を1つの `TradingServices` に集約する
（5モジュールに分割すると同じ依存と可変modelを相互参照する結合が増えるため、
凝集した単一クラスとした）。純粋な抽出であり挙動は main.py の旧クロージャと同一。
"""
from datetime import date, datetime, timedelta
from typing import Optional
import uuid

from loguru import logger
from sqlalchemy import func, select

from src.core import clock
from src.core import market_calendar
from src.core import config as cfg
from src.core import trading_mode as tm
from src.core import watchlist as watchlist_store
from src.core.alerts import LEVEL_INFO, LEVEL_WARNING, alert
from src.core.scheduler import TradingScheduler
from src.data import bar_status
from src.data.bar_status import BarStatus
from src.data.database import OrderIntent, Position, Signal, Trade, get_session
from src.data.market_data import load_ohlcv, update_symbol
from src.execution import order_status as st
from src.risk import liquidity
from src.strategy import ml_model
from src.strategy.signal import Signal as TradeSignal, generate as gen_signal


def _select_latest_signals(session, max_age_days: int = 5) -> list:
    """直近のBUYバッチ（signal_scanの日付）のBUYシグナルと、そのバッチ日以降
    （BUYバッチが無ければmax_age_days以内）のSELLシグナルをあわせて、銘柄ごと
    1件にdedupして返す。

    「now - 20時間」のような固定時間窓では、土日・祝日を挟むと前営業日（例: 金曜16:20）の
    シグナルを月曜9:05の発注時に取りこぼす（20時間を超えるため）。そのため「最新のBUY
    生成日そのもの」を基準にすることで、休場日数に関わらず前営業日分を正しく拾う。
    全体が max_age_days を超えて古い場合は陳腐化したシグナルとみなし空リストを返す
    （長期間ジョブが止まっていた場合の誤発注を防ぐ）。

    BUYの対象日とSELLの対象範囲は別々に決める（新規Important C）。v2 paperの
    stop_loss_checkは当日付でSELLを保存するため、BUYと同じ「最新シグナル日」を
    基準にすると、当日9:00にSELLが1件でも保存された途端、前営業日16:20の
    signal_scanバッチ（本来9:05のmorning_executionが拾うべきBUY群）が丸ごと
    対象外になってしまう。これを避けるため、
    - BUYは「最新のBUY生成日（＝直近のsignal_scanバッチ日）」の分だけを対象にし、
    - SELLは「有効なBUYバッチがあればそのバッチ日の00:00以降」、
      「BUYバッチが無い・陳腐化している場合はmax_age_days以内」を対象にする
      （最終ブランチレビュー 新規Important D）。
      SELLの対象範囲をBUYと無関係にmax_age_days全体まで広げると、`Signal`表に
      消費済みを表す列が無いため、数日前に約定済みで役目を終えた古いSELL行が
      dedupで「SELLは生成時刻に関わらずBUYより優先」（Important 4）のルールに
      乗ってしまい、その後に生成された新しいBUYを毎回上書きして消してしまう
      （＝ある銘柄で損切りが1度発火すると、以降max_age_days日ぶんその銘柄への
      新規BUYが無音で潰れる）。下限をBUYバッチ日に揃えることで、BUYバッチより
      前の古いSELLを拾わないようにする。まだ退出が必要な状況（何日も約定できずに
      残っている等）は、stop_loss_checkが5分毎に新しいSELLを保存し直すため、
      直近のBUYバッチ日以降に収まり取りこぼさない。
    - ただしBUY自体もmax_age_daysより古ければ対象外にする（他銘柄の新しい
    SELLがあるからといって、signal_scanが止まって何日も経った古いBUY
    バッチを蘇らせて発注してしまわないようにするため）。この場合SELLの
    下限もmax_age_days窓へフォールバックする。

    dedupは**生成時刻ではなくSELLを優先**する。日中の stop_loss_check が保存した
    SELL（保有保護の退出）を、同日16:20の signal_scan が生成したより新しいBUYで
    上書きして消してしまうと、損切り・トレーリングストップの退出が最低1営業日
    遅れる（最終ブランチレビュー Important 4）。同一銘柄にSELLが1件でもあれば、
    生成時刻に関わらずSELLを採用する。
    """
    latest_at = session.scalar(
        select(func.max(Signal.generated_at)).where(Signal.action.in_(["BUY", "SELL"]))
    )
    if latest_at is None:
        return []
    if (clock.now() - latest_at).days > max_age_days:
        return []

    signals = []
    buy_latest_at = session.scalar(
        select(func.max(Signal.generated_at)).where(Signal.action == "BUY")
    )
    # SELLがmax_age_days以内なら上のガードは常に通ってしまうため、BUY自体の
    # 陳腐化は独立に見る（他銘柄の新しいSELLがあるからといって、何日も前の
    # 止まったsignal_scanバッチを蘇らせて発注してはいけない）。
    if buy_latest_at is not None and (clock.now() - buy_latest_at).days > max_age_days:
        buy_latest_at = None
    if buy_latest_at is not None:
        buy_day_start = datetime.combine(buy_latest_at.date(), datetime.min.time())
        buy_day_end = buy_day_start + timedelta(days=1)
        signals.extend(session.scalars(
            select(Signal)
            .where(
                Signal.action == "BUY",
                Signal.generated_at >= buy_day_start,
                Signal.generated_at < buy_day_end,
            )
            .order_by(Signal.generated_at.desc())
        ).all())

    sell_floor = clock.now() - timedelta(days=max_age_days)
    if buy_latest_at is not None:
        # 有効なBUYバッチがある場合、SELLの下限をmax_age_days全体ではなく
        # そのバッチ日の00:00に引き上げる（新規Important D）。BUYバッチより
        # 古いSELLは既に約定済みで役目を終えている可能性が高く、それを拾うと
        # dedupで新しいBUYを消してしまう退行になる。
        buy_day_start = datetime.combine(buy_latest_at.date(), datetime.min.time())
        sell_floor = buy_day_start
    signals.extend(session.scalars(
        select(Signal)
        .where(
            Signal.action == "SELL",
            Signal.generated_at >= sell_floor,
        )
        .order_by(Signal.generated_at.desc())
    ).all())

    by_symbol: dict = {}
    for s in signals:
        current = by_symbol.get(s.symbol)
        if current is None:
            by_symbol[s.symbol] = s
        elif current.action != "SELL" and s.action == "SELL":
            # 保有保護の退出（SELL）は生成時刻に関わらずBUYより優先する
            by_symbol[s.symbol] = s
    return list(by_symbol.values())


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


def _held_symbols() -> list[str]:
    """現在保有している全銘柄コード（ウォッチリストの所属・アクティブ切替に関わらず）。

    2026-09-19、9434がウォッチリストのアクティブリストから外れていたため
    `stop_loss_check()` の監視対象（`watchlist_store.get_codes()`）に含まれず、
    保有建玉があるのに損切り・トレーリングストップが一切機能していなかった
    実害が発覚した（リストの切替・銘柄整理は日常的に行われるが、保有中の
    銘柄がその対象になり得ることを想定していなかった）。リスク管理（保有建玉の
    監視）は新規シグナル判断（ウォッチリストのアクティブ切替）から独立させ、
    ここでDBの実際の建玉を直接見る。
    """
    with get_session() as session:
        rows = session.scalars(select(Position).where(Position.quantity > 0)).all()
    return [p.symbol for p in rows]


# REJECTED/CANCELLED は実際には成立しなかった発注なので「未購入」扱いにし、
# 同日中の再挑戦を許す（板薄・スプレッド超過等での見送りは次のスロットで拾い直したい）。
_NOT_BOUGHT_STATUSES = (st.REJECTED, st.CANCELLED)


def _bought_today(symbol: str) -> bool:
    """本日中に、この銘柄へのBUYが成立/進行中か（後場スロットの同日二度打ち防止）。

    2026-09-09 追加。後場(afternoon_execution)は従来「保有中の銘柄はBUY対象から除外」
    していたが、これだと前日までの保有銘柄への買い増しも一律止めてしまい、朝
    (skip_existing=False)が許可している買い増しと矛盾していた。実際に止めるべきは
    「朝と後場で同じシグナルセットを二重に約定させること」であり、保有の有無ではない。

    Trade には作成日時列が無いため（filled_at は未約定の間 None）、紐づく
    OrderIntent.created_at で当日判定する。
    """
    today_start = datetime.combine(clock.today(), datetime.min.time())
    with get_session() as session:
        count = session.scalar(
            select(func.count(Trade.id))
            .join(OrderIntent, Trade.intent_id == OrderIntent.id)
            .where(
                Trade.symbol == symbol,
                Trade.side == "BUY",
                Trade.status.notin_(_NOT_BOUGHT_STATUSES),
                OrderIntent.created_at >= today_start,
            )
        ) or 0
    return count > 0


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


def _save_signal(sig: TradeSignal, data_as_of: Optional[date] = None) -> None:
    with get_session() as session:
        session.add(Signal(
            symbol=sig.symbol,
            rule_score=sig.rule_score,
            ml_score=sig.ml_score,
            combined_score=sig.combined_score,
            action=sig.action,
            data_as_of=data_as_of,
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
        # 直近の data_update で得た銘柄ごとの最終足状態。signal_scan が新規候補の
        # 可否判定に使う。data_update より前に signal_scan が動いた場合は空のままで、
        # そのときは全銘柄が「新規候補にしない」と判定される（安全側）。
        self._bar_states: dict[str, BarStatus] = {}
        # data_update が払い出す確定スナップショットの識別子。signal_scan は
        # この ID の入力集合だけを見る。16:00更新→16:20スキャンという時間差にのみ
        # 依存していると、部分更新の最中に入力が入れ替わったことに気付けない。
        self._data_batch_id: Optional[str] = None

    # ─── データ更新 ─────────────────────────────────────
    def data_update(self) -> dict[str, BarStatus]:
        """全リストの銘柄の日足を更新し、銘柄ごとの最終足状態を返す。

        非アクティブリストもMLモデル学習データとして使うため全リストを対象にする。
        更新に失敗した銘柄は missing として記録し、その日の新規候補から外す
        （古い足のまま新しいシグナルを作らないため）。
        """
        years = self.data_conf.get("history_years", 3)
        states: dict[str, BarStatus] = {}
        for sym in watchlist_store.get_all_codes():
            try:
                states[sym] = update_symbol(sym, years=years)
            except Exception as e:
                logger.error(f"データ更新失敗: {sym} {e}")
                states[sym] = BarStatus(
                    symbol=sym, last_bar_session=None,
                    observed_at=clock.now(), is_final=False, state="missing",
                )
        self._data_batch_id = f"{clock.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        self._bar_states = states
        fresh = sum(1 for s in states.values() if s.state == "fresh")
        logger.info(
            f"データ更新完了: batch={self._data_batch_id} 対象={len(states)}銘柄 "
            f"確定={fresh} 未確定={len(states) - fresh}"
        )
        return states

    def _is_fresh_for_new_candidate(self, symbol: str) -> bool:
        """この銘柄を新規候補として扱ってよいか。

        更新結果が無い銘柄は fresh と判定しない（安全側）。`_bar_states` は
        直近の data_update() 実行時点の凍結値であるため、state だけを信じると
        「data_update が今日まだ完了していない／走行中に signal_scan が始まった」
        場合に前営業日分の"fresh"を素通しさせてしまう。呼び出し時点の
        as_of_session と last_bar_session を突き合わせて再検証する。

        なおこの判定は**新規の戦略シグナルにのみ**適用する。保有保護の退出
        （損切り・トレーリング）と既存注文の管理は鮮度に関わらず実行する。
        """
        st = self._bar_states.get(symbol)
        if st is None or st.state != "fresh":
            return False
        expected = bar_status.as_of_session(clock.now())
        return st.last_bar_session == expected

    def _engine_version(self) -> str:
        """評価基盤のどの世代で動くか。未知の値は安全側（legacy）に倒す。

        legacy = 従来の挙動（既定）。v2 = 段階B〜Dで作り直した執行仮定。
        旧方式へいつでも戻せることが段階投入の前提（設計書 §10）。
        """
        value = cfg.get_section("strategy").get("engine_version", "legacy")
        return "v2" if value == "v2" else "legacy"

    def _paper_uses_v2_execution(self) -> bool:
        """paper経路で新しい執行仮定（翌営業日の寄り）を使うか。

        stop_loss_check は paper モードで日足終値を使って損切りを判定しており、
        同一終値での判断・約定という問題が paper 運用にも及んでいた
        （レビューF04）。**過去の日足を再生する paper と、現在の市場を観測する
        paper は入力契約が違う**。前者は日足による約定の近似であり、既存の
        翌朝9:05運用と同じ約定モデルとしては扱わない。
        """
        return self._engine_version() == "v2"

    # ─── ML週次再学習 ───────────────────────────────────
    def ml_retrain(self) -> None:
        logger.info("MLモデル週次再学習を開始...")
        # 学習データはアクティブリストに限定せず全リストの銘柄を対象にする（サンプル数確保のため）。
        # 各銘柄のOHLCVは単純結合せず、銘柄ごとに train_multi() 内で特徴量・ラベルを
        # 作ってから連結する（移動平均/RSI/トリプルバリア法が銘柄境界をまたいで
        # 壊れるのを防ぐため。詳細は ml_model.train_multi() のdocstring参照）。
        dfs = []
        trained_symbols = []
        for sym in watchlist_store.get_all_codes():
            try:
                df = load_ohlcv(sym)
                if len(df) < 200:
                    continue
                dfs.append(df)
                trained_symbols.append(sym)
            except Exception as e:
                logger.error(f"データ読み込み失敗: {sym} {e}")
        if dfs:
            if self._engine_version() == "v2":
                # v2: 候補を作るだけで self.model を差し替えない。
                # 学習成功は候補の生成であって運用モデルの更新ではない（設計書 §9）。
                # 昇格は promotion.promote() による明示的な操作でのみ起きる。
                from src.strategy import policy
                from src.strategy import v2_training
                from src.backtest import execution

                try:
                    result = v2_training.train_v2(
                        {sym: df for sym, df in zip(trained_symbols, dfs)},
                        policy_conf=policy.config_from_settings(),
                        costs=execution.config_from_settings(),
                    )
                    if result.model_id:
                        logger.warning(
                            f"v2候補モデルを作成しました: {result.model_id}"
                            "（運用モデルは変更していません。昇格は明示操作が必要です）"
                        )
                    else:
                        logger.warning(f"v2候補モデルは作られませんでした: {result.skipped_reason}")
                except Exception as e:
                    logger.error(f"v2再学習失敗: {e}")
                return

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

    # ─── 生存確認（ハートビート）─────────────────────────
    def heartbeat(self) -> None:
        """毎営業日の場前に「稼働中です」を通知する。

        異常検知はプロセスが生きていることが前提で、落ちていれば通知も来ない。
        「異常がない」と「死んでいる」を区別できるようにするための能動的な生存信号
        （2026-08-26/27 は通知が来ないことを正常と誤認し、2営業日気づけなかった）。
        """
        try:
            from src.core import heartbeat as hb
            if market_calendar.is_holiday(clock.today()):
                logger.info("ハートビート省略: 本日は休場です")
                return
            mode = self.trading_conf.get("mode", "paper")
            snapshot = self.order_mgr.status_snapshot()
            with get_session() as session:
                count = session.scalar(
                    select(func.count(Position.id)).where(Position.quantity > 0)
                ) or 0
            hb.send(mode, snapshot, count)
        except Exception as e:
            logger.error(f"ハートビートジョブエラー: {e}")

    # ─── 日次レポートのX投稿 ─────────────────────────────
    def post_daily_summary_to_x(self) -> None:
        """当日/週次/月次/総合の損益サマリ・勝率をXへ投稿する（x.enabled=trueのときのみ）。

        基準資金（%算出用）はpaperはtrading.paper_initial_capital、他は
        reference_capital_store（ダッシュボードGUIで設定）から取得する。未設定なら%は省略される。
        """
        try:
            from src.core import reference_capital as ref_capital_store
            from src.core import x_poster
            # 休場日は取引が無く、投稿しても中身が無い（Discord側と同じ扱いに揃える。
            # 対になる処理の片方だけガードが漏れていた）。
            if market_calendar.is_holiday(clock.today()):
                logger.info("X日次レポート省略: 本日は休場です")
                return
            mode = self.trading_conf.get("mode", "paper")
            paper_base = float(self.trading_conf.get("paper_initial_capital", 500_000))
            basis = ref_capital_store.percent_basis(mode, paper_initial_capital=paper_base)
            x_poster.post_daily_report(mode, basis)
        except Exception as e:
            logger.error(f"X日次レポート投稿エラー: {e}")

    # ─── 日次レポートのDiscord投稿 ───────────────────────
    def post_daily_summary_to_discord(self) -> None:
        """当日/週次/月次/総合の損益サマリ・勝率をDiscordへ投稿する
        （discord_report.enabled=trueのときのみ）。基準資金の扱いはpost_daily_summary_to_xと同じ。
        """
        try:
            from src.core import discord_report
            from src.core import reference_capital as ref_capital_store
            if market_calendar.is_holiday(clock.today()):
                logger.info("日次レポート省略: 本日は休場です")
                return
            mode = self.trading_conf.get("mode", "paper")
            paper_base = float(self.trading_conf.get("paper_initial_capital", 500_000))
            basis = ref_capital_store.percent_basis(mode, paper_initial_capital=paper_base)
            discord_report.post_daily_report(mode, basis)
        except Exception as e:
            logger.error(f"Discord日次レポート投稿エラー: {e}")

    def post_weekly_summary_to_discord(self) -> None:
        """1週間の成績（決済件数・週次/月次/総合損益・現在の保有）をDiscordへ投稿する。

        日次レポートは「その日」しか分からないため、週単位で戦略の効き具合を
        振り返れるようにする（discord_report.enabled=true のときのみ）。
        """
        try:
            from src.core import discord_report
            from src.core import reference_capital as ref_capital_store
            mode = self.trading_conf.get("mode", "paper")
            paper_base = float(self.trading_conf.get("paper_initial_capital", 500_000))
            basis = ref_capital_store.percent_basis(mode, paper_initial_capital=paper_base)
            discord_report.post_weekly_report(mode, basis)
        except Exception as e:
            logger.error(f"Discord週次サマリ投稿エラー: {e}")

    # ─── 損切り監視 ─────────────────────────────────────
    def stop_loss_check(self) -> None:
        if not TradingScheduler.is_market_open():
            return
        is_paper = self.trading_conf.get("mode", "paper") == "paper"
        # 監視対象は「ウォッチリスト（新規候補の母集団）」と「実際の保有建玉」の
        # 和集合にする。ウォッチリストのアクティブ切替・銘柄整理は新規シグナル
        # 判断のためのものであり、既に保有している建玉のリスク管理（損切り・
        # トレーリングストップ）はそれとは独立に、保有している限り必ず行う
        # （2026-09-19、9434がアクティブリスト外に出て監視から漏れていた実害を修正）。
        symbols = sorted(set(watchlist_store.get_codes()) | set(_held_symbols()))
        for sym in symbols:
            try:
                qty = _get_position_qty(sym)
                if qty <= 0:
                    continue  # 保有していない銘柄の板取得は無駄なのでスキップ
                if is_paper:
                    # ペーパーモードはリアルタイム板が無いため日足終値で損切り判定する
                    df = load_ohlcv(sym)
                    if self._paper_uses_v2_execution():
                        # v2: 当日の終値で判断して同じ終値で約定する経路を作らない。
                        # 日足しか無い時点では「翌営業日の寄りで退出する」近似に留め、
                        # 判断だけをこの日に行う（実際の退出は翌営業日の
                        # morning_execution が拾う）。
                        price = float(df["open"].iloc[-1]) if len(df) else 0
                    else:
                        price = float(df["close"].iloc[-1]) if len(df) else 0
                else:
                    board = self.client.get_board(sym)
                    price = board.get("CurrentPrice", 0)
                if not price:
                    continue
                should_exit, exit_reason = self.risk.evaluate_exit(sym, price)
                if should_exit:
                    logger.warning(f"退出発動: {sym} ({exit_reason})")
                    label = "利益確定（トレーリングストップ）" if exit_reason == "trailing_stop" else "損切り"
                    if is_paper and self._paper_uses_v2_execution():
                        # v2: 日足の終値/始値で判断した直後に同じ価格帯で約定させない。
                        # signal_scan のSELL経路と同じ合流先（Signal保存）に載せ、
                        # 翌営業日の morning_execution が実際の退出発注（sell）を行う。
                        # ここでは sell_market を呼ばない（判断と執行のセッションを分ける）。
                        _save_signal(TradeSignal(
                            symbol=sym,
                            action="SELL",
                            rule_score=0.0,
                            ml_score=0.0,
                            combined_score=0.0,
                        ))
                        logger.warning(
                            f"退出シグナル保存（翌営業日執行予定）: {sym} reason={exit_reason}"
                        )
                        alert(
                            f"{label}判定（翌営業日寄りで執行予定）",
                            f"{sym} @{price:.0f}円 が退出条件（{exit_reason}）に達しました。"
                            "翌営業日の寄りで退出注文を出します。",
                            level=LEVEL_INFO,
                        )
                        continue
                    # 損切り・トレーリングストップとも確実な約定を優先し成行で発注する
                    # （指値だと急変時に約定しない）。reason経由で日次上限・損失上限等の
                    # 新規発注ゲートをバイパスする（既存リスクを減らす退出操作のため止めない）
                    order_id = self.order_mgr.sell_market(sym, qty, reason=exit_reason)
                    if order_id:
                        alert(f"{label}実行", f"{sym} @{price:.0f}円", level=LEVEL_INFO)
                    else:
                        # 発注が拒否されても「実行」と通知していた（2026-09-08 に発生）。
                        # 9432 の利確が Code 100378 で失敗したのに🟢が飛び、
                        # **売れたと誤認したまま無防備な建玉を持ち続ける**状態になった。
                        # 退出できないのは損失に直結するので critical で知らせる。
                        alert(
                            f"{label}に失敗しました（建玉は残っています）",
                            f"{sym} {qty}株 @{price:.0f}円 の退出注文が拒否されました。"
                            "ログを確認し、必要なら証券会社の画面から手動で決済してください。",
                        )
            except Exception as e:
                logger.error(f"損切りチェックエラー: {sym} {e}")

    # ─── シグナルスキャン ───────────────────────────────
    def signal_scan(self) -> None:
        """16:20（data_update完了後）に翌営業日の売買候補をスキャン。
        ペーパーモードは終値で即時シミュレート"""
        if TradingScheduler.is_maintenance_window():
            return
        if market_calendar.is_holiday(clock.today()):
            # 休場日は前営業日と同じ終値になり、無意味なシグナルがDBに溜まる。
            # 翌営業日の morning_execution が拾う「最新バッチ」を汚さないよう止める。
            logger.info(
                f"シグナルスキャン省略: 本日は休場です（{market_calendar.holiday_name(clock.today())}）"
            )
            return
        logger.info(f"シグナルスキャン開始... batch={self._data_batch_id}")
        is_paper = self.trading_conf.get("mode", "paper") == "paper"
        sectors = watchlist_store.get_sectors()
        paper_base = float(self.trading_conf.get("paper_initial_capital", 500_000))
        excluded_count = 0
        codes = watchlist_store.get_codes()
        for sym in codes:
            try:
                if not self._is_fresh_for_new_candidate(sym):
                    excluded_count += 1
                    st = self._bar_states.get(sym)
                    logger.info(
                        f"新規候補から除外: {sym} "
                        f"（最終足={getattr(st, 'last_bar_session', None)} "
                        f"状態={getattr(st, 'state', 'unknown')}）"
                    )
                    continue
                df = load_ohlcv(sym)
                if len(df) < 30:
                    continue
                sig = gen_signal(sym, df, self.model)
                _save_signal(sig, data_as_of=self._bar_states[sym].last_bar_session)
                if sig.action not in ("BUY", "SELL"):
                    continue
                logger.info(f"シグナル: {sym} → {sig.action} (score={sig.combined_score:.2f})")
                if is_paper:
                    if self._paper_uses_v2_execution():
                        # v2: 引けで判断した注文をその日の終値で約定させない。
                        # 翌営業日の morning_execution が拾うシグナルとして
                        # 保存するだけに留める（判断と執行のセッションを分ける）。
                        continue
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

        if codes and excluded_count == len(codes):
            alert(
                "新規シグナル0件",
                f"全{len(codes)}銘柄が鮮度不足で新規候補から除外されました"
                f"（batch={self._data_batch_id}）。data_updateの遅延・失敗を確認してください。",
                level=LEVEL_WARNING,
            )

    # ─── 前営業日シグナルの発注（朝・後場で共有）─────────────
    def morning_execution(self) -> None:
        """9:05 に前営業日のBUY/SELLシグナルを元に発注する。

        実行対象は paper 以外（live / dry_run / semi_live）。発注の実体は OrderManager が
        モードに応じて分岐する（live=実発注 / dry_run=実発注せず記録のみ / semi_live=承認キュー）。
        paper（legacy）は signal_scan 内で当日終値で即時シミュレートするため morning は不要。
        ただし paper かつ engine_version: v2 のときは例外で、signal_scan / stop_loss_check が
        保存したBUY/SELLシグナルを翌営業日の寄りでここが拾う（_paper_uses_v2_execution）。
        """
        self._execute_pending_signals(source="morning_execution", skip_existing=False)

    def afternoon_execution(self) -> None:
        """12:35（後場寄り）に、朝に見送った銘柄を拾い直す取引頻度向上用スロット。

        朝は資金不足・板薄・スプレッド超過等で見送られる銘柄が出る。同じ前営業日
        シグナルセットを後場でも一度だけ再評価し、寄り付き直後より板が安定した
        タイミングで拾えるようにする（2026-08-25 取引頻度向上のため追加）。

        skip_existing=True で「本日既にBUYが成立/進行中」の銘柄をBUY対象から除外する
        （＝朝と後場で同じシグナルを二重に約定させない）。前日までの保有銘柄は対象外
        （2026-09-09 以前は保有の有無で判定しており、買い増しも一律止めていた。
        朝(skip_existing=False)は買い増しを許可しており、後場だけ全面禁止する理由が
        無かった）。未約定注文がある銘柄は OrderManager._has_pending_order() が別途止める。
        """
        self._execute_pending_signals(source="afternoon_execution", skip_existing=True)

    def catchup_execution(self) -> None:
        """認証切れからの復帰時に、逃した発注機会を場中なら即座に取り返す。

        朝のログインが遅れると 9:05 の morning_execution が401で失敗し、
        その日は 12:35 の後場スロットまで発注機会が無い（2026-08-31 は
        10:58 復帰で朝の発注を逃し、当日の約定が0件だった）。
        復帰した時点が場中なら、その場で未保有銘柄への発注を試みる。

        締切（既定14:00）を過ぎていたら何もしない。朝の発注は「寄り付き直後の
        値動きが落ち着いた時間」を狙う設計であり、引けに近い時間帯の約定は
        想定と異なるため（no_new_buy_minutes_before_close と同じ思想）。

        skip_existing=True で「本日既にBUYが成立/進行中」の銘柄を除外し、既存の
        二重発注ガード（_has_pending_order）と合わせて重複発注を防ぐ。前日までの
        保有銘柄は対象外＝買い増しを許す（afternoon_execution と同じ判断。2026-09-09）。
        """
        if not TradingScheduler.is_market_open():
            logger.info("認証復帰: 場外のため発注のキャッチアップは行いません")
            return
        deadline = self.trading_conf.get("catchup_deadline_hour", 14)
        if deadline and not TradingScheduler.is_before(int(deadline)):
            logger.info(
                f"認証復帰: 締切({deadline}:00)を過ぎているため発注のキャッチアップを見送ります"
            )
            return
        logger.info("認証復帰: 逃した発注機会のキャッチアップを実行します")
        self._execute_pending_signals(source="catchup_execution", skip_existing=True)

    def _execute_pending_signals(self, *, source: str, skip_existing: bool) -> None:
        """直近バッチのBUY/SELLシグナルを元に発注する（morning/afternoon共通本体）。

        source はジョブ名（OrderIntent.source・ログ表記に使う）。
        skip_existing=True のときは、BUY候補のうち本日既にBUYが成立/進行中の銘柄を
        対象から除外する（同日の二度打ち防止。前日までの保有は対象外＝買い増しを許す）。

        paper（v2）はここへ合流するが、legacy paper 経路と異なり板API・余力APIへの
        接続を前提にできない。paper のときは板・余力の取得をローカル日足
        （load_ohlcv）と仮想ウォレット（_paper_available_cash）へフォールバックする
        （signal_scan のlegacy分岐と同じパターン）。接続の無い環境で paper v2 を
        有効にすると client.get_board/get_wallet が必ず失敗し、BUYは全滅・SELLは
        個別exceptで沈黙する事故になっていた（最終ブランチレビュー Important 3）。
        live/dry_run/semi_live はこの分岐に入らず、従来通り client を使う。
        """
        mode = self.trading_conf.get("mode", "paper")
        if not tm.uses_morning_execution(mode) and not (
            mode == "paper" and self._paper_uses_v2_execution()
        ):
            return
        if not TradingScheduler.is_market_open():
            return
        with get_session() as session:
            pending = _select_latest_signals(session)

        if not pending:
            return

        is_paper = mode == "paper"
        buy_signals = [s for s in pending if s.action == "BUY"]
        sell_signals = [s for s in pending if s.action == "SELL"]
        label = "後場" if skip_existing else "朝"

        # ── SELL シグナル: 保有ポジションがあれば売る ─────────────
        for sig in sell_signals:
            try:
                qty = _get_position_qty(sig.symbol)
                if qty <= 0:
                    continue
                if is_paper:
                    # ペーパーモードはリアルタイム板が無いため直近日足の終値を使う
                    # （stop_loss_check のpaper経路と同じ近似。kabuステーション
                    # 接続の無い環境でも発注処理まで進める）
                    df = load_ohlcv(sig.symbol)
                    price = float(df["close"].iloc[-1]) if len(df) else 0
                else:
                    board = self.client.get_board(sig.symbol)
                    price = board.get("CurrentPrice") or board.get("Buy1", {}).get("Price", 0)
                if not price:
                    continue
                self.order_mgr.sell(sig.symbol, float(price), qty,
                                    rationale=_signal_rationale(sig),
                                    source=source)
                logger.info(f"{label}売り発注: {sig.symbol} {qty}株 @{price:.0f}円")
                alert(f"{label}売り発注",
                      f"{sig.symbol} {qty}株 @{price:,.0f}円（売りシグナルによる手仕舞い）",
                      level=LEVEL_INFO)
            except Exception as e:
                logger.error(f"{label}売り発注失敗: {sig.symbol} {e}")

        # ── BUY シグナル: 余力を確認して買う ──────────────────────
        # 同日の二度打ちだけを防ぐ。前日までの保有銘柄への買い増しは許可する
        # （2026-09-09 以前は保有の有無で一律除外しており、朝が許可している
        # 買い増しと矛盾していた）。1銘柄あたりの上限は calc_position_size 側で
        # 既存保有評価額を差し引いた残り枠として計算される。
        if skip_existing:
            buy_signals = [s for s in buy_signals if not _bought_today(s.symbol)]
        if not buy_signals:
            return
        # 大引け間際の新規BUYは薄商い・不利約定を招きやすいので見送る（P0-6。0で無効）
        near_close_min = int(self.liquidity_conf.get("no_new_buy_minutes_before_close", 0) or 0)
        if TradingScheduler.is_near_close(near_close_min):
            logger.info(f"{label}買い見送り: 大引け{near_close_min}分前以降のため新規BUYを抑止")
            return
        if is_paper:
            # 固定額ではなく仮想ウォレット残高で発注サイズを決める
            # （signal_scanのlegacy分岐と同じ関数。client.get_wallet()は呼ばない）
            paper_base = float(self.trading_conf.get("paper_initial_capital", 500_000))
            cash = _paper_available_cash(paper_base)
        else:
            try:
                wallet = self.client.get_wallet()
                cash = float(wallet.get("StockAccountWallet", 0))
            except Exception as e:
                logger.error(f"余力取得失敗: {e}")
                return
        sectors = watchlist_store.get_sectors()
        for sig in buy_signals:
            try:
                if is_paper:
                    df = load_ohlcv(sig.symbol)
                    price = float(df["close"].iloc[-1]) if len(df) else 0
                    board = None
                else:
                    board = self.client.get_board(sig.symbol)
                    price = board.get("CurrentPrice") or board.get("Sell1", {}).get("Price", 0)
                if not price:
                    continue
                sector = sectors.get(sig.symbol, "")
                ok, reason = self.risk.validate_buy(sig.symbol, float(price), cash, sector)
                if not ok:
                    logger.info(f"{label}買い見送り: {sig.symbol} - {reason}")
                    continue
                ok_liq, liq_reason = liquidity.check_liquidity(
                    sig.symbol, df if is_paper else load_ohlcv(sig.symbol),
                    self.liquidity_conf)
                if not ok_liq:
                    logger.info(f"{label}買い見送り: {liq_reason}")
                    continue
                if is_paper:
                    # 板が無いのでスプレッド判定は行わない
                    # （liquidity.check_spread のdocstring: ペーパー/バックテスト
                    # では板が無いため使わない）
                    ok_sp, sp_reason = True, ""
                else:
                    ok_sp, sp_reason = liquidity.check_spread(board, self.liquidity_conf)
                if not ok_sp:
                    logger.info(f"{label}買い見送り: {sig.symbol} - {sp_reason}")
                    continue
                qty = self.risk.calc_position_size(sig.symbol, float(price), cash)
                if qty <= 0:
                    continue
                order_id = self.order_mgr.buy(sig.symbol, float(price), qty, sector=sector,
                                              rationale=_signal_rationale(sig),
                                              source=source)
                if order_id:
                    # 同一スキャン内の以降の銘柄が同じ余力を前提に判定しないよう、
                    # 発注成功分をその場で減算する（複数銘柄の資金二重計上を防ぐ。
                    # RiskManager の未約定引当と二重で守る）
                    cash -= float(price) * qty
                    logger.info(f"{label}買い発注: {sig.symbol} {qty}株 @{price:.0f}円")
                    # 実弾が動いたことは即座に知らせる（退出だけ通知して
                    # エントリーを通知しないのは非対称で、口座で今いくら使われたかが
                    # 日次レポートまで分からない状態になっていた）
                    alert(f"{label}買い発注",
                          f"{sig.symbol} {qty}株 @{price:,.0f}円"
                          f"（約定額 {float(price) * qty:,.0f}円）",
                          level=LEVEL_INFO)
                else:
                    logger.warning(f"{label}買い発注失敗（注文拒否）: {sig.symbol}")
            except Exception as e:
                logger.error(f"{label}買い発注失敗: {sig.symbol} {e}")
