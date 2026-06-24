"""
リスク管理モジュール
- 1銘柄あたりの最大投資額
- 損切りライン
- 最大保有銘柄数
- セクター集中制限
- ギャップリスク考慮
"""
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from sqlalchemy import func, select

from src.core import clock
from src.core import config as cfg
from src.core import halt
from src.data.database import Position, Trade, get_session
from src.data.market_data import latest_closes
from src.execution import order_status as st


@dataclass
class RiskSnapshot:
    """リスク判定に必要なDB読取を1回にまとめたスナップショット（P2-6）。

    validate_buy() は can_place_order / check_max_positions / calc_position_size /
    check_sector_concentration を続けて呼び、それぞれが positions / 未約定BUY / 未解決数 /
    最新終値を重複して問い合わせていた。発注判定の途中で口座状態がバラバラに変わる
    （バッチ非整合）のも避けたいため、判定開始時に一括取得した本スナップショットを各チェックへ
    渡す。snapshot を渡さない呼び出しは従来どおり個別にDBを引く（後方互換）。
    """
    positions: list = field(default_factory=list)       # quantity>0 の Position
    open_buys: list = field(default_factory=list)        # 未約定BUYの Trade
    pos_sector: dict = field(default_factory=dict)       # symbol→sector（全Position）
    unresolved_count: int = 0
    closes: dict = field(default_factory=dict)           # symbol→最新終値


class RiskManager:
    def __init__(self):
        self._daily_order_count = 0
        self._daily_loss_yen = 0.0
        self._conf = cfg.get_section("trading")

    def reset_daily_counters(self) -> None:
        self._daily_order_count = 0
        self._daily_loss_yen = 0.0

    def restore_daily_state(self) -> None:
        """起動時に当日の注文数・実現損失をDBから復元する。

        日次カウンタはメモリ上のみのため、当日損失上限に達して停止した後に
        プロセスを再起動するとセーフティが消えてしまう。これを防ぐため、
        当日（JST）約定済みTradeから注文数と損失額を再構築する。
        """
        today = clock.today()
        with get_session() as session:
            trades = session.scalars(
                select(Trade).where(Trade.filled_at.isnot(None))
            ).all()
            order_count = 0
            loss_yen = 0.0
            for t in trades:
                if t.filled_at and t.filled_at.date() == today:
                    order_count += 1
                    if t.pnl is not None and t.pnl < 0:
                        loss_yen += abs(t.pnl)
        self._daily_order_count = order_count
        self._daily_loss_yen = loss_yen
        if order_count or loss_yen:
            logger.info(
                f"日次リスク状態を復元: 当日注文数={order_count} 当日実現損失={loss_yen:,.0f}円"
            )

    def record_loss(self, pnl: float) -> None:
        """損益を記録する（損失のみ累積）"""
        if pnl < 0:
            self._daily_loss_yen += abs(pnl)

    def build_snapshot(self) -> RiskSnapshot:
        """リスク判定に必要なDB読取を1回のセッションでまとめて取得する（P2-6）。"""
        with get_session() as session:
            positions = list(session.scalars(
                select(Position).where(Position.quantity > 0)
            ).all())
            open_buys = list(session.scalars(
                select(Trade).where(
                    Trade.side == "BUY",
                    Trade.status.in_(tuple(st.OPEN_STATUSES)),
                )
            ).all())
            pos_sector = {p.symbol: p.sector for p in session.scalars(select(Position)).all()}
            unresolved = session.scalar(
                select(func.count(Trade.id)).where(
                    Trade.status.in_(tuple(st.UNRESOLVED_STATUSES))
                )
            ) or 0
        closes = latest_closes([p.symbol for p in positions])
        return RiskSnapshot(
            positions=positions, open_buys=open_buys, pos_sector=pos_sector,
            unresolved_count=unresolved, closes=closes,
        )

    def current_daily_loss(self) -> float:
        """当日の累積実現損失（円・正の値）。異常監視・可視化用。"""
        return self._daily_loss_yen

    def daily_loss_limit(self) -> float:
        """当日損失上限（円。0 で無効）。"""
        return self._conf.get("max_daily_loss", 0)

    def unrealized_pnl(self, snapshot: Optional[RiskSnapshot] = None) -> float:
        """保有建玉の含み損益（円・正=含み益 / 負=含み損）を返す。

        最新終値が取得できない銘柄は損益0扱いで合計から除外する（avg_costで代用すると
        常に0になり「データが無い」ことを隠してしまうため）。除外した銘柄は
        `unpriced_symbols()` で検知できるようにし、health.py がこれを警告として
        表面化する（合計ドローダウンが静かに過小評価されたままにならないようにするため。
        レビュー再指摘 Critical）。
        """
        total, _unpriced = self._unrealized_pnl_with_gaps(snapshot)
        return total

    def unpriced_symbols(self, snapshot: Optional[RiskSnapshot] = None) -> list[str]:
        """保有中だが最新終値が取得できず、含み損益の計算から除外されている銘柄一覧。"""
        _total, unpriced = self._unrealized_pnl_with_gaps(snapshot)
        return unpriced

    def _unrealized_pnl_with_gaps(
        self, snapshot: Optional[RiskSnapshot] = None
    ) -> tuple[float, list[str]]:
        if snapshot is not None:
            positions = snapshot.positions
            closes = snapshot.closes
        else:
            with get_session() as session:
                positions = list(session.scalars(
                    select(Position).where(Position.quantity > 0)
                ).all())
            closes = latest_closes([p.symbol for p in positions])
        total = 0.0
        unpriced: list[str] = []
        for p in positions:
            close = closes.get(p.symbol)
            if close and p.avg_cost:
                total += (close - p.avg_cost) * p.quantity
            elif p.avg_cost:
                unpriced.append(p.symbol)
        return total, unpriced

    def current_total_drawdown(self, snapshot: Optional[RiskSnapshot] = None) -> float:
        """当日の合計ドローダウン（円・正の値）= 実現損失 + 現在の含み損。

        実現損失だけでは「決済前の大きな含み損」を見逃すため（レビュー P0-5）、
        保有建玉の含み損も合算した保守的なドローダウン指標を返す。含み益は
        実現損失を相殺しない（含み益は確定していないため、安全側に倒す）。
        """
        unrealized = self.unrealized_pnl(snapshot)
        return self._daily_loss_yen + max(0.0, -unrealized)

    def is_daily_loss_limit_reached(self) -> tuple[bool, str]:
        """当日損失上限チェック（実現損失のみ）。(over_limit, reason) を返す"""
        limit = self._conf.get("max_daily_loss", 0)
        if limit <= 0:
            return False, ""
        if self._daily_loss_yen >= limit:
            return True, f"当日損失上限({limit:,.0f}円)に達しました"
        return False, ""

    def is_total_loss_limit_reached(
        self, snapshot: Optional[RiskSnapshot] = None
    ) -> tuple[bool, str]:
        """合計ドローダウン（実現損失+含み損）が上限に達したか。(over_limit, reason) を返す。

        含み損を含めることで、含み損を抱えたまま新規発注を続けてリスクを積み増す事故を
        防ぐ（レビュー P0-5）。発注ゲート(can_place_order)とハルト判定(health)の双方で使う。
        """
        limit = self._conf.get("max_daily_loss", 0)
        if limit <= 0:
            return False, ""
        total = self.current_total_drawdown(snapshot)
        if total >= limit:
            realized = self._daily_loss_yen
            unrealized_loss = max(0.0, -self.unrealized_pnl(snapshot))
            return True, (
                f"当日合計ドローダウン上限({limit:,.0f}円)に達しました"
                f"（実現損失{realized:,.0f}円 + 含み損{unrealized_loss:,.0f}円）"
            )
        return False, ""

    def can_place_order(self, snapshot: Optional[RiskSnapshot] = None) -> tuple[bool, str]:
        """注文可能かチェック。(ok, reason) を返す"""
        # 取引停止スイッチ（kill switch）が ON なら全ての新規発注を抑止する。
        # 損切り・緊急決済は sell_market(reason=...) でこのゲート自体をバイパスするため、
        # 停止中でも退出だけは止まらない（既存リスクを増やさない退出は妨げない方針）。
        if halt.is_halted():
            state = halt.get_state()
            return False, f"取引停止中（kill switch ON）: {state.get('reason') or '手動停止'}"
        limit = self._conf.get("daily_order_limit", 100)
        if self._daily_order_count >= limit:
            return False, f"1日の注文上限({limit})に達しました"
        # 実現損失だけでなく含み損も合算した合計ドローダウンで新規発注を抑止する（P0-5）。
        # 退出系(sell_market reason=...)はこのゲート自体をバイパスするため止まらない。
        over, reason = self.is_total_loss_limit_reached(snapshot)
        if over:
            return False, reason
        # 状態不明・キャンセル失敗の注文が残っている間は新規発注を抑止する
        # （実口座と乖離したまま発注すると二重発注・想定外建玉になるため、要人手確認）
        unresolved = (snapshot.unresolved_count if snapshot is not None
                      else self._count_unresolved_orders())
        if unresolved:
            return False, f"未解決の注文が{unresolved}件あります（要確認）"
        return True, ""

    def _count_unresolved_orders(self) -> int:
        """要人手確認の異常注文（UNKNOWN / CANCEL_FAILED）の件数を返す"""
        with get_session() as session:
            return session.scalar(
                select(func.count(Trade.id)).where(
                    Trade.status.in_(tuple(st.UNRESOLVED_STATUSES))
                )
            ) or 0

    def _reserved_buy_by_sector(
        self, snapshot: Optional[RiskSnapshot] = None
    ) -> tuple[float, dict[str, float]]:
        """未約定BUY注文の引当金額 (総額, セクター別内訳) を返す。

        price × 残数（発注数量 - 約定済数量）で算定する。セクターは発注時に記録した
        `Trade.sector` を優先する（新規銘柄はPositionが未作成のため、Position経由の
        解決では漏れる）。`Trade.sector` が無い古いレコードのみPositionへフォールバック。
        """
        if snapshot is not None:
            trades = snapshot.open_buys
            pos_sector = snapshot.pos_sector
        else:
            with get_session() as session:
                trades = session.scalars(
                    select(Trade).where(
                        Trade.side == "BUY",
                        Trade.status.in_(tuple(st.OPEN_STATUSES)),
                    )
                ).all()
                pos_sector = {
                    p.symbol: p.sector
                    for p in session.scalars(select(Position)).all()
                }
        total = 0.0
        by_sector: dict[str, float] = {}
        for t in trades:
            remaining = (t.quantity or 0) - (t.filled_quantity or 0)
            if remaining <= 0 or not t.price:
                continue
            notional = t.price * remaining
            total += notional
            sec = t.sector or pos_sector.get(t.symbol)
            if sec:
                by_sector[sec] = by_sector.get(sec, 0.0) + notional
        return total, by_sector

    def increment_order_count(self) -> None:
        self._daily_order_count += 1

    def calc_position_size(self, symbol: str, price: float,
                           cash_balance: float,
                           snapshot: Optional[RiskSnapshot] = None) -> int:
        """購入株数を計算する（最大投資額を超えない範囲）"""
        max_ratio = self._conf.get("max_position_ratio", 0.20)
        # 未約定BUYの引当を差し引いた実効余力で上限を計算する
        # （未約定中の多重発注で余力を二重に使う事故を防ぐ。main側の逐次減算と二重で守る）
        reserved, _ = self._reserved_buy_by_sector(snapshot)
        available = max(0.0, cash_balance - reserved)
        max_amount = available * max_ratio
        if price <= 0:
            return 0
        lot = 100  # 東証は通常100株単位
        units = int(max_amount / (price * lot))
        quantity = units * lot
        logger.debug(f"{symbol}: 購入可能数={quantity}株 (価格={price:.0f}, 上限={max_amount:.0f}円)")
        return quantity

    def check_max_positions(self, candidate_symbol: Optional[str] = None,
                            snapshot: Optional[RiskSnapshot] = None) -> tuple[bool, str]:
        """最大保有銘柄数チェック。

        保有Positionの数だけでなく、未約定BUYの銘柄（まだPosition化されていない
        新規銘柄も含む）も合算した銘柄集合で判定する。保有3/上限4のときに未約定BUYが
        2件あると、どちらも個別には通って合計5銘柄になってしまう抜けを防ぐ
        （再レビュー P1-1）。`candidate_symbol` を渡すと、その銘柄を追加した場合を
        含めて判定する（既に保有/未約定の銘柄であれば集合は増えないため通る）。
        """
        max_pos = self._conf.get("max_positions", 5)
        if snapshot is not None:
            active_symbols = {p.symbol for p in snapshot.positions}
            reserved_symbols = {t.symbol for t in snapshot.open_buys}
        else:
            with get_session() as session:
                active_symbols = {
                    p.symbol for p in session.scalars(
                        select(Position).where(Position.quantity > 0)
                    ).all()
                }
                reserved_symbols = {
                    t.symbol for t in session.scalars(
                        select(Trade).where(
                            Trade.side == "BUY",
                            Trade.status.in_(tuple(st.OPEN_STATUSES)),
                        )
                    ).all()
                }
        after_symbols = active_symbols | reserved_symbols
        if candidate_symbol:
            after_symbols = after_symbols | {candidate_symbol}
        if len(after_symbols) > max_pos:
            return False, f"最大保有銘柄数({max_pos})に達しています"
        return True, ""

    def check_sector_concentration(self, sector: str,
                                   candidate_notional: float = 0.0,
                                   snapshot: Optional[RiskSnapshot] = None) -> tuple[bool, str]:
        """同一セクターの集中投資チェック（建玉の時価評価額ベース）。

        銘柄数の比率ではなく、quantity × 最新終値 のエクスポージャー金額で判定する
        （小口1銘柄と大型1銘柄が同じ「1銘柄」として扱われる粗さを避けるため）。
        最新終値が取得できない銘柄は avg_cost（取得平均単価）で代用する。
        `candidate_notional` には**これから出す注文自体の金額**を渡す（再レビュー P1-2）。
        既存保有・未約定BUYの引当だけでは「この注文を加えたら超過する」ケースを
        発注前に弾けないため、判定対象の候補金額も合算してから比較する。
        """
        max_ratio = self._conf.get("max_sector_ratio", 0.40)
        if snapshot is not None:
            all_pos = snapshot.positions
        else:
            with get_session() as session:
                all_pos = session.scalars(select(Position).where(Position.quantity > 0)).all()
        # 未約定BUYの引当も「買った後の集中度」として加味する（発注前に上限超過を防ぐ）
        reserved_total, reserved_by_sector = self._reserved_buy_by_sector(snapshot)
        if not all_pos and reserved_total <= 0 and candidate_notional <= 0:
            return True, ""
        closes = snapshot.closes if snapshot is not None else latest_closes([p.symbol for p in all_pos])
        total_value = reserved_total + candidate_notional
        same_sector_value = reserved_by_sector.get(sector, 0.0) + candidate_notional
        for p in all_pos:
            price = closes.get(p.symbol) or p.avg_cost
            value = p.quantity * price
            total_value += value
            if p.sector == sector:
                same_sector_value += value
        if total_value <= 0:
            return True, ""
        ratio = same_sector_value / total_value
        if ratio >= max_ratio:
            return False, f"セクター集中率が上限({max_ratio:.0%})超: {sector}"
        return True, ""

    def evaluate_exit(self, symbol: str, current_price: float) -> tuple[bool, str]:
        """損切り・ブレークイーブン・トレーリングストップを統合した退出判定。

        呼び出しのたびに保有開始以降の最高値(Position.peak_price)を観測値で更新・永続化する
        （5分間隔のstop_loss_check等から呼ばれる前提。市場時間中ずっと監視するのはこの値のため）。

        ロジック:
          1. 基準ライン = avg_cost × (1 + stop_loss_pct)（従来の損切りラインそのまま）
          2. ピーク時の含み益率が breakeven_trigger_pct 以上に達したら、基準ラインを
             avg_cost（ブレークイーブン）まで引き上げる（一度上がった分、元本割れの
             リスクを取らない）。
          3. 同時にトレーリングストップ（ピーク × (1 - trailing_stop_pct)）も有効化し、
             基準ラインと比べて高い方（より安全な方）を採用する（ピークが伸びるほど
             ラインも追随して上がる）。
          breakeven_trigger_pct / trailing_stop_pct が0（未設定）ならこの2機能は無効になり、
          従来の損切りのみの挙動と完全に一致する。

        戻り値: (退出すべきか, 理由)。理由は "stop_loss"（基準ライン到達） /
        "trailing_stop"（ブレークイーブン/トレーリングで保全した利益の確定）。
        """
        stop_pct = self._conf.get("stop_loss_pct", -0.05)
        trailing_pct = self._conf.get("trailing_stop_pct", 0.0)
        breakeven_trigger_pct = self._conf.get("breakeven_trigger_pct", 0.0)

        with get_session() as session:
            pos = session.scalar(select(Position).where(Position.symbol == symbol))
            if pos is None or pos.avg_cost <= 0:
                return False, ""
            avg_cost = pos.avg_cost
            new_peak = max(pos.peak_price or avg_cost, current_price)
            if new_peak != pos.peak_price:
                pos.peak_price = new_peak
                session.commit()
            peak = new_peak

        effective_stop_price = avg_cost * (1 + stop_pct)
        peak_gain_pct = (peak - avg_cost) / avg_cost
        armed = breakeven_trigger_pct > 0 and peak_gain_pct >= breakeven_trigger_pct
        if armed:
            effective_stop_price = max(effective_stop_price, avg_cost)
            if trailing_pct > 0:
                effective_stop_price = max(effective_stop_price, peak * (1 - trailing_pct))

        if current_price <= effective_stop_price:
            pnl_pct = (current_price - avg_cost) / avg_cost
            reason = "trailing_stop" if armed else "stop_loss"
            label = "トレーリングストップ/ブレークイーブン到達" if armed else "損切りライン到達"
            logger.warning(f"{label}: {symbol} 損益率={pnl_pct:.2%} ピーク以降の最高値={peak:.0f}")
            return True, reason
        return False, ""

    def validate_buy(self, symbol: str, price: float,
                     cash_balance: float, sector: Optional[str] = None) -> tuple[bool, str]:
        """買い注文の総合バリデーション。

        判定開始時に一括スナップショットを取り、各チェックへ渡す（DB往復削減・
        判定中の口座状態の揺れを防ぐバッチ整合。P2-6）。
        """
        snap = self.build_snapshot()
        ok, reason = self.can_place_order(snap)
        if not ok:
            return False, reason
        ok, reason = self.check_max_positions(candidate_symbol=symbol, snapshot=snap)
        if not ok:
            return False, reason
        quantity = self.calc_position_size(symbol, price, cash_balance, snapshot=snap)
        if quantity <= 0:
            return False, f"余力不足: {symbol}"
        if sector:
            # この注文自体の金額もセクター集中度に加味する（発注前に超過を弾く。再レビュー P1-2）
            ok, reason = self.check_sector_concentration(
                sector, candidate_notional=price * quantity, snapshot=snap)
            if not ok:
                return False, reason
        return True, ""
