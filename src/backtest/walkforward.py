"""ポートフォリオ walk-forward エンジン。

1営業日を5フェーズで回す。

  ①前日までに決まった注文の執行 → ②保有と現金の更新 →
  ③退出ポリシーの逐次駆動 → ④日末のNAV記録 → ⑤翌日の候補生成

**Tの終値の情報はT+1以降の注文にしか使えない。** 現行 src/backtest/engine.py は
同じ終値でスコア生成と約定を行っており、引け後スキャン→翌朝発注という実運用と
乖離していた（レビューF04）。

判断は src/strategy/policy.py、約定は src/backtest/execution.py、現金と保有は
src/backtest/portfolio.py に委ね、本モジュールは進行と記録だけを担う。
旧 engine.py は strategy.engine_version: legacy の受け皿として無改造で残す。
"""
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest import execution
from src.backtest import portfolio as pf
from src.strategy import policy

_DAILY_COLUMNS = ["session", "nav", "cash", "n_holdings", "realized_pnl"]
_TRADE_COLUMNS = ["symbol", "entry_at", "entry_price", "exit_at", "exit_price",
                  "quantity", "pnl", "reason"]
_REJECTED_COLUMNS = ["session", "symbol", "reason"]
_MODEL_USAGE_COLUMNS = ["model_id", "from_session", "to_session"]


@dataclass(frozen=True)
class MarketData:
    """銘柄ごとの日足・業種・特徴量。

    bars は symbol → 日付インデックスの**生OHLCV** DataFrame。
    約定価格・必要資金・出来高はここから取る（`price_basis="raw"`）。
    銘柄ごとに長さが違ってよい（上場・上場廃止・データ欠損）。

    features は symbol → **特徴量フレーム**（`indicators.build_feature_frame()`
    の出力に `rule_score` 列を足したもの）。調整済み系列
    （`price_basis="adjusted"`）から因果的に作る。判断（`decide`）と
    売りスコアはここから取る。

    **生OHLCVをそのまま判断側へ渡してはいけない。** そうすると `rule_score`
    も特徴量も存在しない行が `row.get(col, 0.0)` で0に埋まり、正の買い閾値の
    下では全候補が落ちる。「取引ゼロの正常なバックテスト」に見えるが、
    実際には特徴量が一度も繋がっていない（外部レビューR03）。
    欠けた特徴量を0で補完しないこと。

    `feature_valid=False` の行（助走期間・欠損）は判断にも売りスコアにも
    使わない。日付が features に無い銘柄はその日は判断対象外とする。
    """
    bars: dict
    sectors: dict
    # 既定は空。空のまま実戦略（make_rule_then_ml）を回すと rule_score が
    # 無いので**例外**になる。0で埋めて静かに全件見送りにはしない。
    features: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DailyRow:
    session: date
    nav: float
    cash: float
    n_holdings: int
    realized_pnl: float


@dataclass(frozen=True)
class WalkForwardResult:
    """実行結果。最終リターンだけでなく、日次NAV・約定・除外理由まで辿れる。"""
    daily: pd.DataFrame
    trades: pd.DataFrame
    rejected: pd.DataFrame
    degraded: bool
    degraded_reasons: list
    model_usage: pd.DataFrame
    run_id: Optional[int] = None


def sessions_between(md: MarketData, start: date, end: date) -> list:
    """全銘柄共通の営業日（各銘柄の足の日付の和集合）を昇順で返す。"""
    all_sessions = set()
    for df in md.bars.values():
        for ts in df.index:
            all_sessions.add(ts.date())
    return sorted(s for s in all_sessions if start <= s <= end)


def _bar_of(md: MarketData, symbol: str, session: date) -> Optional[pd.Series]:
    df = md.bars.get(symbol)
    if df is None:
        return None
    key = pd.Timestamp(session)
    if key not in df.index:
        return None
    return df.loc[key]


def closes_at(md: MarketData, session: date) -> dict:
    """その日の終値。足が無い銘柄は含めない（呼び出し側が取得単価へ落とす）。"""
    out = {}
    for symbol in md.bars:
        row = _bar_of(md, symbol, session)
        if row is not None:
            out[symbol] = float(row["close"])
    return out


def _row_for(md: MarketData, symbol: str, session: date):
    """判断（decide）へ渡す1行。生OHLCVに特徴量とrule_scoreを重ねたもの。

    特徴量が用意されていない銘柄は生の足だけを返す（テスト用のスタブ戦略が
    使う経路）。**実戦略はこの行に `rule_score` が無ければ例外を投げる。**
    0で埋めて「候補ゼロの正常なバックテスト」に見せない（外部レビューR03）。

    特徴量フレームにその日付があり `feature_valid` が False なら、
    その日は判断対象外として None を返す（助走期間・欠損）。
    """
    bar = _bar_of(md, symbol, session)
    if bar is None:
        return None
    feats = md.features.get(symbol)
    if feats is None:
        return bar
    ts = pd.Timestamp(session)
    if ts not in feats.index:
        return None
    frow = feats.loc[ts]
    if "feature_valid" in feats.columns and not bool(frow["feature_valid"]):
        return None
    merged = dict(bar)
    merged.update({k: frow[k] for k in feats.columns})
    return pd.Series(merged)


def _exit_score(md: MarketData, symbol: str, session: date,
                exit_score_fn: Optional[Callable]) -> Optional[float]:
    """保有銘柄の売りスコア。当日までの情報だけで作る。

    これを繋がないと policy の SIGNAL_SELL 条件が一度も成立せず、
    ストップか満了まで持ち続ける挙動になる（外部レビューR10）。
    ラベル生成（dataset.simulate_event）と同じ契約にすること。
    """
    if exit_score_fn is None:
        return None
    row = _row_for(md, symbol, session)
    if row is None:
        return None
    return exit_score_fn(symbol, row)


def _observation(md: MarketData, symbol: str, session: date,
                 score: Optional[float] = None) -> Optional[policy.Observation]:
    row = _bar_of(md, symbol, session)
    if row is None:
        return None
    return policy.Observation(
        session=session, open=float(row["open"]), high=float(row["high"]),
        low=float(row["low"]), close=float(row["close"]), score=score,
    )


def run_walkforward(md: MarketData, start: date, end: date, *,
                    initial_capital: float,
                    decide: Callable,
                    policy_conf: policy.PolicyConfig,
                    costs: execution.CostConfig,
                    sizing: pf.SizingConfig,
                    liquidity: execution.LiquidityConfig,
                    model=None,
                    exit_score_fn: Optional[Callable] = None,
                    ) -> WalkForwardResult:
    """日次5フェーズでポートフォリオを進める。

      ①前日までに決まった注文の執行 → ②保有と現金の更新 →
      ③退出ポリシーの逐次駆動 → ④日末のNAV記録 → ⑤翌日の候補生成

    **Tの終値の情報はT+1以降の注文にしか使えない。** decide() が返した候補は
    その日には約定せず、翌営業日の寄りで執行される。

    decide は判断規則（戦略バージョン）。
    `decide(session, rows, model, ctx) -> list[portfolio.Candidate]`。

    **model は「その判断時点で利用可能だったモデル」でなければならない。**
    本タスクの時点では引数の `model` を全期間で使う。これは**診断用の
    固定モデル実行**であり、採否の根拠にできる walk-forward 成績ではない。
    再学習の結線（`retrain` / `train_model`）は後続タスクで足す。
    固定モデル実行を昇格の根拠に使わせないための degraded 判定も
    そのタスクで入れる（外部レビューR04）。

    exit_score_fn は保有銘柄の売りスコア。`exit_score_fn(symbol, row) -> float|None`。
    渡さないと policy の SIGNAL_SELL が一度も成立しない（外部レビューR10）。
    """
    sessions = sessions_between(md, start, end)
    state = pf.empty_portfolio(initial_capital)
    daily: list = []
    trades: list = []
    rejected: list = []
    pending_buys: list = []
    pending_exits: list = []

    for session in sessions:
        realized = 0.0
        closes = closes_at(md, session)
        # 出来高の枠はこの営業日ぶん。買いと売りで共有する（外部レビューR09）
        budget = execution.VolumeBudget(liquidity)

        # ① 前日までに決まった成行退出を、この日の寄りで約定させる
        state, exit_trades, pending_exits = _settle_pending_exits(
            state, md, session, pending_exits, costs, budget)
        for t in exit_trades:
            realized += t["pnl"]
        trades.extend(exit_trades)

        # ① 前日までに決まった買いを、この日の寄りで約定させる（②保有と現金の更新）
        #
        # 数量は**前日の終値**で決めてある。約定は翌朝の寄りなので、
        # ギャップアップすると必要額が枠を超える。約定時点の価格で
        # 買える株数まで縮め、それでも1単元に届かなければ見送る。
        # ここを飛ばすと現金が負のままバックテストが進み、NAVも成績も
        # 意味を失う（外部レビューR08）。
        for order in pending_buys:
            bar = _bar_of(md, order.symbol, session)
            if bar is None:
                rejected.append({"session": session, "symbol": order.symbol,
                                 "reason": "この営業日の足が無く約定できません"})
                continue
            obs = _observation(md, order.symbol, session)

            # 約定価格（寄り×(1+slip)）を先に求めてから数量を決める
            fill_price = execution.buy_fill_price(obs.open, costs)
            affordable = pf.max_affordable_quantity(
                state.cash, fill_price, costs.commission_pct)
            # 1銘柄あたりの上限・セクター集中の上限も約定価格で引き直す
            capped = min(
                order.quantity,
                affordable,
                pf.calc_quantity(state, order.symbol, fill_price, sizing),
            )
            capped = (capped // pf.LOT_SIZE) * pf.LOT_SIZE
            if capped < pf.LOT_SIZE:
                rejected.append({
                    "session": session, "symbol": order.symbol,
                    "reason": (
                        f"約定価格{fill_price:,.1f}では買付余力・上限を満たせません"
                        f"（予定{order.quantity:,}株／現金{state.cash:,.0f}円）"
                    )})
                continue
            if capped < order.quantity:
                rejected.append({
                    "session": session, "symbol": order.symbol,
                    "reason": (
                        f"約定価格{fill_price:,.1f}で数量を縮小"
                        f"（{order.quantity:,}株→{capped:,}株）"
                    )})

            result = execution.entry_fill_limited(
                obs, capped, costs, budget,
                symbol=order.symbol, volume=int(bar["volume"]))
            if result.unfilled_reason:
                rejected.append({"session": session, "symbol": order.symbol,
                                 "reason": result.unfilled_reason})
            if result.fill is None:
                continue
            # 出来高の枠は entry_fill_limited() へ渡した budget が既に
            # 退出（①③フェーズ）と共有した状態で消費している（外部レビューR09）。
            # ここで budget.allow() を再度呼ぶと同じ枠を二重に消費してしまうので
            # 呼ばない。実際に約定した数量は result.filled_quantity。
            state = pf.apply_buy(
                state, order.symbol, result.filled_quantity, result.fill.price,
                order.sector, session, costs.commission_pct)
        pending_buys = []

        # ③ 退出ポリシーの逐次駆動
        state, stop_trades, new_pending_exits = _drive_exits(
            state, md, session, policy_conf, costs, budget, exit_score_fn)
        for t in stop_trades:
            realized += t["pnl"]
        trades.extend(stop_trades)
        pending_exits = pending_exits + new_pending_exits

        # ④ 日末のNAV記録
        daily.append(DailyRow(
            session=session, nav=pf.nav(state, closes), cash=state.cash,
            n_holdings=len(state.holdings), realized_pnl=realized,
        ))

        # ⑤ 翌日の候補生成（この日の引けの情報だけを使う）
        # 判断へ渡すのは**特徴量を重ねた行**。生OHLCVだけを渡すと
        # rule_score も特徴量も無い行になる（外部レビューR03）
        rows = {s: _row_for(md, s, session) for s in md.bars}
        rows = {s: r for s, r in rows.items() if r is not None}
        ctx = {"sectors": md.sectors, "portfolio": state, "closes": closes}
        candidates = decide(session, rows, model, ctx)
        if candidates:
            orders, rejects = pf.allocate(state, candidates, sizing, closes)
            pending_buys = orders
            for r in rejects:
                rejected.append({"session": session, "symbol": r.symbol,
                                 "reason": r.reason})

    return WalkForwardResult(
        daily=pd.DataFrame([vars(r) for r in daily], columns=_DAILY_COLUMNS),
        trades=pd.DataFrame(trades, columns=_TRADE_COLUMNS),
        rejected=pd.DataFrame(rejected, columns=_REJECTED_COLUMNS),
        degraded=False,
        degraded_reasons=[],
        model_usage=pd.DataFrame(columns=_MODEL_USAGE_COLUMNS),
    )


def _holding_state(h: pf.Holding) -> policy.HoldingState:
    """ポートフォリオの保有から、退出ポリシーが使う状態を作る。"""
    return policy.HoldingState(
        symbol=h.symbol, entry_at=h.entry_at, avg_cost=h.avg_cost,
        quantity=h.quantity, peak_price=h.peak_price,
        sessions_held=h.sessions_held,
    )


def _drive_exits(portfolio_state: pf.Portfolio, md: MarketData, session: date,
                 policy_conf: policy.PolicyConfig,
                 costs: execution.CostConfig,
                 budget: Optional[execution.VolumeBudget] = None,
                 exit_score_fn: Optional[Callable] = None) -> tuple:
    """保有ごとに退出ポリシーを1営業日ぶん進める。

    戻り値: (次のポートフォリオ, 当日約定した取引のリスト, 翌営業日へ繰り越す退出意図)

    保有の peak_price / sessions_held は **policy.step() が返す次の状態で更新する**。
    portfolio.advance_session() は「その日の足が無くポリシーを回せない保有」にだけ
    使う。同じ規則（未来のピークを遡ってストップに使わない）の実装を2つ
    持たないため。

    STOP の意図はその日のうちに約定する（execution.exit_fill が
    min(open, trigger) で処理する）。MARKET の意図（売りシグナル・満了）は
    翌営業日の寄りで約定するので繰越キューへ入れる。

    budget を省略した場合は無制限（`LiquidityConfig()`既定）の
    `VolumeBudget`を都度作る。単体テストや出来高制約を検証しない呼び出しで
    毎回`VolumeBudget`を組み立てずに済ませるため。日次ループ本体
    （`run_walkforward`）は約定が起きる営業日単位で明示的に共有インスタンスを
    渡すこと（`execution.VolumeBudget`のクラスdocstring参照）。
    """
    if budget is None:
        budget = execution.VolumeBudget(execution.LiquidityConfig())
    trades: list = []
    pending_exits: list = []
    holdings = dict(portfolio_state.holdings)
    current = portfolio_state

    for symbol, holding in list(portfolio_state.holdings.items()):
        # 売りスコアを繋ぐ。None のままだと policy の SIGNAL_SELL 条件が
        # 一度も成立しない（外部レビューR10）
        score = _exit_score(md, symbol, session, exit_score_fn)
        obs = _observation(md, symbol, session, score=score)
        if obs is None:
            # 足が無い日はポリシーを回せない。経過だけ進めて持ち越す
            holdings[symbol] = replace(
                holding, sessions_held=holding.sessions_held + 1)
            continue

        next_state, intent = policy.step(_holding_state(holding), obs, policy_conf)
        holdings[symbol] = replace(
            holding, peak_price=next_state.peak_price,
            sessions_held=next_state.sessions_held)

        if intent is None:
            continue
        if intent.order_type == "MARKET":
            pending_exits.append((symbol, intent))
            continue

        # ストップ退出にも出来高の枠を掛ける。掛けないと900株保有・
        # 当日出来高1,000株・参加率10%でも全量売れてしまう（外部レビューR09）
        bar = _bar_of(md, symbol, session)
        volume = int(bar["volume"]) if bar is not None else 0
        result = execution.exit_fill_limited(
            intent, obs, None, holding.quantity, costs, budget,
            symbol=symbol, volume=volume)
        if result.fill is None:
            # 売れなかった。保有はそのまま残し、退出意図を翌営業日へ持ち越す
            pending_exits.append((symbol, intent))
            continue

        sold = result.filled_quantity
        current = replace(current, holdings=holdings)
        current, realized = pf.apply_sell(
            current, symbol, sold, result.fill.price, costs.commission_pct)
        holdings = dict(current.holdings)
        trades.append({
            "symbol": symbol, "entry_at": holding.entry_at,
            "entry_price": holding.avg_cost,
            # 実現損益はこちらの原価から出ている（買付手数料込み・外部レビューR20）
            "entry_cost_basis": holding.avg_cost_with_fees,
            "exit_at": result.fill.at,
            "exit_price": result.fill.price, "quantity": sold,
            "pnl": realized, "reason": intent.reason,
        })
        if sold < holding.quantity:
            # 売れ残りは保有に残っている。同じ意図を翌営業日へ持ち越す
            pending_exits.append((symbol, intent))

    current = replace(current, holdings=holdings)
    return current, trades, pending_exits


def _settle_pending_exits(portfolio_state: pf.Portfolio, md: MarketData,
                          session: date, pending_exits: list,
                          costs: execution.CostConfig,
                          budget: execution.VolumeBudget) -> tuple:
    """前営業日に決まった成行退出を、この日の寄りで約定させる。

    戻り値: (次のポートフォリオ, 約定した取引のリスト, 約定できなかった意図)

    出来高の枠は買いと共有する（`VolumeBudget`）。売り切れなかったぶんは
    保有に残し、同じ意図を翌営業日へ持ち越す（外部レビューR09）。
    退出フェーズを買いより先に置いているので、枠は退出が先に取る。
    """
    trades: list = []
    carried: list = []
    current = portfolio_state

    for symbol, intent in pending_exits:
        holding = current.holdings.get(symbol)
        obs = _observation(md, symbol, session)
        if holding is None:
            continue
        if obs is None:
            carried.append((symbol, intent))
            continue
        bar = _bar_of(md, symbol, session)
        volume = int(bar["volume"]) if bar is not None else 0
        result = execution.exit_fill_limited(
            intent, obs, obs, holding.quantity, costs, budget,
            symbol=symbol, volume=volume)
        if result.fill is None:
            carried.append((symbol, intent))
            continue
        sold = result.filled_quantity
        current, realized = pf.apply_sell(
            current, symbol, sold, result.fill.price, costs.commission_pct)
        trades.append({
            "symbol": symbol, "entry_at": holding.entry_at,
            "entry_price": holding.avg_cost,
            "entry_cost_basis": holding.avg_cost_with_fees,
            "exit_at": session,
            "exit_price": result.fill.price, "quantity": sold,
            "pnl": realized, "reason": intent.reason,
        })
        if sold < holding.quantity:
            carried.append((symbol, intent))
    return current, trades, carried
