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
import json
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest import execution
from src.backtest import portfolio as pf
from src.core import clock
from src.strategy import policy

_DAILY_COLUMNS = ["session", "nav", "cash", "n_holdings", "realized_pnl"]
_TRADE_COLUMNS = ["symbol", "entry_at", "entry_price", "entry_cost_basis",
                  "exit_at", "exit_price", "quantity", "pnl", "reason"]
_REJECTED_COLUMNS = ["session", "symbol", "reason"]
_MODEL_USAGE_COLUMNS = ["model_id", "from_session", "to_session", "n_train_events"]


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


@dataclass(frozen=True)
class RetrainConfig:
    """期間中の再学習スケジュール。

    every_sessions は再学習の間隔（営業日）。0で無効。
    warmup_sessions は最初の学習までに必要な助走期間。
    実運用が週次で再学習するなら every_sessions=5 で同じ周期になる。
    """
    every_sessions: int = 0
    warmup_sessions: int = 0


@dataclass(frozen=True)
class RunSnapshot:
    """実行開始時に固定する来歴。

    config_hash は同一性の確認には使えるが**復元には使えない**ため、
    設定の実体（config_json）も併せて持つ（spec §8）。
    """
    strategy_version: str
    config_hash: str
    config_json: str
    dataset_id: Optional[str] = None
    code_version: Optional[str] = None
    execution_model_version: Optional[str] = None


def save_run(result: WalkForwardResult, snapshot: RunSnapshot, *,
             symbol_label: str, start: date, end: date,
             initial_capital: float, costs: execution.CostConfig) -> int:
    """実行結果と来歴を保存し、run_id を返す。

    日次NAVは equity_curve_json に、モデル使用履歴は RunModelUsage に入れる。
    degraded な実行も**保存する**（比較から外すのは読む側の責任で、
    「失敗した実行があったこと」自体は残す）。
    """
    from src.data.database import BacktestRun, RunModelUsage, get_session

    final_capital = float(result.daily["nav"].iloc[-1]) if len(result.daily) else initial_capital
    total_return = (final_capital - initial_capital) / initial_capital if initial_capital else 0.0
    curve = [{"date": r["session"].isoformat(), "equity": round(float(r["nav"]), 0)}
             for _, r in result.daily.iterrows()]

    with get_session() as session:
        run = BacktestRun(
            symbol=symbol_label, start_date=start, end_date=end,
            initial_capital=initial_capital, final_capital=final_capital,
            total_return=total_return,
            trade_count=len(result.trades),
            slippage_pct=costs.slippage_pct, commission_pct=costs.commission_pct,
            created_at=clock.now(),
            equity_curve_json=json.dumps(curve, ensure_ascii=False),
            strategy_version=snapshot.strategy_version,
            config_hash=snapshot.config_hash,
            config_json=snapshot.config_json,
            dataset_id=snapshot.dataset_id,
            code_version=snapshot.code_version,
            execution_model_version=snapshot.execution_model_version,
            degraded=1 if result.degraded else 0,
        )
        session.add(run)
        session.flush()
        run_id = run.id
        for _, u in result.model_usage.iterrows():
            session.add(RunModelUsage(
                run_id=run_id, model_id=u["model_id"],
                from_session=u["from_session"], to_session=u["to_session"],
                n_train_events=int(u["n_train_events"]),
            ))
        session.commit()
    return run_id


ON_FAILURE_RULE_ONLY = "rule_only"
ON_FAILURE_HALT_NEW = "halt_new"


@dataclass(frozen=True)
class StrategyConfig:
    """判断規則の設定。

    on_model_failure は「MLが無い・推論に失敗したとき」の動作を**明示する**。
    現行 signal.py:90 は暗黙にルール重みだけが残り、ML欠落時の縮尺が変わって
    同じ閾値でも売買判断が変わっていた（レビューF08）。
    """
    buy_threshold: float
    rule_weight: float = 0.5
    ml_weight: float = 0.5
    on_model_failure: str = ON_FAILURE_RULE_ONLY


def _candidate(symbol: str, row, ctx: dict, score: float) -> pf.Candidate:
    return pf.Candidate(symbol=symbol, sector=ctx["sectors"].get(symbol, ""),
                        price=float(row["close"]), score=score)


def make_weighted_blend(conf: StrategyConfig, score_fn: Callable) -> Callable:
    """案1: 既存の加重合成。`rule × rule_weight + (p−0.5)×2 × ml_weight`。"""
    def decide(session, rows, model, ctx):
        out = []
        for symbol, row in rows.items():
            rule, proba = score_fn(symbol, row)
            ml = (proba - 0.5) * 2 if proba is not None else 0.0
            blended = rule * conf.rule_weight + ml * conf.ml_weight
            if blended >= conf.buy_threshold:
                out.append(_candidate(symbol, row, ctx, blended))
        return out
    return decide


def make_rule_only(conf: StrategyConfig, score_fn: Callable) -> Callable:
    """案2: 縮尺を明示したルール単独。重みで割り引かない。"""
    def decide(session, rows, model, ctx):
        out = []
        for symbol, row in rows.items():
            rule, _ = score_fn(symbol, row)
            if rule >= conf.buy_threshold:
                out.append(_candidate(symbol, row, ctx, rule))
        return out
    return decide


def make_rule_then_ml(conf: StrategyConfig, score_fn: Callable) -> Callable:
    """案3: ルールで候補を作り、MLで買う・見送るの順位を決める。

    **まず試すのはこれ。** 全日付を機械的に買い・売りへ変換せず、
    ルールが候補としたイベントに対してのみMLの追加効果を測るため（spec §8）。
    MLが無い・失敗した場合の動作は on_model_failure で明示する。
    """
    def decide(session, rows, model, ctx):
        gated = []
        for symbol, row in rows.items():
            rule, proba = score_fn(symbol, row)
            if rule < conf.buy_threshold:
                continue
            gated.append((symbol, row, rule, proba))

        usable = [g for g in gated if g[3] is not None]
        if len(usable) < len(gated):
            if conf.on_model_failure == ON_FAILURE_HALT_NEW:
                return []
            # rule_only: ML無しの候補はルールスコアで順位付けする
            return [_candidate(s, r, ctx, rule) for s, r, rule, _ in gated]

        ranked = sorted(usable, key=lambda g: g[3], reverse=True)
        return [_candidate(s, r, ctx, proba) for s, r, _, proba in ranked]
    return decide


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


def opens_at(md: MarketData, session: date) -> dict:
    """その日の寄り値。足が無い銘柄は含めない（呼び出し側が取得単価へ落とす）。

    買い約定はこの日の寄りで起きるため、約定判定の時点で分かっている情報は
    終値ではなく寄り値まで（Tの終値の情報はT+1以降の注文にしか使えない）。
    """
    out = {}
    for symbol in md.bars:
        row = _bar_of(md, symbol, session)
        if row is not None:
            out[symbol] = float(row["open"])
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
                    retrain: Optional[RetrainConfig] = None,
                    train_model: Optional[Callable] = None,
                    ) -> WalkForwardResult:
    """日次5フェーズでポートフォリオを進める。

      ①前日までに決まった注文の執行 → ②保有と現金の更新 →
      ③退出ポリシーの逐次駆動 → ④日末のNAV記録 → ⑤翌日の候補生成

    **Tの終値の情報はT+1以降の注文にしか使えない。** decide() が返した候補は
    その日には約定せず、翌営業日の寄りで執行される。

    decide は判断規則（戦略バージョン）。
    `decide(session, rows, model, ctx) -> list[portfolio.Candidate]`。

    **model は「その判断時点で利用可能だったモデル」でなければならない。**
    `retrain`/`train_model` を渡さない場合は引数の `model` を全期間で使う
    （診断用の固定モデル実行）。`retrain` と `train_model` を両方渡すと、
    助走期間のあとは `train_model(as_of)` が返すモデルへ順次切り替わる。
    締切の適用（`validation.training_inputs()` を通すこと）は `train_model`
    の実装側の責任であり、この関数はいつ呼ぶかだけを管理する。

    exit_score_fn は保有銘柄の売りスコア。`exit_score_fn(symbol, row) -> float|None`。
    渡さないと policy の SIGNAL_SELL が一度も成立しない（外部レビューR10）。

    retrain は期間中の再学習スケジュール（`RetrainConfig`）。
    train_model は `train_model(as_of: date) -> tuple[object, int]`。
    その時点までに確定した情報だけで学習し `(モデル, 学習イベント数)` を
    返す関数。どちらか一方でも省略すると再学習しない
    （固定モデル実行のまま）。
    """
    sessions = sessions_between(md, start, end)
    state = pf.empty_portfolio(initial_capital)
    daily: list = []
    trades: list = []
    rejected: list = []
    pending_buys: list = []
    pending_exits: list = []

    model_usage: list = []
    current_model = model
    current_model_id: Optional[str] = None
    model_since: Optional[date] = None
    current_n_train = 0
    degraded_reasons: list = []

    if train_model is None or retrain is None or retrain.every_sessions <= 0:
        if model is not None:
            # 過去の全日付を1つのモデルで判断する実行。
            # そのモデルが評価期間より後のデータで学習されていないことを
            # この関数は確かめられない。診断には使えるが、walk-forward成績
            # として昇格の根拠にはできない（外部レビューR04）。
            degraded_reasons.append(
                "再学習が結線されていないため全期間を単一モデルで判断した"
                "（診断用。昇格の根拠にできない）")

    for index, session in enumerate(sessions):
        # ⓪ 再学習（この時点までに確定した情報だけで学習する）
        if (retrain is not None and train_model is not None
                and retrain.every_sessions > 0
                and index >= retrain.warmup_sessions
                and (index - retrain.warmup_sessions) % retrain.every_sessions == 0):
            if current_model_id is not None:
                model_usage.append({
                    "model_id": current_model_id, "from_session": model_since,
                    "to_session": sessions[index - 1],
                    "n_train_events": current_n_train,
                })
            try:
                current_model, current_n_train = train_model(session)
                current_model_id = str(current_model)
                model_since = session
            except Exception as e:
                degraded_reasons.append(f"{session}: 再学習に失敗しました: {e}")
                logger.warning(f"バックテスト中の再学習に失敗: {session} {e}")

        realized = 0.0
        closes = closes_at(md, session)
        # 買い約定（この日の寄り）の判定に使う。closesと違い、9:00時点で
        # 実際に分かっている価格のみを含む（新規Important A: 終値を混ぜると
        # T+1以降にしか使えないはずの情報が同日の約定判定に混入するlook-ahead）。
        opens = opens_at(md, session)
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
            # 1銘柄あたりの上限（calc_quantity）も約定価格で引き直す。
            # セクター集中の上限は calc_quantity() が見ないため、下の
            # check_sector_concentration() で別途約定価格・約定数量で引き直す
            # （Minor 2。以前はここのコメントだけがセクターも引き直すと主張していた）。
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

            # セクター集中の上限を約定価格・約定数量で引き直す。allocate()時点
            # （前日終値ベース）は通過していても、ギャップアップで分子（約定金額）・
            # 分母（総資金）が動いた後は超過することがある（Minor 2）。
            # 既存保有の評価には必ず寄り値（opens）を使う。closesを渡すと
            # 9:00の約定判定が同日の終値（T+1以降にしか使えないはずの情報）に
            # 依存するlook-aheadになる（新規Important A）。
            notional = fill_price * capped
            sector_ok, sector_reason = pf.check_sector_concentration(
                state, order.sector, notional, opens, sizing)
            if not sector_ok:
                original_sector_reason = sector_reason
                shrunk = capped
                while shrunk >= pf.LOT_SIZE and not sector_ok:
                    shrunk -= pf.LOT_SIZE
                    sector_ok, sector_reason = pf.check_sector_concentration(
                        state, order.sector, fill_price * shrunk, opens, sizing)
                if shrunk < pf.LOT_SIZE:
                    # 0株まで縮めた時点で集中率自体は解消し得るため、その場合
                    # sector_reasonは空になる。空の却下理由を残さないよう、
                    # 縮小前に超過していた事実を1件の見送り理由にまとめる
                    # （新規Minor B）。
                    rejected.append({
                        "session": session, "symbol": order.symbol,
                        "reason": (
                            f"約定価格{fill_price:,.1f}で{original_sector_reason}"
                            f"のため単元未満まで縮小し見送り（{capped:,}株→0株）"
                        )})
                    continue
                if shrunk < capped:
                    rejected.append({
                        "session": session, "symbol": order.symbol,
                        "reason": (
                            f"約定価格{fill_price:,.1f}でセクター集中率上限のため"
                            f"数量を再縮小（{capped:,}株→{shrunk:,}株）"
                        )})
                capped = shrunk

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
        try:
            candidates = decide(session, rows, current_model, ctx)
        except Exception as e:
            degraded_reasons.append(f"{session}: 判断に失敗しました: {e}")
            logger.warning(f"バックテスト中の判断に失敗: {session} {e}")
            candidates = []
        if candidates:
            orders, rejects = pf.allocate(state, candidates, sizing, closes)
            pending_buys = orders
            for r in rejects:
                rejected.append({"session": session, "symbol": r.symbol,
                                 "reason": r.reason})

    if current_model_id is not None:
        model_usage.append({
            "model_id": current_model_id, "from_session": model_since,
            "to_session": sessions[-1], "n_train_events": current_n_train,
        })

    return WalkForwardResult(
        daily=pd.DataFrame([vars(r) for r in daily], columns=_DAILY_COLUMNS),
        trades=pd.DataFrame(trades, columns=_TRADE_COLUMNS),
        rejected=pd.DataFrame(rejected, columns=_REJECTED_COLUMNS),
        degraded=bool(degraded_reasons),
        degraded_reasons=degraded_reasons,
        model_usage=pd.DataFrame(model_usage, columns=_MODEL_USAGE_COLUMNS),
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
    その日の足が無くポリシーを回せない保有は、`sessions_held` のみをインラインで
    +1 して持ち越す（`peak_price` は据え置き。値の付かない日にピークを進めない
    ため）。`portfolio.advance_session()` はここでは呼ばない。同関数は価格辞書から
    `peak_price` も更新してしまうため、足が無い日に使うと不適切（外部レビューI-1）。

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
