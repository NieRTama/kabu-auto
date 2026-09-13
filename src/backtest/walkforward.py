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
                    liquidity: execution.LiquidityConfig) -> WalkForwardResult:
    """日次5フェーズでポートフォリオを進める。

    decide は判断規則（戦略バージョン）。
    `decide(session, rows, model, ctx) -> list[portfolio.Candidate]` の形で、
    その日の引けの情報から**翌営業日に出す**候補を返す。
    """
    sessions = sessions_between(md, start, end)
    portfolio_state = pf.empty_portfolio(initial_capital)
    daily: list = []

    for session in sessions:
        realized = 0.0
        closes = closes_at(md, session)

        # ④ 日末評価（①②③⑤は後続タスクで足す）
        portfolio_state = pf.advance_session(portfolio_state, closes)
        daily.append(DailyRow(
            session=session,
            nav=pf.nav(portfolio_state, closes),
            cash=portfolio_state.cash,
            n_holdings=len(portfolio_state.holdings),
            realized_pnl=realized,
        ))

    return WalkForwardResult(
        daily=pd.DataFrame([vars(r) for r in daily], columns=_DAILY_COLUMNS),
        trades=pd.DataFrame(columns=_TRADE_COLUMNS),
        rejected=pd.DataFrame(columns=_REJECTED_COLUMNS),
        degraded=False,
        degraded_reasons=[],
        model_usage=pd.DataFrame(columns=_MODEL_USAGE_COLUMNS),
    )
