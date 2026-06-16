"""
バックテストエンジン

時系列ウォークフォワード方式でシグナル戦略の過去実績を検証する。
テクニカル指標は全期間で一括計算し、各日のシグナルにはその日までの
データのみを参照することでルックアヘッドバイアスを防ぐ。

- ルールベースのみ、またはルールベース+ML（開始前データで学習）を選択可
- 1銘柄×数年分のバックテストは数秒で完了する
"""
import json
import math
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.core import config as cfg
from src.data.database import BacktestRun, BacktestTradeRecord, get_session
from src.data.market_data import load_ohlcv
from src.strategy import ml_model
from src.strategy.indicators import FEATURE_COLS, build_features
from src.strategy.signal import compute_rule_score


def run_backtest(
    symbol: str,
    start: date,
    end: date,
    initial_capital: float = 500_000.0,
    use_ml: bool = False,
) -> int:
    """バックテストを実行してDBに保存し、run_id を返す"""
    strat_conf = cfg.get_section("strategy")
    trade_conf = cfg.get_section("trading")
    stop_loss_pct = trade_conf.get("stop_loss_pct", -0.05)
    max_pos_ratio = trade_conf.get("max_position_ratio", 0.20)
    buy_thr = strat_conf.get("buy_threshold", 0.6)
    sell_thr = strat_conf.get("sell_threshold", -0.6)
    ml_weight = strat_conf.get("ml_weight", 0.5)
    rule_weight = strat_conf.get("rule_weight", 0.5)

    # ─── 全履歴をロード（指標計算のウォームアップ分を含む）─────────
    full_df = load_ohlcv(symbol, limit=2000)
    if len(full_df) < 80:
        raise ValueError(f"データ不足: {symbol} ({len(full_df)}件、最低80件必要)")

    full_df.index = pd.to_datetime(full_df.index)

    # テスト期間だけ取り出す
    mask = (full_df.index.date >= start) & (full_df.index.date <= end)
    if not mask.any():
        raise ValueError(f"指定期間にデータがありません: {start} ~ {end}")

    # ─── 指標を全期間で一括計算（毎日再計算を避けて高速化）──────────
    featured_df = build_features(full_df)

    # ─── MLモデル: テスト開始前データのみで学習 ──────────────────
    model = None
    effective_ml_weight = 0.0
    effective_rule_weight = 1.0
    if use_ml:
        pre_df = full_df[full_df.index.date < start]
        if len(pre_df) >= 200:
            try:
                model = ml_model.train(pre_df)
                effective_ml_weight = ml_weight
                effective_rule_weight = rule_weight
                logger.info(f"バックテスト用MLモデル学習完了: {len(pre_df)}件")
            except Exception as e:
                logger.warning(f"ML学習失敗 → ルールベースのみで継続: {e}")
        else:
            logger.warning(
                f"ML学習データ不足 ({len(pre_df)}件、最低200件必要) → ルールベースのみ"
            )

    # ─── シミュレーション ────────────────────────────────────────
    cash = initial_capital
    pos_qty = 0
    pos_avg_cost = 0.0
    pos_entry_date: Optional[date] = None
    sim_trades = []
    equity_curve = []

    for dt in full_df.index[mask]:
        dt_date = dt.date()
        today_row = full_df.loc[dt]
        close_price = float(today_row["close"])
        low_price = float(today_row["low"])

        # ── 損切りチェック（当日のLow価格で判定）─────────────────
        if pos_qty > 0:
            drawdown = (low_price - pos_avg_cost) / pos_avg_cost
            if drawdown <= stop_loss_pct:
                stop_price = round(pos_avg_cost * (1 + stop_loss_pct), 2)
                pnl = round((stop_price - pos_avg_cost) * pos_qty, 0)
                cash += stop_price * pos_qty
                sim_trades.append(_make_trade(
                    symbol, pos_entry_date, pos_avg_cost,
                    dt_date, stop_price, pos_qty, pnl, "STOP_LOSS",
                ))
                pos_qty = 0
                equity_curve.append({"date": dt_date.isoformat(), "equity": round(cash, 0)})
                continue

        # ── シグナル生成（当日以前の事前計算済み特徴量を使用）────────
        if dt not in featured_df.index:
            # 指標計算に必要な期間に満たない（序盤のNA除去行）
            equity_curve.append({
                "date": dt_date.isoformat(),
                "equity": round(cash + pos_qty * close_price, 0),
            })
            continue

        feat_slice = featured_df[featured_df.index <= dt]
        if len(feat_slice) < 2:
            equity_curve.append({
                "date": dt_date.isoformat(),
                "equity": round(cash + pos_qty * close_price, 0),
            })
            continue

        r_score = compute_rule_score(feat_slice)

        ml_s = 0.0
        if model is not None:
            try:
                today_feats = featured_df.loc[[dt], FEATURE_COLS]
                proba = float(model.predict_proba(today_feats)[0][1])
                ml_s = (proba - 0.5) * 2
            except Exception:
                pass

        combined = r_score * effective_rule_weight + ml_s * effective_ml_weight

        # ── 売り判定 ──────────────────────────────────────────────
        if combined <= sell_thr and pos_qty > 0:
            pnl = round((close_price - pos_avg_cost) * pos_qty, 0)
            cash += close_price * pos_qty
            sim_trades.append(_make_trade(
                symbol, pos_entry_date, pos_avg_cost,
                dt_date, close_price, pos_qty, pnl, "SIGNAL_SELL",
            ))
            pos_qty = 0

        # ── 買い判定 ──────────────────────────────────────────────
        if combined >= buy_thr and pos_qty == 0:
            budget = cash * max_pos_ratio
            qty = int(budget / close_price / 100) * 100  # 100株単位
            if qty >= 100 and close_price * qty <= cash:
                cash -= close_price * qty
                pos_qty = qty
                pos_avg_cost = close_price
                pos_entry_date = dt_date

        equity_curve.append({
            "date": dt_date.isoformat(),
            "equity": round(cash + pos_qty * close_price, 0),
        })

    # ── 期間終了時の強制決済 ──────────────────────────────────────
    if pos_qty > 0:
        last_idx = full_df.index[mask][-1]
        last_close = float(full_df.loc[last_idx, "close"])
        pnl = round((last_close - pos_avg_cost) * pos_qty, 0)
        cash += last_close * pos_qty
        sim_trades.append(_make_trade(
            symbol, pos_entry_date, pos_avg_cost,
            last_idx.date(), last_close, pos_qty, pnl, "END_OF_PERIOD",
        ))

    # ── パフォーマンス指標 ────────────────────────────────────────
    pnls = [t["pnl"] for t in sim_trades]
    total_return = round((cash - initial_capital) / initial_capital, 4)
    max_dd = _max_drawdown(equity_curve, initial_capital)
    sharpe = _sharpe_ratio(equity_curve)
    win_rate = round(sum(1 for p in pnls if p > 0) / len(pnls), 3) if pnls else 0.0

    # ── DBに保存 ──────────────────────────────────────────────────
    with get_session() as session:
        run = BacktestRun(
            symbol=symbol,
            start_date=start,
            end_date=end,
            initial_capital=initial_capital,
            final_capital=round(cash, 0),
            total_return=total_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            trade_count=len(sim_trades),
            use_ml=1 if model is not None else 0,
            created_at=datetime.now(),
            equity_curve_json=json.dumps(equity_curve),
        )
        session.add(run)
        session.flush()
        run_id = run.id
        for t in sim_trades:
            session.add(BacktestTradeRecord(run_id=run_id, **t))
        session.commit()

    logger.info(
        f"バックテスト完了: {symbol} {start}~{end} "
        f"リターン={total_return:.1%} MDD={max_dd:.1%} "
        f"シャープ={sharpe:.2f} 取引数={len(sim_trades)}"
    )
    return run_id


def _make_trade(
    symbol: str, entry_date, entry_price: float,
    exit_date, exit_price: float,
    quantity: int, pnl: float, exit_reason: str,
) -> dict:
    return {
        "symbol": symbol,
        "entry_date": entry_date,
        "entry_price": entry_price,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "quantity": quantity,
        "pnl": pnl,
        "exit_reason": exit_reason,
    }


def _max_drawdown(equity_curve: list, initial_capital: float) -> float:
    peak = initial_capital
    max_dd = 0.0
    for e in equity_curve:
        eq = e["equity"]
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
    return round(max_dd, 4)


def _sharpe_ratio(equity_curve: list) -> float:
    equities = [e["equity"] for e in equity_curve]
    if len(equities) < 2:
        return 0.0
    rets = [
        (equities[i] - equities[i - 1]) / equities[i - 1]
        for i in range(1, len(equities))
        if equities[i - 1] > 0
    ]
    if len(rets) < 2:
        return 0.0
    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=1))
    return round(mean_r / std_r * math.sqrt(252), 3) if std_r > 0 else 0.0
