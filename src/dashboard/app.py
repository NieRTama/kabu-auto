"""
Webダッシュボード（FastAPI）
- ポジション・損益確認
- 取引履歴
- シグナル履歴
- 緊急全ポジション決済ボタン
- バックテスト実行・結果表示
"""
import asyncio
import json
import secrets
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import func, select

from src.core import config as cfg
from src.data.database import (
    BacktestRun, BacktestTradeRecord, ModelMetrics, OHLCV, Position, Signal, Trade, get_session,
)

app = FastAPI(title="kabu-auto Dashboard")

_order_manager = None
_ml_retrain_fn = None
_emergency_token: Optional[str] = None
_system_status = {
    "running": False,
    "ws_connected": False,
    "last_update": None,
    "mode": "paper",
}

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def set_ml_retrain_fn(fn) -> None:
    global _ml_retrain_fn
    _ml_retrain_fn = fn


def set_order_manager(om) -> None:
    global _order_manager, _emergency_token
    _order_manager = om
    conf_token = cfg.get_section("dashboard").get("emergency_token", "")
    _emergency_token = conf_token if conf_token else secrets.token_urlsafe(16)
    logger.info(f"緊急決済トークン (X-Emergency-Token ヘッダーに設定): {_emergency_token}")


async def _verify_emergency_token(x_emergency_token: str = Header(...)) -> None:
    if _emergency_token is None or x_emergency_token != _emergency_token:
        raise HTTPException(status_code=403, detail="Invalid emergency token")


def update_status(running: bool, ws_connected: bool, mode: str) -> None:
    _system_status.update({
        "running": running,
        "ws_connected": ws_connected,
        "last_update": datetime.utcnow().isoformat(),
        "mode": mode,
    })


def _compute_max_drawdown(cumulative_values: list) -> float:
    """累積損益リストから最大ドローダウン率を返す（負の値、例: -0.12 = -12%）"""
    if len(cumulative_values) < 2:
        return 0.0
    peak = cumulative_values[0]
    max_dd = 0.0
    for v in cumulative_values:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
    return round(max_dd, 4)


def _compute_sharpe(daily_pnl: list) -> float:
    """日次損益リストから年率換算シャープレシオを返す"""
    if len(daily_pnl) < 2:
        return 0.0
    arr = np.array(daily_pnl, dtype=float)
    std = float(arr.std())
    if std == 0:
        return 0.0
    return round(float(arr.mean() / std * (252 ** 0.5)), 2)


@app.get("/", response_class=HTMLResponse)
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>kabu-auto Dashboard</h1><p>frontend/index.html が見つかりません</p>")


@app.get("/api/status")
async def get_status():
    return _system_status


@app.get("/api/positions")
async def get_positions():
    with get_session() as session:
        positions = session.scalars(
            select(Position).where(Position.quantity > 0)
        ).all()
        result = []
        for p in positions:
            latest = session.scalar(
                select(OHLCV).where(OHLCV.symbol == p.symbol).order_by(OHLCV.date.desc())
            )
            latest_price = latest.close if latest else None
            unrealized_pnl = (
                round((latest_price - p.avg_cost) * p.quantity, 0)
                if latest_price and p.avg_cost else None
            )
            return_pct = (
                round((latest_price - p.avg_cost) / p.avg_cost, 4)
                if latest_price and p.avg_cost else None
            )
            result.append({
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_cost": p.avg_cost,
                "sector": p.sector,
                "opened_at": p.opened_at.isoformat() if p.opened_at else None,
                "latest_price": latest_price,
                "unrealized_pnl": unrealized_pnl,
                "return_pct": return_pct,
            })
    return result


@app.get("/api/trades")
async def get_trades(limit: int = 50):
    with get_session() as session:
        trades = session.scalars(
            select(Trade).order_by(Trade.id.desc()).limit(limit)
        ).all()
    return [
        {
            "order_id": t.order_id,
            "symbol": t.symbol,
            "side": t.side,
            "quantity": t.quantity,
            "price": t.price,
            "filled_at": t.filled_at.isoformat() if t.filled_at else None,
            "status": t.status,
            "pnl": t.pnl,
        }
        for t in trades
    ]


@app.get("/api/pnl_summary")
async def get_pnl_summary():
    with get_session() as session:
        total_pnl = session.scalar(func.sum(Trade.pnl).filter(Trade.pnl.isnot(None))) or 0.0
        win_count = session.scalar(
            func.count(Trade.id).filter(Trade.pnl > 0)
        ) or 0
        loss_count = session.scalar(
            func.count(Trade.id).filter(Trade.pnl < 0)
        ) or 0
    total = win_count + loss_count
    win_rate = win_count / total if total > 0 else 0.0
    return {
        "total_pnl": round(total_pnl, 0),
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_rate, 3),
    }


@app.get("/api/signals")
async def get_signals(limit: int = 50):
    with get_session() as session:
        signals = session.scalars(
            select(Signal).order_by(Signal.id.desc()).limit(limit)
        ).all()
    return [
        {
            "symbol": s.symbol,
            "generated_at": s.generated_at.isoformat() if s.generated_at else None,
            "rule_score": s.rule_score,
            "ml_score": s.ml_score,
            "combined_score": s.combined_score,
            "action": s.action,
        }
        for s in signals
    ]


@app.get("/api/pnl_chart")
async def get_pnl_chart():
    """日次累積損益データ（Chart.js用）"""
    with get_session() as session:
        trades = session.scalars(
            select(Trade).where(Trade.pnl.isnot(None), Trade.filled_at.isnot(None))
            .order_by(Trade.filled_at)
        ).all()
    cumulative = 0.0
    data = []
    for t in trades:
        cumulative += t.pnl or 0
        data.append({
            "date": t.filled_at.strftime("%Y-%m-%d") if t.filled_at else None,
            "pnl": round(cumulative, 0),
        })
    return data


@app.get("/api/pnl/daily")
async def get_pnl_daily(days: int = 90):
    """日次損益（棒グラフ用）と累積損益を返す"""
    with get_session() as session:
        trades = session.scalars(
            select(Trade).where(Trade.pnl.isnot(None), Trade.filled_at.isnot(None))
            .order_by(Trade.filled_at)
        ).all()

    daily: dict = {}
    for t in trades:
        date_str = t.filled_at.strftime("%Y-%m-%d")
        daily[date_str] = daily.get(date_str, 0.0) + (t.pnl or 0.0)

    sorted_dates = sorted(daily.keys())
    if len(sorted_dates) > days:
        sorted_dates = sorted_dates[-days:]

    cumulative = 0.0
    result = []
    for d in sorted_dates:
        daily_pnl = round(daily[d], 0)
        cumulative += daily_pnl
        result.append({"date": d, "daily_pnl": daily_pnl, "cumulative_pnl": round(cumulative, 0)})
    return result


@app.get("/api/pnl/enhanced_summary")
async def get_pnl_enhanced_summary():
    """今日/MTD/YTD/シャープレシオ/最大ドローダウン/含み損益合計を返す"""
    today = datetime.utcnow().date()
    month_start = today.replace(day=1)
    year_start = today.replace(month=1, day=1)

    with get_session() as session:
        trades = session.scalars(
            select(Trade).where(Trade.pnl.isnot(None), Trade.filled_at.isnot(None))
            .order_by(Trade.filled_at)
        ).all()
        positions = session.scalars(select(Position).where(Position.quantity > 0)).all()
        total_unrealized = 0.0
        for p in positions:
            latest = session.scalar(
                select(OHLCV).where(OHLCV.symbol == p.symbol).order_by(OHLCV.date.desc())
            )
            if latest and latest.close and p.avg_cost:
                total_unrealized += (latest.close - p.avg_cost) * p.quantity

    daily: dict = {}
    for t in trades:
        d = t.filled_at.strftime("%Y-%m-%d")
        daily[d] = daily.get(d, 0.0) + (t.pnl or 0.0)

    total_pnl = sum(daily.values())
    today_pnl = daily.get(today.isoformat(), 0.0)
    mtd_pnl = sum(v for k, v in daily.items() if k >= month_start.isoformat())
    ytd_pnl = sum(v for k, v in daily.items() if k >= year_start.isoformat())

    win_count = sum(1 for t in trades if (t.pnl or 0) > 0)
    loss_count = sum(1 for t in trades if (t.pnl or 0) < 0)
    total = win_count + loss_count

    sorted_dates = sorted(daily.keys())
    cumulative_values: list = []
    daily_pnl_list: list = []
    cum = 0.0
    for d in sorted_dates:
        pnl = daily[d]
        cum += pnl
        cumulative_values.append(cum)
        daily_pnl_list.append(pnl)

    return {
        "total_pnl": round(total_pnl, 0),
        "today_pnl": round(today_pnl, 0),
        "mtd_pnl": round(mtd_pnl, 0),
        "ytd_pnl": round(ytd_pnl, 0),
        "total_unrealized_pnl": round(total_unrealized, 0),
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_count / total, 3) if total > 0 else 0.0,
        "max_drawdown": _compute_max_drawdown(cumulative_values),
        "sharpe_ratio": _compute_sharpe(daily_pnl_list),
    }


@app.post("/api/emergency_close")
async def emergency_close(_: None = Depends(_verify_emergency_token)):
    """緊急全ポジション決済（X-Emergency-Token ヘッダー必須）"""
    if _order_manager is None:
        raise HTTPException(status_code=503, detail="OrderManagerが初期化されていません")
    try:
        _order_manager.close_all_positions()
        logger.warning("緊急全ポジション決済を実行しました")
        return {"status": "ok", "message": "全ポジション決済を開始しました"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── MLモデルメトリクス ───────────────────────────────────────────


@app.get("/api/model/metrics")
async def get_model_metrics(limit: int = 20):
    """学習履歴（CV精度・サンプル数）を時系列で返す"""
    with get_session() as session:
        records = session.scalars(
            select(ModelMetrics).order_by(ModelMetrics.id.desc()).limit(limit)
        ).all()
    return [
        {
            "id": r.id,
            "trained_at": r.trained_at.isoformat() if r.trained_at else None,
            "cv_mean_accuracy": r.cv_mean_accuracy,
            "cv_std_accuracy": r.cv_std_accuracy,
            "n_samples": r.n_samples,
            "n_estimators": r.n_estimators,
            "trigger": r.trigger,
        }
        for r in records
    ]


@app.get("/api/model/latest")
async def get_model_latest():
    """最新の学習結果（特徴量重要度含む）を返す"""
    with get_session() as session:
        record = session.scalar(
            select(ModelMetrics).order_by(ModelMetrics.id.desc())
        )
    if record is None:
        return None
    fi = json.loads(record.feature_importances_json) if record.feature_importances_json else {}
    return {
        "id": record.id,
        "trained_at": record.trained_at.isoformat() if record.trained_at else None,
        "cv_mean_accuracy": record.cv_mean_accuracy,
        "cv_std_accuracy": record.cv_std_accuracy,
        "n_samples": record.n_samples,
        "n_estimators": record.n_estimators,
        "trigger": record.trigger,
        "feature_importances": fi,
    }


@app.post("/api/model/retrain")
async def retrain_model():
    """MLモデルを手動再学習する（ウォッチリスト全銘柄のデータを使用）"""
    if _ml_retrain_fn is None:
        raise HTTPException(status_code=503, detail="再学習関数が設定されていません")
    try:
        await asyncio.to_thread(_ml_retrain_fn)
    except Exception as e:
        logger.error(f"手動再学習失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "message": "再学習が完了しました"}


# ─── バックテスト ──────────────────────────────────────────────────


class BacktestRequest(BaseModel):
    symbol: str
    start: str        # "YYYY-MM-DD"
    end: str          # "YYYY-MM-DD"
    initial_capital: float = 500_000.0
    use_ml: bool = False


@app.get("/api/watchlist")
async def get_watchlist():
    """config.yaml のウォッチリスト銘柄を返す"""
    return cfg.get_section("trading").get("watchlist", [])


@app.post("/api/backtest/run")
async def start_backtest(req: BacktestRequest):
    """バックテストを実行してrun_idを返す（数秒〜十数秒かかる）"""
    from src.backtest.engine import run_backtest
    try:
        start_d = date.fromisoformat(req.start)
        end_d = date.fromisoformat(req.end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"日付形式エラー: {e}")
    if start_d >= end_d:
        raise HTTPException(status_code=400, detail="開始日は終了日より前にしてください")
    try:
        run_id = await asyncio.to_thread(
            run_backtest, req.symbol, start_d, end_d, req.initial_capital, req.use_ml
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"バックテスト失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"run_id": run_id, "status": "completed"}


@app.get("/api/backtest/runs")
async def get_backtest_runs(limit: int = 20):
    """直近のバックテスト実行一覧を返す"""
    with get_session() as session:
        runs = session.scalars(
            select(BacktestRun).order_by(BacktestRun.id.desc()).limit(limit)
        ).all()
    return [
        {
            "id": r.id,
            "symbol": r.symbol,
            "start_date": r.start_date.isoformat() if r.start_date else None,
            "end_date": r.end_date.isoformat() if r.end_date else None,
            "initial_capital": r.initial_capital,
            "final_capital": r.final_capital,
            "total_return": r.total_return,
            "max_drawdown": r.max_drawdown,
            "sharpe_ratio": r.sharpe_ratio,
            "win_rate": r.win_rate,
            "trade_count": r.trade_count,
            "use_ml": bool(r.use_ml),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in runs
    ]


@app.get("/api/backtest/{run_id}")
async def get_backtest_detail(run_id: int):
    """バックテスト詳細（エクイティカーブ＋取引一覧）を返す"""
    with get_session() as session:
        run = session.scalar(select(BacktestRun).where(BacktestRun.id == run_id))
        if run is None:
            raise HTTPException(status_code=404, detail="バックテスト結果が見つかりません")
        trades = session.scalars(
            select(BacktestTradeRecord)
            .where(BacktestTradeRecord.run_id == run_id)
            .order_by(BacktestTradeRecord.entry_date)
        ).all()
        equity_curve = json.loads(run.equity_curve_json) if run.equity_curve_json else []

    return {
        "run": {
            "id": run.id,
            "symbol": run.symbol,
            "start_date": run.start_date.isoformat() if run.start_date else None,
            "end_date": run.end_date.isoformat() if run.end_date else None,
            "initial_capital": run.initial_capital,
            "final_capital": run.final_capital,
            "total_return": run.total_return,
            "max_drawdown": run.max_drawdown,
            "sharpe_ratio": run.sharpe_ratio,
            "win_rate": run.win_rate,
            "trade_count": run.trade_count,
            "use_ml": bool(run.use_ml),
        },
        "equity_curve": equity_curve,
        "trades": [
            {
                "entry_date": t.entry_date.isoformat() if t.entry_date else None,
                "entry_price": t.entry_price,
                "exit_date": t.exit_date.isoformat() if t.exit_date else None,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
            }
            for t in trades
        ],
    }
