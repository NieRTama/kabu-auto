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

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import func, select

from src.core import config as cfg
from src.data.database import (
    BacktestRun, BacktestTradeRecord, Position, Signal, Trade, get_session,
)

app = FastAPI(title="kabu-auto Dashboard")

_order_manager = None
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
    return [
        {
            "symbol": p.symbol,
            "quantity": p.quantity,
            "avg_cost": p.avg_cost,
            "sector": p.sector,
            "opened_at": p.opened_at.isoformat() if p.opened_at else None,
        }
        for p in positions
    ]


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
