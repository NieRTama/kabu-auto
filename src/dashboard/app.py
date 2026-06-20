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
import os
import secrets
import socket
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import numpy as np

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import func, select

from src.core import (
    config as cfg, watchlist as watchlist_store, risk_profile as risk_profile_store,
    auth as auth_store, clock,
)
from src.data.database import (
    BacktestRun, BacktestTradeRecord, ModelMetrics, Position, Signal, Trade, get_session,
)
from src.data.market_data import latest_closes

app = FastAPI(title="kabu-auto Dashboard")

_order_manager = None
_ml_retrain_fn = None
_data_update_fn = None
_emergency_token: Optional[str] = None
_dashboard_token: Optional[str] = None
_auth_required: bool = False
_system_status = {
    "running": False,
    "ws_connected": False,
    "last_update": None,
    "mode": "paper",
}

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _get_lan_ip() -> str:
    """LAN内からアクセス可能なIPアドレスを自動検出する（実際には通信しない）"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def init_auth() -> None:
    """ダッシュボードのアクセス認証を初期化する。

    - `dashboard.api_token`（または環境変数 KABU_DASHBOARD_TOKEN）が設定されていれば
      その値で認証を有効化する。
    - 未設定でも host が localhost 以外（例: 0.0.0.0 = LAN公開）ならトークンを自動生成して
      認証を強制する（無認証でLANに建玉・損益を晒さないため）。
    - host が localhost かつトークン未設定なら、利便性のため認証なし（ローカル専用）。

    トークンはログには出さず、コンソールに一度だけアクセスURLとして表示する。
    """
    global _dashboard_token, _auth_required
    dash = cfg.get_section("dashboard")
    # ログイン認証情報（Authファイル）を読み込む
    auth_store.load(
        dash.get("auth_file", "data/auth.json"),
        session_ttl_hours=dash.get("session_ttl_hours"),
    )
    token = os.environ.get("KABU_DASHBOARD_TOKEN") or dash.get("api_token", "")
    host = dash.get("host", "127.0.0.1")
    is_local = host in ("127.0.0.1", "localhost", "::1")
    if token:
        _dashboard_token = token
        _auth_required = True
    elif not is_local:
        _dashboard_token = secrets.token_urlsafe(16)
        _auth_required = True
        logger.warning(
            "ダッシュボードをLAN公開(host={})していますが api_token 未設定のため"
            "アクセストークンを自動生成しました。".format(host)
        )
    else:
        _auth_required = False
        _dashboard_token = None

    if _auth_required:
        port = dash.get("port", 8080)
        if is_local:
            display_host = "localhost"
        elif host == "0.0.0.0":
            # 0.0.0.0 はワイルドカード待受アドレスであり、LAN端末からは接続できない。
            # 実際にLAN内から到達可能な自機IPを解決して表示する。
            display_host = _get_lan_ip()
        else:
            display_host = host
        setup_hint = (
            "  初回はログイン画面でユーザーID・パスワードを作成してください（初期設定）。\n"
            if not auth_store.is_configured()
            else "  ブラウザで上記URLを開き、ユーザーID・パスワードでログインしてください。\n"
        )
        # トークンはログ平文に残さず、コンソールへ一度だけ表示する
        print(
            "\n==== ダッシュボードへのアクセスには認証が必要です ====\n"
            f"  URL: http://{display_host}:{port}/\n"
            f"{setup_hint}"
            f"  curl等のプログラムアクセスは X-API-Token ヘッダーにトークンを指定:\n"
            f"    {_dashboard_token}\n"
            "=======================================================\n",
            flush=True,
        )


# ログインなしでアクセスできる（認証の入口となる）パス
_AUTH_EXEMPT_PATHS = ("/login", "/api/login", "/api/setup", "/api/auth_status")


def _has_valid_token(request: Request) -> bool:
    """X-API-Token ヘッダー / ?token= クエリ / kabu_token Cookie のいずれかが有効か。"""
    provided = (
        request.headers.get("X-API-Token")
        or request.query_params.get("token")
        or request.cookies.get("kabu_token")
    )
    return bool(provided and _dashboard_token is not None
                and secrets.compare_digest(provided, _dashboard_token))


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    """認証ミドルウェア。

    認証が有効な場合（LAN公開時など）、以下のいずれかでのみ通す:
      - 有効なログインセッション Cookie（kabu_session）
      - 有効な X-API-Token / ?token= / kabu_token Cookie（curl等プログラム用）
    未認証のHTMLナビゲーションは /login へリダイレクトし、API/XHRは401を返す。
    認証有効時は /docs・/openapi.json も保護対象とし、未認証の第三者にAPI仕様を露出させない。
    """
    if not _auth_required:
        return await call_next(request)

    path = request.url.path
    if path in _AUTH_EXEMPT_PATHS:
        return await call_next(request)

    session_token = request.cookies.get("kabu_session")
    authorized = auth_store.validate_session(session_token) or _has_valid_token(request)

    if not authorized:
        accepts_html = "text/html" in request.headers.get("accept", "")
        if accepts_html and not path.startswith("/api/"):
            return RedirectResponse(url="/login", status_code=303)
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    response = await call_next(request)
    if request.query_params.get("token"):
        # ブラウザでトークン付きURLを直開きした場合、以後の fetch 用に Cookie へ保存する
        response.set_cookie("kabu_token", _dashboard_token, httponly=True, samesite="strict")
    return response


def set_ml_retrain_fn(fn) -> None:
    global _ml_retrain_fn
    _ml_retrain_fn = fn


def set_data_update_fn(fn) -> None:
    global _data_update_fn
    _data_update_fn = fn


def set_order_manager(om) -> None:
    global _order_manager, _emergency_token
    _order_manager = om
    conf_token = cfg.get_section("dashboard").get("emergency_token", "")
    _emergency_token = conf_token if conf_token else secrets.token_urlsafe(16)
    if not conf_token:
        # 自動生成したトークンはログ平文に残さず、コンソールへ一度だけ表示する
        print(
            "\n==== 緊急決済トークン（X-Emergency-Token ヘッダー用）====\n"
            f"  {_emergency_token}\n"
            "  恒久運用する場合は config.yaml の dashboard.emergency_token に設定してください。\n"
            "=====================================================\n",
            flush=True,
        )
    init_auth()


async def _verify_emergency_token(x_emergency_token: str = Header(...)) -> None:
    if _emergency_token is None or not secrets.compare_digest(x_emergency_token, _emergency_token):
        raise HTTPException(status_code=403, detail="Invalid emergency token")


def update_status(running: bool, ws_connected: bool, mode: str) -> None:
    _system_status.update({
        "running": running,
        "ws_connected": ws_connected,
        "last_update": clock.now().isoformat(),  # JST naive
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


# ─── ログイン認証 ──────────────────────────────────────────────────────────


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """ログイン（初回はID・パスワード作成）画面を返す。"""
    page = FRONTEND_DIR / "login.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>kabu-auto ログイン</h1><p>frontend/login.html が見つかりません</p>")


@app.get("/api/auth_status")
async def auth_status():
    """認証が必要か・認証情報が登録済みかを返す（ログイン画面の表示切替用）。"""
    return {"configured": auth_store.is_configured(), "auth_required": _auth_required}


class CredentialsRequest(BaseModel):
    username: str
    password: str


def _issue_session(response: JSONResponse) -> JSONResponse:
    token = auth_store.create_session()
    response.set_cookie("kabu_session", token, httponly=True, samesite="strict")
    return response


@app.post("/api/setup")
async def setup_credentials(req: CredentialsRequest):
    """初期設定：ユーザーID・パスワードを作成する（未設定時のみ）。成功時はログイン状態にする。"""
    try:
        auth_store.create_user(req.username, req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _issue_session(JSONResponse({"status": "ok", "message": "初期設定が完了しました"}))


@app.post("/api/login")
async def login(req: CredentialsRequest):
    """ユーザーID・パスワードでログインする。"""
    if not auth_store.is_configured():
        raise HTTPException(status_code=400, detail="初期設定が未完了です")
    if not auth_store.verify(req.username, req.password):
        raise HTTPException(status_code=401, detail="ユーザーIDまたはパスワードが違います")
    return _issue_session(JSONResponse({"status": "ok", "message": "ログインしました"}))


@app.post("/api/logout")
async def logout(request: Request):
    """ログアウト（セッション破棄・Cookie削除）。"""
    auth_store.destroy_session(request.cookies.get("kabu_session"))
    response = JSONResponse({"status": "ok", "message": "ログアウトしました"})
    response.delete_cookie("kabu_session")
    return response


@app.get("/api/status")
async def get_status():
    """システム稼働状態と運用上の重要情報（モード・発注可否・LAN公開等）を返す。

    ダッシュボードはこの情報を元に LIVE 赤バッジ・LAN公開警告・発注停止表示を出す。
    """
    from src.core import trading_mode as tm
    dash = cfg.get_section("dashboard")
    kabu = cfg.get_section("kabu_station")
    mode = _system_status.get("mode", "paper")
    host = dash.get("host", "127.0.0.1")
    lan_exposed = host == "0.0.0.0"
    status = {
        **_system_status,
        "risk_profile": risk_profile_store.get_active(),
        "mode_description": tm.description(mode),
        "places_real_orders": tm.places_real_orders(mode),
        "endpoint": kabu.get("base_url", ""),
        "lan_exposed": lan_exposed,
        "funding_source": ("口座余力(/wallet)" if tm.reads_broker_api(mode)
                           else "ペーパー初期資金"),
    }
    if _order_manager is not None:
        try:
            status.update(_order_manager.status_snapshot())
        except Exception as e:
            logger.warning(f"status_snapshot 取得失敗: {e}")
    return status


@app.get("/api/risk_profile")
async def get_risk_profile():
    """アクティブなリスクプロファイルと選択可能な全プロファイルを返す。

    組み込み（low_risk/high_risk）とカスタムを区別できるよう builtin リストも返す。
    """
    profiles = risk_profile_store.get_profiles()
    return {
        "active": risk_profile_store.get_active(),
        "profiles": profiles,
        "builtin": [n for n in profiles if risk_profile_store.is_builtin(n)],
    }


def _requires_live_confirmation(confirm: bool) -> None:
    """実発注モードでのプロファイル選択/cloneは二重確認を要求する（11: ライブ安全）。"""
    from src.core import trading_mode as tm
    mode = _system_status.get("mode", "paper")
    if tm.places_real_orders(mode) and not confirm:
        raise HTTPException(
            status_code=409,
            detail=f"{tm.description(mode)}でのリスクプロファイル変更には確認が必要です"
                   "（confirm=true を指定してください）",
        )


class RiskProfileRequest(BaseModel):
    name: str
    confirm: bool = False  # 実発注モードでの切替に必要な二重確認


@app.post("/api/risk_profile")
async def set_risk_profile(req: RiskProfileRequest):
    """リスクプロファイルを切り替える。発注サイズ・損切り幅・売買閾値などに即時反映。

    実発注モード（live/semi_live）では confirm=true が必要（二重確認）。
    """
    _requires_live_confirmation(req.confirm)
    try:
        result = risk_profile_store.set_active(req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


class CustomProfileRequest(BaseModel):
    name: str
    params: dict


@app.post("/api/risk_profiles")
async def create_custom_profile(req: CustomProfileRequest):
    """カスタムリスクプロファイルを新規作成する（値はサーバ側で検証する）。"""
    try:
        return risk_profile_store.create_custom(req.name, req.params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class UpdateProfileRequest(BaseModel):
    params: dict


@app.put("/api/risk_profiles/{name}")
async def update_custom_profile(name: str, req: UpdateProfileRequest):
    """カスタムリスクプロファイルを更新する（組み込みは編集不可）。"""
    try:
        return risk_profile_store.update_custom(name, req.params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/risk_profiles/{name}")
async def delete_custom_profile(name: str):
    """カスタムリスクプロファイルを削除する（組み込み・アクティブは削除不可）。"""
    try:
        remaining = risk_profile_store.delete_custom(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "remaining": remaining}


class CloneProfileRequest(BaseModel):
    new_name: str


@app.post("/api/risk_profiles/{name}/clone")
async def clone_profile(name: str, req: CloneProfileRequest):
    """既存プロファイルを複製して新しいカスタムを作る。"""
    try:
        return risk_profile_store.clone(name, req.new_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/risk_profiles/export")
async def export_risk_profile(name: str):
    """プロファイルをJSONファイルとしてダウンロードする。"""
    try:
        data = risk_profile_store.export_profile(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    filename = f"risk_profile_{name}.json".replace("/", "_")
    encoded = quote(filename)
    return JSONResponse(
        content=data,
        headers={
            "Content-Disposition": f"attachment; filename=\"risk_profile_export.json\"; filename*=UTF-8''{encoded}"
        },
    )


class ProfileImportRequest(BaseModel):
    name: str
    params: dict
    overwrite: bool = False


@app.post("/api/risk_profiles/import")
async def import_risk_profile(req: ProfileImportRequest):
    """外部のプロファイル定義を取り込んでカスタムとして保存する。"""
    try:
        return risk_profile_store.import_profile(req.name, req.params, overwrite=req.overwrite)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/risk_profiles/compare")
async def compare_risk_profiles(a: str, b: str):
    """2つのプロファイルを比較して差分を返す。"""
    try:
        return risk_profile_store.compare(a, b)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/risk_profiles/history")
async def get_risk_profile_history(limit: int = 50):
    """リスクプロファイルの変更履歴を新しい順に返す。"""
    return risk_profile_store.get_history(limit)


@app.get("/api/positions")
async def get_positions():
    with get_session() as session:
        positions = session.scalars(
            select(Position).where(Position.quantity > 0)
        ).all()
    # 銘柄ごとに最新終値を個別取得する(N+1クエリ)のを避け、まとめて1クエリで取得する
    closes = latest_closes([p.symbol for p in positions])
    result = []
    for p in positions:
        latest_price = closes.get(p.symbol)
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
            "rationale": t.rationale,
        }
        for t in trades
    ]


# ─── 日次レポート・取引ジャーナル（Phase 5 / 7.4・7.9）────────────────────


def _exit_value(t: Trade) -> Optional[float]:
    """決済（SELL）の約定総額。成行は price=0 で filled_price に実単価が入る。"""
    px = t.filled_price if t.filled_price else t.price
    if not px:
        return None
    return px * (t.filled_quantity or t.quantity or 0)


@app.get("/api/report/daily")
async def get_daily_report(days: int = 30):
    """日次の取引活動レポート（新しい順）。

    各日について確定損益・約定件数・売買内訳・勝敗・勝率を集計する。
    既存の `/api/pnl/daily`（損益のみ）に対し、こちらは「何件・どちら向き・勝敗」の
    取引活動の側面を加えたもの。DRY_RUN は実取引でないため除外する。
    """
    with get_session() as session:
        trades = session.scalars(
            select(Trade).where(
                Trade.filled_at.isnot(None),
                Trade.status.in_(("FILLED", "PARTIALLY_FILLED")),
            ).order_by(Trade.filled_at)
        ).all()

    daily: dict = {}
    for t in trades:
        d = t.filled_at.strftime("%Y-%m-%d")
        row = daily.setdefault(d, {
            "date": d, "realized_pnl": 0.0, "trade_count": 0,
            "buy_count": 0, "sell_count": 0, "win_count": 0, "loss_count": 0,
        })
        row["trade_count"] += 1
        if t.side == "BUY":
            row["buy_count"] += 1
        else:
            row["sell_count"] += 1
        if t.pnl is not None:
            row["realized_pnl"] += t.pnl
            if t.pnl > 0:
                row["win_count"] += 1
            elif t.pnl < 0:
                row["loss_count"] += 1

    result = []
    for d in sorted(daily.keys(), reverse=True)[:days]:
        row = daily[d]
        decided = row["win_count"] + row["loss_count"]
        row["realized_pnl"] = round(row["realized_pnl"], 0)
        row["win_rate"] = round(row["win_count"] / decided, 3) if decided else None
        result.append(row)
    return result


@app.get("/api/journal")
async def get_trade_journal(limit: int = 100):
    """取引ジャーナル: 損益が確定した取引（決済）を新しい順に、リターン率・メモ付きで返す。

    リターン率は約定総額と損益から取得原価を逆算して算出する
    （取得原価 = 約定総額 − 損益）。メモは `PUT /api/journal/{order_id}/note` で編集できる。
    """
    with get_session() as session:
        trades = session.scalars(
            select(Trade).where(Trade.pnl.isnot(None))
            .order_by(Trade.filled_at.desc().nullslast(), Trade.id.desc())
            .limit(limit)
        ).all()
    out = []
    for t in trades:
        exit_val = _exit_value(t)
        cost_basis = (exit_val - t.pnl) if (exit_val is not None and t.pnl is not None) else None
        return_pct = (round(t.pnl / cost_basis, 4)
                      if cost_basis and cost_basis > 0 else None)
        out.append({
            "order_id": t.order_id,
            "symbol": t.symbol,
            "side": t.side,
            "sector": t.sector,
            "quantity": t.filled_quantity or t.quantity,
            "exit_price": t.filled_price if t.filled_price else t.price,
            "pnl": round(t.pnl, 0) if t.pnl is not None else None,
            "return_pct": return_pct,
            "filled_at": t.filled_at.isoformat() if t.filled_at else None,
            "note": t.note,
            "rationale": t.rationale,
        })
    return out


class TradeNoteRequest(BaseModel):
    note: str


@app.put("/api/journal/{order_id}/note")
async def set_trade_note(order_id: str, req: TradeNoteRequest):
    """取引にメモを付ける（取引ジャーナル用）。"""
    with get_session() as session:
        trade = session.scalar(select(Trade).where(Trade.order_id == order_id))
        if trade is None:
            raise HTTPException(status_code=404, detail="取引が見つかりません")
        trade.note = req.note[:1000]
        session.commit()
    return {"status": "ok", "order_id": order_id}


@app.get("/api/health")
async def get_health():
    """運用上の異常（未解決注文・損失上限接近・kill switch等）の現在一覧を返す（7.5）。"""
    if _order_manager is None:
        return {"anomalies": []}
    from src.core import health
    try:
        anomalies = health.check_anomalies(_order_manager._risk)
    except Exception as e:
        logger.warning(f"異常検知の取得失敗: {e}")
        anomalies = []
    return {"anomalies": anomalies}


@app.get("/api/performance")
async def get_performance():
    """パフォーマンス分析（7.8）: 期待値・プロフィットファクター・連勝連敗・銘柄/セクター別。"""
    from src.analytics.performance import compute_performance
    with get_session() as session:
        trades = session.scalars(
            select(Trade).where(Trade.pnl.isnot(None)).order_by(Trade.filled_at)
        ).all()
    return compute_performance(trades)


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
    today = clock.today()  # filled_at（JST naive）と同じ基準で当日判定する
    month_start = today.replace(day=1)
    year_start = today.replace(month=1, day=1)

    with get_session() as session:
        trades = session.scalars(
            select(Trade).where(Trade.pnl.isnot(None), Trade.filled_at.isnot(None))
            .order_by(Trade.filled_at)
        ).all()
        positions = session.scalars(select(Position).where(Position.quantity > 0)).all()
    # 銘柄ごとに最新終値を個別取得する(N+1クエリ)のを避け、まとめて1クエリで取得する
    closes = latest_closes([p.symbol for p in positions])
    total_unrealized = 0.0
    for p in positions:
        close = closes.get(p.symbol)
        if close and p.avg_cost:
            total_unrealized += (close - p.avg_cost) * p.quantity

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


# ─── 取引停止スイッチ（kill switch）────────────────────────────────────────


class HaltRequest(BaseModel):
    reason: str = ""
    close_positions: bool = False  # True で停止と同時に全ポジションを成行決済する


@app.get("/api/halt")
async def get_halt_status():
    """取引停止スイッチの現在状態を返す（停止中か・理由・停止時刻）。"""
    from src.core import halt
    return halt.get_state()


@app.post("/api/halt")
async def engage_halt(req: HaltRequest, _: None = Depends(_verify_emergency_token)):
    """取引停止スイッチを作動させる（X-Emergency-Token 必須）。

    新規発注を全停止し、未約定BUYをキャンセルする。close_positions=True なら
    全ポジションを成行決済する。損切り・緊急決済は停止中も実行可能。
    """
    if _order_manager is None:
        raise HTTPException(status_code=503, detail="OrderManagerが初期化されていません")
    try:
        result = _order_manager.halt_trading(req.reason, close_positions=req.close_positions)
        return {"status": "ok", "message": "取引を停止しました", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/halt")
async def release_halt(_: None = Depends(_verify_emergency_token)):
    """取引停止スイッチを解除する（X-Emergency-Token 必須）。

    未解決注文（UNKNOWN/CANCEL_FAILED）が残っている間は解除できない（409）。
    """
    if _order_manager is None:
        raise HTTPException(status_code=503, detail="OrderManagerが初期化されていません")
    result = _order_manager.resume_trading()
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result.get("reason"))
    return {"status": "ok", "message": "取引を再開しました", **result}


# ─── ブローカー側逆指値ストップ（4.3）──────────────────────────────────────


class StopLossRequest(BaseModel):
    symbol: str
    quantity: Optional[int] = None      # 省略時は保有数量
    trigger_price: Optional[float] = None  # 省略時は avg_cost × (1 + stop_loss_pct)


@app.post("/api/stop_loss")
async def place_broker_stop(req: StopLossRequest, _: None = Depends(_verify_emergency_token)):
    """保有ポジションにブローカー側の逆指値ストップを置く（X-Emergency-Token 必須）。

    数量・トリガー価格を省略すると、保有数量と「取得平均 ×(1+損切り率)」から自動算出する。
    PC/アプリ停止時でも証券会社側で損切りが効く保険（ライブ/セミライブのみ実発注）。
    """
    if _order_manager is None:
        raise HTTPException(status_code=503, detail="OrderManagerが初期化されていません")
    with get_session() as session:
        pos = session.scalar(select(Position).where(Position.symbol == req.symbol))
    if pos is None or pos.quantity <= 0:
        raise HTTPException(status_code=400, detail=f"保有ポジションがありません: {req.symbol}")
    quantity = req.quantity or pos.quantity
    if req.trigger_price is not None:
        trigger = req.trigger_price
    else:
        stop_pct = cfg.get_section("trading").get("stop_loss_pct", -0.05)
        trigger = round(pos.avg_cost * (1 + stop_pct), 1)
    order_id = _order_manager.place_stop_loss(req.symbol, quantity, trigger)
    if not order_id:
        raise HTTPException(status_code=400,
                            detail="逆指値ストップを発注できませんでした（ペーパー/dry_run、または発注拒否）")
    return {"status": "ok", "order_id": order_id, "symbol": req.symbol,
            "quantity": quantity, "trigger_price": trigger}


# ─── semi_live 発注承認キュー ───────────────────────────────────────────────


@app.get("/api/approvals")
async def get_pending_approvals():
    """semi_live モードの承認待ち計画注文を返す。"""
    if _order_manager is None:
        raise HTTPException(status_code=503, detail="OrderManagerが初期化されていません")
    return _order_manager.list_pending_approvals()


@app.post("/api/approvals/{approval_id}/approve")
async def approve_pending_order(approval_id: int):
    """承認待ちの計画注文を承認し、実APIへ発注する（semi_live）。"""
    if _order_manager is None:
        raise HTTPException(status_code=503, detail="OrderManagerが初期化されていません")
    result = _order_manager.approve_order(approval_id)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("reason"))
    return {"status": "ok", "message": "承認して発注しました", **result}


@app.post("/api/approvals/{approval_id}/reject")
async def reject_pending_order(approval_id: int):
    """承認待ちの計画注文を却下する（発注しない）。"""
    if _order_manager is None:
        raise HTTPException(status_code=503, detail="OrderManagerが初期化されていません")
    result = _order_manager.reject_order(approval_id)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("reason"))
    return {"status": "ok", "message": "却下しました", **result}


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


@app.post("/api/data/update")
async def update_market_data():
    """過去データを手動で取得する（yfinance経由）"""
    if _data_update_fn is None:
        raise HTTPException(status_code=503, detail="データ更新関数が設定されていません")
    try:
        await asyncio.to_thread(_data_update_fn)
    except Exception as e:
        logger.error(f"データ更新失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "message": "データ更新が完了しました"}


# ─── ウォッチリスト ──────────────────────────────────────────────────


class WatchlistEntry(BaseModel):
    code: str
    name: str = ""


@app.get("/api/watchlist")
async def get_watchlist():
    """ウォッチリスト銘柄（コード・会社名）を返す"""
    return watchlist_store.get_all()


@app.post("/api/watchlist")
async def add_watchlist_entry(entry: WatchlistEntry):
    """ウォッチリストに銘柄を追加し、過去データを自動取得する。

    同一リスト内に同じ銘柄が既にある場合は 409 Conflict を返す（6.9 重複防止）。
    """
    try:
        result = watchlist_store.add(entry.code, entry.name)
    except watchlist_store.DuplicateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    from src.data.market_data import lookup_sector, update_symbol
    code = watchlist_store.normalize_code(entry.code)
    years = cfg.get_section("data").get("history_years", 3)
    try:
        await asyncio.to_thread(update_symbol, code, years)
    except Exception as e:
        logger.error(f"過去データ取得失敗: {code} {e}")
    try:
        sector = await asyncio.to_thread(lookup_sector, code)
        watchlist_store.update_sector(code, sector)
    except Exception as e:
        logger.warning(f"セクター取得失敗: {code} {e}")
    return result


@app.delete("/api/watchlist/{code}")
async def delete_watchlist_entry(code: str):
    """ウォッチリストから銘柄を削除"""
    return watchlist_store.remove(code)


# ─── 複数ウォッチリストの管理（作成・切替・削除・改名） ──────────────────────


@app.get("/api/watchlists")
async def get_watchlists():
    """全ウォッチリスト名とアクティブなリスト名を返す"""
    return {"active": watchlist_store.get_active_list_name(), "names": watchlist_store.get_list_names()}


class WatchlistNameRequest(BaseModel):
    name: str


@app.post("/api/watchlists")
async def create_watchlist(req: WatchlistNameRequest):
    """新規の空ウォッチリストを作成し、アクティブに切り替える"""
    try:
        names = watchlist_store.create_list(req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"active": watchlist_store.get_active_list_name(), "names": names}


@app.delete("/api/watchlists/{name}")
async def delete_watchlist(name: str):
    """ウォッチリストを削除する（最後の1つは削除不可。アクティブリストの場合は自動切替）"""
    try:
        return watchlist_store.delete_list(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/watchlists/active")
async def set_active_watchlist(req: WatchlistNameRequest):
    """アクティブなウォッチリストを切り替える（次回のジョブ実行から反映される）"""
    try:
        watchlist_store.switch_active(req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"active": req.name}


class WatchlistRenameRequest(BaseModel):
    new_name: str


@app.put("/api/watchlists/{name}")
async def rename_watchlist(name: str, req: WatchlistRenameRequest):
    """ウォッチリスト名を変更する"""
    try:
        new_name = watchlist_store.rename_list(name, req.new_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"active": watchlist_store.get_active_list_name(), "names": watchlist_store.get_list_names(), "renamed_to": new_name}


# ─── ウォッチリストの外部出力（エクスポート）・取込（インポート） ────────────


@app.get("/api/watchlist/export")
async def export_watchlist(name: Optional[str] = None):
    """指定リスト（省略時はアクティブリスト）をJSONファイルとしてダウンロードする"""
    try:
        data = watchlist_store.export_list(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    filename = f"watchlist_{data['name']}.json".replace("/", "_")
    # HTTPヘッダーはLatin-1必須のため、日本語等のリスト名は RFC 6266 の
    # filename*=UTF-8'' percent-encoding で渡す（ASCII filename はフォールバック用）
    encoded = quote(filename)
    return JSONResponse(
        content=data,
        headers={
            "Content-Disposition": f"attachment; filename=\"watchlist_export.json\"; filename*=UTF-8''{encoded}"
        },
    )


class WatchlistImportRequest(BaseModel):
    name: str
    entries: list[dict]
    overwrite: bool = False


@app.post("/api/watchlist/import")
async def import_watchlist(req: WatchlistImportRequest):
    """エクスポートされたJSON（{"name", "entries"}）からウォッチリストを取込み、アクティブに切り替える"""
    try:
        names = watchlist_store.import_list(req.name, req.entries, overwrite=req.overwrite)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"active": watchlist_store.get_active_list_name(), "names": names}


@app.get("/api/symbol_names")
async def get_symbol_names():
    """銘柄コード→会社名マッピングを返す（ダッシュボード表示用）"""
    return watchlist_store.get_names()


@app.get("/api/symbol_lookup/{code}")
async def lookup_symbol_name(code: str):
    """yfinanceから銘柄コードに対応する会社名を取得する（ウォッチリスト追加フォームの自動入力用）"""
    from src.data.market_data import lookup_company_name
    code = watchlist_store.normalize_code(code)
    name = await asyncio.to_thread(lookup_company_name, code)
    return {"code": code, "name": name}


# ─── バックテスト ──────────────────────────────────────────────────


class BacktestRequest(BaseModel):
    symbol: str
    start: str        # "YYYY-MM-DD"
    end: str          # "YYYY-MM-DD"
    initial_capital: float = 500_000.0
    use_ml: bool = False
    buy_threshold: Optional[float] = None   # 省略時はアクティブなリスクプロファイルの値を使用
    sell_threshold: Optional[float] = None  # ライブ/ペーパー取引の設定には影響しない（この実行のみ）


@app.get("/api/backtest/default_thresholds")
async def get_backtest_default_thresholds():
    """現在アクティブな買い/売り閾値（バックテストのデフォルト値表示用）を返す"""
    strat = cfg.get_section("strategy")
    return {"buy_threshold": strat.get("buy_threshold", 0.6), "sell_threshold": strat.get("sell_threshold", -0.6)}


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
            run_backtest, req.symbol, start_d, end_d, req.initial_capital, req.use_ml,
            req.buy_threshold, req.sell_threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"バックテスト失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"run_id": run_id, "status": "completed"}


def _serialize_run(r: BacktestRun) -> dict:
    return {
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
        "buy_threshold": r.buy_threshold,
        "sell_threshold": r.sell_threshold,
        "archived": bool(r.archived),
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


@app.get("/api/backtest/runs")
async def get_backtest_runs(page: int = 1, page_size: int = 15,
                            include_archived: bool = False):
    """バックテスト実行一覧をページングして返す（新しい順）。

    既定はアーカイブ済みを除外する（include_archived=true で含める）。1ページ15件。
    戻り値: {"runs": [...], "total": int, "page": int, "page_size": int, "total_pages": int}
    """
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    cond = [] if include_archived else [
        (BacktestRun.archived.is_(None)) | (BacktestRun.archived == 0)
    ]
    with get_session() as session:
        total = session.scalar(
            select(func.count(BacktestRun.id)).where(*cond)
        ) or 0
        runs = session.scalars(
            select(BacktestRun).where(*cond)
            .order_by(BacktestRun.id.desc())
            .offset((page - 1) * page_size).limit(page_size)
        ).all()
        items = [_serialize_run(r) for r in runs]
    total_pages = (total + page_size - 1) // page_size if total else 0
    return {
        "runs": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


class BacktestDeleteRequest(BaseModel):
    ids: list[int]


def _delete_runs(ids: list[int]) -> int:
    """指定したバックテスト実行とその取引明細を1トランザクションで削除する。

    実行(BacktestRun)と取引明細(BacktestTradeRecord)を一括で消し、孤児レコードを残さない。
    戻り値: 実際に削除した実行件数。
    """
    if not ids:
        return 0
    with get_session() as session:
        runs = session.scalars(
            select(BacktestRun).where(BacktestRun.id.in_(ids))
        ).all()
        found = [r.id for r in runs]
        if found:
            session.query(BacktestTradeRecord).filter(
                BacktestTradeRecord.run_id.in_(found)
            ).delete(synchronize_session=False)
            for r in runs:
                session.delete(r)
        session.commit()
    return len(found)


@app.delete("/api/backtest/{run_id}")
async def delete_backtest_run(run_id: int):
    """バックテスト実行を1件削除する（取引明細も同時削除）。"""
    deleted = _delete_runs([run_id])
    if deleted == 0:
        raise HTTPException(status_code=404, detail="バックテスト結果が見つかりません")
    return {"status": "ok", "deleted": deleted}


@app.post("/api/backtest/delete")
async def delete_backtest_runs(req: BacktestDeleteRequest):
    """バックテスト実行を複数まとめて削除する（取引明細も同時削除・1トランザクション）。"""
    deleted = _delete_runs(req.ids)
    return {"status": "ok", "deleted": deleted}


def _set_archived(run_id: int, archived: bool) -> bool:
    with get_session() as session:
        run = session.scalar(select(BacktestRun).where(BacktestRun.id == run_id))
        if run is None:
            return False
        run.archived = 1 if archived else 0
        session.commit()
    return True


@app.post("/api/backtest/{run_id}/archive")
async def archive_backtest_run(run_id: int):
    """バックテスト実行をアーカイブする（一覧から除外。履歴は保持）。"""
    if not _set_archived(run_id, True):
        raise HTTPException(status_code=404, detail="バックテスト結果が見つかりません")
    return {"status": "ok", "archived": True}


@app.post("/api/backtest/{run_id}/unarchive")
async def unarchive_backtest_run(run_id: int):
    """バックテスト実行のアーカイブを解除する。"""
    if not _set_archived(run_id, False):
        raise HTTPException(status_code=404, detail="バックテスト結果が見つかりません")
    return {"status": "ok", "archived": False}


@app.get("/api/backtest/{run_id}/divergence")
async def get_backtest_divergence(run_id: int):
    """バックテストの取引と、同一銘柄の実取引の乖離（勝率・平均リターン・純損益）を返す（7.7）。"""
    from src.analytics.performance import compute_divergence
    with get_session() as session:
        run = session.scalar(select(BacktestRun).where(BacktestRun.id == run_id))
        if run is None:
            raise HTTPException(status_code=404, detail="バックテスト結果が見つかりません")
        bt_trades = session.scalars(
            select(BacktestTradeRecord).where(BacktestTradeRecord.run_id == run_id)
        ).all()
        actual = session.scalars(
            select(Trade).where(Trade.symbol == run.symbol, Trade.pnl.isnot(None))
        ).all()
        result = compute_divergence(bt_trades, actual)
    result["symbol"] = run.symbol
    return result


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
            "buy_threshold": run.buy_threshold,
            "sell_threshold": run.sell_threshold,
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
