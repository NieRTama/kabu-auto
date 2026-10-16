"""総資産（現金＋保有評価額）の計算と日次記録。ダッシュボードと Discord の各メッセージが共用する。

現金は live・dry_run・semi_live では口座余力（`/wallet/cash`）、paper では仮想残高。
価格を取得できない銘柄は評価額に含めず `unpriced` に載せる（取得単価で代用すると含み損益が
0に見え、価格欠落を隠すため。pnl_report.build_holdings と同じ方針）。余力を取得できない場合は
total を None にする（評価額だけを総資産として扱わない）。

口座余力の取得は本体プロセス（kabu-auto）の中でだけ行う。別プロセスがトークンを発行すると
本体側のトークンが無効になるため、別プロセス（scripts/daily_ops_report.py）は記録値を読む。
"""
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

from loguru import logger
from sqlalchemy import select

from src.core import clock
from src.core import trading_mode as tm

# 口座余力のキャッシュ（時刻, 値）。ダッシュボードは10秒ごとに全APIを叩くため、
# 毎回 /wallet/cash を呼ぶと証券会社APIの実行回数制限（429）に近づく（2026-09-11 の事例）。
_wallet_cache: Optional[tuple] = None
WALLET_CACHE_SECONDS = 30

# 総資産の日次記録 {"YYYY-MM-DD": 総資産}。前日比の算出に使う（その日の最後の計算値が残る）。
HISTORY_PATH = Path("data/total_assets_history.json")
HISTORY_KEEP_DAYS = 90


def _broker_cash(client) -> float:
    global _wallet_cache
    now = time.monotonic()
    if _wallet_cache and now - _wallet_cache[0] < WALLET_CACHE_SECONDS:
        return _wallet_cache[1]
    if client is None:
        raise RuntimeError("kabuステーションAPIクライアント未接続")
    cash = float(client.get_wallet().get("StockAccountWallet", 0))
    _wallet_cache = (now, cash)
    return cash


def _load_history() -> dict:
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _save_history(hist: dict) -> None:
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = HISTORY_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(hist, ensure_ascii=False, indent=0), encoding="utf-8")
        os.replace(tmp, HISTORY_PATH)
    except OSError as e:
        logger.warning(f"総資産の日次記録に失敗しました: {e}")


def previous_record(today: date) -> tuple:
    """当日より前で最新の記録 (日付, 総資産)。無ければ (None, None)。"""
    hist = _load_history()
    past = sorted(k for k in hist if k < today.isoformat())
    return (date.fromisoformat(past[-1]), hist[past[-1]]) if past else (None, None)


def recorded(day: date) -> Optional[float]:
    """その日の記録値（無ければ None）。"""
    return _load_history().get(day.isoformat())


def _record(today: date, total: float) -> None:
    hist = _load_history()
    key = today.isoformat()
    if hist.get(key) == total:
        return
    hist[key] = total
    for k in sorted(hist)[:-HISTORY_KEEP_DAYS]:
        del hist[k]
    _save_history(hist)


def previous_business_day(d: date) -> date:
    from src.core import market_calendar
    d -= timedelta(days=1)
    while market_calendar.is_holiday(d):
        d -= timedelta(days=1)
    return d


def change_label(today: date, prev_date: Optional[date]) -> Optional[str]:
    """比較対象が前営業日なら「前日比」、それ以外は「M/D比」。"""
    if prev_date is None:
        return None
    if prev_date == previous_business_day(today):
        return "前日比"
    return f"{prev_date.month}/{prev_date.day}比"


def compute(mode: str, *, client, price_fn: Callable[[list], dict],
            paper_base: float, basis: float, record: bool = True) -> dict:
    """総資産と内訳・前日比・基準資金比を返す。record=True なら当日分を記録する。"""
    from src.data.database import Position, get_session
    with get_session() as session:
        positions = session.scalars(select(Position).where(Position.quantity > 0)).all()
    prices = price_fn([p.symbol for p in positions]) if positions else {}
    market_value = cost = 0.0
    unpriced = []
    for p in positions:
        price = prices.get(p.symbol)
        if not price:
            unpriced.append(p.symbol)
            continue
        market_value += price * p.quantity
        cost += p.avg_cost * p.quantity

    cash, cash_error = None, None
    if tm.reads_broker_api(mode):
        cash_source = "broker"
        try:
            cash = _broker_cash(client)
        except Exception as e:
            cash_error = str(e)[:200]
    else:
        from src.services.trading import _paper_available_cash
        cash_source = "paper"
        cash = _paper_available_cash(paper_base)

    total = round(cash + market_value, 0) if cash is not None else None
    today = clock.today()
    prev_date, prev_total = previous_record(today)
    if record and total is not None:
        _record(today, total)
    return {
        "total": total,
        "cash": round(cash, 0) if cash is not None else None,
        "market_value": round(market_value, 0),
        "unrealized": round(market_value - cost, 0),
        "positions": len(positions),
        "unpriced": unpriced,
        "cash_source": cash_source,
        "cash_error": cash_error,
        "basis": basis or None,
        "basis_change": round(total - basis, 0) if total is not None and basis else None,
        "prev_total": prev_total,
        "prev_label": change_label(today, prev_date),
        "day_change": round(total - prev_total, 0) if total is not None and prev_total is not None else None,
    }


def _signed(v: float) -> str:
    return f"{'+' if v > 0 else '-' if v < 0 else '±'}{abs(v):,.0f}円"


def format_line(d: dict) -> str:
    """Discord 向けの1行（例: 総資産: 492,380円（前日比 +3,160円 ／ 基準資金比 -7,620円））。"""
    if d.get("total") is None:
        reason = d.get("cash_error") or "口座余力を取得できません"
        return f"総資産: 取得不可（{reason}）"
    parts = []
    if d.get("day_change") is not None:
        parts.append(f"{d['prev_label']} {_signed(d['day_change'])}")
    if d.get("basis_change") is not None:
        parts.append(f"基準資金比 {_signed(d['basis_change'])}")
    tail = f"（{' ／ '.join(parts)}）" if parts else ""
    line = f"総資産: {d['total']:,.0f}円{tail}"
    if d.get("unpriced"):
        line += f"\n※価格取得不可のため総資産に未算入: {', '.join(d['unpriced'])}"
    return line
