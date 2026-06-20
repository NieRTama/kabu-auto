"""
異常検知・アラート（Phase 5 / 7.5）。

「システムが口座状態を誤読して誤った前提で発注すること」が最大のリスクという思想に基づき、
定期的に運用上の異常を検知して通知する。同じ異常を毎回通知して埋もれさせないよう、
状態が「新たに発生したとき」だけ alert を出し、解消したらログに記録する（エッジtrigger）。

検知する異常:
  - 未解決注文（UNKNOWN / CANCEL_FAILED）の残存（critical。新規発注が抑止される）
  - 当日損失が上限に接近/到達（warning/critical）
  - 取引停止スイッチ（kill switch）が ON のまま（warning）
"""
import threading

from loguru import logger
from sqlalchemy import func, select

from src.core import halt
from src.core.alerts import alert
from src.data.database import Trade, get_session
from src.execution import order_status as st

WARNING = "warning"
CRITICAL = "critical"

# 当日損失が上限のこの割合に達したら警告する（到達前の予兆検知）
LOSS_WARN_RATIO = 0.8

_alerted_keys: set = set()
_lock = threading.Lock()


def check_anomalies(risk) -> list[dict]:
    """現在の異常一覧 [{key, level, message}] を返す（副作用なし）。"""
    items: list[dict] = []

    with get_session() as session:
        unresolved = session.scalar(
            select(func.count(Trade.id)).where(
                Trade.status.in_(tuple(st.UNRESOLVED_STATUSES))
            )
        ) or 0
    if unresolved:
        items.append({
            "key": "unresolved_orders", "level": CRITICAL,
            "message": f"未解決注文が{unresolved}件あります（UNKNOWN/CANCEL_FAILED）。"
                       "実口座を確認し解消するまで新規発注は抑止されます",
        })

    limit = risk.daily_loss_limit()
    if limit and limit > 0:
        loss = risk.current_daily_loss()
        ratio = loss / limit
        if ratio >= 1.0:
            items.append({
                "key": "daily_loss_limit", "level": CRITICAL,
                "message": f"当日損失が上限に到達: {loss:,.0f} / {limit:,.0f}円",
            })
        elif ratio >= LOSS_WARN_RATIO:
            items.append({
                "key": "daily_loss_warn", "level": WARNING,
                "message": f"当日損失が上限の{ratio:.0%}に接近: {loss:,.0f} / {limit:,.0f}円",
            })

    if halt.is_halted():
        items.append({
            "key": "halted", "level": WARNING,
            "message": f"取引停止スイッチがONです（{halt.get_state().get('reason') or '手動停止'}）",
        })

    return items


def run_and_alert(risk) -> list[dict]:
    """異常を検知し、新規発生分のみ alert を送る（解消はログ）。現在の異常一覧を返す。"""
    items = check_anomalies(risk)
    current = {i["key"] for i in items}
    with _lock:
        new_items = [i for i in items if i["key"] not in _alerted_keys]
        recovered = _alerted_keys - current
        _alerted_keys.clear()
        _alerted_keys.update(current)
    for i in new_items:
        alert("異常検知", i["message"])
    for key in recovered:
        logger.info(f"異常が解消しました: {key}")
    return items


def reset() -> None:
    """通知済み状態をリセットする（テスト・再起動時用）。"""
    with _lock:
        _alerted_keys.clear()
