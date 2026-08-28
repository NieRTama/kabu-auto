"""生存確認（ハートビート）を定期送信する。

全ての異常検知には共通の弱点がある: **プロセス自体が落ちたら通知も来ない**。
「異常がない」と「死んでいる」が区別できないため、通知が来ないことを
「正常」と誤認してしまう。実際 2026-08-26/27 は2営業日にわたり
「何も通知が来ない＝正常」と誤認し、認証切れで停止していたことに気づけなかった。

そこで毎営業日の場前に「稼働中です」を能動的に送る。これがあると
**通知が来ないこと自体が異常**として扱えるようになる。

送る内容は運用判断に必要な最小限（モード・発注可否・認証状態・建玉数）に絞り、
秘密情報は含めない。
"""
from loguru import logger

from src.core import broker_auth
from src.core import trading_mode as tm
from src.core.alerts import alert


def build_message(mode: str, snapshot: dict, position_count: int) -> str:
    """ハートビート本文を組み立てる（純粋関数。テストしやすいよう副作用なし）。"""
    can_order = snapshot.get("can_place_order")
    block = snapshot.get("block_reason") or ""
    auth = "有効" if not broker_auth.is_expired() else "切れ（要ログイン）"
    order_state = "可" if can_order else f"不可（{block}）"
    return (
        f"モード: {tm.description(mode)}\n"
        f"発注: {order_state}\n"
        f"kabuステーション認証: {auth}\n"
        f"保有銘柄: {position_count}件 / "
        f"未解決注文: {snapshot.get('unresolved_orders', 0)}件"
    )


def send(mode: str, snapshot: dict, position_count: int) -> None:
    """稼働中であることを通知する。失敗しても運用は止めない。"""
    try:
        alert("稼働中です（定期報告）", build_message(mode, snapshot, position_count))
    except Exception as e:
        logger.error(f"ハートビート送信に失敗しました: {e}")
