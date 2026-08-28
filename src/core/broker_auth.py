"""kabuステーションの認証状態（ログイン切れ）を保持する。

kabuステーションのログイン認証には有効期限があり、PCを起動したままでも
日をまたぐと切れる（アプリ自身が "Code 10016: ログイン認証の有効期間が切れました。
再ログインしてください。" を持つ）。実際 2026-08-26・08-27 は毎朝8:30の
トークン更新が401で失敗し、以後終日1,300回超の401を出しながら
「起動しているが証券会社と通信できない抜け殻」で動き続けていた。

このモジュールは「今ログイン切れかどうか」という状態だけを持つ。
発注抑止は RiskManager.can_place_order() が is_expired() を見て行い、
再ログイン待ちのリトライは main 側のジョブが担う（関心の分離。
halt.py と同じ役割分担にしてある）。

永続化はしない。プロセス再起動時は必ずトークン取得を試すため、
その結果で状態が決まる（古い状態を引きずらない）。
"""
from typing import Optional

from loguru import logger

from src.core import clock

_expired: bool = False
_detail: str = ""
_since: Optional[str] = None


def mark_expired(detail: str = "") -> None:
    """認証切れを記録する（トークン更新失敗時に呼ぶ）。

    既に切れている場合はログを重ねない（毎朝の失敗後、リトライのたびに
    記録するとログが埋まるため）。
    """
    global _expired, _detail, _since
    if _expired:
        _detail = detail or _detail
        return
    _expired = True
    _detail = detail
    _since = clock.now().isoformat()
    logger.critical(
        "kabuステーションのログイン認証が切れています。再ログインするまで新規発注を停止します"
        f"（検知: {_since}）: {detail}"
    )


def mark_valid() -> None:
    """認証が有効になったことを記録する（トークン取得成功時に呼ぶ）。"""
    global _expired, _detail, _since
    if _expired:
        logger.warning("kabuステーションの認証が回復しました。取引を再開します")
    _expired = False
    _detail = ""
    _since = None


def is_expired() -> bool:
    """現在ログイン切れか（新規発注を抑止すべきか）。"""
    return _expired


def get_state() -> dict:
    return {"expired": _expired, "detail": _detail, "since": _since}


def reset() -> None:
    """テスト用にモジュール状態を初期化する。"""
    global _expired, _detail, _since
    _expired = False
    _detail = ""
    _since = None
