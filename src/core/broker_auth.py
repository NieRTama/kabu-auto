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

## トークン更新直後の一過性401（2026-09-08）

`KabuClient._token` はスレッド間で共有されるインスタンス変数で、更新に排他制御が
無い。`auth_recovery_check`（5分間隔）がトークンを更新した直後、別スレッドの
`reconcile_positions_with_broker`（15秒間隔）が**更新前のトークンで送信済みだった
リクエスト**がサーバーに到達し、401で拒否されることがある
（kabuステーション側ログで実測: トークン更新の2.2ミリ秒後に別リクエストが
`Code=4001009 APIキー不一致` で拒否）。300秒(5分)は15秒の倍数のため、
**この競合は周期的に必ず発生する**。

以前は個々のAPI呼び出しの401を無視していたため無害だったが、
「場中の401を認証切れとして記録する」ようにした際（2026-09-02の事故の修正）、
この一過性の競合が**5分ごとに新規発注を止める**副作用に変わった。実際には
接続できているのに、次の`auth_recovery_check`まで「切れ」表示が固定される。

`mark_valid()` の直後 `_RACE_GRACE_SECONDS` 以内の `mark_expired()` は、
本物の認証切れではなく通り抜け中のリクエストの可能性が高いため記録しない
（ログには残す。次のAPI呼び出しでも401が続けば、猶予後に正しく検知される）。
"""
import time
from typing import Optional

from loguru import logger

from src.core import clock

_expired: bool = False
_detail: str = ""
_since: Optional[str] = None
_last_valid_at: Optional[float] = None

# トークン更新直後、このN秒以内の401は一過性の競合とみなして無視する。
# 本物の認証切れなら次のAPI呼び出しでも401が続くため、猶予後に検知される。
_RACE_GRACE_SECONDS = 3.0


def mark_expired(detail: str = "") -> None:
    """認証切れを記録する（トークン更新失敗時に呼ぶ）。

    既に切れている場合はログを重ねない（毎朝の失敗後、リトライのたびに
    記録するとログが埋まるため）。
    """
    global _expired, _detail, _since
    if _expired:
        _detail = detail or _detail
        return
    if _last_valid_at is not None and time.monotonic() - _last_valid_at < _RACE_GRACE_SECONDS:
        # トークン更新直後の一過性401（上記モジュールdocstring参照）。
        # 本物の切れなら次の呼び出しでも401が続き、猶予後に検知される。
        logger.debug(f"トークン更新直後の一過性401を無視（猶予中）: {detail}")
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
    global _expired, _detail, _since, _last_valid_at
    if _expired:
        logger.warning("kabuステーションの認証が回復しました。取引を再開します")
    _expired = False
    _detail = ""
    _since = None
    _last_valid_at = time.monotonic()


def is_expired() -> bool:
    """現在ログイン切れか（新規発注を抑止すべきか）。"""
    return _expired


def get_state() -> dict:
    return {"expired": _expired, "detail": _detail, "since": _since}


def reset() -> None:
    """テスト用にモジュール状態を初期化する。"""
    global _expired, _detail, _since, _last_valid_at
    _expired = False
    _detail = ""
    _since = None
    _last_valid_at = None
