"""ログイン認証切れからの自動復帰を、時間制限なく継続的に試みる。

## 背景（2026-08-29 に実際に起きた設計欠陥）

当初は「トークン更新に失敗したら30分だけ待つ」実装だった。しかしこれは
**一度タイムアウトしたら翌朝まで復帰しない**という致命的な穴を持っていた:

  08:30 認証切れ検知 → 通知 → 30分待機
  09:00 タイムアウト。以後いくらログインしても復帰せず、その日は終日発注不可

「通知を見て→ログインしたのに動かない」では、せっかくの通知が意味を失う。
朝寝坊・通知の見落としといった現実的な事象で丸一日の取引機会を失う。

## 方針

認証切れの間は**時間制限なく**、一定間隔で再接続を試み続ける。ログインされた
時点で自動復帰する（何時であっても）。復帰後は次に切れるまで何もしない。

呼び出しは短時間で必ず返る（ブロッキングしない）。スケジューラの定期ジョブから
呼ばれる前提で、待機はジョブ間隔が担う。これにより
「待機スレッドが生き残る／タイムアウトで諦める」という状態を持たなくて済む。
"""
from typing import Callable, Optional

from loguru import logger

from src.core import broker_auth


def attempt_recovery(
    connect: Callable[[], object],
    *,
    on_recovered: Optional[Callable[[], None]] = None,
    is_expired: Callable[[], bool] = broker_auth.is_expired,
    mark_valid: Callable[[], None] = broker_auth.mark_valid,
) -> bool:
    """認証切れなら再接続を1回試す。復帰したら True。

    認証が有効なときは何もしない（毎回APIを叩いてトークンを無駄に再発行しない）。
    失敗しても例外を投げず False を返す（次回のジョブ実行で再試行するため）。

    connect: 接続を試みる呼び出し（例: client.refresh_token）
    on_recovered: 復帰したときに1度だけ呼ぶ（通知用）
    """
    if not is_expired():
        return False

    try:
        connect()
    except Exception as e:
        # 認証切れ中は失敗が続くのが通常。ログを重ねないよう DEBUG に落とす
        # （切れたこと自体は broker_auth.mark_expired が CRITICAL で記録済み）。
        logger.debug(f"認証復帰の試行に失敗（次回再試行）: {e}")
        return False

    mark_valid()
    logger.warning("kabuステーションの認証が回復しました。取引を再開します")
    if on_recovered is not None:
        try:
            on_recovered()
        except Exception as e:
            logger.warning(f"認証回復の通知に失敗しました（復帰処理は完了）: {e}")
    return True
