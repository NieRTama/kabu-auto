"""kabuステーションの起動・ログイン待ち（起動時の接続リトライ）。

背景: kabu-auto をOS起動時に自動実行すると、kabuステーション(KabuS.exe)が
まだ起動・ログインされていないため必ず接続に失敗する。実発注モードは
fail-closed で即座に起動を中断する設計のため、「自動起動したはずが動いていない」
という事故になっていた（2026-08-28 実際に発生）。

kabuステーションのログイン自体は人手が要る（2段階認証）ため自動化しない。
代わりに「ログインされるまで静かに待つ」ことで、ユーザーが好きなタイミングで
ログインすれば自動的に稼働を開始できるようにする。

このモジュールは待機ループだけを担う純粋なユーティリティで、KabuClient や
main の起動シーケンスを知らない（テスト容易性のため sleep/通知も注入可能）。
"""
import time
from typing import Callable, Optional

from loguru import logger


def wait_for_broker(
    connect: Callable[[], object],
    *,
    timeout_seconds: float,
    unlimited: bool = False,
    initial_interval: float = 30.0,
    max_interval: float = 60.0,
    on_wait_start: Optional[Callable[[], None]] = None,
    on_connected: Optional[Callable[[float], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """`connect()` が成功するまで待機する。成功したら True、時間切れなら False。

    connect: 接続を試みる呼び出し（例: client.refresh_token）。例外を投げたら失敗とみなす。
    timeout_seconds: 待機を諦めるまでの秒数。0以下なら1回だけ試して結果を返す（従来動作）。
    unlimited: True なら timeout_seconds を無視して接続できるまで待ち続ける。
        起動時にこれを使う。時間制限を設けると「30分以内にログインしないと
        起動しない」ことになり、朝寝坊や通知の見落としで丸一日動かない
        （2026-08-29 に稼働中の復帰で同じ穴を踏んだ）。
    on_wait_start: 1回目の失敗直後に一度だけ呼ぶ（Discord通知などの副作用用）。
    on_connected: 待機の末に接続できたとき、待機秒数を引数に呼ぶ。

    待機中のログは冗長にならないよう、最初の失敗時と以後5分ごとのみ出力する
    （毎回出すと自動起動から人がログインするまでの間ログが埋まる）。
    """
    try:
        connect()
        return True
    except Exception as e:
        first_error = e

    if timeout_seconds <= 0 and not unlimited:
        logger.error(f"kabuステーション接続に失敗しました: {first_error}")
        return False

    started = monotonic()
    limit_text = "接続できるまで待機します" if unlimited else f"最大{timeout_seconds / 60:.0f}分待機します"
    logger.warning(
        f"kabuステーションに接続できません（未起動またはログイン前）。{limit_text}: {first_error}"
    )
    if on_wait_start is not None:
        try:
            on_wait_start()
        except Exception as e:  # 通知の失敗で待機を止めない
            logger.warning(f"待機開始の通知に失敗しました（待機は継続）: {e}")

    interval = initial_interval
    last_log = started
    while True:
        elapsed = monotonic() - started
        if not unlimited and elapsed >= timeout_seconds:
            logger.critical(
                f"kabuステーションへ接続できないまま{timeout_seconds / 60:.0f}分が経過しました。"
                "kabuステーションを起動してログインしてから、再度実行してください"
            )
            return False

        remaining = interval if unlimited else min(interval, max(0.0, timeout_seconds - elapsed))
        sleep(remaining)
        interval = min(interval * 2, max_interval)

        try:
            connect()
        except Exception as e:
            now = monotonic()
            # 待機中は5分に1回だけ状況を残す（ログのノイズ抑制）
            if now - last_log >= 300:
                last_log = now
                logger.info(
                    f"kabuステーションのログイン待ち（経過 {(now - started) / 60:.0f}分）: {e}"
                )
            continue

        waited = monotonic() - started
        logger.info(f"kabuステーションへ接続しました（待機 {waited / 60:.1f}分）。起動を続行します")
        if on_connected is not None:
            try:
                on_connected(waited)
            except Exception as e:
                logger.warning(f"接続の通知に失敗しました（起動は継続）: {e}")
        return True
