"""kabuステーションのプロセス断を「鳴らしすぎず・見落とさず」知らせる判断。

## なぜ切り出すか

この判断を main.py のクロージャに直接書くと、テストが「ソース文字列が含まれるか」
しか見られず、休場日・判定不能・再通知間隔・自動起動の有無といった分岐を検証できない。
実際、クロージャに書いた版はレビューで複数の穴（週末に🔴が約100件、判定不能が
スロットルを解除、文面が構成と食い違う）が見つかった。判断だけをここに置き、
通知の送信は呼び出し側（alert）に任せる。

## 何を守るか

- **鳴らしすぎない**: 夜間・休場日は鳴らさない。対応不要な🔴が届き続けると、
  本物の🔴を無視する癖がつく（2026-09-05 に token_refresh で同じ問題を潰した）。
- **見落としで終日失わない**: 一度きりにせず、落ちている間は一定間隔で再通知する。
- **判定できない回を「復帰」と誤解しない**: tasklist の失敗でスロットルが
  解除されると、鳴る間隔が縮む。
"""
from datetime import datetime, time as dtime
from typing import Optional, Tuple

from src.core import market_calendar

#: 落ちている間の再通知の間隔。一度きりだと、その1通を見落として終日気づけない。
DOWN_REPEAT_SECONDS = 1800

#: 通知してよい時間帯。場が関係する時間だけに絞る（前場前〜大引け直後）。
NOTIFY_FROM = dtime(8, 0)
NOTIFY_TO = dtime(15, 40)


def in_notify_window(now: datetime) -> bool:
    """通知してよい時刻か（営業日の、場が関係する時間帯のみ）。

    夜間や休場日にアプリを閉じるのは通常の操作であって異常ではない。
    ここを見ないと、金曜の夜に閉じただけで週末中ずっと🔴が鳴り続ける
    （5分間隔ジョブ × 週末＝100通規模）。
    """
    if not market_calendar.is_business_day(now.date()):
        return False
    return NOTIFY_FROM <= now.time() <= NOTIFY_TO


class BrokerDownNotifier:
    """プロセス断を知らせるかどうかを決める（送信はしない）。"""

    def __init__(self, *, repeat_seconds: int = DOWN_REPEAT_SECONDS):
        self._repeat = repeat_seconds
        self._last_at: float = 0.0

    def should_notify(self, *, alive: Optional[bool], now: float,
                      notify_window: bool) -> bool:
        """今このタイミングで知らせるべきか。

        alive は3値:
          True  … 起動している。落ちた記録を戻す（次に落ちれば即通知）
          False … 落ちている。窓の中かつ間隔を空けていれば知らせる
          None  … 判定できなかった。**何もしない**（通知もリセットもしない。
                  リセットすると tasklist の一過性の失敗で間隔が縮む）
        """
        if alive is True:
            self._last_at = 0.0
            return False
        if alive is None or not notify_window:
            return False
        if self._last_at and now - self._last_at < self._repeat:
            return False
        self._last_at = now
        return True

    def check(self, *, alive: Optional[bool], now: float,
              notify_window: bool, auto_launch: bool,
              launch_result: Optional[Tuple[bool, str]] = None
              ) -> Optional[Tuple[str, str]]:
        """通知すべきなら (タイトル, 本文) を返す。不要なら None。"""
        # 起動できた回は🟢「起動しました」を別途出しているので、🔴で追い打ちしない。
        # スロットルの枠も消費しない（消費すると、次に本当に異常な回が黙る）。
        if launch_result is not None and launch_result[0]:
            return None
        if not self.should_notify(alive=alive, now=now, notify_window=notify_window):
            return None
        return down_message(auto_launch=auto_launch, launch_result=launch_result)


def down_message(*, auto_launch: bool,
                 launch_result: Optional[Tuple[bool, str]]) -> Tuple[str, str]:
    """プロセス断の通知文を作る。**実際に起きたことだけを書く**。

    「自動起動を試みます」と先に約束すると、実行ファイル未検出・日次上限・
    クールダウンで起動できなかったときに、30分おきに約束だけを繰り返しながら
    一度も起動せず、利用者は待たされたまま終日取引を失う
    （コミット a4a591d で潰した「問題を隠すログ」と同型）。
    """
    head = "kabuステーションのプロセスが落ちています。\n"
    if not auto_launch:
        body = head + (
            "アプリを起動してログインしてください"
            "（自動起動は設定で無効にしています）。"
        )
    elif launch_result is None:
        body = head + (
            "自動起動は行われませんでした。"
            "手動で起動してログインしてください。"
        )
    elif launch_result[0]:
        body = head + (
            f"{launch_result[1]}\n認証（ログイン）は人が行う必要があります。"
        )
    else:
        body = head + (
            f"自動起動できませんでした: {launch_result[1]}\n"
            "手動で起動してログインしてください。"
        )
    return "kabuステーションが起動していません", body
