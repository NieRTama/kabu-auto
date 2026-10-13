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
from typing import Callable, Optional, Tuple

from loguru import logger

from src.core import broker_auth, market_calendar
from src.core.alerts import LEVEL_INFO
from src.core.auth_recovery import RecoveryResult, attempt_recovery

#: 落ちている間の再通知の間隔。一度きりだと、その1通を見落として終日気づけない。
DOWN_REPEAT_SECONDS = 1800

#: 通知してよい時間帯。場が関係する時間だけに絞る（前場前〜大引け直後）。
NOTIFY_FROM = dtime(8, 0)
NOTIFY_TO = dtime(15, 40)

#: 日中の自動復帰（完全自動ログイン）を許可し始める時刻。
#: 08:30の定時ジョブ(broker_full_login_job)が起動〜ログインを担当するため、
#: それより前に落ちているのは夜間に閉じた通常状態であり、二重にログインさせない。
FULL_LOGIN_RECOVERY_FROM = dtime(8, 40)


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


def should_full_login_recover(*, alive: Optional[bool], auth: RecoveryResult,
                               in_window: bool, full_login_enabled: bool,
                               auto_launch: bool, now_time: dtime) -> bool:
    """kabuステーションの認証切れから完全自動ログインで復帰させてよいか。

    auto_launch_broker は preflight の起動経路の排他性チェックにより
    broker_full_login_enabled と同時にtrueにできない構成のため、
    auto_launch側の自動起動が無効な運用では日中の断が起きても
    「🔴を出すだけで誰も起動しない」状態になっていた
    （2026-09-28朝、KabuSが落ちて08:30の定時ジョブもmissedで終日手動になりかけた）。

    `alive is True`（プロセスは生きたまま認証だけ失効。1.33.1と同型）も対象に
    含める（2026-09-30）。完全自動ログイン（broker_full_login.run()）はWSL側
    スクリプトが既存のKabuS.exeを一旦killしてから再起動・再ログインする設計
    のため、二重起動にはならない。`alive is None`（tasklist判定不能）だけは
    動いているものを誤ってkillしかねないため引き続き見送る。

    08:40以降に限る（FULL_LOGIN_RECOVERY_FROM参照）: 08:30の定時ジョブが
    朝の起動〜ログインを担当するため、それより前は夜間に閉じた通常状態であり
    二重にログインさせない。休場日のガードは in_window（in_notify_window）が
    営業日判定を含んでいるため別途不要。

    生存中は **認証が実際に切れていて再接続も失敗したとき（auth=FAILED）だけ**
    対象にする（2026-10-13）。認証が有効（NOT_NEEDED）なのに「未回復」と読んで、
    ログイン済みのKabuSを08:40以降5分ごとにkillして再ログインし、トークンを
    無効化し続けた（日次上限3回で止まるまで発注不可）。プロセスが落ちている場合
    （alive is False）は、場中のクラッシュが401ではなく接続エラーになり認証切れと
    して記録されないため、認証状態に関わらず対象のまま。
    """
    if alive is None:
        return False
    if auth is RecoveryResult.RECOVERED:
        return False
    if alive is True and auth is not RecoveryResult.FAILED:
        return False
    if not in_window:
        return False
    if not full_login_enabled:
        return False
    if auto_launch:
        return False
    return now_time >= FULL_LOGIN_RECOVERY_FROM


def auth_valid_now(refresh_token: Callable[[], object]) -> bool:
    """KabuSを強制終了する前の最終確認。トークンが取れるなら認証は生きている。

    判定ロジック（should_full_login_recover 等）とは独立した安全装置。判定が一度
    誤っても、壊れていないKabuSを完全自動ログインで止めることはなくなる
    （2026-10-13、判定の誤りでログイン済みのKabuSを2回強制再ログインした）。
    """
    try:
        refresh_token()
    except Exception:
        return False
    broker_auth.mark_valid()
    return True


def recovery_tick(*, refresh_token: Callable[[], object],
                  on_recovered: Callable[[], None],
                  probe_running: Callable[[], Optional[bool]],
                  now: datetime, runtime: dict,
                  launch_broker: Callable[[], Tuple[bool, str]],
                  full_login: Callable[[], Tuple[bool, str]],
                  notifier: "BrokerDownNotifier",
                  alert: Callable[..., None], monotonic: float) -> None:
    """5分ごとの認証復帰チェック1回分（main.py の auth_recovery_check の本体）。

    main() のクロージャに直接書いていたため、1日の時系列で回すテストが書けず、
    「ログイン済みで認証も有効」という普段の状態での誤作動を見逃した（2026-10-13）。
    依存を引数で受け取り、tests/test_recovery_scenario.py で時系列のまま検証する。
    """
    auth = attempt_recovery(refresh_token, on_recovered=on_recovered)
    recovered = auth is RecoveryResult.RECOVERED

    # KabuS.exe の生存確認は**認証の状態に関わらず毎回**行う。ここが唯一の
    # 生存確認であり、場中のクラッシュは接続エラーになるだけで 401 ではない
    # （kabu_client は401のときだけ mark_expired する）。認証切れの判定の
    # 後ろに置くと、本来の対象であるクラッシュ復帰に永久に到達しない。
    # auto_launch の既定は False（起動元を人に一本化する運用）。キーが欠けた設定で
    # 自動起動が黙って復活すると、Windowsの自動起動と競合して二重起動が再発する。
    auto_launch = bool(runtime.get("auto_launch_broker", False))
    alive = probe_running()
    in_window = in_notify_window(now)

    # 通知より先に起動を試す。通知文へ「実際にどうなったか」を載せるため
    # （先に「試みます」と書くと、起動できなかった事実が隠れる）。
    # 夜間・休場日に閉じるのは通常の操作なので、通知と同じ窓の中だけで起動する。
    # 起動するのは、プロセスが見つからず**かつAPIにも到達できない**ときだけ。
    # tasklist の判定だけを根拠にすると、プロセス名の変更などで誤って False に
    # なった場合、APIが正常稼働しているのに5分ごとに追加起動してしまう。
    launch_result = None
    if alive is False and not recovered and auto_launch and in_window:
        launch_result = launch_broker()
        if launch_result[0]:
            alert(
                "kabuステーションを起動しました",
                f"{launch_result[1]}\n認証アプリで承認してください。"
                "承認後は自動で取引を再開します。",
                level=LEVEL_INFO,
            )
        else:
            logger.warning(f"kabuステーションの自動起動は行われませんでした: {launch_result[1]}")

    # 日中に落ちた／認証が切れたときの完全自動ログイン復帰。auto_launch_broker は
    # preflight の起動経路の排他性チェックにより broker_full_login_enabled と併用できない
    # ため、auto_launch側が無効な運用（現行）では上のブロックが一度も起動を試みず、
    # 🔴を出すだけで誰も起動しない穴があった（2026-09-28朝の実例）。
    full_login_enabled = bool(runtime.get("broker_full_login_enabled", False))
    if should_full_login_recover(
        alive=alive, auth=auth, in_window=in_window,
        full_login_enabled=full_login_enabled, auto_launch=auto_launch,
        now_time=now.time(),
    ):
        if auth_valid_now(refresh_token):
            logger.warning(
                "完全自動ログインの条件を満たしましたが、トークンを取得できたため"
                "実行しません（認証は有効。KabuSを止めない）"
            )
        else:
            launch_result = full_login()
            if launch_result[0]:
                alert("kabuステーションを完全自動ログインで復帰しました",
                      launch_result[1], level=LEVEL_INFO)
            else:
                logger.warning(
                    f"kabuステーションの完全自動ログイン復帰に失敗しました: {launch_result[1]}"
                )
            # down_message() は auto_launch=True のときだけ launch_result の内容
            # （成功文言 / 「自動起動できませんでした: …」）を使う分岐を持つ。
            # この経路を使った回はそれに乗せるため、以降の判定では auto_launch=True 扱いにする。
            auto_launch = True

    notice = notifier.check(
        alive=alive, now=monotonic, notify_window=in_window,
        auto_launch=auto_launch, launch_result=launch_result,
    )
    if notice:
        alert(*notice)
