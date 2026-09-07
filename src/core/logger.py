import logging
import os
import sys
from pathlib import Path

from loguru import logger

from src.core import config as cfg

_WARN_LEVEL = "WARNING"


class _InterceptHandler(logging.Handler):
    """標準 logging のレコードを loguru へ流すハンドラ。

    APScheduler は **標準 logging** で出力する。橋渡しが無いと
    stderr（＝コンソール窓）にしか出ず、
      - `log/kabu_auto_warning_*.log` に残らない（後から追えない・ローテーションもされない）
      - `error_rate` の件数（health_check のエラー率監視）にも入らない
    という状態になり、**その瞬間コンソール窓を見ている人にしか気づけない**。

    アラート用の件数（error_rate）へは**レベル別**に載せる（`_countable_for_alerts`）。
    routine に出る WARNING（遅延・スキップ）は誤った🔴の発生源になるので数えず、
    異常時にしか出ない ERROR/CRITICAL（`Job "..." raised an exception`）だけを数える。

    実際 2026-09-06 に `Execution of job "RemoteControl.poll_once" skipped:
    maximum number of running instances reached (1)` がコンソールにだけ出ており、
    ログには1行も残っていなかった。同じ経路で
    `Job "morning_execution" raised an exception` も出るため、
    朝の発注ジョブが例外で死に続けても気づけない構造だった。
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # logging 内部のフレームを飛ばし、実際の呼び出し元の位置を出す。
            # depth は「この emit フレームから何段上か」で数える。誤ると発生元が
            # `logging:callHandlers` と記録され、どこで何が起きたか分からなくなる。
            frame, depth = sys._getframe(0), 0
            while frame is not None and (
                depth == 0 or frame.f_code.co_filename == logging.__file__
            ):
                frame = frame.f_back
                depth += 1

            # `stdlib_source` を付けて「アプリ自身のログではない」と分かるようにする。
            # error_rate のシンクはこれを見て除外する（下の setup() を参照）。
            logger.opt(depth=depth, exception=record.exc_info).bind(
                stdlib_source=record.name
            ).log(level, record.getMessage())
        except Exception:
            # logging.Handler.handle() は emit() の例外を握らない（握るのは各標準
            # ハンドラの emit 内部の handleError）。ここで漏らすと「ログを1行出すだけ」
            # のつもりが、APScheduler のジョブ実行中や HTTP 呼び出し中の処理を中断させる。
            # 橋渡しを入れる前は logging.lastResort が握り潰していた挙動なので退行させない。
            self.handleError(record)


# 橋渡しする対象を**明示列挙**する。root には付けない。
#
# root に付けると外部ライブラリの ERROR が error_rate（health_check のエラー率監視）へ
# 積まれ、実害が無いのに🔴が飛ぶ。実例: yfinance は取得失敗を標準logging の ERROR で出すが
# （yfinance/base.py "Failed to get ticker ... reason: ..."）、market_data 側は例外を捕捉して
# リトライ・スキップし正常に処理している。data_update は約40銘柄を回すため、Yahoo 側の
# 一時障害だけで15分に10件（既定閾値）を容易に超える。websocket-client の再接続 ERROR も同様。
#
# 監視に載せたいのは「ジョブそのものが動いていない」ことを示すものだけなので APScheduler に絞る。
# 子ロガー（apscheduler.scheduler / apscheduler.executors.*）は伝播でここに集まる。
#   - apscheduler.scheduler      … "Execution of job ... skipped: maximum number of running instances"
#   - apscheduler.executors.<n>  … 'Job "..." raised an exception'（exc_info 付き）
#
# 注意: uvicorn は対象にできない。uvicorn は "uvicorn"/"uvicorn.access" に propagate=False を
# 設定するため、どこに付けても伝播で捕まえられない（ダッシュボードの未捕捉例外は
# コンソール窓限定のまま）。捕まえたくなったら uvicorn 側のロガーへ直接付ける必要がある。
_BRIDGED_LOGGERS = ("apscheduler",)


def _countable_for_alerts(record) -> bool:
    """アラート用の件数（error_rate）に数えてよいログか。

    橋渡ししたライブラリのログのうち **WARNING だけを除外**する。
      - 除外する理由: APScheduler は routine な遅延・スキップでも WARNING を出す。
        15秒間隔の `reconcile_orders` が詰まるだけで15分に60件出得て、警告の閾値
        50件（`src/core/health.py`）を単独で超え、実害が無いのに🔴が飛ぶ。
      - ERROR/CRITICAL は残す理由: `Job "..." raised an exception` は異常時にしか出ない。
        `morning_execution` は try/except で包まれておらず（main.py）、
        `_execute_pending_signals` の `get_session()` も try の外にあるため、
        DBロック等で例外死すると**発注ゼロのまま**になる。これを数えないと、
        ログを見に行かない限り誰も気づけない（heartbeat は「稼働中」を返すだけ、
        liveness は別のAPI呼び出しで緑のままなので、どちらもこの故障を拾わない）。
    """
    if "stdlib_source" not in record["extra"]:
        return True                      # アプリ自身のログは従来どおり全部数える
    return record["level"].no > logger.level(_WARN_LEVEL).no


def bridge_stdlib_logging() -> None:
    """標準 logging（APScheduler）の出力を loguru 側へ合流させる。

    既存のハンドラは消さない（pytest のログ捕捉など、他が付けたものを壊さないため）。
    同じハンドラを二重登録しないよう、既に居れば何もしない。
    """
    for name in _BRIDGED_LOGGERS:
        lg = logging.getLogger(name)
        if not any(isinstance(h, _InterceptHandler) for h in lg.handlers):
            lg.addHandler(_InterceptHandler())
        if lg.level == logging.NOTSET or lg.level > logging.WARNING:
            lg.setLevel(logging.WARNING)


def disable_console_quick_edit() -> bool:
    """Windowsコンソールの「簡易編集モード」を切る。

    戻り値は「**コンソール由来の停止リスクが無い状態か**」。
    そもそもコンソールを持たない実行（サービス化・出力のリダイレクト・CI）や
    非Windowsは、止まりようが無いので True を返す。False を返すのは
    「コンソールはあるのに簡易編集を切れなかった」場合だけで、
    呼び出し側はそのときだけ警告する（毎回警告すると通知の価値が下がる）。

    簡易編集モードが有効なコンソールは、**窓の中をクリックして選択した瞬間に
    そこへ出力するプロセスを停止させる**。解除するまで戻らない。

    2026-09-07、これで取引システム全体が約3時間停止した。停止の連鎖はこうなる:

        stop_loss_check が logger.error()
          → loguru がハンドラのロックを取得
          → stderr(=コンソール)へ write ……選択中のため戻らない
          → **ロックを握ったまま**なので、以後ログを出す全スレッドが待機
          → APScheduler の本体ループも WebSocket も停止

    9:05の朝発注・損切り監視・認証の自動復帰がすべて動かず、実弾を持ったまま
    無防備になった。**ログを1行出そうとしただけで全機能が止まる**構造だった。

    ここでモードを切っても、右クリックメニューからのコピーは使える。
    コンソールを持たない実行（サービス化・出力のリダイレクト）や非Windowsでは
    何もせず False を返す（失敗しても起動を妨げない）。
    """
    if os.name != "nt":
        return True                 # 非Windowsはこの停止の仕方をしない
    try:
        import ctypes

        STD_INPUT_HANDLE = -10
        ENABLE_QUICK_EDIT_MODE = 0x0040
        # QuickEdit を落とすときは EXTENDED_FLAGS を必ず立てる必要がある
        # （立てないと SetConsoleMode がマウス関連ビットを無視する）。
        ENABLE_EXTENDED_FLAGS = 0x0080

        k32 = ctypes.windll.kernel32
        handle = k32.GetStdHandle(STD_INPUT_HANDLE)
        if handle == 0 or handle == -1:
            return True             # コンソールが無い＝止まりようが無い
        mode = ctypes.c_uint32()
        if not k32.GetConsoleMode(handle, ctypes.byref(mode)):
            return True             # コンソールが無い（リダイレクト等）
        if not mode.value & ENABLE_QUICK_EDIT_MODE:
            return True             # 既に無効
        new_mode = (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS
        return bool(k32.SetConsoleMode(handle, new_mode))
    except Exception:
        return False                # 失敗しても起動は続ける


def dated_log_path(log_file: str, tag: str = "") -> str:
    """設定されたログパスのファイル名末尾に日付（と任意のタグ）を入れたパターンを返す。

    例: dated_log_path("data/kabu_auto.log") -> "data/kabu_auto_{time:YYYY-MM-DD}.log"
        dated_log_path("data/kabu_auto.log", "warning") -> "data/kabu_auto_warning_{time:YYYY-MM-DD}.log"
    loguru は {time} を含むパスに対し、ファイル生成（＝日次ローテーション）のたびに
    その日の日付でファイル名を確定する。
    """
    p = Path(log_file)
    name_tag = f"_{tag}" if tag else ""
    return str(p.with_name(f"{p.stem}{name_tag}_{{time:YYYY-MM-DD}}{p.suffix}"))


def setup() -> None:
    conf = cfg.get_section("logging")
    level = conf.get("level", "INFO")
    log_file = conf.get("file", "data/kabu_auto.log")
    retention = conf.get("retention", "15 days")

    # ログ格納フォルダ（例: log/）が無ければ作成する
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger.remove()

    # diagnose=False は必須。loguru の既定(True)は例外行に現れる変数の**値**を
    # トレースバックへ展開するため、発注処理の payload
    # （`{"OrderID": ..., "Password": ...}`、src/api/kabu_client.py）が
    # そのままログへ平文で残る。ログは15日保持されるので実際の漏洩になる
    # （Knowledge.md 12章「機密情報の扱い」）。例外の型・発生箇所は残るので調査には困らない。
    _no_leak = {"diagnose": False}

    # コンソールの簡易編集モードを切る（クリック選択による停止を防ぐ。上の関数を参照）
    quick_edit_disabled = disable_console_quick_edit()

    # enqueue=True は**必須**。コンソールへの書き込みは詰まることがあり
    # （簡易編集モードでの選択・Ctrl+S・端末側の停止）、同期のままだと
    # loguru のハンドラロックを握ったまま止まって**アプリ全体が凍結**する
    # （2026-09-07 に約3時間停止。上の disable_console_quick_edit() 参照）。
    # enqueue にすると書き込みは専用スレッドへ渡され、詰まっても呼び出し側は進む。
    # 簡易編集モードを切っただけでは Ctrl+S 等の別経路が残るため、両方入れる。
    logger.add(sys.stderr, level=level, colorize=True, enqueue=True, **_no_leak,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    # ログは1日単位で区切り（毎日0時にローテーション）、ファイル名末尾に日付を付与する。
    # retention で古いログを自動削除する（既定: 15日周期）。
    # INFO以下（DEBUG/INFO）と WARNING以上（WARNING/ERROR/CRITICAL）を別ファイルに分ける。
    logger.add(
        dated_log_path(log_file),
        level=level,
        rotation="00:00",
        retention=retention,
        encoding="utf-8",
        **_no_leak,
        filter=lambda record: record["level"].no < logger.level(_WARN_LEVEL).no,
    )
    logger.add(
        dated_log_path(log_file, tag="warning"),
        level=_WARN_LEVEL,
        rotation="00:00",
        retention=retention,
        encoding="utf-8",
        **_no_leak,
    )

    # WARNING以上を時間窓で数えるシンク（health_check がこの件数を見て異常を通知する）。
    # 個々のバグは予測できないが「壊れたらエラーが増える」のは普遍的に成り立つため、
    # 未知の障害に対する防御になる（2026-08 の認証切れ・KeyError はどちらも
    # 既存の異常検知をすり抜けたが、エラー数では捉えられていた）。
    #
    # ERROR だけでなく WARNING も数えるのは、リトライ前提の失敗が WARNING で
    # 記録されるため（2026-09-02: 401が499件出たが ERROR は28件しかなく、
    # 15分あたり3.5件で閾値10に届かず検知できなかった）。閾値はレベル別に持つ。
    from src.core import error_rate
    logger.add(error_rate.make_sink(), level=_WARN_LEVEL, filter=_countable_for_alerts)

    # APScheduler が標準 logging へ出すジョブのスキップ・例外も、上のシンク群
    # （ファイル・警告カウント）に載せる。載せないとコンソール窓にしか出ない。
    bridge_stdlib_logging()

    logger.info("Logger initialized")
    if not quick_edit_disabled:
        # 切れなかった場合は、コンソールをクリックすると止まる危険が残る。
        # 黙って進むと「なぜか固まる」の再来になるので、起動時に一度だけ知らせる。
        logger.warning(
            "コンソールの簡易編集モードを無効化できませんでした。"
            "窓の中をクリックして選択すると出力が止まります（Escapeで解除）"
        )
