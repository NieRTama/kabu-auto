"""標準logging（APScheduler 等）の出力を loguru へ取り込むことの回帰テスト。

背景: APScheduler はジョブのスキップ・ジョブ内の例外を**標準 logging** で出す。
      橋渡しが無いと stderr（コンソール窓）にしか出ず、ログファイルにも
      error_rate（health_check のエラー率監視）にも入らない。つまり
      **その瞬間コンソール窓を見ている人にしか気づけない**。

      2026-09-06 に `Execution of job "RemoteControl.poll_once" skipped:
      maximum number of running instances reached (1)` がコンソールにだけ出ており、
      ログファイルには1行も無い状態で発見した。ジョブが例外で死に続けても
      同じく気づけない構造だったため塞ぐ。

      アラート用の件数へはレベル別に載せる。routine に出る WARNING（遅延・スキップ）は
      誤った🔴の発生源になるので数えず、異常時にしか出ない ERROR/CRITICAL だけ数える。
"""
import logging
import time
from unittest.mock import patch

import pytest
from loguru import logger as loguru_logger

import src.core.logger as log_setup
from src.core import error_rate

SKIP_MESSAGE = (
    'Execution of job "RemoteControl.poll_once (trigger: interval[0:00:30])" '
    "skipped: maximum number of running instances reached (1)"
)


@pytest.fixture
def initialized_logger(tmp_path):
    """本番と同じ経路で logger.setup() を通し、後始末までする。

    setup() は loguru だけでなく**標準logging側にも**ハンドラとレベルを設定する。
    戻さないと後続のテストへ漏れ、単独では通るのにこのモジュールの後だと落ちる、
    という順序依存を作るため、標準logging 側も元に戻す。
    """
    conf = {"logging": {"level": "INFO", "file": str(tmp_path / "kabu_auto.log")}}
    bridged = [logging.getLogger(name) for name in log_setup._BRIDGED_LOGGERS]
    saved_levels = {lg.name: lg.level for lg in bridged}
    error_rate.reset()
    with patch.object(log_setup.cfg, "get_section", lambda s: conf.get(s, {})):
        log_setup.setup()
    try:
        yield tmp_path
    finally:
        loguru_logger.remove()
        error_rate.reset()
        for lg in bridged:
            for h in [h for h in lg.handlers
                      if isinstance(h, log_setup._InterceptHandler)]:
                lg.removeHandler(h)
            lg.setLevel(saved_levels[lg.name])


def _warning_log_text(tmp_path) -> str:
    loguru_logger.complete()
    files = list(tmp_path.glob("kabu_auto_warning_2*.log"))
    assert len(files) == 1, f"WARNINGログが1本できるはず: {files}"
    return files[0].read_text(encoding="utf-8")


def test_apscheduler_skip_warning_reaches_warning_log_file(initialized_logger):
    """ジョブのスキップ警告が WARNING ログファイルに残ること。"""
    logging.getLogger("apscheduler.scheduler").warning(SKIP_MESSAGE)

    assert "maximum number of running instances reached" in _warning_log_text(
        initialized_logger
    )


def test_apscheduler_warnings_do_not_feed_the_alert_counter(initialized_logger):
    """APScheduler の WARNING は error_rate に積まないこと（誤った🔴を防ぐ）。

    APScheduler は routine な遅延・スキップでも WARNING を出す。15秒間隔の
    reconcile_orders が詰まるだけで15分に60件出得て、警告の閾値50件
    （src/core/health.py）を単独で超え、実害が無いのに🔴が飛ぶ。
    """
    for _ in range(3):
        logging.getLogger("apscheduler.scheduler").warning(SKIP_MESSAGE)
    loguru_logger.complete()

    snap = error_rate.snapshot(now=time.monotonic(), window_seconds=900)
    assert snap["warning_count"] == 0, f"WARNINGとして数えない: {snap['latest_warning']}"


def test_job_exceptions_still_feed_the_alert_counter(initialized_logger):
    """ジョブ例外（ERROR）は数えること。

    routine に出る WARNING と違い、'Job "..." raised an exception' は異常だけで出る。
    morning_execution は try/except で包まれておらず（main.py:400）、
    `_execute_pending_signals` の `get_session()` も try の外にあるため、
    DBロック等で毎朝例外死すると**発注ゼロのまま**になる。ここを数えないと、
    ログを見に行かない限り誰も気づけない。
    """
    logging.getLogger("apscheduler.executors.default").error(
        'Job "morning_execution" raised an exception'
    )
    loguru_logger.complete()

    snap = error_rate.snapshot(now=time.monotonic(), window_seconds=900)
    assert snap["count"] == 1
    assert "morning_execution" in snap["latest"]


def test_app_own_logs_are_still_counted(initialized_logger):
    """アプリ自身のログはこれまで通り数えること（監視を殺さない）。"""
    loguru_logger.warning("建玉照合に失敗しました（次回再試行）")
    loguru_logger.error("発注に失敗しました")
    loguru_logger.complete()

    snap = error_rate.snapshot(now=time.monotonic(), window_seconds=900)
    assert snap["count"] == 1
    assert snap["warning_count"] == 1


def test_job_exception_keeps_level_and_traceback(initialized_logger):
    """ジョブ内の例外が ERROR のまま記録され、トレースバックも残ること。

    APScheduler は `logger.error(..., exc_info=True)` でジョブ例外を出す。
    レベルが落ちたり例外情報が消えると、朝の発注ジョブが死んでも原因が追えない。
    """
    try:
        raise ValueError("発注ジョブが壊れた")
    except ValueError:
        logging.getLogger("apscheduler.executors.default").error(
            'Job "morning_execution" raised an exception', exc_info=True
        )

    text = _warning_log_text(initialized_logger)
    assert 'Job "morning_execution" raised an exception' in text
    assert "ValueError" in text and "発注ジョブが壊れた" in text
    assert "| ERROR" in text, "ERROR のまま記録されること（WARNING に落ちない）"


def test_setup_twice_does_not_duplicate_records(initialized_logger, tmp_path):
    """setup() が複数回呼ばれても同じログが二重に出ないこと（ハンドラの重複登録防止）。"""
    conf = {"logging": {"level": "INFO", "file": str(tmp_path / "kabu_auto.log")}}
    with patch.object(log_setup.cfg, "get_section", lambda s: conf.get(s, {})):
        log_setup.setup()

    logging.getLogger("apscheduler.scheduler").warning("重複確認メッセージ")

    assert _warning_log_text(tmp_path).count("重複確認メッセージ") == 1


def test_library_errors_are_not_counted_by_error_rate(initialized_logger):
    """外部ライブラリの ERROR を警告カウントに積まないこと（誤った🔴を防ぐ）。

    yfinance は取得失敗を標準logging の ERROR で出す（yfinance/base.py の
    "Failed to get ticker ... reason: ..."）。一方 market_data 側は例外を捕捉して
    リトライ・スキップし、正常に処理を続けている。これを error_rate に積むと、
    data_update が約40銘柄を回す都合で Yahoo 側の一時障害だけで15分に10件
    （health_check の既定閾値）を容易に超え、実害が無いのに🔴が飛ぶ。

    監視に載せたいのは「ジョブそのものが動いていない」ことを示すものだけ。
    """
    logging.getLogger("yfinance").error(
        "Failed to get ticker '7203.T' reason: Expecting value"
    )
    logging.getLogger("websocket").error("connection closed")
    loguru_logger.complete()

    snap = error_rate.snapshot(now=time.monotonic(), window_seconds=900)
    assert snap["count"] == 0, f"ライブラリのERRORは数えない: {snap['latest']}"
    assert snap["warning_count"] == 0


def test_bridge_failure_does_not_break_the_caller(initialized_logger):
    """橋渡しの内部で例外が出ても、ログを呼んだ側へ伝播しないこと。

    logging.Handler.handle() は emit() の例外を握らない（握るのは各ハンドラの
    emit 内部の handleError）。ここで漏らすと「ログを1行出すだけ」のつもりが、
    APScheduler のジョブ実行中や HTTP 呼び出し中の処理を中断させる。
    橋渡しを入れる前は logging.lastResort が握り潰していた挙動なので、退行させない。
    """
    with patch.object(log_setup.logger, "opt", side_effect=RuntimeError("sink爆発")):
        # 例外が出れば、この呼び出し自体が失敗する
        logging.getLogger("apscheduler.scheduler").warning(SKIP_MESSAGE)


def test_record_points_at_the_library_not_logging_internals(initialized_logger):
    """発生元が logging の内部ではなく、実際に出したライブラリとして記録されること。

    フレームの遡り数を誤ると `logging:callHandlers:1706` と記録され、
    どこで何が起きたのか分からないログになる（調査を誤らせる）。
    """
    logging.getLogger("apscheduler.scheduler").warning(SKIP_MESSAGE)

    text = _warning_log_text(initialized_logger)
    assert "logging:callHandlers" not in text, f"発生元が logging 内部になっている: {text}"


def _always_fails(**kwargs):
    raise RuntimeError("500 Server Error")


def _send_order_like(secret):
    """実コード src/api/kabu_client.py と同じ形。

    例外が出る行に payload が現れるため、loguru の diagnose が有効だと
    その値（＝パスワードを含む dict）がトレースバックに展開される。
    """
    payload = {"OrderID": "X1", "Password": secret}
    return _always_fails(json=payload)


def test_exception_log_does_not_leak_local_variable_values(initialized_logger):
    """例外ログにローカル変数の値（＝APIパスワード）を書かないこと。

    loguru の既定は diagnose=True で、例外行に現れる変数の**値**を展開する。
    APScheduler のジョブ例外を橋渡しすると、発注処理の payload
    （`{"OrderID": ..., "Password": ...}`）が 15日保持のログへ平文で残る。
    Knowledge.md 12章（機密情報・実際に事故った）に直撃するため退行させない。
    """
    secret = "SUPER_SECRET_PASSWORD_12345"
    try:
        _send_order_like(secret)
    except RuntimeError:
        logging.getLogger("apscheduler.executors.default").error(
            'Job "morning_execution" raised an exception', exc_info=True
        )

    text = _warning_log_text(initialized_logger)
    assert "Job \"morning_execution\" raised an exception" in text
    assert "RuntimeError" in text, "例外の型と発生箇所は残すこと（調査に必要）"
    assert secret not in text, "ローカル変数の値をログに展開してはいけない"
