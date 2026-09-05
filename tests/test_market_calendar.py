"""
東証の営業日判定（market_calendar）のテスト

is_market_open() は当初「土日」しか除外しておらず、祝日を「場中」と誤判定していた。
祝日には板取得が全て失敗するため、エラー率監視とサイレント故障検知が
誤発報する（2026-08 に昼休み・夜間で同種の誤検知を2度経験している）。
"""
from datetime import date, datetime

import pytest
import pytz

from src.core import market_calendar as mc
from src.core.scheduler import TradingScheduler

TZ = pytz.timezone("Asia/Tokyo")


class TestWeekend:
    def test_saturday_is_holiday(self):
        assert mc.is_holiday(date(2026, 8, 29)) is True   # 土

    def test_sunday_is_holiday(self):
        assert mc.is_holiday(date(2026, 8, 30)) is True   # 日

    def test_weekday_is_business_day(self):
        assert mc.is_business_day(date(2026, 9, 1)) is True  # 火


class TestNationalHolidays:
    """国民の祝日（jpholiday が判定）"""

    @pytest.mark.parametrize("d,name", [
        (date(2026, 9, 21), "敬老の日"),
        (date(2026, 9, 22), "国民の休日"),
        (date(2026, 9, 23), "秋分の日"),
        (date(2026, 11, 3), "文化の日"),
        (date(2026, 11, 23), "勤労感謝の日"),
    ])
    def test_holiday_is_detected(self, d, name):
        assert mc.is_holiday(d) is True, f"{d} ({name}) が休場と判定されない"
        assert mc.is_business_day(d) is False

    def test_holiday_name_is_returned(self):
        assert mc.holiday_name(date(2026, 9, 21)) == "敬老の日"

    def test_substitute_holiday(self):
        """振替休日も休場（祝日が日曜と重なった場合の翌月曜）"""
        # 2026-05-03(憲法記念日)は日曜 → 05-06が振替休日
        assert mc.is_holiday(date(2026, 5, 6)) is True


class TestYearEndNewYear:
    """年末年始（12/31〜1/3）は祝日ではないため個別に判定する必要がある"""

    @pytest.mark.parametrize("d", [
        date(2026, 12, 31),
        date(2027, 1, 1),   # 元日（祝日でもある）
        date(2027, 1, 2),   # 祝日ではない
    ])
    def test_year_end_is_holiday(self, d):
        assert mc.is_holiday(d) is True, f"{d} が休場と判定されない"

    def test_year_end_name(self):
        assert mc.holiday_name(date(2026, 12, 31)) == "年末年始休場"

    def test_january_fourth_is_business_day_if_weekday(self):
        """1/4 は営業日（曜日次第）。2027-01-04 は月曜"""
        assert mc.is_holiday(date(2027, 1, 4)) is False


class TestBusinessDayName:
    def test_returns_none_for_business_day(self):
        assert mc.holiday_name(date(2026, 9, 1)) is None

    def test_weekend_name(self):
        assert mc.holiday_name(date(2026, 8, 29)) == "土日"


class TestMarketOpenIntegration:
    """is_market_open が祝日を除外すること（本丸の回帰防止）"""

    def _at(self, y, m, d, hh, mm):
        return TZ.localize(datetime(y, m, d, hh, mm))

    def test_closed_on_holiday_during_trading_hours(self):
        """祝日の場中時間帯でも「開いていない」と判定する"""
        assert TradingScheduler.is_market_open(self._at(2026, 9, 21, 10, 0)) is False

    def test_open_on_business_day(self):
        assert TradingScheduler.is_market_open(self._at(2026, 9, 1, 10, 0)) is True

    def test_closed_during_lunch_on_business_day(self):
        assert TradingScheduler.is_market_open(self._at(2026, 9, 1, 12, 0)) is False

    def test_closed_on_year_end(self):
        assert TradingScheduler.is_market_open(self._at(2026, 12, 31, 10, 0)) is False

    def test_near_close_false_on_holiday(self):
        """祝日は引け間際判定も False（新規BUY抑止の対象外＝そもそも動かない）"""
        assert TradingScheduler.is_near_close(30, now=self._at(2026, 9, 21, 15, 20)) is False


class TestJobGuards:
    """休場日に動くべきでないジョブがガードされていること（ソース検証）"""

    def _source(self, name):
        import inspect
        import src.services.trading as svc
        return inspect.getsource(getattr(svc.TradingServices, name))

    def test_signal_scan_skips_on_holiday(self):
        """休場日は前営業日と同じ終値になり、無意味なシグナルでDBを汚す"""
        assert "market_calendar.is_holiday" in self._source("signal_scan")

    def test_heartbeat_skips_on_holiday(self):
        assert "market_calendar.is_holiday" in self._source("heartbeat")

    def test_daily_report_skips_on_holiday(self):
        assert "market_calendar.is_holiday" in self._source("post_daily_summary_to_discord")

    def test_data_update_still_runs_on_holiday(self):
        """データ更新は害がなく、翌営業日を軽くするので止めない（意図の固定）"""
        assert "market_calendar.is_holiday" not in self._source("data_update")

    def test_x_report_skips_on_holiday(self):
        """対になるDiscord側にガードがあるのにX側だけ漏れていた（2026-09-05 発見）"""
        assert "market_calendar.is_holiday" in self._source("post_daily_summary_to_x")


class TestTokenRefreshHolidayGuard:
    """休場日にトークン更新で🔴を出さないこと（2026-09-05 の誤発報の回帰防止）。

    土曜8:30に「kabuステーションの再ログインが必要です」が届いた。休場日は取引が
    無く認証が切れていても異常ではないのに、毎週末🔴が飛ぶ状態だった。
    **対応不要な🔴が定期的に届くと、本物の🔴を無視する癖がつく**のが実害。
    """

    def _main_source(self):
        with open("main.py", encoding="utf-8") as f:
            return f.read()

    def test_token_refresh_checks_holiday(self):
        import re
        src = self._main_source()
        fn = re.search(r"def token_refresh\(\):.*?(?=\n    def )", src, re.S)
        assert fn, "token_refresh が見つからない"
        assert "market_calendar.is_holiday" in fn.group(0)

    def test_token_refresh_returns_before_alert(self):
        """休場判定が alert より前にあること（順序が逆だと通知が出てしまう）"""
        import re
        fn = re.search(r"def token_refresh\(\):.*?(?=\n    def )",
                       self._main_source(), re.S).group(0)
        assert fn.index("is_holiday") < fn.index("alert("), "休場判定は通知より前に置く"


class TestStartupSkipsBrokerWaitOnHoliday:
    """休場日は起動時のログイン待ちをしないこと（2026-09-05 の実害の回帰防止）。

    土曜に再起動したところ、kabuステーション未ログインのため無制限待機に入り、
    スケジューラが起動しないまま🔴「ログイン待ち」が届いた。休場日は取引が無く
    接続を必須にする理由が無いうえ、待ち続けると db_backup などの日次ジョブまで
    止まってしまう。
    """

    def _startup_source(self):
        import re
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        # 待機処理からfail-closed判定までを切り出す
        m = re.search(r"wait_minutes = .*?order_mgr\.sync_on_startup\(\)", src, re.S)
        assert m, "起動シーケンスを特定できない"
        return m.group(0)

    def test_wait_is_skipped_on_holiday(self):
        src = self._startup_source()
        assert "market_calendar.is_holiday" in src
        # 設定キー名 wait_for_broker_minutes ではなく実際の呼び出しで位置を見る
        call = "broker_wait.wait_for_broker("
        assert call in src
        # 休場判定が待機呼び出しより前にあること（順序が逆なら待ってしまう）
        assert src.index("market_calendar.is_holiday") < src.index(call)

    def test_live_mode_does_not_abort_on_holiday(self):
        """休場日は fail-closed の中断を掛けない。

        掛けたままだと接続できない休場日に起動そのものができず、
        「待つ」を止めた意味が無くなる（起動中断に化けるだけ）。
        """
        src = self._startup_source()
        assert "places_real_orders(mode) and not is_closed_today" in src, \
            "休場日は実発注モードでも起動を中断しない条件が必要"

    def test_auth_stays_expired_when_not_connected(self):
        """接続できないまま起動した場合は認証切れのままにする（発注ゲートを開けない）"""
        src = self._startup_source()
        assert "broker_auth.mark_expired" in src

    def test_preflight_skips_api_check_on_holiday(self):
        """プリフライトのAPI疎通も休場日は見ない。

        待機とfail-closedを直しても、プリフライトが疎通失敗で中断するため
        休場日は結局起動できなかった（同じ欠陥が3箇所目）。
        """
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        assert "skip_api=is_closed_today" in src

    def test_preflight_still_checks_config_on_holiday(self):
        """省略するのは疎通だけ。ポート競合・モード不整合は休場日でも止める"""
        import inspect
        from src.core import preflight
        src = inspect.getsource(preflight.run_preflight)
        body = src[src.index("checks: list"):]
        for name in ("_check_port", "_check_endpoint_mode_consistency"):
            assert name in body, f"{name} が skip_api で一緒に飛んでいる"
        # skip_api の分岐が API疎通だけを包んでいること
        assert body.index("if skip_api") < body.index("_check_api_connectivity")
        assert body.index("_check_api_connectivity") < body.index("_check_port")


class TestScheduledJobsHaveDayRestriction:
    """cronジョブに曜日指定があること（漏れを構造的に検出する）。

    token_refresh だけ day_of_week が無く、周囲のジョブが全て mon-fri だったため
    レビューでも見落とされた。**「毎日動いてよい」ものを明示的に列挙**し、
    それ以外に曜日指定が無ければ落とす。新しいジョブを追加したときも自動で掛かる。
    """

    # 毎日動かす意図があるジョブと、その理由
    DAILY_BY_DESIGN = {
        # 取得は害がなく、翌営業日の処理を軽くする（設計書 1.24.4 で意図を固定）
        "data_update",
        # バックアップは休場日も取る（データ保全は取引の有無と無関係）
        "db_backup",
    }

    def _cron_jobs(self):
        import re
        with open("src/core/scheduler.py", encoding="utf-8") as f:
            src = f.read()
        jobs = {}
        for m in re.finditer(r'cb\["(\w+)"\],\s*"cron",(.*?)id="(\w+)"', src, re.S):
            body, jid = m.group(2), m.group(3)
            jobs[jid] = bool(re.search(r'day_of_week="[^"]+"', body))
        return jobs

    def test_all_cron_jobs_restrict_day_of_week(self):
        jobs = self._cron_jobs()
        assert jobs, "cronジョブを1つも検出できていない（正規表現が壊れている）"
        missing = [j for j, has_dow in jobs.items()
                   if not has_dow and j not in self.DAILY_BY_DESIGN]
        assert not missing, (
            "曜日指定が無いcronジョブがあります（休場日に動いてよいなら "
            f"DAILY_BY_DESIGN に理由付きで追加すること）: {missing}"
        )

    def test_token_refresh_is_weekday_only(self):
        assert self._cron_jobs().get("token_refresh") is True
