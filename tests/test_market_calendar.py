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
