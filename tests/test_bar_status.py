"""確定足判定（src/data/bar_status.py）のテスト

背景: is_business_day() は営業日か否かを返すだけで、その足が確定済みかは
分からない。場中の未確定足・引け後の配信待ち・休場を区別する必要がある。
"""
from datetime import date, datetime

from src.data import bar_status


class TestAsOfSession:
    def test_after_close_returns_same_day(self):
        """引け後（15:00 + 猶予20分 = 15:20 以降）はその営業日が確定セッション"""
        # 2026-09-10 は木曜（営業日）
        now = datetime(2026, 9, 10, 16, 0)
        assert bar_status.as_of_session(now) == date(2026, 9, 10)

    def test_during_session_returns_previous_business_day(self):
        """場中は当日足がまだ確定していないので前営業日が基準"""
        now = datetime(2026, 9, 10, 11, 0)
        assert bar_status.as_of_session(now) == date(2026, 9, 9)

    def test_within_grace_after_close_returns_previous(self):
        """引け直後の猶予時間内は、配信待ちとみなして前営業日を基準にする"""
        now = datetime(2026, 9, 10, 15, 10)  # 15:00引け + 猶予20分未満
        assert bar_status.as_of_session(now) == date(2026, 9, 9)

    def test_on_holiday_returns_last_business_day(self):
        """休場日は直近の営業日が基準"""
        # 2026-09-12 は土曜
        now = datetime(2026, 9, 12, 16, 0)
        assert bar_status.as_of_session(now) == date(2026, 9, 11)

    def test_monday_morning_returns_friday(self):
        """月曜の場中は前営業日＝金曜"""
        # 2026-09-14 は月曜
        now = datetime(2026, 9, 14, 10, 0)
        assert bar_status.as_of_session(now) == date(2026, 9, 11)


class TestClassify:
    def test_matching_session_is_fresh(self):
        now = datetime(2026, 9, 10, 16, 0)
        st = bar_status.classify("7203", date(2026, 9, 10), now)
        assert st.state == "fresh"
        assert st.is_final is True
        assert st.symbol == "7203"
        assert st.last_bar_session == date(2026, 9, 10)
        assert st.observed_at == now

    def test_older_session_is_stale(self):
        now = datetime(2026, 9, 10, 16, 0)
        st = bar_status.classify("7203", date(2026, 9, 9), now)
        assert st.state == "stale"
        assert st.is_final is True

    def test_no_bar_is_missing(self):
        now = datetime(2026, 9, 10, 16, 0)
        st = bar_status.classify("7203", None, now)
        assert st.state == "missing"
        assert st.is_final is False

    def test_future_session_is_provisional(self):
        """基準セッションより新しい足は、まだ確定していない場中の足とみなす"""
        now = datetime(2026, 9, 10, 11, 0)  # 基準は 9/9
        st = bar_status.classify("7203", date(2026, 9, 10), now)
        assert st.state == "provisional"
        assert st.is_final is False
