"""
config の password 警告（C-1）と clock の JST naive 挙動（C-2）のテスト
"""
from datetime import datetime

import src.core.clock as clock
import src.core.config as cfg


class TestClock:
    def test_now_is_naive(self):
        """clock.now() は tzinfo を持たない naive datetime を返す"""
        assert clock.now().tzinfo is None

    def test_now_close_to_jst_wall_clock(self):
        """clock.now() は実際のJST壁時計（zoneinfo基準）とほぼ一致する

        システムのローカル時刻と比較すると、CI(UTC)とJST環境で結果が変わってしまうため、
        zoneinfo で直接JSTを求めた値と比較する。
        """
        from zoneinfo import ZoneInfo

        jst_now = datetime.now(ZoneInfo("Asia/Tokyo")).replace(tzinfo=None)
        delta = abs((clock.now() - jst_now).total_seconds())
        assert delta < 5

    def test_today_is_date(self):
        assert clock.today() == clock.now().date()


class TestConfigPasswordWarning:
    def test_warns_when_yaml_has_password(self, tmp_path, caplog):
        p = tmp_path / "config.yaml"
        p.write_text('kabu_station:\n  password: "leaked"\n', encoding="utf-8")
        import loguru
        # loguru はstdの caplog に流れないため、簡易にハンドラを差し込む
        messages = []
        handler_id = loguru.logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            cfg.load(str(p))
        finally:
            loguru.logger.remove(handler_id)
        assert any("password" in m for m in messages)

    def test_no_warning_without_password(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text('kabu_station:\n  base_url: "x"\n', encoding="utf-8")
        import loguru
        messages = []
        handler_id = loguru.logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            cfg.load(str(p))
        finally:
            loguru.logger.remove(handler_id)
        assert not any("password" in m for m in messages)
