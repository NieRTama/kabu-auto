"""
通知の重要度（alert level）とエントリー通知のテスト

## 背景

1. 退出（損切り・利確）は通知されるのに、エントリー（買い）は通知されず
   ログにしか残らなかった。実弾が動いたのに知らされないのは非対称で、
   「口座で今いくら使われたか」が日次レポートまで分からなかった。

2. 通知が全て同じ見た目だったため、スマホの一覧で緊急度を判断できなかった。
   アラート疲れ対策の定石に沿い、行動が変わる4段階の記号を付ける。
   段階を増やすと結局読み分けなくなるため4つに絞る。
"""
from unittest.mock import MagicMock, patch

import pytest

import src.core.alerts as mod


def _capture(level=None, title="テスト", message="本文"):
    """alert() が実際に送る本文を取り出す。"""
    provider = MagicMock()
    provider.name = "discord"
    with patch.object(mod, "build_providers", return_value=[provider]):
        if level is None:
            mod.alert(title, message)
        else:
            mod.alert(title, message, level=level)
    return provider.send.call_args[0][0]


class TestSeverityMarks:
    def test_critical_is_default(self):
        """従来の呼び出し（引数なし）は critical 扱い＝安全側に倒す"""
        assert _capture().startswith("🔴")

    def test_each_level_has_distinct_mark(self):
        marks = {
            mod.LEVEL_CRITICAL: "🔴",
            mod.LEVEL_WARNING: "🟡",
            mod.LEVEL_INFO: "🟢",
            mod.LEVEL_ROUTINE: "⚪",
        }
        for level, mark in marks.items():
            assert _capture(level).startswith(mark), f"{level} の記号が違う"
        assert len(set(marks.values())) == 4, "記号が重複していたら見分けられない"

    def test_unknown_level_falls_back_to_critical(self):
        """未知の値でも通知は落とさず、安全側（要対応）として扱う"""
        assert _capture("bogus").startswith("🔴")

    def test_title_and_message_are_preserved(self):
        text = _capture(mod.LEVEL_INFO, title="買い発注", message="9432 100株")
        assert "買い発注" in text and "9432 100株" in text
        assert "【kabu-auto】" in text

    def test_level_does_not_change_delivery(self):
        """重要度は見た目だけを変え、送信先や送信可否は変えない"""
        provider = MagicMock()
        provider.name = "discord"
        with patch.object(mod, "build_providers", return_value=[provider]):
            mod.alert("t", "m", level=mod.LEVEL_ROUTINE)
        provider.send.assert_called_once()


class TestRoutineAndInfoAreDowngraded:
    """対応不要の通知が critical のままだと、本物の緊急通知が埋もれる"""

    def test_heartbeat_is_routine(self):
        import src.core.heartbeat as hb
        with patch.object(hb, "alert") as alert_mock:
            hb.send("live", {"can_place_order": True, "block_reason": "",
                             "unresolved_orders": 0}, 0)
        assert alert_mock.call_args.kwargs.get("level") == mod.LEVEL_ROUTINE

    def test_exit_execution_is_info(self):
        """損切り・利確の実行報告は結果の共有であり、対応は不要"""
        import inspect
        import src.services.trading as svc
        src = inspect.getsource(svc.TradingServices.stop_loss_check)
        assert "level=LEVEL_INFO" in src


class TestEntryIsNotified:
    """エントリー（買い）と手仕舞い（売り）が通知されること"""

    def _source(self):
        import inspect
        import src.services.trading as svc
        return inspect.getsource(svc.TradingServices._execute_pending_signals)

    def test_buy_sends_alert(self):
        src = self._source()
        assert "買い発注" in src
        assert 'alert(f"{label}買い発注"' in src, "買い発注時に通知していない"

    def test_sell_sends_alert(self):
        assert 'alert(f"{label}売り発注"' in self._source(), "売り発注時に通知していない"

    def test_entry_alerts_are_info_level(self):
        """約定報告は対応不要なので INFO（critical にすると緊急通知が埋もれる）"""
        src = self._source()
        # 買い・売りの2箇所とも INFO で送っている
        assert src.count("level=LEVEL_INFO") >= 2

    def test_buy_alert_includes_amount(self):
        """いくら使われたかが分かること（通知の目的そのもの）"""
        assert "約定額" in self._source()
