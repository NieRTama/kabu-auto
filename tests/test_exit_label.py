"""退出通知の文面の出し分けのテスト（表示のみ。売買ルールは変えない）。

2026-10-23: 9434 がブレークイーブン（建値ストップ）で 242.5円→242円 で売られ -60円だったのに
「利益確定（トレーリングストップ）実行」と通知された。ブレークイーブンとトレーリングは
同じ reason=trailing_stop で判定されるため、売値と取得単価で出し分ける。
"""
from src.services.trading import _exit_label


def test_breakeven_exit_at_or_below_cost_is_not_called_profit():
    assert _exit_label("trailing_stop", 242.0, 242.5) == "建値で撤退（ブレークイーブン）"
    assert _exit_label("trailing_stop", 242.5, 242.5) == "建値で撤退（ブレークイーブン）"


def test_trailing_exit_above_cost_is_profit_taking():
    assert _exit_label("trailing_stop", 260.0, 242.5) == "利益確定（トレーリングストップ）"


def test_other_reasons_unchanged():
    assert _exit_label("stop_loss", 200.0, 242.5) == "損切り"
    assert _exit_label("time_stop", 245.0, 242.5) == "保有期限による入れ替え"
    assert _exit_label("unknown", 245.0, 242.5) == "損切り"


def test_unknown_cost_falls_back_to_reason_label():
    assert _exit_label("trailing_stop", 260.0, None) == "利益確定（トレーリングストップ）"
