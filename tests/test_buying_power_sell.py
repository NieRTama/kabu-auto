"""売り代金の反映遅れの補正（総資産用）のテスト。

2026-11-06 実例: 15:15 に売った 9434 の代金 46,860円が 17:30 の口座余力に入っておらず、
総資産が 470,760円ではなく 423,900円（前日比 -71,760円）と記録された。
"""
import pytest

from src.core import buying_power as bp


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(bp, "_last_raw", None)
    monkeypatch.setattr(bp, "_unreflected", [])
    monkeypatch.setattr(bp, "_unreflected_sells", [])


def test_sale_counts_until_wallet_changes_and_order_sizing_ignores_it():
    assert bp.adjust(94_260.0) == 94_260.0
    bp.note_sell_fill(46_860.0)
    assert bp.adjust(94_260.0) == 94_260.0, "発注の余力には足さない"
    assert bp.unreflected_sell_amount() == 46_860.0
    bp.adjust(141_120.0)
    assert bp.unreflected_sell_amount() == 0.0


def test_sale_memo_expires_next_day(monkeypatch):
    """余力へ先に入っていた場合でも、二重に足すのは当日中だけ"""
    from datetime import date
    bp.adjust(94_260.0)
    monkeypatch.setattr(bp.clock, "today", lambda: date(2026, 11, 6))
    bp.note_sell_fill(46_860.0)
    monkeypatch.setattr(bp.clock, "today", lambda: date(2026, 11, 9))
    assert bp.unreflected_sell_amount() == 0.0
