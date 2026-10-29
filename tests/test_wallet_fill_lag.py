"""買い約定が口座余力へ遅れて反映される間の二重計上のテスト（2026-10-29 実例）。

10/29 09:05 に 9434 を 92,480円で買い約定したが、余力（/wallet/cash）は約定前の 227,860円 のまま
しばらく減らず、ダッシュボードの総資産が約9万円過大（約58万円）に出た。10/28 も同様に +17.8万円。
"""
import pytest

from src.core import buying_power, total_assets
from src.data.database import Position, get_session


class FakeClient:
    def __init__(self, cash):
        self.cash = cash

    def get_wallet(self):
        return {"StockAccountWallet": self.cash}


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    monkeypatch.setattr(buying_power, "_last_raw", None)
    monkeypatch.setattr(buying_power, "_unreflected", [])
    monkeypatch.setattr(total_assets, "_wallet_cache", None)
    monkeypatch.setattr(total_assets, "HISTORY_PATH", tmp_path / "hist.json")


def _total(client):
    total_assets._wallet_cache = None
    return total_assets.compute("live", client=client, price_fn=lambda syms: {"9434": 231.2},
                                paper_base=0, basis=0, record=False)["total"]


def test_total_not_double_counted_until_wallet_reflects_buy(isolated_db):
    client = FakeClient(227_860)
    assert _total(client) == 227_860  # 約定前（保有なし）

    with get_session() as s:
        s.add(Position(symbol="9434", quantity=400, avg_cost=231.2))
        s.commit()
    buying_power.note_buy_fill(92_480)

    # 余力が約定前の値のまま → 代金を差し引く（修正前は 227,860 + 92,480 で過大）
    assert _total(client) == 227_860 - 92_480 + 92_480

    # 余力に反映された → 差し引きをやめる
    client.cash = 135_380
    assert _total(client) == 135_380 + 92_480


def test_adjust_used_by_trading_budget():
    assert buying_power.adjust(227_860) == 227_860
    buying_power.note_buy_fill(92_480)
    assert buying_power.adjust(227_860) == 135_380
    assert buying_power.adjust(135_380) == 135_380
    assert buying_power._unreflected == []


def test_fill_before_any_wallet_read_is_not_adjusted():
    buying_power.note_buy_fill(92_480)
    assert buying_power.adjust(227_860) == 227_860
