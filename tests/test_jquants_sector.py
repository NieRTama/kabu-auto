"""J-Quants の17業種でセクター・会社名を取る（2026-11-18）。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data import market_data as md

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import migrate_sector_s17 as mig  # noqa: E402

ROW = {"Code": "86970", "CoName": "日本取引所グループ", "S17Nm": "金融（除く銀行）"}


def _resp(rows):
    r = MagicMock()
    r.json.return_value = {"data": rows}
    return r


def test_sector_is_s17_and_never_falls_back_to_yfinance(monkeypatch):
    monkeypatch.setenv("JQUANTS_API_KEY", "dummy")
    with patch.object(md.requests, "get", return_value=_resp([ROW])) as get, \
         patch.object(md.yf, "Ticker") as yft:
        assert md.lookup_sector("8697") == "金融（除く銀行）"
        assert get.call_args.kwargs["headers"] == {"x-api-key": "dummy"}
    with patch.object(md.requests, "get", side_effect=OSError("down")), \
         patch.object(md.yf, "Ticker") as yft:
        assert md.lookup_sector("8697") == ""
        yft.assert_not_called()  # 英語の分類を混ぜない


def test_company_name_prefers_jquants_then_yfinance(monkeypatch):
    monkeypatch.setenv("JQUANTS_API_KEY", "dummy")
    with patch.object(md.requests, "get", return_value=_resp([ROW])):
        assert md.lookup_company_name("8697") == "日本取引所グループ"
    yft = MagicMock()
    yft.return_value.info = {"longName": "Japan Exchange Group"}
    with patch.object(md.requests, "get", return_value=_resp([])), patch.object(md.yf, "Ticker", yft):
        assert md.lookup_company_name("8697") == "Japan Exchange Group"


def test_no_key_means_no_request(monkeypatch):
    monkeypatch.delenv("JQUANTS_API_KEY", raising=False)
    with patch.object(md.requests, "get") as get:
        assert md.jquants_master("8697") == {}
        get.assert_not_called()


def test_migration_plan_covers_all_lists_and_empty_position_sector():
    wl = {"active": "A", "lists": {
        "A": [{"code": "8306", "name": "x", "sector": "Financial Services"}],
        "B": [{"code": "8306", "name": "x", "sector": "Financial Services"},
              {"code": "9999", "name": "y", "sector": "Industrials"}],  # 照会できず→変更しない
    }}
    s17 = {"8306": "銀行", "9432": "情報通信・サービスその他"}
    wl_c, pos_c = mig.plan(wl, [("9432", None), ("8306", "銀行")], s17)
    assert [(c[0], c[1]) for c in wl_c] == [("A", "8306"), ("B", "8306")]
    assert pos_c == [("positions", "9432", None, "情報通信・サービスその他")]
    mig.apply_watchlists(wl, s17)
    assert wl["lists"]["B"][1]["sector"] == "Industrials"
    assert all(e["sector"] == "銀行" for es in wl["lists"].values() for e in es if e["code"] == "8306")
