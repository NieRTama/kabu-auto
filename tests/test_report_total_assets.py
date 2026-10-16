"""Discord 日次・週次レポートに総資産の行が入ることのテスト（TradingServices 経由）。"""
import json

import pytest

import src.services.trading as trading
from src.core import discord_report, total_assets
from src.risk.manager import RiskManager


@pytest.fixture
def services(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr(total_assets, "HISTORY_PATH", tmp_path / "hist.json")
    monkeypatch.setattr(total_assets, "_wallet_cache", None)
    monkeypatch.setattr(trading.market_calendar, "is_holiday", lambda d: False)
    svc = trading.TradingServices(client=None, risk=RiskManager(), order_mgr=None)
    svc.trading_conf = {"mode": "paper", "paper_initial_capital": 500_000}
    return svc


def test_daily_report_gets_assets_line_and_records_history(services, monkeypatch):
    sent = {}
    monkeypatch.setattr(discord_report, "post_daily_report",
                        lambda mode, basis, assets_line=None: sent.setdefault("line", assets_line))
    services.post_daily_summary_to_discord()
    assert sent["line"].startswith("総資産: 500,000円")  # paper・保有なし
    assert list(json.loads(total_assets.HISTORY_PATH.read_text()).values()) == [500000.0]


def test_weekly_report_gets_assets_line(services, monkeypatch):
    sent = {}
    monkeypatch.setattr(discord_report, "post_weekly_report",
                        lambda mode, basis, assets_line=None: sent.setdefault("line", assets_line))
    services.post_weekly_summary_to_discord()
    assert sent["line"].startswith("総資産: ")


def test_assets_failure_does_not_block_report(services, monkeypatch):
    monkeypatch.setattr(total_assets, "compute", lambda *a, **k: 1 / 0)
    sent = {}
    monkeypatch.setattr(discord_report, "post_daily_report",
                        lambda mode, basis, assets_line=None: sent.setdefault("line", assets_line))
    services.post_daily_summary_to_discord()
    assert "line" in sent and sent["line"] is None  # 総資産なしでもレポート自体は送る
