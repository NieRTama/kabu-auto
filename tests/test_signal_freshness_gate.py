"""シグナルのデータ基準日記録と、鮮度による新規候補の除外

背景: 生成日時しか持たないため、古い足から作られたシグナルを区別できなかった。
また確定していない足でも新規候補として保存されていた（F01）。
"""
from datetime import date

import pytest
from sqlalchemy import select

from src.core import clock
from src.core import config as cfg
from src.data import database as db
from src.data.database import Signal, get_session
from src.services import trading
from src.strategy.signal import Signal as TradeSignal


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


class TestSignalDataAsOf:
    def test_data_as_of_is_saved_separately_from_generated_at(self, isolated_db):
        sig = TradeSignal(symbol="7203", action="BUY", rule_score=0.3,
                          ml_score=0.1, combined_score=0.4)
        trading._save_signal(sig, data_as_of=date(2026, 9, 10))

        with get_session() as session:
            row = session.scalar(select(Signal))
        assert row.data_as_of == date(2026, 9, 10)
        # 生成日時は保存時刻（clock.now）で、データ基準日とは独立に決まる
        assert row.generated_at is not None
        assert row.generated_at.date() == clock.today()

    def test_data_as_of_is_nullable_for_legacy_rows(self, isolated_db):
        sig = TradeSignal(symbol="7203", action="HOLD", rule_score=0.0,
                          ml_score=0.0, combined_score=0.0)
        trading._save_signal(sig)

        with get_session() as session:
            row = session.scalar(select(Signal))
        assert row.data_as_of is None
