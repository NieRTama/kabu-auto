"""
RiskManager.check_max_positions() / check_sector_concentration() の
未約定BUY・候補注文金額の取り込みテスト（再レビュー P1-1/P1-2対応）
"""
import pytest

import src.core.config as cfg
import src.data.database as db
import src.execution.order_status as st
from src.data.database import Position, Trade, get_session
from src.risk.manager import RiskManager


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    try:
        yield tmp_path
    finally:
        db._engine = None
        db._Session = None


def _add_position(symbol, quantity, avg_cost, sector=""):
    with get_session() as session:
        session.add(Position(symbol=symbol, quantity=quantity, avg_cost=avg_cost, sector=sector))
        session.commit()


def _add_trade(symbol, side, status, quantity, price, sector=None):
    with get_session() as session:
        session.add(Trade(
            order_id=f"{symbol}-{side}-{status}-{sector}",
            symbol=symbol, side=side, status=status, sector=sector,
            quantity=quantity, price=price,
        ))
        session.commit()


class TestCheckMaxPositions:
    def test_under_limit_with_no_pending_ok(self, isolated_db):
        _add_position("1111", 100, 1000.0)
        risk = RiskManager()
        risk._conf = {"max_positions": 3}
        ok, _ = risk.check_max_positions(candidate_symbol="2222")
        assert ok is True

    def test_pending_buy_counts_toward_limit(self, isolated_db):
        """保有2 + 未約定BUY2件で上限3を超えるため、新規候補は拒否される"""
        _add_position("1111", 100, 1000.0)
        _add_position("2222", 100, 1000.0)
        _add_trade("3333", "BUY", st.PENDING, 100, 1000.0)
        risk = RiskManager()
        risk._conf = {"max_positions": 3}
        ok, reason = risk.check_max_positions(candidate_symbol="4444")
        assert ok is False
        assert "最大保有銘柄数" in reason

    def test_candidate_already_held_does_not_increase_count(self, isolated_db):
        """候補銘柄が既に保有中なら集合は増えないため、上限到達中でも通る
        （既存ポジションへの追加買いは新規銘柄ではないため）"""
        _add_position("1111", 100, 1000.0)
        _add_position("2222", 100, 1000.0)
        _add_position("3333", 100, 1000.0)
        risk = RiskManager()
        risk._conf = {"max_positions": 3}
        ok, _ = risk.check_max_positions(candidate_symbol="1111")
        assert ok is True

    def test_candidate_already_pending_does_not_increase_count(self, isolated_db):
        _add_position("1111", 100, 1000.0)
        _add_position("2222", 100, 1000.0)
        _add_trade("3333", "BUY", st.PENDING, 100, 1000.0)
        risk = RiskManager()
        risk._conf = {"max_positions": 3}
        ok, _ = risk.check_max_positions(candidate_symbol="3333")
        assert ok is True

    def test_no_candidate_just_checks_current_set(self, isolated_db):
        _add_position("1111", 100, 1000.0)
        _add_position("2222", 100, 1000.0)
        _add_position("3333", 100, 1000.0)
        risk = RiskManager()
        risk._conf = {"max_positions": 3}
        ok, _ = risk.check_max_positions()
        assert ok is True  # candidate無しなので現状3=上限3でも超過ではない


class TestSectorConcentrationCandidateNotional:
    def test_candidate_notional_alone_can_trigger_over_limit(self, isolated_db):
        """既存保有が無くても、候補注文単体の金額が分母の大半を占めれば超過判定になる"""
        risk = RiskManager()
        risk._conf = {"max_sector_ratio": 0.40}
        ok, reason = risk.check_sector_concentration("A", candidate_notional=100_000.0)
        assert ok is False  # 候補のみ→そのセクターの比率は100%
        assert "A" in reason

    def test_candidate_notional_pushes_existing_balanced_sector_over_limit(self, isolated_db):
        """既存保有では均等(33%)でも、候補注文を加えると上限超になるケース"""
        for sym, sec in [("1111", "A"), ("2222", "B"), ("3333", "C")]:
            _add_position(sym, 100, 1000.0, sector=sec)
        risk = RiskManager()
        risk._conf = {"max_sector_ratio": 0.40}
        ok, _ = risk.check_sector_concentration("A")
        assert ok is True  # 候補無しならOK
        # Aセクターへの新規候補20万円を加味 → A=(10万+20万)/(30万+20万)=60% → NG
        ok, reason = risk.check_sector_concentration("A", candidate_notional=200_000.0)
        assert ok is False
        assert "A" in reason

    def test_zero_candidate_notional_is_backward_compatible(self, isolated_db):
        """candidate_notional省略時は従来どおりの挙動（既存保有・引当のみで判定）"""
        _add_position("1111", 100, 1000.0, sector="A")
        risk = RiskManager()
        risk._conf = {"max_sector_ratio": 0.40}
        ok, _ = risk.check_sector_concentration("A")
        assert ok is False  # 単一保有=100%濃度


class TestValidateBuyUsesCandidateNotional:
    def test_validate_buy_rejects_when_order_itself_breaches_sector_limit(self, isolated_db):
        """既存保有が無くても、validate_buyが計算する注文金額がセクター上限を超えるなら拒否する"""
        risk = RiskManager()
        risk._conf = {
            "daily_order_limit": 100, "max_daily_loss": 0, "max_positions": 10,
            "max_position_ratio": 1.0, "max_sector_ratio": 0.40,
        }
        # cash=100万、price=1000 → calc_position_size次第で大きな金額になり、
        # セクター単独保有なので比率100% > 40% で拒否されるはず
        ok, reason = risk.validate_buy("9999", 1000.0, 1_000_000.0, sector="A")
        assert ok is False
        assert "セクター集中率" in reason
