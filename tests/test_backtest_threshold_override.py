"""
バックテストの閾値オーバーライド機能のテスト

経緯: 7813のバックテスト（2026-01-01~2026-05-31）で合成スコアの最大値(0.176)が
アクティブなリスクプロファイルの買い閾値(0.18)にわずかに届かず取引が0件になった。
ライブ/ペーパー取引の設定（risk_profile）を変えずに、バックテストだけ閾値を
探索的に上書きできるようにする機能（run_backtest の buy_threshold/sell_threshold
引数）を検証する。compute_rule_score をモックし、指標計算の内部詳細に依存せず
「閾値を下回れば取引が発生しない／上回れば発生する」という結線だけを確認する。
"""
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

import src.core.config as cfg
import src.data.database as db
from src.data.database import BacktestRun, get_session
from src.data.market_data import upsert_ohlcv


SYMBOL = "9999"


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    try:
        # 200日分の穏やかな（急落のない）合成OHLCVを投入する。
        # ローソク足の値自体は使わない（compute_rule_score をモックするため）が、
        # build_features() のローリングウィンドウNaNを抜けるのに十分な本数が必要。
        start = date(2025, 1, 1)
        rows = []
        price = 1000.0
        for i in range(200):
            price *= 1 + 0.001 * ((i % 7) - 3)  # 軽い上下動
            rows.append({
                "date": start + timedelta(days=i),
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price, "volume": 100000,
            })
        df = pd.DataFrame(rows).set_index("date")
        upsert_ohlcv(SYMBOL, df)
        yield tmp_path
    finally:
        # 後続テストが本物の data/kabu_auto.db に対して動くよう、グローバルな
        # DB接続状態をリセットする（次回 get_session() 呼び出し時に再初期化される）
        db._engine = None
        db._Session = None


def _last_n_days(n: int) -> tuple:
    end = date(2025, 1, 1) + timedelta(days=199)
    start = end - timedelta(days=n)
    return start, end


class TestBacktestThresholdOverride:
    def test_default_threshold_uses_config_value(self, isolated_db):
        """オーバーライド無しの場合、config.yaml の strategy.buy_threshold が使われる"""
        from src.backtest.engine import run_backtest

        with patch("src.backtest.engine.compute_rule_score", return_value=0.2):
            start, end = _last_n_days(60)
            run_id = run_backtest(SYMBOL, start, end, use_ml=False)

        with get_session() as session:
            run = session.get(BacktestRun, run_id)
            expected = cfg.get_section("strategy")["buy_threshold"]
            assert run.buy_threshold == expected
            assert run.trade_count == 0, "合成スコア0.2はデフォルト閾値(0.25)未満のため取引は発生しないはず"

    def test_override_below_score_triggers_trade(self, isolated_db):
        """買い閾値を合成スコア未満まで下げると取引が発生する"""
        from src.backtest.engine import run_backtest

        with patch("src.backtest.engine.compute_rule_score", return_value=0.2):
            start, end = _last_n_days(60)
            run_id = run_backtest(
                SYMBOL, start, end, use_ml=False,
                buy_threshold=0.15, sell_threshold=-0.9,
            )

        with get_session() as session:
            run = session.get(BacktestRun, run_id)
            assert run.buy_threshold == 0.15
            assert run.sell_threshold == -0.9
            assert run.trade_count >= 1, "合成スコア0.2が閾値0.15を上回るため取引が発生するはず"

    def test_override_does_not_mutate_config(self, isolated_db):
        """オーバーライドは config の strategy セクションを書き換えない（ライブ設定への影響なし）"""
        from src.backtest.engine import run_backtest

        before = cfg.get_section("strategy")["buy_threshold"]
        with patch("src.backtest.engine.compute_rule_score", return_value=0.2):
            start, end = _last_n_days(60)
            run_backtest(SYMBOL, start, end, use_ml=False, buy_threshold=0.01)
        after = cfg.get_section("strategy")["buy_threshold"]
        assert before == after, "バックテストの閾値オーバーライドが config を汚染してはいけない"
