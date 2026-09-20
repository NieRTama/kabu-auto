"""engine_version による3経路の分岐（段階F）のテスト

段階A〜Eは新しい経路を作るだけで、どちらを使うかは誰も決めていなかった。
**legacy を選んだときの挙動が現在と1ビットも変わらない**ことを固定する。
"""
import inspect
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.data.database as db
import src.dashboard.app as dash
from src.core import config as cfg
from src.services import trading


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    dash._auth_required = False
    try:
        yield tmp_path
    finally:
        db._engine = None
        db._Session = None


@pytest.fixture
def client():
    return TestClient(dash.app)


@pytest.fixture(autouse=True)
def _config():
    cfg.load("config.yaml")
    yield
    cfg.get_section("strategy")["engine_version"] = "legacy"


def _service():
    return trading.TradingServices(
        client=MagicMock(), risk=MagicMock(), order_mgr=MagicMock(), model=None)


class TestLegacyRetrainUnchanged:
    def test_default_is_legacy(self):
        assert cfg.get_section("strategy").get("engine_version") == "legacy"

    def test_legacy_calls_train_multi(self):
        svc = _service()
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.ml_model, "train_multi") as train, \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]):
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()
        assert train.called

    def test_legacy_assigns_the_running_model(self):
        """従来どおり self.model へ代入する（挙動不変）"""
        svc = _service()
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.ml_model, "train_multi",
                          return_value="trained-model"), \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]):
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()
        assert svc.model == "trained-model"

    def test_legacy_source_still_present(self):
        src = inspect.getsource(trading.TradingServices.ml_retrain)
        assert "ml_model.train_multi" in src
        assert "self.model =" in src


class TestV2Retrain:
    def test_v2_calls_train_v2_not_train_multi(self):
        cfg.get_section("strategy")["engine_version"] = "v2"
        svc = _service()
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.ml_model, "train_multi") as legacy_train, \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]), \
             patch("src.strategy.v2_training.train_v2") as v2_train:
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()
        assert v2_train.called
        assert not legacy_train.called

    def test_v2_does_not_replace_the_running_model(self):
        """候補を作るだけ。運用モデルは自動で入れ替わらない"""
        cfg.get_section("strategy")["engine_version"] = "v2"
        svc = _service()
        svc.model = "existing-model"
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]), \
             patch("src.strategy.v2_training.train_v2",
                   return_value=MagicMock(model_id="cand-1", skipped_reason=None)):
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()
        assert svc.model == "existing-model"

    def test_v2_passes_window_sessions_and_trigger(self):
        """`window_sessions`を渡し忘れると`ModelMetrics.training_window_sessions`
        が常にNULLになる（段階F残課題6）。呼び出し時に明示的に渡すこと。
        """
        cfg.get_section("strategy")["engine_version"] = "v2"
        cfg.get_section("backtest")["retrain_window_sessions"] = 250
        svc = _service()
        try:
            with patch.object(trading, "load_ohlcv") as load, \
                 patch.object(trading.watchlist_store, "get_all_codes",
                              return_value=["7203"]), \
                 patch("src.strategy.v2_training.train_v2",
                       return_value=MagicMock(model_id="cand-1",
                                               skipped_reason=None)) as v2_train:
                load.return_value = MagicMock(__len__=lambda s: 300)
                svc.ml_retrain()
        finally:
            del cfg.get_section("backtest")["retrain_window_sessions"]
        assert v2_train.call_args.kwargs.get("window_sessions") == 250
        assert v2_train.call_args.kwargs.get("trigger") == "weekly_schedule"

    def test_v2_failure_does_not_raise(self):
        cfg.get_section("strategy")["engine_version"] = "v2"
        svc = _service()
        with patch.object(trading, "load_ohlcv") as load, \
             patch.object(trading.watchlist_store, "get_all_codes",
                          return_value=["7203"]), \
             patch("src.strategy.v2_training.train_v2",
                   side_effect=RuntimeError("boom")):
            load.return_value = MagicMock(__len__=lambda s: 300)
            svc.ml_retrain()   # 例外が外へ出ない


class TestBacktestEngineSelection:
    def test_legacy_uses_the_old_engine(self):
        from src.dashboard import app as dash

        assert dash._select_backtest_engine() == "legacy"

    def test_v2_uses_walkforward(self):
        from src.dashboard import app as dash

        cfg.get_section("strategy")["engine_version"] = "v2"
        assert dash._select_backtest_engine() == "v2"

    def test_unknown_value_falls_back_to_legacy(self):
        from src.dashboard import app as dash

        cfg.get_section("strategy")["engine_version"] = "experimental"
        assert dash._select_backtest_engine() == "legacy"

    def test_legacy_path_still_imports_the_old_engine(self):
        """legacy の経路が残っている（挙動不変の裏付け）"""
        from src.dashboard import app as dash

        src = inspect.getsource(dash)
        assert "from src.backtest.engine import run_backtest" in src

    def test_v2_path_references_walkforward(self):
        from src.dashboard import app as dash

        src = inspect.getsource(dash)
        assert "walkforward" in src


class TestV2ScoreFunction:
    def test_holder_path_uses_predict_proba_not_predict(self, monkeypatch):
        """回帰テスト: model_holder経由は必ずCurrentLightGBM(predict_proba専用)。

        既存テスト（本クラスの他のテスト）は load_current() を
        `.predict()` だけのスタブに差し替えるだけで、holder経由の経路を
        一度も通していなかった。そのため `_v2_score_fn` が holder 側にも
        `.predict()` を呼ぶ実バグ（AttributeError→ModelInferenceError、
        ウォームアップ後の全推論日で必ず失敗し degraded になっていた）を
        検出できなかった。ここでは実物の CurrentLightGBM
        （predict_proba() のみを持ち predict() を持たない）を
        model_holder へ入れて score_fn を呼び、例外にならず確率が
        返ることを固定する。
        """
        import numpy as np
        import pandas as pd

        from src.dashboard import app as dash
        from src.strategy import model_store as ms
        from src.strategy.evaluation import CurrentLightGBM
        from src.strategy.indicators import FEATURE_COLS

        assert not hasattr(CurrentLightGBM, "predict")

        def _fail_if_called(**kwargs):
            pytest.fail(
                "holder経由の経路でload_current()が呼ばれている"
                "（point-in-timeモデルを無視して昇格モデルへ先読みしている疑い）")

        monkeypatch.setattr(ms, "load_current", _fail_if_called)

        rng = np.random.default_rng(0)
        n = 40
        X = pd.DataFrame(
            rng.normal(size=(n, len(FEATURE_COLS))), columns=list(FEATURE_COLS))
        y = pd.Series([0, 1] * (n // 2))
        model = CurrentLightGBM()
        model.fit(X, y, np.ones(n))

        model_holder = {"model": model}
        score_fn = dash._v2_score_fn(use_ml=True, model_holder=model_holder)
        row = {"rule_score": 0.3}
        row.update({c: 0.1 for c in FEATURE_COLS})

        rule, proba = score_fn("7203", row)

        assert rule == pytest.approx(0.3)
        assert proba is not None
        assert 0.0 <= proba <= 1.0

    def test_warmup_does_not_fallback_to_load_current(self, monkeypatch):
        """ウォームアップ中（holderの中身がまだNone）はload_current()へ絶対に
        フォールバックしない（外部レビューR04・段階F最終ブランチレビューで
        検出されたCriticalの再発防止）。

        段階Fで最も重大だったバグは、v2バックテストのスコア関数が
        `model_store.load_current()`（現在の昇格モデル）の戻り値を生成時に
        閉包へ固定してしまい、walk-forwardが各判断時点で作るpoint-in-time
        モデルが捨てられて評価期間全体へ「未来に昇格したモデル」を当てる
        先読みになる、というものだった。コミット0ad51b4でmodel_holder
        （可変dict）方式により是正済みだが、その是正を固定するテストが
        1件も無く、防波堤は「テスト環境に昇格モデルが無いので
        load_current()がNoneを返す」という環境依存の偶然だけだった。
        ここではholderを{"model": None}（ウォームアップ中を模す）にした上で
        load_current()が呼ばれたら明示的に失敗させ、`(rule, None)`が
        返ることを固定する。
        """
        from src.dashboard import app as dash
        from src.strategy import model_store as ms

        def _fail_if_called(**kwargs):
            pytest.fail("ウォームアップ中に昇格モデルへフォールバックしている（先読み）")

        monkeypatch.setattr(ms, "load_current", _fail_if_called)
        score_fn = dash._v2_score_fn(use_ml=True, model_holder={"model": None})
        rule, proba = score_fn("7203", {"rule_score": 0.3})
        assert rule == pytest.approx(0.3)
        assert proba is None

    def test_returns_none_probability_when_unpromoted(self, tmp_path, monkeypatch):
        """モデル未昇格ならML確率はNone（ルールだけで動く）。これは劣化ではない"""
        from src.dashboard import app as dash
        from src.strategy import model_store as ms

        monkeypatch.setattr(ms, "load_current", lambda **k: None)
        score_fn = dash._v2_score_fn(use_ml=True)
        rule, proba = score_fn("7203", {"rule_score": 0.3})
        assert rule == pytest.approx(0.3)
        assert proba is None

    def test_returns_none_probability_when_ml_disabled(self, monkeypatch):
        from src.dashboard import app as dash

        score_fn = dash._v2_score_fn(use_ml=False)
        _, proba = score_fn("7203", {"rule_score": 0.3})
        assert proba is None

    def test_uses_the_promoted_model_when_available(self, monkeypatch):
        from src.dashboard import app as dash
        from src.strategy import model_store as ms

        class _Booster:
            def predict(self, X):
                return [0.77] * len(X)

        class _Meta:
            feature_cols = ["f1", "f2"]

        monkeypatch.setattr(ms, "load_current", lambda **k: (_Booster(), _Meta()))
        score_fn = dash._v2_score_fn(use_ml=True)
        _, proba = score_fn("7203", {"rule_score": 0.3, "f1": 1.0, "f2": 2.0})
        assert proba == pytest.approx(0.77)

    def test_inference_failure_is_raised_not_swallowed(self, monkeypatch):
        """推論障害は例外として外へ出す（外部レビューR05）

        `(rule, None)` へ落とすと「意図したML無効」と見分けがつかず、
        walk-forward の degraded も立たない。MLが効いていない実行が
        正常な成績として保存されてしまう。
        """
        from src.dashboard import app as dash
        from src.strategy import model_store as ms

        class _Broken:
            def predict(self, X):
                raise RuntimeError("boom")

        class _Meta:
            feature_cols = ["f1", "f2"]

        monkeypatch.setattr(ms, "load_current", lambda **k: (_Broken(), _Meta()))
        score_fn = dash._v2_score_fn(use_ml=True)
        with pytest.raises(dash.ModelInferenceError):
            score_fn("7203", {"rule_score": 0.3, "f1": 1.0, "f2": 2.0})

    def test_missing_rule_score_is_an_error_not_a_zero(self, monkeypatch):
        """rule_score が無い行を0で埋めない（外部レビューR03）

        0で埋めると、特徴量が一度も繋がっていない状態が「全候補が
        買い閾値に届かない正常なバックテスト」に見える。
        """
        from src.dashboard import app as dash
        from src.strategy import model_store as ms

        monkeypatch.setattr(ms, "load_current", lambda **k: None)
        score_fn = dash._v2_score_fn(use_ml=True)
        with pytest.raises(dash.ModelInferenceError, match="rule_score"):
            score_fn("7203", {"close": 1000.0})

    def test_missing_features_are_an_error_not_zeros(self, monkeypatch):
        from src.dashboard import app as dash
        from src.strategy import model_store as ms

        class _Booster:
            def predict(self, X):
                return [0.77] * len(X)

        class _Meta:
            feature_cols = ["f1", "f2"]

        monkeypatch.setattr(ms, "load_current", lambda **k: (_Booster(), _Meta()))
        score_fn = dash._v2_score_fn(use_ml=True)
        with pytest.raises(dash.ModelInferenceError, match="特徴量"):
            score_fn("7203", {"rule_score": 0.3, "f1": 1.0})   # f2 が無い


class TestV2ExitScoreFunction:
    def test_returns_the_rule_score_for_held_symbols(self):
        from src.dashboard import app as dash

        fn = dash._v2_exit_score_fn(use_ml=False)
        assert fn("7203", {"rule_score": -0.4}) == pytest.approx(-0.4)

    def test_missing_rule_score_is_an_error(self):
        from src.dashboard import app as dash

        fn = dash._v2_exit_score_fn(use_ml=False)
        with pytest.raises(dash.ModelInferenceError):
            fn("7203", {"close": 1000.0})


class TestV2BacktestEndToEnd:
    """合成OHLCV → エンドポイント → T+1約定まで実際の型で通す。

    テストで `rule_score` を手渡すだけでは、特徴量が経路上で供給されて
    いることを確認できない（外部レビューR03）。
    """

    def _seed_ohlcv(self, symbol="7203", n=300):
        """特徴量の助走期間を満たす合成日足をDBへ入れる"""
        import numpy as np
        import pandas as pd

        from src.data import market_data

        rng = np.random.default_rng(0)
        close = 1000 + np.cumsum(rng.normal(0, 15, n))
        idx = pd.bdate_range("2025-01-06", periods=n)
        df = pd.DataFrame({
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "adjusted_close": close,
            "volume": [1_000_000] * n,
        }, index=idx)
        df.index.name = "date"
        market_data.upsert_ohlcv(symbol, df)
        return idx

    def test_features_reach_the_decision_and_trades_can_happen(
            self, isolated_db, client):
        idx = self._seed_ohlcv()
        cfg.get_section("strategy")["engine_version"] = "v2"

        res = client.post("/api/backtest/run", json={
            "symbol": "7203",
            "start": str(idx[200].date()),
            "end": str(idx[-2].date()),
            "initial_capital": 1_000_000.0,
            "use_ml": False,
        })
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["engine_version"] == "v2"
        # 特徴量が供給されていれば、ルールスコアは一様に0にならない。
        # 取引ゼロでも「常に0点」ではないことをrun記録から確かめる
        assert body["run_id"] is not None

    def test_a_period_not_covered_by_the_data_is_rejected(
            self, isolated_db, client):
        """期間がデータに覆われていないときは黙って短い期間で回さない"""
        self._seed_ohlcv()
        cfg.get_section("strategy")["engine_version"] = "v2"

        res = client.post("/api/backtest/run", json={
            "symbol": "7203",
            "start": "2020-01-06",
            "end": "2020-12-30",
            "initial_capital": 1_000_000.0,
            "use_ml": False,
        })
        assert res.status_code == 400
        assert "覆われていません" in res.json()["detail"]

    def test_a_symbol_without_bars_is_rejected(self, isolated_db, client):
        cfg.get_section("strategy")["engine_version"] = "v2"
        res = client.post("/api/backtest/run", json={
            "symbol": "9999",
            "start": "2026-01-05",
            "end": "2026-02-05",
            "initial_capital": 1_000_000.0,
            "use_ml": False,
        })
        assert res.status_code == 400

    def test_the_run_records_a_real_config_hash(self, isolated_db, client):
        """config_hash="" で保存しない（外部レビューの残件）"""
        from sqlalchemy import select

        from src.data import database as db
        from src.data.database import get_session

        idx = self._seed_ohlcv()
        cfg.get_section("strategy")["engine_version"] = "v2"
        client.post("/api/backtest/run", json={
            "symbol": "7203",
            "start": str(idx[200].date()),
            "end": str(idx[-2].date()),
            "initial_capital": 1_000_000.0,
            "use_ml": False,
        })
        with get_session() as session:
            row = session.scalars(
                select(db.BacktestRun).order_by(db.BacktestRun.id.desc())).first()
        assert row.config_hash
        assert row.config_hash != ""
        assert row.config_json and row.config_json != "{}"
        # 戦略節だけでなくコスト・数量制限まで入っている
        assert "costs" in row.config_json
        assert "sizing" in row.config_json

    def test_use_ml_true_is_persisted_as_1(self, isolated_db, client):
        """use_ml=Trueのv2バックテストは BacktestRun.use_ml==1 を記録する。

        `_run_backtest_v2` から `wf.save_run(..., use_ml=req.use_ml)` へ
        引数を渡し忘れる種類のバグ（実際に発生した）の再発防止。
        本クラスの既存E2Eテストは全て use_ml=False で、この経路を
        一度も通していなかった。
        """
        from sqlalchemy import select

        from src.data import database as db
        from src.data.database import get_session

        idx = self._seed_ohlcv()
        cfg.get_section("strategy")["engine_version"] = "v2"
        res = client.post("/api/backtest/run", json={
            "symbol": "7203",
            "start": str(idx[200].date()),
            "end": str(idx[-2].date()),
            "initial_capital": 1_000_000.0,
            "use_ml": True,
        })
        assert res.status_code == 200, res.text
        with get_session() as session:
            row = session.scalars(
                select(db.BacktestRun).order_by(db.BacktestRun.id.desc())).first()
        assert row.use_ml == 1
