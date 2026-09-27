"""昇格済み候補モデルのshadow記録の配線（src/services/shadow_recording.py）

**観察のみ。実発注には一切影響しない。** 記録の失敗が本来の判断
（買い/売り/様子見）や paper執行を妨げないことも、このファイルで固定する。
"""
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.backtest import execution
from src.core import clock
from src.core import config as cfg
from src.data import database as db
from src.data.bar_status import BarStatus
from src.data.database import OHLCV, Signal, get_session
from src.services import shadow_recording
from src.services import trading
from src.strategy import dataset as ds
from src.strategy import model_store as ms
from src.strategy import policy
from src.strategy import shadow
from src.strategy.indicators import FEATURE_COLS, build_features
from src.strategy.signal import Signal as TradeSignal


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    return tmp_path


class _FakeClassifier:
    """sklearn LGBMClassifier の「形」だけを持つ最小の贋物。

    predict() はクラスラベル、predict_proba() は (n, 2) の確率を返す。
    実機の LGBMClassifier で確認した挙動と同じ
    （predict -> [1] / predict_proba -> [[0.4211, 0.5789]]）。
    """

    def __init__(self, positives):
        self._p = np.asarray(positives, dtype=float)

    def predict(self, X):
        return (self._p[:len(X)] >= 0.5).astype(int)

    def predict_proba(self, X):
        p = self._p[:len(X)]
        return np.column_stack([1.0 - p, p])


class _OneColumnProbaClassifier:
    """単一クラスしか見なかった分類器（predict_proba が (n, 1) を返す）"""

    def predict_proba(self, X):
        return np.ones((len(X), 1), dtype=float)


class _FixedProbaModel:
    """候補モデル役。predict() が正例確率をそのまま返す（Booster と同じ契約）"""

    def __init__(self, probabilities):
        self._p = np.asarray(probabilities, dtype=float)

    def predict(self, X):
        return self._p[:len(X)]


class TestLegacyModelProbaAdapter:
    def test_predict_returns_probabilities_not_class_labels(self):
        """素の predict() はラベルを返す。確率として記録してはいけない"""
        model = _FakeClassifier([0.5789, 0.1])
        X = pd.DataFrame({"f": [1.0, 2.0]})

        assert list(model.predict(X)) == [1, 0]
        got = shadow_recording._LegacyModelProbaAdapter(model).predict(X)
        assert list(got) == pytest.approx([0.5789, 0.1])

    def test_rejects_a_single_class_probability_matrix(self):
        """確率が2列そろわないときは既定値で埋めず例外にする

        黙って0.5等で埋めると「現行が何を予測したか」が捏造される
        （Knowledge.md §10「欠落を既定値で埋めると未結線が正常な結果に化ける」）。
        """
        adapter = shadow_recording._LegacyModelProbaAdapter(
            _OneColumnProbaClassifier())
        with pytest.raises(ValueError):
            adapter.predict(pd.DataFrame({"f": [1.0]}))

    def test_satisfies_the_contract_shadow_compare_requires(self):
        comparisons = shadow.compare(
            ["7203:20260910"], pd.DataFrame({"f": [1.0]}),
            current=shadow_recording._LegacyModelProbaAdapter(
                _FakeClassifier([0.8])),
            candidate=_FixedProbaModel([0.2]),
            threshold=shadow_recording.SHADOW_THRESHOLD)

        assert comparisons[0].current_probability == pytest.approx(0.8)
        assert comparisons[0].current_takes is True
        assert comparisons[0].candidate_takes is False
        assert comparisons[0].agreement == shadow.AGREEMENT_ONLY_CURRENT

    def test_threshold_is_the_observation_default(self):
        assert shadow_recording.SHADOW_THRESHOLD == 0.5


class _ConstantCandidate:
    """model_store.save_candidate() が定数モデルとして保存できる最小の形。

    読み戻すと model_store.ConstantModel になり、
    predict(X) -> np.full(len(X), probability) を返す。
    """

    is_constant = True

    def __init__(self, probability):
        self.constant_probability = float(probability)


def _current_contract() -> str:
    """いま設定から決まるラベル契約ID（固定文字列を書かない）"""
    return ds.make_label_contract_id(policy.config_from_settings(),
                                     execution.config_from_settings())


def _promote_constant_candidate(models_dir, *, model_id="v2-test-0001",
                                probability=0.7, feature_cols=None):
    """tmp配下に候補を保存して現行（＝昇格済み）に設定する"""
    meta = ms.ModelMeta(
        model_id=model_id,
        trained_at=clock.now(),
        symbols=["7203"],
        label_definition="net_return>0",
        feature_cols=list(FEATURE_COLS) if feature_cols is None else feature_cols,
        label_contract_id=_current_contract(),
    )
    ms.save_candidate(_ConstantCandidate(probability), meta,
                      base_dir=str(models_dir))
    ms.set_current(model_id, base_dir=str(models_dir))
    return model_id


class TestPrepare:
    def test_returns_none_when_no_candidate_is_promoted(self, isolated_db, tmp_path):
        """未昇格は異常ではない。静かにスキップする"""
        assert shadow_recording.prepare(
            None, base_dir=str(tmp_path / "models")) is None

    def test_loads_the_promoted_candidate(self, isolated_db, tmp_path):
        models_dir = tmp_path / "models"
        _promote_constant_candidate(models_dir)

        batch = shadow_recording.prepare(None, base_dir=str(models_dir))

        assert batch is not None
        assert batch.candidate_model_id == "v2-test-0001"
        assert batch.candidate.predict(pd.DataFrame({"f": [1.0]})) == \
            pytest.approx([0.7])
        assert batch.threshold == 0.5
        assert batch.label_contract_id == _current_contract()
        assert batch.event_ids == []
        assert batch.rows == []

    def test_run_id_is_one_batch_per_day(self, isolated_db, tmp_path):
        models_dir = tmp_path / "models"
        _promote_constant_candidate(models_dir)

        with patch.object(clock, "today", return_value=date(2026, 9, 10)):
            batch = shadow_recording.prepare(None, base_dir=str(models_dir))

        assert batch.evaluation_run_id == "shadow-2026-09-10"

    def test_wraps_the_current_model_in_the_probability_adapter(
            self, isolated_db, tmp_path):
        models_dir = tmp_path / "models"
        _promote_constant_candidate(models_dir)

        batch = shadow_recording.prepare(_FakeClassifier([0.9]),
                                         base_dir=str(models_dir))

        assert isinstance(batch.current, shadow_recording._LegacyModelProbaAdapter)
        assert batch.current.predict(pd.DataFrame({"f": [1.0]})) == \
            pytest.approx([0.9])

    def test_no_current_model_is_recorded_as_such(self, isolated_db, tmp_path):
        """現行モデル未ロード（学習前）でも候補だけは記録できる"""
        models_dir = tmp_path / "models"
        _promote_constant_candidate(models_dir)

        batch = shadow_recording.prepare(None, base_dir=str(models_dir))

        assert batch.current is None
        assert batch.current_model_id is None

    def test_refuses_to_record_when_the_feature_definition_disagrees(
            self, isolated_db, tmp_path):
        """特徴量定義が食い違う候補は黙って推論しない（fail-closed）"""
        models_dir = tmp_path / "models"
        _promote_constant_candidate(models_dir, feature_cols=["rsi", "macd"])

        assert shadow_recording.prepare(None, base_dir=str(models_dir)) is None


class TestLegacyModelId:
    def test_is_none_without_a_current_model(self):
        assert shadow_recording._legacy_model_id(None) is None

    def test_uses_the_sha256_of_the_saved_pickle(self, tmp_path, monkeypatch):
        """週次再学習で中身が入れ替わるので、版を sha256 で区別する"""
        monkeypatch.chdir(tmp_path)
        meta_path = tmp_path / "models" / "lgb_model.meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            '{"sha256": "6958ed7c114b609a1175e26703a0c60867c2a5620df1fde301da36'
            '022a71cb3a", "trained_at": "2026-09-18T11:14:50.350372"}',
            encoding="utf-8")

        got = shadow_recording._legacy_model_id(_FakeClassifier([0.5]))

        assert got == "legacy-6958ed7c114b"

    def test_falls_back_when_the_sidecar_is_missing(self, tmp_path, monkeypatch):
        """メタが読めなくても記録自体は続ける（判断を止めない）"""
        monkeypatch.chdir(tmp_path)

        got = shadow_recording._legacy_model_id(_FakeClassifier([0.5]))

        assert got == shadow_recording.LEGACY_MODEL_ID_UNKNOWN


def _ohlcv_frame(*, periods=120, end=date(2026, 9, 10)) -> pd.DataFrame:
    """合成の日足（日付インデックス）。

    ma_long の既定が75本なので、最終行で全特徴量が有効になるよう120本入れる。
    """
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=periods)
    close = np.array([1000.0 + 50.0 * np.sin(i / 7.0) + i * 0.5
                      for i in range(periods)])
    return pd.DataFrame({
        "open": close - 3.0,
        "high": close + 6.0,
        "low": close - 6.0,
        "close": close,
        "volume": np.array([100_000 + 300 * i for i in range(periods)]),
    }, index=idx)


def _batch(*, current=None, current_model_id=None,
           run_id="shadow-2026-09-10", candidate_probability=0.7):
    return shadow_recording.ShadowBatch(
        evaluation_run_id=run_id,
        candidate=_FixedProbaModel([candidate_probability] * 16),
        candidate_model_id="v2-test-0001",
        current=current,
        current_model_id=current_model_id,
        label_contract_id="lc_test",
    )


class TestCollect:
    def test_event_id_is_the_symbol_and_the_decision_session(self, isolated_db):
        batch = _batch()

        assert shadow_recording.collect(batch, "7203", _ohlcv_frame()) is True
        assert batch.event_ids == ["7203:20260910"]

    def test_uses_the_same_feature_row_the_current_model_sees(self, isolated_db):
        """gen_signal が使う build_features() の最終行と同一であること

        別の行を渡すと「同じ入力に対する2つの判断」ではなくなる。
        """
        df = _ohlcv_frame()
        batch = _batch()
        shadow_recording.collect(batch, "7203", df)

        expected = build_features(df)[list(FEATURE_COLS)].iloc[[-1]]
        assert list(batch.rows[0].columns) == list(FEATURE_COLS)
        assert batch.rows[0].to_numpy() == pytest.approx(expected.to_numpy())

    def test_skips_a_symbol_whose_features_are_not_ready(self, isolated_db):
        """助走期間が足りない銘柄は記録しない（欠損を0で埋めない）"""
        batch = _batch()

        assert shadow_recording.collect(batch, "7203",
                                        _ohlcv_frame(periods=40)) is False
        assert batch.event_ids == []
        assert batch.rows == []

    def test_accumulates_every_symbol(self, isolated_db):
        batch = _batch()
        shadow_recording.collect(batch, "7203", _ohlcv_frame())
        shadow_recording.collect(batch, "9984", _ohlcv_frame())

        assert batch.event_ids == ["7203:20260910", "9984:20260910"]
        assert len(batch.rows) == 2


class TestFlush:
    def test_writes_one_row_per_collected_symbol(self, isolated_db):
        batch = _batch()
        shadow_recording.collect(batch, "7203", _ohlcv_frame())
        shadow_recording.collect(batch, "9984", _ohlcv_frame())

        n = shadow_recording.flush(batch)

        assert n == 2
        got = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert set(got["event_id"]) == {"7203:20260910", "9984:20260910"}
        assert set(got["candidate_model_id"]) == {"v2-test-0001"}
        assert got["candidate_probability"].tolist() == pytest.approx([0.7, 0.7])
        assert set(got["threshold"]) == {0.5}
        assert set(got["label_contract_id"]) == {"lc_test"}

    def test_records_both_sides_when_a_current_model_exists(self, isolated_db):
        batch = _batch(
            current=shadow_recording._LegacyModelProbaAdapter(
                _FakeClassifier([0.9, 0.9])),
            current_model_id="legacy-6958ed7c114b")
        shadow_recording.collect(batch, "7203", _ohlcv_frame())
        shadow_recording.collect(batch, "9984", _ohlcv_frame())

        shadow_recording.flush(batch)

        got = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert set(got["current_model_id"]) == {"legacy-6958ed7c114b"}
        assert got["current_probability"].tolist() == pytest.approx([0.9, 0.9])
        assert set(got["agreement"]) == {shadow.AGREEMENT_BOTH_TAKE}

    def test_records_the_unpromoted_current_as_its_own_state(self, isolated_db):
        batch = _batch()
        shadow_recording.collect(batch, "7203", _ohlcv_frame())

        shadow_recording.flush(batch)

        got = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert got["current_model_id"].isna().all()
        assert got["current_probability"].isna().all()
        assert set(got["agreement"]) == {shadow.AGREEMENT_ONLY_CANDIDATE}

    def test_nothing_collected_writes_nothing(self, isolated_db):
        assert shadow_recording.flush(_batch()) == 0
        assert len(shadow.load_shadow_comparisons("shadow-2026-09-10")) == 0

    def test_flushing_the_same_run_twice_replaces_the_rows(self, isolated_db):
        """同じ日に2回走っても上書きで整合する（run単位の置換）"""
        first = _batch()
        shadow_recording.collect(first, "7203", _ohlcv_frame())
        shadow_recording.collect(first, "9984", _ohlcv_frame())
        shadow_recording.flush(first)

        second = _batch(candidate_probability=0.2)
        shadow_recording.collect(second, "7203", _ohlcv_frame())
        shadow_recording.collect(second, "9984", _ohlcv_frame())
        shadow_recording.flush(second)

        got = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert len(got) == 2
        assert got["candidate_probability"].tolist() == pytest.approx([0.2, 0.2])

    def test_one_flush_per_scan_keeps_every_symbol(self, isolated_db):
        """銘柄ごとに flush してはいけないことを、結果の差で示す

        record_shadow() は同一 evaluation_run_id の既存行を削除してから
        挿入する。銘柄ごとに flush すると最後の1銘柄しか残らない。
        """
        per_symbol = _batch()
        shadow_recording.collect(per_symbol, "7203", _ohlcv_frame())
        shadow_recording.flush(per_symbol)
        per_symbol.event_ids.clear()
        per_symbol.rows.clear()
        shadow_recording.collect(per_symbol, "9984", _ohlcv_frame())
        shadow_recording.flush(per_symbol)

        wrong = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert set(wrong["event_id"]) == {"9984:20260910"}  # 7203が消える

        batched = _batch(run_id="shadow-2026-09-11")
        shadow_recording.collect(batched, "7203", _ohlcv_frame())
        shadow_recording.collect(batched, "9984", _ohlcv_frame())
        shadow_recording.flush(batched)

        right = shadow.load_shadow_comparisons("shadow-2026-09-11")
        assert set(right["event_id"]) == {"7203:20260910", "9984:20260910"}


def _fresh_states(symbols, session_date=date(2026, 9, 10)):
    return {s: BarStatus(symbol=s, last_bar_session=session_date,
                         observed_at=datetime(2026, 9, 10, 16, 20),
                         is_final=True, state="fresh")
            for s in symbols}


class TestSignalScanWiring:
    """配線をソースで固定する（既存ロジックを変えていないことを含む）"""

    def _source(self):
        import inspect
        return inspect.getsource(trading.TradingServices.signal_scan)

    def test_existing_decision_line_is_unchanged(self):
        src = self._source()
        assert "sig = gen_signal(sym, df, self.model)" in src
        assert "_save_signal(sig, data_as_of=self._bar_states[sym].last_bar_session)" in src

    def test_the_scan_never_reassigns_the_operating_model(self):
        """候補を self.model へ入れない（shadowは観察のみ）"""
        assert "self.model =" not in self._source()

    def test_calls_prepare_once_before_the_loop(self):
        src = self._source()
        assert src.index("shadow_recording.prepare(self.model)") < \
            src.index("for sym in codes:")

    def test_collects_before_the_action_filter(self):
        """HOLDの銘柄も記録対象にする（絞り込まない）"""
        src = self._source()
        assert src.index("shadow_recording.collect(shadow_batch, sym, df)") < \
            src.index('if sig.action not in ("BUY", "SELL"):')

    def test_flushes_once_after_the_loop(self):
        src = self._source()
        assert src.index("shadow_recording.flush(shadow_batch)") > \
            src.index("シグナルスキャンエラー")


class TestShadowFailureDoesNotAffectTrading:
    """shadow記録の失敗が本来の判断・発注を妨げないこと"""

    def _services(self, model=None):
        risk = MagicMock()
        risk.validate_buy.return_value = (True, "ok")
        risk.calc_position_size.return_value = 100
        svc = trading.TradingServices(client=MagicMock(), risk=risk,
                                      order_mgr=MagicMock(), model=model)
        svc.trading_conf = dict(svc.trading_conf)
        svc.trading_conf["mode"] = "paper"
        return svc

    def _scan(self, svc, symbols):
        svc._bar_states = _fresh_states(symbols)
        with patch.object(trading.clock, "now",
                          return_value=datetime(2026, 9, 10, 16, 20)), \
             patch.object(trading.clock, "today", return_value=date(2026, 9, 10)), \
             patch.object(trading.watchlist_store, "get_codes",
                          return_value=list(symbols)), \
             patch.object(trading.watchlist_store, "get_sectors",
                          return_value={}), \
             patch.object(trading.TradingScheduler, "is_maintenance_window",
                          return_value=False), \
             patch.object(trading, "load_ohlcv", return_value=_ohlcv_frame()), \
             patch.object(trading, "gen_signal",
                          side_effect=lambda sym, df, model: TradeSignal(
                              symbol=sym, action="BUY", rule_score=0.6,
                              ml_score=0.2, combined_score=0.4)), \
             patch.object(trading.liquidity, "check_liquidity",
                          return_value=(True, "")):
            svc.signal_scan()

    def test_a_failing_collect_does_not_block_the_paper_order(self, isolated_db):
        svc = self._services()
        with patch.object(shadow_recording, "prepare",
                          return_value=_batch()), \
             patch.object(shadow_recording, "collect",
                          side_effect=RuntimeError("shadow boom")):
            self._scan(svc, ["7203"])

        assert svc.order_mgr.buy.called

    def test_a_failing_collect_does_not_stop_the_other_symbols(self, isolated_db):
        svc = self._services()
        with patch.object(shadow_recording, "prepare",
                          return_value=_batch()), \
             patch.object(shadow_recording, "collect",
                          side_effect=RuntimeError("shadow boom")):
            self._scan(svc, ["7203", "9984"])

        with get_session() as session:
            saved = {r.symbol for r in session.scalars(select(Signal)).all()}
        assert saved == {"7203", "9984"}

    def test_a_failing_prepare_does_not_stop_the_scan(self, isolated_db):
        svc = self._services()
        with patch.object(shadow_recording, "prepare",
                          side_effect=RuntimeError("load boom")):
            self._scan(svc, ["7203"])

        assert svc.order_mgr.buy.called

    def test_a_failing_flush_does_not_stop_the_scan(self, isolated_db):
        svc = self._services()
        with patch.object(shadow_recording, "prepare",
                          return_value=_batch()), \
             patch.object(shadow_recording, "flush",
                          side_effect=RuntimeError("save boom")):
            self._scan(svc, ["7203"])

        assert svc.order_mgr.buy.called

    def test_no_candidate_means_no_shadow_calls(self, isolated_db):
        """未昇格なら collect も flush も呼ばれない"""
        svc = self._services()
        with patch.object(shadow_recording, "prepare", return_value=None), \
             patch.object(shadow_recording, "collect") as collect, \
             patch.object(shadow_recording, "flush") as flush:
            self._scan(svc, ["7203"])

        assert not collect.called
        assert not flush.called
        assert svc.order_mgr.buy.called


def _seed_ohlcv(symbol, *, periods=120, end=date(2026, 9, 10)):
    """合成の日足をDBへ入れる（load_ohlcv 経由で実際に読ませる）"""
    frame = _ohlcv_frame(periods=periods, end=end)
    with get_session() as session:
        for ts, row in frame.iterrows():
            session.add(OHLCV(
                symbol=symbol, date=ts.date(),
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=int(row["volume"]), adjusted_close=float(row["close"])))
        session.commit()


class TestSignalScanRecordsShadow:
    """signal_scan を実際に走らせ、DBに入った行の件数と中身を確かめる"""

    def _run(self, svc, symbols):
        svc._bar_states = _fresh_states(symbols)
        with patch.object(trading.clock, "now",
                          return_value=datetime(2026, 9, 10, 16, 20)), \
             patch.object(trading.clock, "today", return_value=date(2026, 9, 10)), \
             patch.object(trading.watchlist_store, "get_codes",
                          return_value=list(symbols)), \
             patch.object(trading.watchlist_store, "get_sectors",
                          return_value={}), \
             patch.object(trading.TradingScheduler, "is_maintenance_window",
                          return_value=False):
            svc.signal_scan()

    def _services(self, model=None):
        return trading.TradingServices(client=MagicMock(), risk=MagicMock(),
                                       order_mgr=MagicMock(), model=model)

    def test_records_one_row_per_scanned_symbol(self, isolated_db, tmp_path,
                                                monkeypatch):
        _seed_ohlcv("7203")
        _seed_ohlcv("9984")
        _promote_constant_candidate(tmp_path / "models")
        # prepare() は既定の相対パス "models" を見る。実リポジトリを汚さない
        # よう、cfg.load / db.init を済ませた後に作業ディレクトリを移す
        monkeypatch.chdir(tmp_path)

        self._run(self._services(), ["7203", "9984"])

        got = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert len(got) == 2
        assert set(got["event_id"]) == {"7203:20260910", "9984:20260910"}
        assert set(got["candidate_model_id"]) == {"v2-test-0001"}
        assert got["candidate_probability"].tolist() == pytest.approx([0.7, 0.7])
        assert set(got["threshold"]) == {0.5}
        assert set(got["label_contract_id"]) == {_current_contract()}
        # 現行モデル未ロード（model=None）なので「現行が無かった」状態が残る
        assert got["current_model_id"].isna().all()
        assert set(got["agreement"]) == {shadow.AGREEMENT_ONLY_CANDIDATE}

    def test_records_the_current_models_probability_too(self, isolated_db,
                                                        tmp_path, monkeypatch):
        _seed_ohlcv("7203")
        _promote_constant_candidate(tmp_path / "models")
        monkeypatch.chdir(tmp_path)

        self._run(self._services(model=_FakeClassifier([0.9])), ["7203"])

        got = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert len(got) == 1
        # ラベル(1)ではなく確率(0.9)が入っていること＝アダプタが効いている
        assert got["current_probability"].tolist() == pytest.approx([0.9])
        assert got["current_model_id"].tolist() == \
            [shadow_recording.LEGACY_MODEL_ID_UNKNOWN]
        assert got["agreement"].tolist() == [shadow.AGREEMENT_BOTH_TAKE]

    def test_the_candidate_never_touches_the_signal(self, isolated_db, tmp_path,
                                                    monkeypatch):
        """保存されたシグナルが、候補の有無で変わらないこと"""
        _seed_ohlcv("7203")
        monkeypatch.chdir(tmp_path)
        svc = self._services()
        self._run(svc, ["7203"])
        with get_session() as session:
            without = [(r.action, r.combined_score)
                       for r in session.scalars(select(Signal)).all()]
            session.query(Signal).delete()
            session.commit()

        _promote_constant_candidate(tmp_path / "models")
        self._run(self._services(), ["7203"])

        with get_session() as session:
            with_candidate = [(r.action, r.combined_score)
                              for r in session.scalars(select(Signal)).all()]
        assert with_candidate == without
        assert len(shadow.load_shadow_comparisons("shadow-2026-09-10")) == 1

    def test_no_candidate_records_nothing_and_still_saves_signals(
            self, isolated_db, tmp_path, monkeypatch):
        _seed_ohlcv("7203")
        monkeypatch.chdir(tmp_path)   # models/current.json が無い状態

        self._run(self._services(), ["7203"])

        assert len(shadow.load_shadow_comparisons("shadow-2026-09-10")) == 0
        with get_session() as session:
            assert session.scalar(select(Signal)) is not None

    def test_a_symbol_without_enough_history_is_skipped(self, isolated_db,
                                                        tmp_path, monkeypatch):
        """特徴量が揃わない銘柄は記録しない（他の銘柄は記録する）"""
        _seed_ohlcv("7203")
        _seed_ohlcv("9984", periods=40)
        _promote_constant_candidate(tmp_path / "models")
        monkeypatch.chdir(tmp_path)

        self._run(self._services(), ["7203", "9984"])

        got = shadow.load_shadow_comparisons("shadow-2026-09-10")
        assert set(got["event_id"]) == {"7203:20260910"}

    def test_running_twice_in_a_day_stays_consistent(self, isolated_db, tmp_path,
                                                     monkeypatch):
        _seed_ohlcv("7203")
        _seed_ohlcv("9984")
        _promote_constant_candidate(tmp_path / "models")
        monkeypatch.chdir(tmp_path)

        self._run(self._services(), ["7203", "9984"])
        self._run(self._services(), ["7203", "9984"])

        assert len(shadow.load_shadow_comparisons("shadow-2026-09-10")) == 2
        with get_session() as session:
            preds = list(session.scalars(select(db.Prediction)).all())
        assert len(preds) == 2
        assert {p.purpose for p in preds} == {"shadow"}
        assert {p.fold_index for p in preds} == {-1}

    def test_no_outcome_rows_are_written(self, isolated_db, tmp_path, monkeypatch):
        """実績（勝敗）は予測時点で未確定。ここでは書かない"""
        _seed_ohlcv("7203")
        _promote_constant_candidate(tmp_path / "models")
        monkeypatch.chdir(tmp_path)

        self._run(self._services(), ["7203"])

        with get_session() as session:
            assert list(session.scalars(select(db.PredictionOutcome)).all()) == []

    def test_no_orders_are_created(self, isolated_db, tmp_path, monkeypatch):
        """shadowは記録だけ。注文・建玉・約定を一切作らない"""
        _seed_ohlcv("7203")
        _promote_constant_candidate(tmp_path / "models")
        monkeypatch.chdir(tmp_path)
        svc = self._services()

        self._run(svc, ["7203"])

        assert not svc.order_mgr.buy.called
        assert not svc.order_mgr.sell.called
        assert not svc.order_mgr.sell_market.called
        with get_session() as session:
            assert list(session.scalars(select(db.Trade)).all()) == []
            assert list(session.scalars(select(db.Position)).all()) == []
            assert list(session.scalars(select(db.OrderIntent)).all()) == []
