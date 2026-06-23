"""
ニュースセンチメント特徴量パイプラインのテスト（torch 不要・純pandas部分）。

検証項目:
  - trading_date_for: 16:00 JST カットオフ・週末繰り上げ（リーク防止の要）
  - join_news_features: 行を落とさない・欠損は中立0・列が揃う
  - load_news_frame: DB往復で後ろ向き移動平均が正しく出る・ニュース無しは None
  - active_feature_cols / build_features: フラグでニュース列が増減し、dropnaは基本列のみ
"""
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

import src.core.config as cfg
import src.data.database as db
from src.data.database import NewsSentiment, get_session
from src.strategy import news_features as nf
from src.strategy.indicators import (
    BASE_TECHNICAL_COLS, NEWS_FEATURE_COLS, active_feature_cols, build_features,
)


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


@pytest.fixture
def news_off():
    """use_news_features を OFF に固定（既定）。"""
    cfg.load("config.yaml")
    cfg.get_section("strategy")["use_news_features"] = False
    yield


@pytest.fixture
def news_on():
    """use_news_features を ON にする。"""
    cfg.load("config.yaml")
    cfg.get_section("strategy")["use_news_features"] = True
    yield
    cfg.get_section("strategy")["use_news_features"] = False


# ─── trading_date_for（リーク防止の要）────────────────────────────────────────

class TestTradingDateFor:
    def test_before_cutoff_same_day(self):
        # 月曜 10:00 公開 → 同じ月曜
        d = nf.trading_date_for(datetime(2026, 6, 22, 10, 0), cutoff_hour=16)
        assert d == date(2026, 6, 22)

    def test_after_cutoff_next_day(self):
        # 月曜 16:30 公開 → 翌火曜（大引け後は当日には反映できない）
        d = nf.trading_date_for(datetime(2026, 6, 22, 16, 30), cutoff_hour=16)
        assert d == date(2026, 6, 23)

    def test_exactly_cutoff_is_next_day(self):
        d = nf.trading_date_for(datetime(2026, 6, 22, 16, 0), cutoff_hour=16)
        assert d == date(2026, 6, 23)

    def test_friday_after_cutoff_rolls_to_monday(self):
        # 金曜 17:00 → 土日を飛ばして月曜
        d = nf.trading_date_for(datetime(2026, 6, 19, 17, 0), cutoff_hour=16)
        assert d == date(2026, 6, 22)  # 2026-06-22 は月曜

    def test_saturday_rolls_to_monday(self):
        d = nf.trading_date_for(datetime(2026, 6, 20, 9, 0), cutoff_hour=16)
        assert d == date(2026, 6, 22)


# ─── join_news_features ───────────────────────────────────────────────────────

class TestJoinNewsFeatures:
    def _feat(self, n=10):
        idx = pd.bdate_range("2026-06-01", periods=n)
        return pd.DataFrame({"close": np.arange(n, dtype=float)}, index=idx)

    def test_none_news_fills_zero_and_keeps_rows(self):
        feat = self._feat(5)
        out = nf.join_news_features(feat, None)
        assert len(out) == 5  # 行は落ちない
        for col in NEWS_FEATURE_COLS:
            assert col in out.columns
            assert (out[col] == 0.0).all()

    def test_join_aligns_by_date_and_zero_fills_gaps(self):
        feat = self._feat(5)  # 2026-06-01 .. 06-05（営業日）
        # 06-02 と 06-04 だけニュース特徴量がある news_df を用意
        news_df = pd.DataFrame(
            {c: 0.0 for c in NEWS_FEATURE_COLS},
            index=[date(2026, 6, 2), date(2026, 6, 4)],
        )
        news_df.loc[date(2026, 6, 2), "macro_sent"] = 0.5
        news_df.loc[date(2026, 6, 4), "macro_sent"] = -0.3
        out = nf.join_news_features(feat, news_df)
        assert len(out) == 5
        vals = dict(zip([d.date() for d in out.index], out["macro_sent"]))
        assert vals[date(2026, 6, 2)] == 0.5
        assert vals[date(2026, 6, 4)] == -0.3
        # ニュースが無い日は0
        assert vals[date(2026, 6, 1)] == 0.0
        assert vals[date(2026, 6, 3)] == 0.0


# ─── load_news_frame（DB往復・後ろ向き移動平均）────────────────────────────────

class TestLoadNewsFrame:
    def test_returns_none_when_no_news(self, isolated_db):
        assert nf.load_news_frame("7203") is None

    def test_rolling_means_are_backward_looking(self, isolated_db):
        # マクロセンチメントを3営業日分入れる
        with get_session() as session:
            for d, s in [(date(2026, 6, 1), 1.0),
                         (date(2026, 6, 2), 0.0),
                         (date(2026, 6, 3), -1.0)]:
                session.add(NewsSentiment(
                    scope="macro", symbol=nf.MACRO_SYMBOL, date=d,
                    sentiment_score=s, article_count=1, source="rss"))
            session.commit()

        frame = nf.load_news_frame("7203")
        assert frame is not None
        assert list(frame.columns) == NEWS_FEATURE_COLS
        # ma3 は当日を含む過去3日平均（後ろ向き）。06-03 では (1+0-1)/3 = 0
        assert frame.loc[date(2026, 6, 3), "macro_sent_ma3"] == pytest.approx(0.0)
        # 06-01 では当日のみ → 1.0
        assert frame.loc[date(2026, 6, 1), "macro_sent_ma3"] == pytest.approx(1.0)
        # 変化量: 06-02 は 0 - 1 = -1
        assert frame.loc[date(2026, 6, 2), "macro_sent_chg"] == pytest.approx(-1.0)


# ─── active_feature_cols / build_features のフラグ挙動 ─────────────────────────

class TestActiveFeatureCols:
    def test_off_returns_base_only(self, news_off):
        assert active_feature_cols() == BASE_TECHNICAL_COLS

    def test_on_adds_news_cols(self, news_on):
        cols = active_feature_cols()
        assert cols == BASE_TECHNICAL_COLS + NEWS_FEATURE_COLS
        assert len(cols) == len(BASE_TECHNICAL_COLS) + len(NEWS_FEATURE_COLS)


class TestBuildFeaturesWithNews:
    def _ohlcv(self, n=120):
        idx = pd.bdate_range("2025-06-01", periods=n)
        rng = np.random.default_rng(0)
        close = 1000 + np.cumsum(rng.normal(0, 5, n))
        return pd.DataFrame({
            "open": close, "high": close + 5, "low": close - 5,
            "close": close, "volume": rng.integers(1e5, 1e6, n).astype(float),
        }, index=idx)

    def test_off_has_no_news_cols(self, news_off):
        out = build_features(self._ohlcv())
        for col in NEWS_FEATURE_COLS:
            assert col not in out.columns
        for col in BASE_TECHNICAL_COLS:
            assert col in out.columns

    def test_on_with_news_df_adds_cols_without_dropping_rows(self, news_on):
        df = self._ohlcv()
        base = build_features(df)  # news 無しの行数（dropna 基準）
        # 一部の日にだけニュース特徴量を与える
        news_df = pd.DataFrame(
            {c: 0.0 for c in NEWS_FEATURE_COLS},
            index=[d.date() for d in base.index[:5]],
        )
        out = build_features(df, news_df=news_df)
        # ニュース列が付与され、かつ行数は base と同じ（ニュース列で dropna しない）
        for col in NEWS_FEATURE_COLS:
            assert col in out.columns
        assert len(out) == len(base)

    def test_on_without_news_df_still_zero_fills(self, news_on):
        # フラグON・news_df=None でも build_training_set 等が active 列を要求するため
        # ニュース列は0で埋まって存在する必要がある
        from src.strategy.news_features import join_news_features
        out = join_news_features(build_features(self._ohlcv()), None)
        for col in NEWS_FEATURE_COLS:
            assert col in out.columns
            assert (out[col] == 0.0).all()
