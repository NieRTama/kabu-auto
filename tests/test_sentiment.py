"""
sentiment.py のテスト。

torch/transformers 未導入の環境（CI＝requirements.txt のみ）でも、import が成功し
採点が中立(0.0)へグレースフルに縮退することを検証する（コア機能を壊さない）。
torch が有る環境では実モデルでの採点も軽く確認する。
"""
import pytest

from src.strategy import sentiment
from tests.conftest import HAS_TORCH


@pytest.fixture(autouse=True)
def _reset():
    sentiment.reset_cache()
    yield
    sentiment.reset_cache()


class TestGracefulDegradation:
    def test_empty_text_is_neutral(self):
        assert sentiment.score_text("") == 0.0
        assert sentiment.score_text("   ") == 0.0

    @pytest.mark.skipif(HAS_TORCH, reason="torch 未導入時の縮退を検証するテスト")
    def test_unavailable_without_torch(self):
        assert sentiment.available() is False
        # モデルが無くても例外を投げず中立を返す
        assert sentiment.score_text("日経平均が大幅高で取引を終えた") == 0.0
        assert sentiment.score_texts(["a", "b"]) == [0.0, 0.0]

    def test_label_to_polarity_mapping(self):
        assert sentiment._label_to_polarity("positive") == 1.0
        assert sentiment._label_to_polarity("NEGATIVE") == -1.0
        assert sentiment._label_to_polarity("neutral") == 0.0
        assert sentiment._label_to_polarity("LABEL_2") == 1.0
        assert sentiment._label_to_polarity("5 stars") == 1.0
        assert sentiment._label_to_polarity("1 star") == -1.0
        assert sentiment._label_to_polarity("???") == 0.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch/transformers 未導入")
class TestRealModel:
    def test_scores_in_range(self):
        if not sentiment.available():
            pytest.skip("モデル重みをロードできない環境")
        s = sentiment.score_text("業績好調で株価は大幅に上昇した")
        assert -1.0 <= s <= 1.0
