"""
日本語ニュースのセンチメント採点（ローカル FinBERT）。

torch / transformers は重い依存のため requirements-ml.txt に分離し、本モジュール内で
**遅延 import** する。未インストールでも import 自体は成功し、`available()` が False を
返すだけにする（コア機能・既存テストを壊さないため）。

採点は [-1.0, +1.0]（負=ネガティブ / 正=ポジティブ）。モデルは初回ロード時に一度だけ
構築してプロセス内にキャッシュする。さらに本文→スコアの結果も LRU キャッシュして、
同一見出しの再採点を避ける。
"""
from functools import lru_cache
from typing import Optional

from loguru import logger

from src.core import config as cfg

# 遅延ロードしたパイプラインを保持（None=未ロード, False=ロード失敗）
_pipeline = None


def _model_name() -> str:
    return cfg.get_section("sentiment").get(
        "model_name", "christian-phu/bert-finetuned-japanese-sentiment")


def available() -> bool:
    """センチメント採点が利用可能か（torch/transformers が import でき、モデルが構築できるか）。"""
    return _get_pipeline() is not None


def _get_pipeline():
    """transformers のテキスト分類パイプラインを遅延構築してキャッシュする。

    依存が無い／モデル構築に失敗した場合は None を記録して二度と試みない（_pipeline=False）。
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline or None  # False は None として返す
    try:
        from transformers import pipeline  # 遅延 import（torch も連鎖的にロードされる）
    except Exception as e:
        logger.warning(f"センチメント採点を無効化（transformers/torch 未導入）: {e}")
        _pipeline = False
        return None
    try:
        _pipeline = pipeline(
            "sentiment-analysis",
            model=_model_name(),
            top_k=None,  # 全ラベルのスコアを返す（旧 return_all_scores=True）
        )
        logger.info(f"センチメントモデルをロード: {_model_name()}")
    except Exception as e:
        logger.warning(f"センチメントモデルの構築に失敗（採点を無効化）: {e}")
        _pipeline = False
        return None
    return _pipeline or None


def _label_to_polarity(label: str) -> float:
    """モデルのラベル名を符号（+1/0/-1）に対応付ける。

    モデルにより "positive"/"negative"/"neutral" や "LABEL_0/1/2"、星評価などがあるため、
    代表的な表記を吸収する。未知ラベルは中立(0)扱い。
    """
    s = label.strip().lower()
    if s in ("positive", "pos", "label_2", "good", "bullish"):
        return 1.0
    if s in ("negative", "neg", "label_0", "bad", "bearish"):
        return -1.0
    if s in ("neutral", "neu", "label_1"):
        return 0.0
    # "5 stars".."1 star" のような星評価
    if "star" in s:
        try:
            stars = int(s.split()[0])
            return (stars - 3) / 2.0  # 1→-1, 3→0, 5→+1
        except (ValueError, IndexError):
            return 0.0
    return 0.0


@lru_cache(maxsize=4096)
def score_text(text: str) -> float:
    """1テキストのセンチメントを [-1, 1] で返す。採点不能なら 0.0（中立）。

    返値 = Σ(ラベル極性 × そのラベル確率)。例: positive 0.8 / negative 0.2 → 0.6。
    """
    if not text or not text.strip():
        return 0.0
    pipe = _get_pipeline()
    if pipe is None:
        return 0.0
    try:
        # top_k=None のとき、結果は [[{label, score}, ...]] の入れ子で返る
        result = pipe(text[:512])  # モデルの最大長に安全側で切る
        scores = result[0] if result and isinstance(result[0], list) else result
        polarity = sum(_label_to_polarity(d["label"]) * float(d["score"]) for d in scores)
        return max(-1.0, min(1.0, polarity))
    except Exception as e:
        logger.warning(f"センチメント採点失敗: {e}")
        return 0.0


def score_texts(texts: list[str]) -> list[float]:
    """複数テキストをまとめて採点する（キャッシュ越しに1件ずつ）。"""
    return [score_text(t) for t in texts]


def reset_cache() -> None:
    """テスト用: パイプラインと採点キャッシュをリセットする。"""
    global _pipeline
    _pipeline = None
    score_text.cache_clear()
