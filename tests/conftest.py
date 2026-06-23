"""pytest 設定・共通フィクスチャ"""
import importlib.util

import pytest


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# 重いMLスタック（requirements-ml.txt）の有無。CI（requirements.txt のみ導入）では
# 未導入のため、これらに依存するテストは skip する。
HAS_TORCH = _has_module("torch") and _has_module("transformers")
HAS_FEEDPARSER = _has_module("feedparser")

requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="torch/transformers 未導入（requirements-ml.txt）")
requires_feedparser = pytest.mark.skipif(
    not HAS_FEEDPARSER, reason="feedparser 未導入（requirements-ml.txt）")
