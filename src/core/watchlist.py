"""
ウォッチリスト（銘柄コード→会社名）の永続化管理。

config.yaml は起動時に一度だけ読み込む静的設定だが、ウォッチリストは
ダッシュボードのGUIから追加・削除できるようにするため、専用のJSONファイル
（デフォルト: watchlist.json）で管理し、変更を即座にディスクへ反映する。
スケジューラの各ジョブは実行ごとに get_codes() を呼ぶため、GUIでの変更は
プロセス再起動なしで次回のジョブ実行から反映される。
"""
import json
import unicodedata
from pathlib import Path
from typing import Optional

from loguru import logger

_path: Path = Path("watchlist.json")
_cache: Optional[list[dict]] = None


def normalize_code(code: str) -> str:
    """全角数字（IME入力時など）を半角に正規化する（例: '７２０３' → '7203'）"""
    return unicodedata.normalize("NFKC", code.strip())


def load(path: str = "watchlist.json") -> list[dict]:
    global _cache, _path
    _path = Path(path)
    if not _path.exists():
        _cache = []
        logger.warning(f"ウォッチリストファイルなし: {_path}（空リストで起動）")
        return _cache
    with open(_path, encoding="utf-8") as f:
        _cache = json.load(f)
    logger.info(f"ウォッチリスト読み込み: {len(_cache)}件 ({_path})")
    return _cache


def _get() -> list[dict]:
    if _cache is None:
        load()
    return _cache


def get_all() -> list[dict]:
    """[{"code": "7203", "name": "トヨタ自動車"}, ...] のリストを返す"""
    return list(_get())


def get_codes() -> list[str]:
    return [e["code"] for e in _get()]


def get_names() -> dict[str, str]:
    return {e["code"]: e.get("name", "") for e in _get()}


def add(code: str, name: str = "") -> list[dict]:
    code = normalize_code(code)
    if not code:
        raise ValueError("銘柄コードを入力してください")
    entries = _get()
    for e in entries:
        if e["code"] == code:
            e["name"] = name or e["name"]
            break
    else:
        entries.append({"code": code, "name": name})
    _save(entries)
    return entries


def remove(code: str) -> list[dict]:
    code = normalize_code(code)
    entries = [e for e in _get() if e["code"] != code]
    _save(entries)
    return entries


def _save(entries: list[dict]) -> None:
    global _cache
    _cache = entries
    with open(_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    logger.info(f"ウォッチリストを更新: {len(entries)}件 → {_path}")
