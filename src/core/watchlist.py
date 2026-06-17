"""
ウォッチリスト（銘柄コード→会社名→セクター）の永続化管理。複数リスト対応。

config.yaml は起動時に一度だけ読み込む静的設定だが、ウォッチリストは
ダッシュボードのGUIから追加・削除・切替できるようにするため、専用のJSONファイル
（デフォルト: watchlists.json）で管理し、変更を即座にディスクへ反映する。
スケジューラの各ジョブは実行ごとに get_codes() 等を呼ぶため、GUIでの変更は
プロセス再起動なしで次回のジョブ実行から反映される。

複数のウォッチリスト（例: "メイン" "高配当株" "グロース株"）を保持できる。
取引（signal_scan/morning_execution/stop_loss_check 等）は常に「アクティブな
1つのリスト」に対して動作する。GUIでリストを切り替えると、次回のジョブ実行から
新しいアクティブリストが使われる。

保存形式:
{
  "active": "メイン",
  "lists": {
    "メイン": [{"code": "7203", "name": "トヨタ自動車", "sector": "Consumer Cyclical"}, ...],
    "高配当株": [...]
  }
}

旧形式（フラットなリストのみの watchlist.json）からの自動移行に対応する。
"""
import json
import unicodedata
from pathlib import Path
from typing import Optional

from loguru import logger

DEFAULT_LIST_NAME = "メイン"

_path: Path = Path("watchlists.json")
_legacy_path: Path = Path("watchlist.json")
_data: Optional[dict] = None


def normalize_code(code: str) -> str:
    """全角数字（IME入力時など）を半角に正規化する（例: '７２０３' → '7203'）"""
    return unicodedata.normalize("NFKC", code.strip())


def _normalize_entries(entries: list[dict]) -> list[dict]:
    return [
        {
            "code": normalize_code(e.get("code", "")),
            "name": e.get("name", ""),
            "sector": e.get("sector", ""),
        }
        for e in entries
        if e.get("code")
    ]


def load(path: str = "watchlists.json", legacy_path: str = "watchlist.json") -> dict:
    """ウォッチリスト群を読み込む。新形式ファイルが無く旧形式ファイルがある場合は自動移行する。"""
    global _data, _path, _legacy_path
    _path = Path(path)
    _legacy_path = Path(legacy_path)

    if _path.exists():
        with open(_path, encoding="utf-8") as f:
            _data = json.load(f)
        _data.setdefault("lists", {})
        if not _data["lists"]:
            _data["lists"][DEFAULT_LIST_NAME] = []
        if _data.get("active") not in _data["lists"]:
            _data["active"] = next(iter(_data["lists"]))
        total = sum(len(v) for v in _data["lists"].values())
        logger.info(
            f"ウォッチリスト読み込み: {len(_data['lists'])}リスト・計{total}件 ({_path})"
        )
        return _data

    if _legacy_path.exists():
        with open(_legacy_path, encoding="utf-8") as f:
            legacy_entries = json.load(f)
        _data = {"active": DEFAULT_LIST_NAME, "lists": {DEFAULT_LIST_NAME: _normalize_entries(legacy_entries)}}
        _save()
        logger.info(f"旧形式の {_legacy_path} を新形式 {_path} に移行しました")
        return _data

    _data = {"active": DEFAULT_LIST_NAME, "lists": {DEFAULT_LIST_NAME: []}}
    logger.warning(f"ウォッチリストファイルなし: {_path}（空リストで起動）")
    _save()
    return _data


def _get_data() -> dict:
    if _data is None:
        load()
    return _data


def _active_entries() -> list[dict]:
    d = _get_data()
    return d["lists"][d["active"]]


# ─── アクティブリストへの操作（既存呼び出し元との互換API） ──────────────────


def get_all() -> list[dict]:
    """アクティブリストの [{"code": "7203", "name": "トヨタ自動車", "sector": "..."}, ...] を返す"""
    return list(_active_entries())


def get_codes() -> list[str]:
    return [e["code"] for e in _active_entries()]


def get_names() -> dict[str, str]:
    return {e["code"]: e.get("name", "") for e in _active_entries()}


def get_sectors() -> dict[str, str]:
    """{"7203": "Consumer Cyclical", ...} のセクターマッピングを返す（セクター集中リスクチェック用）"""
    return {e["code"]: e.get("sector", "") for e in _active_entries()}


def add(code: str, name: str = "") -> list[dict]:
    code = normalize_code(code)
    if not code:
        raise ValueError("銘柄コードを入力してください")
    entries = _active_entries()
    for e in entries:
        if e["code"] == code:
            e["name"] = name or e["name"]
            break
    else:
        entries.append({"code": code, "name": name, "sector": ""})
    _save()
    return entries


def update_sector(code: str, sector: str) -> None:
    """銘柄のセクターを更新する（取得失敗時は空文字を渡せば何もしない）"""
    if not sector:
        return
    code = normalize_code(code)
    entries = _active_entries()
    for e in entries:
        if e["code"] == code:
            e["sector"] = sector
            break
    _save()


def remove(code: str) -> list[dict]:
    code = normalize_code(code)
    d = _get_data()
    d["lists"][d["active"]] = [e for e in _active_entries() if e["code"] != code]
    _save()
    return d["lists"][d["active"]]


# ─── 複数リストの管理 ────────────────────────────────────────────────────


def get_list_names() -> list[str]:
    return list(_get_data()["lists"].keys())


def get_active_list_name() -> str:
    return _get_data()["active"]


def create_list(name: str) -> list[str]:
    """新規の空リストを作成し、アクティブに切り替える。戻り値: 全リスト名"""
    name = name.strip()
    if not name:
        raise ValueError("リスト名を入力してください")
    d = _get_data()
    if name in d["lists"]:
        raise ValueError(f"同名のリストが既に存在します: {name}")
    d["lists"][name] = []
    d["active"] = name
    _save()
    return get_list_names()


def delete_list(name: str) -> dict:
    """リストを削除する。アクティブリストを削除した場合は残りの先頭リストへ自動切替する。
    戻り値: {"active": ..., "names": [...]}"""
    d = _get_data()
    if name not in d["lists"]:
        raise ValueError(f"存在しないリストです: {name}")
    if len(d["lists"]) <= 1:
        raise ValueError("最後の1リストは削除できません")
    del d["lists"][name]
    if d["active"] == name:
        d["active"] = next(iter(d["lists"]))
    _save()
    return {"active": d["active"], "names": get_list_names()}


def switch_active(name: str) -> str:
    """アクティブなウォッチリストを切り替える。次回のジョブ実行から反映される。"""
    d = _get_data()
    if name not in d["lists"]:
        raise ValueError(f"存在しないリストです: {name}")
    d["active"] = name
    _save()
    logger.warning(f"アクティブなウォッチリストを切替: {name}")
    return name


def rename_list(old_name: str, new_name: str) -> str:
    new_name = new_name.strip()
    d = _get_data()
    if old_name not in d["lists"]:
        raise ValueError(f"存在しないリストです: {old_name}")
    if not new_name:
        raise ValueError("新しいリスト名を入力してください")
    if new_name != old_name and new_name in d["lists"]:
        raise ValueError(f"同名のリストが既に存在します: {new_name}")
    d["lists"][new_name] = d["lists"].pop(old_name)
    if d["active"] == old_name:
        d["active"] = new_name
    _save()
    return new_name


# ─── 外部出力（エクスポート）・取込（インポート） ────────────────────────────


def export_list(name: Optional[str] = None) -> dict:
    """指定リスト（省略時はアクティブリスト）を外部保存用の辞書として返す。

    {"name": "メイン", "entries": [{"code": ..., "name": ..., "sector": ...}, ...]}
    ダッシュボードの /api/watchlist/export がこれをファイルダウンロードとして返す。
    """
    d = _get_data()
    target = name or d["active"]
    if target not in d["lists"]:
        raise ValueError(f"存在しないリストです: {target}")
    return {"name": target, "entries": d["lists"][target]}


def import_list(name: str, entries: list[dict], overwrite: bool = False) -> list[str]:
    """エクスポートされたJSON（{"name", "entries"}の entries 部分）からリストを作成する。

    既存と同名の場合、overwrite=True でなければ ValueError。
    取込後はアクティブリストを切り替える（すぐに内容を確認できるようにするため）。
    戻り値: 全リスト名
    """
    name = name.strip()
    if not name:
        raise ValueError("リスト名を入力してください")
    d = _get_data()
    if name in d["lists"] and not overwrite:
        raise ValueError(f"同名のリストが既に存在します: {name}（上書きする場合は overwrite を指定）")
    d["lists"][name] = _normalize_entries(entries)
    d["active"] = name
    _save()
    logger.info(f"ウォッチリストを取込: {name}（{len(d['lists'][name])}件）")
    return get_list_names()


def _save() -> None:
    with open(_path, "w", encoding="utf-8") as f:
        json.dump(_data, f, ensure_ascii=False, indent=2)
    logger.info(f"ウォッチリストを更新: {_data['active']} → {_path}")
