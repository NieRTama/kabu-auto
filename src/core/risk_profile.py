"""
リスクプロファイル管理（ハイリスク・ハイリターン ⇔ ローリスク・ローリターン ＋ カスタム）。

config.yaml の `risk_profiles` に組み込みプロファイル（"low_risk" / "high_risk"）を定義し、
さらにユーザーが任意名のカスタムプロファイルを risk_profile.json に作成・clone・取込できる。
アクティブなプロファイルのパラメータは config の trading / strategy セクションへ反映され、
発注サイズ・損切り幅・売買閾値などに即座に反映される（既定はローリスク側＝安全側）。

設計（Phase 4 / 6.2-6.8・11.2・11.6）:
- 組み込み名（low_risk/high_risk）は維持し、カスタムで上書き・削除はできない（11.2）。
- 全プロファイルは保存・適用前に共有バリデーション（RiskProfileSchema）を通す。
  無効値はエラーで弾き、丸めない（11.6。誤設定のまま起動・運用しないため）。
- 切替・カスタム変更は履歴として risk_profile.json に記録する（変更履歴）。

risk_profile.json 形式:
{
  "active": "low_risk",
  "custom": {"my_profile": {max_position_ratio: ..., ...}},
  "history": [{"at": "...", "action": "switch", "from": "low_risk", "to": "high_risk"}]
}
旧形式（{"active": "..."} のみ）からの読み込みにも対応する。
"""
import json
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator

from src.core import clock
from src.core import config as cfg

# 各パラメータをどちらの config セクションへ反映するか
_STRATEGY_KEYS = {"buy_threshold", "sell_threshold"}
_TRADING_KEYS = {
    "max_position_ratio", "stop_loss_pct", "max_positions",
    "max_sector_ratio", "max_daily_loss",
}
_ALL_KEYS = _STRATEGY_KEYS | _TRADING_KEYS

_HISTORY_LIMIT = 100  # 履歴の保持件数上限（古いものから切り捨て）

_path: Path = Path("risk_profile.json")
_active: Optional[str] = None
_custom: dict = {}
_history: list = []


class RiskProfileSchema(BaseModel):
    """リスクプロファイルの共有バリデーション（11.6: 無効値は弾く・丸めない）。

    全7項目が必須。範囲外・型不一致・未知キーはすべてエラーにする（extra="forbid"）。
    """
    model_config = ConfigDict(extra="forbid")

    max_position_ratio: float
    stop_loss_pct: float
    max_positions: int
    max_sector_ratio: float
    max_daily_loss: float
    buy_threshold: float
    sell_threshold: float

    @field_validator("max_position_ratio", "max_sector_ratio")
    @classmethod
    def _ratio_range(cls, v, info):
        if not (0 < v <= 1):
            raise ValueError(f"{info.field_name} は 0 < x ≦ 1 で指定してください: {v}")
        return v

    @field_validator("stop_loss_pct")
    @classmethod
    def _stop_loss_negative(cls, v):
        if not (-1 < v < 0):
            raise ValueError(f"stop_loss_pct は -1 < x < 0（負の値）で指定してください: {v}")
        return v

    @field_validator("max_positions")
    @classmethod
    def _positions_positive(cls, v):
        if v < 1:
            raise ValueError(f"max_positions は1以上で指定してください: {v}")
        return v

    @field_validator("max_daily_loss")
    @classmethod
    def _daily_loss_nonneg(cls, v):
        if v < 0:
            raise ValueError(f"max_daily_loss は0以上で指定してください: {v}")
        return v

    @field_validator("buy_threshold")
    @classmethod
    def _buy_positive(cls, v):
        if v <= 0:
            raise ValueError(f"buy_threshold は正の値で指定してください: {v}")
        return v

    @field_validator("sell_threshold")
    @classmethod
    def _sell_negative(cls, v):
        if v >= 0:
            raise ValueError(f"sell_threshold は負の値で指定してください: {v}")
        return v


def validate_profile(params: dict) -> dict:
    """プロファイルのパラメータを検証して正規化した辞書を返す。無効なら ValueError。"""
    try:
        model = RiskProfileSchema(**params)
    except Exception as e:
        raise ValueError(f"リスクプロファイルの値が不正です: {e}")
    return model.model_dump()


# ─── プロファイルの取得 ────────────────────────────────────────────────────


def _builtin_profiles() -> dict:
    """config.yaml で定義された組み込みプロファイル（low_risk/high_risk）。"""
    return cfg.get_section("risk_profiles")


def get_profiles() -> dict:
    """組み込み＋カスタムの全プロファイルを返す（カスタムは組み込みを上書きしない）。"""
    merged = dict(_builtin_profiles())
    for name, params in _custom.items():
        if name not in merged:  # 組み込み名は保護（11.2）
            merged[name] = params
    return merged


def is_builtin(name: str) -> bool:
    return name in _builtin_profiles()


def _default_profile_name() -> str:
    profiles = get_profiles()
    configured = cfg.get().get("active_risk_profile")
    if configured and configured in profiles:
        return configured
    return next(iter(profiles), "")


def _apply(params: dict) -> None:
    """プロファイルのパラメータを config の trading / strategy 辞書へ反映する。"""
    trading = cfg.get_section("trading")
    strategy = cfg.get_section("strategy")
    for key, value in params.items():
        if key in _STRATEGY_KEYS:
            strategy[key] = value
        elif key in _TRADING_KEYS:
            trading[key] = value
        else:
            logger.warning(f"未知のリスクプロファイル項目を無視: {key}")


# ─── 読み込み・永続化 ──────────────────────────────────────────────────────


def load(path: str = "risk_profile.json") -> str:
    """永続化されたアクティブプロファイル・カスタム・履歴を読み込んで適用し、名前を返す。

    risk_profiles が未定義の場合は何もしない（後方互換: config の素の値で動作）。
    アクティブプロファイルは適用前に検証し、無効なら起動を失敗させる（11.6 fail-closed）。
    """
    global _path, _active, _custom, _history
    _path = Path(path)
    _custom = {}
    _history = []
    profiles = _builtin_profiles()
    if not profiles:
        logger.info("risk_profiles 未定義のため config の素の値で動作します")
        _active = None
        return ""

    if _path.exists():
        try:
            with open(_path, encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved.get("custom"), dict):
                _custom = saved["custom"]
            if isinstance(saved.get("history"), list):
                _history = saved["history"]
            saved_active = saved.get("active")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"risk_profile.json 読み込み失敗（既定を使用）: {e}")
            saved_active = None
    else:
        saved_active = None

    available = get_profiles()
    name = saved_active if saved_active in available else _default_profile_name()

    params = available.get(name, {})
    # 適用前に検証（無効なプロファイルで運用に入らない）
    validate_profile(params)
    _apply(params)
    _active = name
    logger.info(f"リスクプロファイル適用: {name}")
    return name


def _persist() -> None:
    _path.parent.mkdir(parents=True, exist_ok=True)
    with open(_path, "w", encoding="utf-8") as f:
        json.dump(
            {"active": _active, "custom": _custom, "history": _history},
            f, ensure_ascii=False, indent=2,
        )


def _record_history(action: str, **fields) -> None:
    _history.append({"at": clock.now().isoformat(), "action": action, **fields})
    if len(_history) > _HISTORY_LIMIT:
        del _history[:-_HISTORY_LIMIT]


# ─── アクティブ切替 ────────────────────────────────────────────────────────


def get_active() -> Optional[str]:
    return _active


def set_active(name: str) -> dict:
    """アクティブプロファイルを切り替えて適用・永続化する。

    戻り値: {"active": name, "params": {...}}
    """
    global _active
    profiles = get_profiles()
    if name not in profiles:
        raise ValueError(f"未知のリスクプロファイル: {name}")
    params = validate_profile(profiles[name])
    _apply(params)
    prev, _active = _active, name
    _record_history("switch", **{"from": prev, "to": name})
    _persist()
    logger.warning(f"リスクプロファイルを切替: {name} → {params}")
    return {"active": name, "params": params}


# ─── カスタムプロファイルの作成・編集・削除・clone ─────────────────────────


def _check_custom_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("プロファイル名を入力してください")
    if len(name) > 50:
        raise ValueError("プロファイル名が長すぎます（50文字以内）")
    if is_builtin(name):
        raise ValueError(f"組み込みプロファイル名は使用できません: {name}")
    return name


def create_custom(name: str, params: dict) -> dict:
    """新しいカスタムプロファイルを作成する（既存と同名なら ValueError）。"""
    name = _check_custom_name(name)
    if name in _custom:
        raise ValueError(f"同名のカスタムプロファイルが既に存在します: {name}")
    validated = validate_profile(params)
    _custom[name] = validated
    _record_history("create", to=name)
    _persist()
    logger.info(f"カスタムリスクプロファイルを作成: {name}")
    return {"name": name, "params": validated}


def update_custom(name: str, params: dict) -> dict:
    """既存のカスタムプロファイルを更新する（組み込みは編集不可）。

    更新対象がアクティブな場合は即座に再適用する。
    """
    if name not in _custom:
        raise ValueError(f"編集可能なカスタムプロファイルが見つかりません: {name}")
    validated = validate_profile(params)
    _custom[name] = validated
    if _active == name:
        _apply(validated)
    _record_history("update", to=name)
    _persist()
    logger.info(f"カスタムリスクプロファイルを更新: {name}")
    return {"name": name, "params": validated}


def delete_custom(name: str) -> list[str]:
    """カスタムプロファイルを削除する（組み込み・アクティブは削除不可）。戻り値: 残りカスタム名。"""
    if name not in _custom:
        raise ValueError(f"カスタムプロファイルが見つかりません: {name}")
    if _active == name:
        raise ValueError("アクティブなプロファイルは削除できません。先に別のプロファイルへ切り替えてください")
    del _custom[name]
    _record_history("delete", to=name)
    _persist()
    logger.info(f"カスタムリスクプロファイルを削除: {name}")
    return list(_custom.keys())


def clone(src_name: str, new_name: str) -> dict:
    """既存プロファイル（組み込み/カスタム問わず）を複製して新しいカスタムを作る。"""
    profiles = get_profiles()
    if src_name not in profiles:
        raise ValueError(f"複製元プロファイルが見つかりません: {src_name}")
    result = create_custom(new_name, dict(profiles[src_name]))
    _record_history("clone", **{"from": src_name, "to": new_name})
    _persist()
    return result


# ─── エクスポート・インポート・比較 ────────────────────────────────────────


def export_profile(name: str) -> dict:
    """プロファイルを外部保存用の辞書として返す（YAML/JSONダウンロード用）。"""
    profiles = get_profiles()
    if name not in profiles:
        raise ValueError(f"プロファイルが見つかりません: {name}")
    return {"name": name, "params": profiles[name]}


def import_profile(name: str, params: dict, overwrite: bool = False) -> dict:
    """外部のプロファイル定義を取り込んでカスタムとして保存する。

    組み込み名では取り込めない（11.2）。同名カスタムは overwrite=True でのみ上書き。
    """
    name = _check_custom_name(name)
    if name in _custom and not overwrite:
        raise ValueError(f"同名のカスタムプロファイルが既に存在します: {name}（上書きは overwrite 指定）")
    validated = validate_profile(params)
    _custom[name] = validated
    _record_history("import", to=name)
    _persist()
    logger.info(f"カスタムリスクプロファイルを取込: {name}")
    return {"name": name, "params": validated}


def compare(name_a: str, name_b: str) -> dict:
    """2つのプロファイルを比較し、差分を返す。

    戻り値: {"a","b","params_a","params_b","diff": {key: [a_val, b_val], ...}}
    """
    profiles = get_profiles()
    for n in (name_a, name_b):
        if n not in profiles:
            raise ValueError(f"プロファイルが見つかりません: {n}")
    pa, pb = profiles[name_a], profiles[name_b]
    diff = {}
    for key in sorted(_ALL_KEYS):
        va, vb = pa.get(key), pb.get(key)
        if va != vb:
            diff[key] = [va, vb]
    return {"a": name_a, "b": name_b, "params_a": pa, "params_b": pb, "diff": diff}


def get_history(limit: int = 50) -> list:
    """変更履歴を新しい順に返す。"""
    return list(reversed(_history[-limit:]))
