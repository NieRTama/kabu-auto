"""
リスクプロファイル（ハイリスク・ハイリターン ⇔ ローリスク・ローリターン）の切替管理。

config.yaml の `risk_profiles` に複数のプロファイル（例: conservative / balanced /
aggressive）を定義しておき、ダッシュボードから日次の成績を見て切り替えられるようにする。
アクティブなプロファイル名は risk_profile.json に永続化し、プロセス再起動後も維持する。

切替時は、プロファイルのパラメータを config の trading / strategy セクション辞書へ
インプレースで反映する。RiskManager は `cfg.get_section("trading")` の辞書参照を保持し、
strategy 側は generate() ごとに `cfg.get_section("strategy")` を読み直すため、
辞書の中身を書き換えるだけで発注サイズ・損切り幅・売買閾値などに即座に反映される。
"""
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from src.core import config as cfg

# 各パラメータをどちらの config セクションへ反映するか
_STRATEGY_KEYS = {"buy_threshold", "sell_threshold"}
_TRADING_KEYS = {
    "max_position_ratio", "stop_loss_pct", "max_positions",
    "max_sector_ratio", "max_daily_loss",
}

_path: Path = Path("risk_profile.json")
_active: Optional[str] = None


def get_profiles() -> dict:
    """config.yaml で定義された全プロファイルを返す"""
    return cfg.get_section("risk_profiles")


def _default_profile_name() -> str:
    profiles = get_profiles()
    # config の active_risk_profile → balanced → 最初に定義されたもの の順で既定を決める
    configured = cfg.get().get("active_risk_profile")
    if configured and configured in profiles:
        return configured
    if "balanced" in profiles:
        return "balanced"
    return next(iter(profiles), "")


def _apply(params: dict) -> None:
    """プロファイルのパラメータを config の trading / strategy 辞書へ反映する"""
    trading = cfg.get_section("trading")
    strategy = cfg.get_section("strategy")
    for key, value in params.items():
        if key in _STRATEGY_KEYS:
            strategy[key] = value
        elif key in _TRADING_KEYS:
            trading[key] = value
        else:
            logger.warning(f"未知のリスクプロファイル項目を無視: {key}")


def load(path: str = "risk_profile.json") -> str:
    """永続化されたアクティブプロファイルを読み込んで適用し、その名前を返す。

    risk_profiles が未定義の場合は何もしない（後方互換: config の素の値で動作）。
    """
    global _path, _active
    _path = Path(path)
    profiles = get_profiles()
    if not profiles:
        logger.info("risk_profiles 未定義のため config の素の値で動作します")
        _active = None
        return ""

    name = _default_profile_name()
    if _path.exists():
        try:
            with open(_path, encoding="utf-8") as f:
                saved = json.load(f).get("active")
            if saved in profiles:
                name = saved
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"risk_profile.json 読み込み失敗（既定を使用）: {e}")

    _apply(profiles[name])
    _active = name
    logger.info(f"リスクプロファイル適用: {name}")
    return name


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
    _apply(profiles[name])
    _active = name
    with open(_path, "w", encoding="utf-8") as f:
        json.dump({"active": name}, f, ensure_ascii=False, indent=2)
    logger.warning(f"リスクプロファイルを切替: {name} → {profiles[name]}")
    return {"active": name, "params": profiles[name]}
