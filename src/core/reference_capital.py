"""
モード別の基準資金（損益%算出用）の管理。

paper モードは `trading.paper_initial_capital` を基準に使う（既存の挙動）。
live / dry_run / semi_live は実運用での入出金を自動追跡する手段が無いため、
ユーザーがダッシュボードGUIから明示的に基準資金を設定する（reference_capital.json に永続化）。
未設定（0円）のモードは%算出ができないため、利用側（pnl_report等）は
get(mode) が0を返した場合は%を省略すること。
"""
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from src.core import trading_mode as tm

_path: Path = Path("reference_capital.json")
_values: dict = {}


def _default_values() -> dict:
    return {mode: 0.0 for mode in tm.VALID_MODES if mode != tm.PAPER}


def load(path: str = "reference_capital.json") -> dict:
    """永続化された基準資金設定を読み込む。ファイルが無ければ全モード未設定(0)で初期化する。"""
    global _path, _values
    _path = Path(path)
    _values = _default_values()
    if _path.exists():
        try:
            with open(_path, encoding="utf-8") as f:
                saved = json.load(f)
            for mode, amount in (saved or {}).items():
                if mode in _values and isinstance(amount, (int, float)) and amount >= 0:
                    _values[mode] = float(amount)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"reference_capital.json 読み込み失敗（既定=未設定で起動）: {e}")
    return dict(_values)


def _persist() -> None:
    _path.parent.mkdir(parents=True, exist_ok=True)
    with open(_path, "w", encoding="utf-8") as f:
        json.dump(_values, f, ensure_ascii=False, indent=2)


def get(mode: str) -> float:
    """指定モードの基準資金（円）を返す。未設定/paperは0（=%算出不可の意味で使う側が解釈）。"""
    return _values.get(mode, 0.0)


def get_all() -> dict:
    return dict(_values)


def set_value(mode: str, amount: float) -> dict:
    """指定モードの基準資金を設定する。paperは対象外（config側のpaper_initial_capitalを使うため）。"""
    if mode == tm.PAPER:
        raise ValueError("paper モードは trading.paper_initial_capital を使うため設定不要です")
    if mode not in tm.VALID_MODES:
        raise ValueError(f"未知のモードです: {mode}")
    if amount < 0:
        raise ValueError("基準資金は0以上を指定してください")
    _values[mode] = float(amount)
    _persist()
    logger.info(f"基準資金を設定: {mode} = {amount:,.0f}円")
    return dict(_values)


def percent_basis(mode: str, paper_initial_capital: Optional[float] = None) -> float:
    """%算出に使う基準資金を返す（paperはpaper_initial_capital、他は設定値）。0なら%算出不可。"""
    if mode == tm.PAPER:
        return float(paper_initial_capital or 0.0)
    return get(mode)
