"""
取引停止スイッチ（kill switch）の永続フラグ管理。

ライブ運用中に「異常を察知したら、まず全ての新規発注を止めて手動確認する」ための
緊急ブレーキ。フラグは JSON ファイル（既定: data/trading_halt.json）へ永続化し、
プロセス再起動を跨いでも停止状態を維持する（=明示的に解除するまで止め続ける）。

設計方針:
- このモジュールは「停止フラグ」という状態だけを持つ純粋な永続化層。
  実際の発注抑止は RiskManager.can_place_order() が is_halted() を見て行い、
  未約定BUYのキャンセルや成行決済といった副作用は OrderManager 側が担う
  （関心の分離。halt.py は client や DB を知らない）。
- 損切り・緊急決済（reason="stop_loss"/"emergency"）はリスクを減らす退出操作なので、
  停止中でも止めない。これは sell_market() の reason バイパスと同じ思想。

保存形式:
{"halted": true, "reason": "...", "halted_at": "2026-06-20T10:00:00"}
"""
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from src.core import clock

_path: Path = Path("data/trading_halt.json")
_state: Optional[dict] = None


def _default_state() -> dict:
    return {"halted": False, "reason": "", "halted_at": None}


def load(path: str = "data/trading_halt.json") -> dict:
    """永続化された停止状態を読み込む。ファイルが無ければ非停止で初期化する。"""
    global _path, _state
    _path = Path(path)
    if _path.exists():
        try:
            with open(_path, encoding="utf-8") as f:
                data = json.load(f)
            _state = {**_default_state(), **data}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"trading_halt.json 読み込み失敗（非停止で起動）: {e}")
            _state = _default_state()
    else:
        _state = _default_state()
    if _state.get("halted"):
        logger.critical(
            f"取引停止スイッチが ON の状態で起動しました（理由: {_state.get('reason')}）。"
            "新規発注は抑止されます。ダッシュボードまたは resume で解除してください。"
        )
    return _state


def _get_state() -> dict:
    if _state is None:
        load()
    return _state


def is_halted() -> bool:
    """新規発注を停止すべきか（kill switch が ON か）を返す。"""
    return bool(_get_state().get("halted"))


def get_state() -> dict:
    """現在の停止状態（halted/reason/halted_at）のコピーを返す。"""
    return dict(_get_state())


def _save() -> None:
    _path.parent.mkdir(parents=True, exist_ok=True)
    with open(_path, "w", encoding="utf-8") as f:
        json.dump(_state, f, ensure_ascii=False, indent=2)


def engage(reason: str = "") -> dict:
    """取引を停止する（kill switch ON）。新規発注がブロックされるようになる。"""
    global _state
    _state = {
        "halted": True,
        "reason": reason or "手動停止",
        "halted_at": clock.now().isoformat(),
    }
    _save()
    logger.critical(f"取引停止スイッチ ON: {_state['reason']}")
    return dict(_state)


def release() -> dict:
    """取引停止を解除する（kill switch OFF）。新規発注が再び可能になる。"""
    global _state
    _state = _default_state()
    _save()
    logger.warning("取引停止スイッチ OFF（取引を再開しました）")
    return dict(_state)
