"""口座余力（/wallet/cash の StockAccountWallet）の、買い約定の反映遅れを補正する。

2026-10-29 実測: 買いが約定しても余力はしばらく減らない（10/29 09:05 約定 → 11:29 には反映済み）。
その間は保有に入った株と、まだ余力から引かれていない代金を二重に数え、総資産が過大になり
（10/28 +17.8万円・10/29 +9.2万円）、発注側の実効余力（未約定BUY引当は約定で消える）も過大になる。
約定時点で最後に見た余力と代金を覚え、余力の値が変わるまでその代金を差し引く。
"""
import threading
from typing import Optional

from loguru import logger

from src.core import clock

_lock = threading.Lock()
_last_raw: Optional[float] = None  # 最後に見た余力（APIの生の値）
_unreflected: list = []  # [(約定時点で最後に見た余力, 代金)]
_unreflected_sells: list = []  # [(約定時点で最後に見た余力, 代金, 約定日)]


def note_buy_fill(amount: float) -> None:
    """ライブの買い約定（増分）を記録する。"""
    with _lock:
        # ponytail: 起動後まだ余力を見ていなければ基準が無いので補正しない（起動直後の約定のみ過大のまま）
        if _last_raw is None or amount <= 0:
            return
        _unreflected.append((_last_raw, amount))


def note_sell_fill(amount: float) -> None:
    """ライブの売り約定（増分）を記録する。

    2026-11-06 実測: 15:15 の売り代金 46,860円が 17:30 の余力に入らず、総資産が過少に記録された。
    総資産の計算でだけ足す（unreflected_sell_amount）。発注の余力には足さない（証券会社が
    まだ使えると言っていない金額で発注しない）。
    """
    with _lock:
        if _last_raw is None or amount <= 0:
            return
        _unreflected_sells.append((_last_raw, amount, clock.today()))


def unreflected_sell_amount() -> float:
    """余力にまだ入っていない当日の売り代金の合計。

    ponytail: 余力の値が変われば反映済みとみなす。買いの反映と重なると早めに消え（過少に戻るだけ）、
    約定の同期より先に余力へ入ると当日中は二重に足す（日をまたげば消える）。
    """
    today = clock.today()
    with _lock:
        return sum(a for b, a, d in _unreflected_sells if b == _last_raw and d == today)


def adjust(raw: float) -> float:
    """APIの余力から、まだ反映されていない買い代金を差し引いた値を返す。"""
    global _last_raw
    with _lock:
        if raw != _last_raw:
            if _last_raw is not None:
                logger.info(f"口座余力が変化: {_last_raw:,.0f}円 → {raw:,.0f}円"
                            f"（未反映の買い代金 {len(_unreflected)}件）")
            _last_raw = raw
        # ponytail: 余力の値が変われば反映済みとみなす。売りの入金と同時に変わると早めに外れる。
        # 約定の同期より先に余力が減っていた場合は二重に差し引くが、余力が次に変わるまでの間だけ。
        _unreflected[:] = [(b, a) for b, a in _unreflected if b == raw]
        _unreflected_sells[:] = [e for e in _unreflected_sells if e[0] == raw]
        return raw - sum(a for _, a in _unreflected)
