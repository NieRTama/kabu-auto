"""過去検証の執行アダプタ — 意図を約定へ変換する。

policy.py は「いつ・なぜ退出するか」までを決め、約定価格には触れない。
本モジュールが、その意図と利用可能な日足から「いくらで約定したか」を
仮定として決め、スリッページと手数料を**一元的に**控除する。

Tの引けで判断し T+1 の寄りで約定する。現行 src/backtest/engine.py:111-188 は
同じ終値でスコア生成と約定を行っており、実運用（引け後スキャン→翌朝発注）と
乖離していた（レビューF04）。

足の型は policy.Observation を再利用する（session/open/high/low/close で
同じ形のため、型を二重に定義しない）。
"""
from dataclasses import dataclass
from datetime import date
from typing import Optional

from src.core import config as cfg
from src.strategy.policy import ExitIntent, Observation, ORDER_TYPE_STOP, ORDER_TYPE_MARKET


@dataclass(frozen=True)
class Fill:
    """約定。price は**コスト控除前**ではなくスリッページ込みの約定価格。

    手数料はここには含めない（数量に対する金額として net_return で控除する）。
    スリッページを二重に引かないため、控除の責務はこのモジュールに閉じる。
    """
    at: date
    price: float
    quantity: int
    reason: str


@dataclass(frozen=True)
class CostConfig:
    """約定コストの仮定。"""
    slippage_pct: float    # backtest.slippage_pct（片道）
    commission_pct: float  # backtest.commission_pct（片道）


def config_from_settings() -> CostConfig:
    """config.yaml の backtest 節から CostConfig を作る（唯一の読み出し口）。"""
    conf = cfg.get_section("backtest")
    return CostConfig(
        slippage_pct=conf.get("slippage_pct", 0.0),
        commission_pct=conf.get("commission_pct", 0.0),
    )


def buy_fill_price(price: float, costs: CostConfig) -> float:
    """買い約定価格（スリッページ分だけ不利＝高く約定する）。

    既存 src/backtest/engine.py:276-278 と同じ規約（小数2桁で丸める）。
    """
    return round(price * (1 + costs.slippage_pct), 2)


def sell_fill_price(price: float, costs: CostConfig) -> float:
    """売り約定価格（スリッページ分だけ不利＝安く約定する）。

    既存 src/backtest/engine.py:281-283 と同じ規約（小数2桁で丸める）。
    """
    return round(price * (1 - costs.slippage_pct), 2)


def entry_fill(next_bar: Observation, quantity: int, costs: CostConfig) -> Fill:
    """Tの引けで決めた買いを、T+1 の寄りで約定させる。

    判断した日の終値では約定しない。実運用は引け後にスキャンし翌朝に発注する
    ため、シグナルを作れる時点と約定できる時点が違う（レビューF04）。
    """
    return Fill(
        at=next_bar.session,
        price=buy_fill_price(next_bar.open, costs),
        quantity=quantity,
        reason="ENTRY",
    )


def exit_fill(intent: ExitIntent, bar: Observation, next_bar: Optional[Observation],
              quantity: int, costs: CostConfig) -> Optional[Fill]:
    """退出意図を約定へ変換する。約定できなければ None を返す。

    STOP（基準線への到達）:
        通常は基準線で約定したとみなす。ただし**寄りが既に基準線を割っていたら
        min(open, trigger_price) で約定する**。現行 engine.py:119-123 は
        基準線ちょうどで約定できる前提になっており、ギャップダウンに楽観的。

    MARKET（売りシグナル・満了）:
        翌営業日の寄りで成行約定する。満了日の終値で判断して同じ終値で約定する
        経路を作らない（レビューF04）。翌足が無ければ未約定。
    """
    if intent.order_type == ORDER_TYPE_STOP:
        if intent.trigger_price is None:
            raise ValueError("STOP の意図には trigger_price が必要です")
        raw = min(bar.open, intent.trigger_price)
        return Fill(
            at=bar.session,
            price=sell_fill_price(raw, costs),
            quantity=quantity,
            reason=intent.reason,
        )

    if intent.order_type != ORDER_TYPE_MARKET:
        raise ValueError(f"未知の order_type です: {intent.order_type}")
    if next_bar is None:
        # 足が尽きた＝この意図は約定していない。呼び出し側は未成熟として扱う
        return None
    return Fill(
        at=next_bar.session,
        price=sell_fill_price(next_bar.open, costs),
        quantity=quantity,
        reason=intent.reason,
    )


def net_return(entry: Fill, exit_: Fill, costs: CostConfig) -> float:
    """コスト控除後の純収益率。

    スリッページは entry_fill / exit_fill の約定価格に既に織り込まれているため、
    ここで二重に引かない。手数料だけを売買それぞれの約定代金に対して控除する。
    控除の責務をこのモジュールに閉じることで、期待値の式（spec §8）の末尾で
    コストを再度引く二重計上を防ぐ。

    entry と exit の quantity が一致しない場合（部分決済）は、呼び出し側が
    按分・分割の責任を持つべきなので ValueError にする。buy_amount が0以下
    （価格または数量が不正）の場合も、本物の0%リターンと区別するため
    0.0を返さず例外にする。
    """
    if entry.quantity != exit_.quantity:
        raise ValueError(
            f"entry と exit の数量が一致しません: {entry.quantity} != {exit_.quantity}"
        )
    buy_amount = entry.price * entry.quantity
    if buy_amount <= 0:
        raise ValueError(f"buy_amount が不正です: price={entry.price}, quantity={entry.quantity}")
    sell_amount = exit_.price * exit_.quantity
    commission = (buy_amount + sell_amount) * costs.commission_pct
    return (sell_amount - buy_amount - commission) / buy_amount
