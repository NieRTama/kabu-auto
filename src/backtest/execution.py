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

from src.core import config as cfg
from src.strategy.policy import Observation


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
