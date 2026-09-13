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
from src.backtest.portfolio import LOT_SIZE
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


@dataclass(frozen=True)
class LiquidityConfig:
    """執行できる量の上限。

    max_volume_share は「その日の出来高に対して自分が占めてよい割合」。
    0 で無制限（従来どおり欲しい数量は必ず買える前提）。
    """
    max_volume_share: float = 0.0


@dataclass(frozen=True)
class FillResult:
    """約定の結果。**未約定と部分約定を明示的に表現する。**

    現行エンジンは「欲しい数量は必ず買える」前提で、薄商い銘柄の執行可能性を
    織り込んでいなかった（spec §8）。filled_quantity < requested_quantity なら
    部分約定、fill が None なら未約定。
    """
    fill: Optional[Fill]
    requested_quantity: int
    filled_quantity: int
    unfilled_reason: Optional[str]


def entry_fill_limited(next_bar: Observation, quantity: int, costs: CostConfig,
                       liquidity: LiquidityConfig, *, volume: int) -> FillResult:
    """出来高の制約を織り込んでエントリーを約定させる。

    約定価格の規約は entry_fill() と同じ（T+1の寄り × (1+slip)）。違うのは
    「いくつ約定できたか」だけ。段階B前半の entry_fill() は挙動を変えない
    （段階B後半の dataset.py がその挙動に依存しているため）。

    部分約定も単元単位に切り捨てる。1単元にも満たなければ未約定とする。
    """
    if liquidity.max_volume_share <= 0:
        return FillResult(
            fill=entry_fill(next_bar, quantity, costs),
            requested_quantity=quantity,
            filled_quantity=quantity,
            unfilled_reason=None,
        )

    allowed = int(volume * liquidity.max_volume_share)
    allowed_lots = (allowed // LOT_SIZE) * LOT_SIZE
    fillable = min(quantity, allowed_lots)

    if fillable < LOT_SIZE:
        return FillResult(
            fill=None,
            requested_quantity=quantity,
            filled_quantity=0,
            unfilled_reason=(
                f"出来高{volume:,}株の{liquidity.max_volume_share:.0%}では"
                f"単元({LOT_SIZE}株)に満たないため約定できません"
            ),
        )

    reason = None
    if fillable < quantity:
        reason = (
            f"出来高{volume:,}株の{liquidity.max_volume_share:.0%}まで"
            f"（{quantity:,}株のうち{fillable:,}株を約定）"
        )
    return FillResult(
        fill=entry_fill(next_bar, fillable, costs),
        requested_quantity=quantity,
        filled_quantity=fillable,
        unfilled_reason=reason,
    )


class VolumeBudget:
    """同じ営業日・同じ銘柄の出来高枠を、買いと売りで共有する台帳。

    出来高の制約は「その日その銘柄で自分が動かせる株数の上限」であって、
    買い専用の枠ではない。買いだけに掛けて売りを無制限にすると、
    900株保有・当日出来高1,000株・参加率上限10%でも全量売れてしまう
    （外部レビューR09）。

    **共有の規約**: 枠は `int(volume * max_volume_share)` を単元へ切り捨てた
    株数。walk-forward の1営業日の中で、操作が起きた順（退出→買い→
    ストップ退出）に消費する。順序を決めておかないと、同じ入力で結果が
    変わる。枠を使い切った後の注文は未約定として理由付きで記録し、
    翌営業日へ持ち越す判断は呼び出し側が行う。

    可変オブジェクトである点に注意。1営業日ぶんを1インスタンスで使い、
    日をまたいで持ち回らない。
    """

    def __init__(self, liquidity: LiquidityConfig):
        self._share = liquidity.max_volume_share
        self._remaining: dict = {}

    def unlimited(self) -> bool:
        return self._share <= 0

    def allow(self, symbol: str, volume: int, quantity: int) -> int:
        """`symbol` で `quantity` 株のうち何株まで動かせるかを返し、枠を減らす。

        単元未満は返さない（0 になる）。
        """
        if self.unlimited():
            return quantity
        if symbol not in self._remaining:
            allowed = int(max(0, volume) * self._share)
            self._remaining[symbol] = (allowed // LOT_SIZE) * LOT_SIZE
        fillable = min(quantity, self._remaining[symbol])
        fillable = (fillable // LOT_SIZE) * LOT_SIZE
        if fillable < LOT_SIZE:
            return 0
        self._remaining[symbol] -= fillable
        return fillable

    def remaining(self, symbol: str) -> Optional[int]:
        """残り枠（無制限なら None）。テストと明細の記録に使う。"""
        if self.unlimited():
            return None
        return self._remaining.get(symbol)


def exit_fill_limited(intent: ExitIntent, bar: Observation,
                      next_bar: Optional[Observation], quantity: int,
                      costs: CostConfig, budget: VolumeBudget, *,
                      symbol: str, volume: int) -> FillResult:
    """出来高の制約を織り込んで退出を約定させる。

    約定価格の規約は exit_fill() と同じ（STOPは基準線とギャップの安いほう、
    MARKETは翌寄り）。違うのは「いくつ売れたか」だけ。
    段階B前半の exit_fill() は挙動を変えない（dataset.py が依存しているため）。

    売れ残りは `requested_quantity - filled_quantity` として返す。
    保有を減らさずに残し、退出意図を翌営業日へ持ち越すのは呼び出し側の仕事。
    損切りを出しても全量は売れない状況を、黙って全量売却にしない
    （外部レビューR09）。
    """
    # ExitIntent は銘柄を持たない（退出の意図だけを表す型）。枠は銘柄ごとなので
    # 呼び出し側から symbol を受け取る。
    fillable = budget.allow(symbol, volume, quantity)
    if fillable < LOT_SIZE:
        return FillResult(
            fill=None,
            requested_quantity=quantity,
            filled_quantity=0,
            unfilled_reason=(
                f"出来高{volume:,}株の枠では単元({LOT_SIZE}株)に満たないため"
                f"売却できません"
            ),
        )

    fill = exit_fill(intent, bar, next_bar, fillable, costs)
    if fill is None:
        return FillResult(
            fill=None, requested_quantity=quantity, filled_quantity=0,
            unfilled_reason="翌営業日の足が無く成行退出を約定できません",
        )

    reason = None
    if fillable < quantity:
        reason = (
            f"出来高の枠まで（{quantity:,}株のうち{fillable:,}株を売却）"
        )
    return FillResult(
        fill=fill,
        requested_quantity=quantity,
        filled_quantity=fillable,
        unfilled_reason=reason,
    )
