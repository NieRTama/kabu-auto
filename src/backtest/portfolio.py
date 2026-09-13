"""ポートフォリオの状態機械 — 現金と保有だけを持つ。

**DB・ネットワーク・設定ファイルに触らない純粋な状態機械にする。** 設定は
引数で受け取る。こうしておくと、資金競合や上限判定を実データなしで
決定的にテストできる。

判定式（1銘柄上限・既存保有の差し引き・最大保有銘柄数・セクター集中率）は
src/risk/manager.py の実運用実装をそのまま写し取る。式が2つあると、
バックテストで通った条件が運用で弾かれる。

現行 src/backtest/engine.py は単一銘柄の現金と数量しか持たず、複数銘柄の
資金競合を扱えない（レビュー Backtest）。
"""
from dataclasses import dataclass, replace
from datetime import date
from typing import Optional

# 単元株数。src/risk/manager.py:26 の LOT_SIZE と同じ値であること。
LOT_SIZE = 100


class InsufficientCash(ValueError):
    """買付に必要な現金（代金＋手数料）が足りない。

    数量は前日終値で決めるのに約定は翌朝の寄りなので、ギャップアップすると
    必要額が枠を超える。呼び出し側が数量を縮めるか見送るかを決められるよう、
    黙って現金を負にせず例外にする（外部レビューR08）。
    """


@dataclass(frozen=True)
class Holding:
    """1銘柄の保有。

    peak_price と sessions_held は退出ポリシー（src/strategy/policy.py の
    HoldingState）が必要とする状態で、実運用も Position.peak_price として
    永続化している。
    """
    symbol: str
    quantity: int
    avg_cost: float            # 約定価格だけの加重平均（退出ポリシーが見る）
    sector: str
    entry_at: date
    peak_price: float
    sessions_held: int
    # 買付手数料まで含めた1株あたり原価。実現損益の計算はこちらを使う。
    # avg_cost と分けるのは、退出ポリシー（policy.HoldingState）と実運用の
    # risk/manager.py が「約定価格に対する騰落率」で損切り線を引いており、
    # そこへ手数料を混ぜると本番と過去検証で損切り位置がずれるため。
    avg_cost_with_fees: float = 0.0


@dataclass(frozen=True)
class Portfolio:
    """現金・保有・未約定買いの引当。

    reserved は「発注済みだがまだ約定していない買いの金額」。現金から差し引いた
    実効余力で新規の枠を決めるために持つ（実運用 position_budget と同じ考え方）。
    """
    cash: float
    holdings: dict
    reserved: float


def empty_portfolio(cash: float) -> Portfolio:
    return Portfolio(cash=cash, holdings={}, reserved=0.0)


def _price_of(h: Holding, prices: dict) -> float:
    """評価に使う価格。取れない銘柄は取得単価で代用する。

    src/risk/manager.py:518 の `closes.get(p.symbol) or p.avg_cost` と同じ規約。
    """
    return prices.get(h.symbol) or h.avg_cost


def holdings_value(pf: Portfolio, prices: dict) -> float:
    """建玉の時価評価額の合計。"""
    return sum(h.quantity * _price_of(h, prices) for h in pf.holdings.values())


def nav(pf: Portfolio, prices: dict) -> float:
    """総資産（現金＋建玉評価額）。

    reserved は現金の内訳であって総資産を減らさないため引かない。
    """
    return pf.cash + holdings_value(pf, prices)


def held_value(pf: Portfolio, symbol: str, price: float) -> float:
    """この銘柄の現在の保有評価額（現在値ベース）。

    簿価ではなく現在値で評価する。簿価だと値上がり銘柄の実質的な集中度を
    過小評価し、値下がり銘柄では過大評価する（実運用 _held_value と同じ）。
    """
    h = pf.holdings.get(symbol)
    return h.quantity * price if h else 0.0


def apply_buy(pf: Portfolio, symbol: str, quantity: int, price: float,
              sector: str, at: date, commission_pct: float) -> Portfolio:
    """買い約定を反映する。

    手数料は約定代金に対して掛かる。スリッページは execution.py の約定価格に
    織り込み済みなのでここでは扱わない（二重に引かないため）。
    買い増しの場合は取得単価を加重平均し、**保有開始日は動かさない**
    （保有期間の数え方を保つため）。
    """
    if quantity <= 0:
        raise ValueError(f"quantity は正の整数: {quantity}")
    amount = price * quantity
    commission = amount * commission_pct
    outlay = amount + commission
    cash = pf.cash - outlay

    # 現金が足りない買いは成立させない。数量は前日終値で決めるのに約定は
    # 翌朝の寄りなので、ギャップアップすると必要額が枠を超える。ここを
    # 通すと現金が負のままバックテストが進み、NAVも成績も意味を失う
    # （外部レビューR08）。縮小するか諦めるかは呼び出し側の判断なので、
    # ここでは拒否だけする。
    if cash < 0:
        raise InsufficientCash(
            f"現金が足りません: {symbol} {quantity}株 × {price} "
            f"＋手数料{commission:,.1f} = {outlay:,.1f} > 現金{pf.cash:,.1f}")

    existing = pf.holdings.get(symbol)
    if existing is None:
        holding = Holding(
            symbol=symbol, quantity=quantity, avg_cost=price, sector=sector,
            entry_at=at, peak_price=price, sessions_held=0,
            avg_cost_with_fees=outlay / quantity,
        )
    else:
        total_qty = existing.quantity + quantity
        avg_cost = (existing.avg_cost * existing.quantity + amount) / total_qty
        # 手数料込み原価も同じ加重平均で積む
        avg_fees = (existing.avg_cost_with_fees * existing.quantity
                    + outlay) / total_qty
        holding = replace(existing, quantity=total_qty, avg_cost=avg_cost,
                          avg_cost_with_fees=avg_fees)

    holdings = dict(pf.holdings)
    holdings[symbol] = holding
    return replace(pf, cash=cash, holdings=holdings)


def apply_sell(pf: Portfolio, symbol: str, quantity: int, price: float,
               commission_pct: float) -> tuple:
    """売り約定を反映し、(次の状態, 実現損益) を返す。

    実現損益は**往復の手数料控除後**。買付手数料は `avg_cost_with_fees`
    （1株あたり原価）として保有に積んであり、売却数量ぶんを按分して引く。

    買付手数料を現金からだけ引いて原価に含めないと、同値で往復したときに
    現金は往復ぶん減るのに実現損益は片道ぶんしか減らない。10万円ぶんを
    片道0.1%で往復すると、現金 −200円に対し実現損益 −100円になる
    （外部レビューR20）。日次明細・取引明細にもこの値を書くので、
    ここがずれると成績表が現金と合わなくなる。

    部分決済では残りの取得単価（`avg_cost` も `avg_cost_with_fees` も）を
    変えない。按分は数量比で行われる。
    """
    existing = pf.holdings.get(symbol)
    if existing is None or existing.quantity < quantity:
        held = existing.quantity if existing else 0
        raise ValueError(f"保有数量が足りません: {symbol} 保有{held}株 < 売却{quantity}株")

    proceeds = price * quantity
    commission = proceeds * commission_pct
    cash = pf.cash + proceeds - commission
    realized = (price - existing.avg_cost_with_fees) * quantity - commission

    holdings = dict(pf.holdings)
    remaining = existing.quantity - quantity
    if remaining > 0:
        holdings[symbol] = replace(existing, quantity=remaining)
    else:
        del holdings[symbol]
    return replace(pf, cash=cash, holdings=holdings), realized


def max_affordable_quantity(cash: float, price: float, commission_pct: float,
                           lot: int = LOT_SIZE) -> int:
    """現金・手数料・単元を満たす最大数量（0 なら見送り）。"""
    if cash <= 0 or price <= 0:
        return 0
    # qty * price * (1 + commission_pct) <= cash を満たす最大の qty を単元ぶんで求める
    # qty = floor(cash / (price * (1 + commission_pct)) / lot) * lot
    max_qty = int(cash / (price * (1 + commission_pct)))
    return (max_qty // lot) * lot


def advance_session(pf: Portfolio, prices: dict) -> Portfolio:
    """1営業日ぶん保有を進める（ピーク更新と経過営業日数の加算）。

    退出ポリシーが「当日の基準線は前営業日終了時点のピークで固定する」
    規約を持つため、ピークの反映はその日の判定が終わった後に行う
    （src/strategy/policy.py の step() と同じ順序）。
    価格が取れない銘柄のピークは動かさない。
    """
    holdings = {}
    for symbol, h in pf.holdings.items():
        price = prices.get(symbol)
        peak = max(h.peak_price, price) if price else h.peak_price
        holdings[symbol] = replace(h, peak_price=peak,
                                   sessions_held=h.sessions_held + 1)
    return replace(pf, holdings=holdings)


@dataclass(frozen=True)
class SizingConfig:
    """資金配分の制限。実運用は trading 節（risk_profile で上書きされる）から来る。"""
    max_position_ratio: float   # 1銘柄あたり最大資金比率
    max_positions: int          # 最大同時保有銘柄数
    max_sector_ratio: float     # 同一セクター最大集中率


def position_budget(pf: Portfolio, conf: SizingConfig) -> float:
    """1銘柄に投じられる上限額。

    src/risk/manager.py:382-396 と同じ式。「**残余力に対する比率**」であり
    総資産に対する比率ではない。未約定買いの引当を差し引いた実効余力に
    比率を掛ける（未約定中の多重発注で余力を二重に使う事故を防ぐ）。
    """
    available = max(0.0, pf.cash - pf.reserved)
    return available * conf.max_position_ratio


def calc_quantity(pf: Portfolio, symbol: str, price: float,
                  conf: SizingConfig) -> int:
    """購入株数（単元切り捨て）。

    src/risk/manager.py:418-439 と同じ式。1銘柄あたりの上限から**既存保有の
    評価額を差し引いた残り枠**まで買える。差し引かないと、上限いっぱい
    保有した銘柄へさらに満額買い増そうとしてしまう。
    """
    if price <= 0:
        return 0
    budget = position_budget(pf, conf)
    remaining = max(0.0, budget - held_value(pf, symbol, price))
    units = int(remaining / (price * LOT_SIZE))
    return units * LOT_SIZE


def check_max_positions(pf: Portfolio, candidate: Optional[str],
                        conf: SizingConfig) -> tuple:
    """最大保有銘柄数チェック。(通るか, 理由) を返す。

    src/risk/manager.py:441-474 と同じ考え方で、**銘柄の集合**で判定する。
    候補が既に保有中なら集合は増えないので通る。
    """
    after = set(pf.holdings)
    if candidate:
        after = after | {candidate}
    if len(after) > conf.max_positions:
        return False, f"最大保有銘柄数({conf.max_positions})に達しています"
    return True, ""


def check_sector_concentration(pf: Portfolio, sector: str,
                               candidate_notional: float, prices: dict,
                               conf: SizingConfig) -> tuple:
    """同一セクターの集中投資チェック。(通るか, 理由) を返す。

    src/risk/manager.py:477-535 と同じ式。

    分子は「同一セクターの建玉評価額 ＋ これから出す注文の金額」。
    分母は**総資金（現金＋建玉評価額）**。投資済み額を分母にすると、保有が
    少ないほど必ず超過し、**保有ゼロでは比率が常に100%になって最初の1銘柄を
    永久に買えない**（2026-09-04に実際に起きた）。買付余力は建玉に変わるだけで
    総資金は増減しないため、分母に候補金額を足すと二重計上になる。

    比率が上限**以上**なら却下する（実運用と同じく >= で判定）。
    """
    positions_value = 0.0
    same_sector_value = candidate_notional
    for h in pf.holdings.values():
        value = h.quantity * _price_of(h, prices)
        positions_value += value
        if h.sector == sector:
            same_sector_value += value

    total_value = pf.cash + positions_value
    if total_value <= 0:
        return True, ""

    ratio = same_sector_value / total_value
    if ratio >= conf.max_sector_ratio:
        return False, (
            f"セクター集中率が上限({conf.max_sector_ratio:.0%})超: {sector} "
            f"（{same_sector_value:,.0f}円 / 総資金{total_value:,.0f}円 = {ratio:.0%}）"
        )
    return True, ""
