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
