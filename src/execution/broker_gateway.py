"""
BrokerGateway — ブローカー（kabuステーション）APIとの境界を一手に引き受ける層
（レビュー C3 の OrderManager 分割。BrokerGateway = broker API calls only）。

従来 OrderManager 内に「現物買い指値」「現物売り指値」「成行売り」「逆指値ストップ」の
ほぼ同一な注文dictが4つ散在し、Password/Exchange/AccountType 等のマジック定数も
重複していた。発注ペイロードの組み立てと送信・取消・照会をここへ集約し、
OrderManager は「いつ・何を発注するか（実行判断）」に専念できるようにする。

パスワードは `config.get_api_password()`（環境変数優先）から取得し、値はログに出さない。
"""
from typing import Optional

from loguru import logger

from src.api.kabu_client import KabuClient
from src.core import config as cfg
from src.execution.broker_constants import (
    AccountType, CashMargin, DelivType, Exchange, FrontOrderType,
    FUND_TYPE_DEFAULT, SecurityType, Side,
)


class BrokerGateway:
    def __init__(self, client: KabuClient):
        self._client = client

    def _base_order(self, side: Side, symbol: str, quantity: int) -> dict:
        """全注文に共通のペイロード骨格（パスワード・口座種別・現物自動振替など）。"""
        return {
            "Password": cfg.get_api_password(),
            "Symbol": symbol,
            "Exchange": Exchange.TOSHO.value,
            "SecurityType": SecurityType.STOCK.value,
            "Side": side.value,
            "CashMargin": CashMargin.CASH.value,
            "DelivType": DelivType.AUTO.value,
            "FundType": FUND_TYPE_DEFAULT,
            "AccountType": AccountType.SPECIFIC.value,
            "Qty": quantity,
            "ExpireDay": 0,  # 当日中
        }

    # ─── 発注 ────────────────────────────────────────────────────────────
    def send_buy_limit(self, symbol: str, price: float, quantity: int) -> dict:
        order = self._base_order(Side.BUY, symbol, quantity)
        order.update({"Price": price, "FrontOrderType": FrontOrderType.LIMIT.value})
        return self._client.send_order(order)

    def send_sell_limit(self, symbol: str, price: float, quantity: int) -> dict:
        order = self._base_order(Side.SELL, symbol, quantity)
        order.update({"Price": price, "FrontOrderType": FrontOrderType.LIMIT.value})
        return self._client.send_order(order)

    def send_sell_market(self, symbol: str, quantity: int) -> dict:
        order = self._base_order(Side.SELL, symbol, quantity)
        order.update({"Price": 0, "FrontOrderType": FrontOrderType.MARKET.value})
        return self._client.send_order(order)

    def send_stop_loss_market(self, symbol: str, quantity: int,
                              trigger_price: float) -> dict:
        """逆指値（トリガー後は成行）の売り。PC/アプリ停止中でも証券会社側で発動する保険。"""
        order = self._base_order(Side.SELL, symbol, quantity)
        order.update({
            "Price": 0,
            "FrontOrderType": FrontOrderType.REVERSE_LIMIT.value,
            "ReverseLimitOrder": {
                "TriggerSec": 1,            # 発注銘柄でトリガー
                "TriggerPrice": trigger_price,
                "UnderOver": 1,             # 1=以下（下落してトリガー価格以下で発動）
                "AfterHitOrderType": 1,     # 1=成行（発動後は確実な約定を優先）
                "AfterHitPrice": 0,
            },
        })
        return self._client.send_order(order)

    # ─── 取消・照会 ──────────────────────────────────────────────────────
    def cancel(self, order_id: str) -> dict:
        return self._client.cancel_order(order_id)

    def get_orders(self) -> list:
        return self._client.get_orders()

    def get_positions(self) -> list:
        return self._client.get_positions()

    @staticmethod
    def is_accepted(result: dict) -> bool:
        """send_order の結果が受理（Result==0）かを返す。"""
        return isinstance(result, dict) and result.get("Result") == 0

    @staticmethod
    def order_id_of(result: dict) -> Optional[str]:
        oid = result.get("OrderId") if isinstance(result, dict) else None
        if not oid:
            logger.error(f"発注応答に OrderId がありません: {result}")
        return oid or None
