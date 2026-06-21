"""
kabuステーション発注APIのマジック定数を型付き列挙にまとめる（レビュー P2-1）。

従来 OrderManager 内の4つの注文dictに直接 1 / 4 / "2" / 20 等が散在しており、
意味が読み取りづらく取り違えのリスクがあった。ここに集約し、BrokerGateway から参照する。
"""
from enum import Enum, IntEnum


class Exchange(IntEnum):
    TOSHO = 1  # 東証


class SecurityType(IntEnum):
    STOCK = 1  # 株式


class Side(str, Enum):
    """kabu APIは売買を文字列 "1"/"2" で表す。"""
    SELL = "1"
    BUY = "2"


class CashMargin(IntEnum):
    CASH = 1  # 現物


class DelivType(IntEnum):
    AUTO = 2  # 自動振替


class AccountType(IntEnum):
    SPECIFIC = 4  # 特定口座


class FrontOrderType(IntEnum):
    MARKET = 10        # 成行
    LIMIT = 20         # 指値
    REVERSE_LIMIT = 30  # 逆指値


class OrderState(IntEnum):
    DONE = 5  # 終了（約定・取消・失効で確定）


# 東証の通常の売買単位（単元株）
BOARD_LOT = 100

# kabu API の FundType 既定（現物・自動）。空白2文字。
FUND_TYPE_DEFAULT = "  "
