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
    """受渡区分。**売買で指定する値が違う。**

    2026-09-08、現物売りの成行注文が
    `{"Code":100378,"Message":"指定された市場でのお取引はお受けできません。"}`
    で拒否された。売買共通で 2（お預り金）を送っていたが、**現物売は 0（指定なし）**。
    エラー文言が「市場」を指すため受渡区分が原因と気づきにくかった。

    旧実装は `AUTO = 2  # 自動振替` としていたが、2 は「お預り金」で自動振替は 1。
    定数名とコメントの両方が誤っていた。
    """
    UNSPECIFIED = 0   # 指定なし（現物売・信用新規）
    AUTO = 1          # 自動振替
    DEPOSIT = 2       # お預り金（現物買）


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

# kabu API の FundType（預り区分）は**売買で値が違う**。
#
# 2026-09-07、システムとして初めて出した現物買い注文が
# `{"Code":1010004,"Message":"預り区分が未設定です。"}` で拒否された。
# 売買共通で "  "（空白2文字）を送っていたが、これは**現物売**でのみ有効な値。
# 現物買では預り区分を明示しなければならない。
#
# 買いが一度も成立していなかったため、この不整合は2か月以上露見しなかった
# （トレーリングストップの KeyError と同じ「一度も通っていない経路」）。
FUND_TYPE_SELL = "  "   # 現物売: 空白2文字
FUND_TYPE_BUY = "02"    # 現物買: 保護預り（信用取引の代用にはしない運用）

# 後方互換のため名前を残す（従来の参照は売り向けの値だった）
FUND_TYPE_DEFAULT = FUND_TYPE_SELL
