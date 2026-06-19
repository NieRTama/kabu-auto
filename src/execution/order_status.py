"""注文ステータス定数

注文のライフサイクルを表す文字列定数を1箇所に集約する。
DBの `Trade.status` 列にそのまま格納する（SQLite運用のため Enum 型は使わず文字列）。

経緯: 旧来は PENDING / FILLED / CANCELLED の3値しか無く、
- キャンセルAPIが失敗したのに CANCELLED にしてしまう（CANCEL_FAILED が無い）
- 起動同期でAPIに見つからない注文を即 CANCELLED にしてしまう（UNKNOWN が無い）
- 部分約定・拒否を表現できない（PARTIALLY_FILLED / REJECTED が無い）
といった、約定状態とDBの乖離を招く問題があった。これを正しく表現するため拡張する。
"""

# 発注直後〜約定/キャンセル待ち
PENDING = "PENDING"
# 一部のみ約定（残数は引き続き未約定）
PARTIALLY_FILLED = "PARTIALLY_FILLED"
# 全数約定
FILLED = "FILLED"
# キャンセル要求済み（証券会社側の確定待ち）
CANCEL_REQUESTED = "CANCEL_REQUESTED"
# キャンセル成立
CANCELLED = "CANCELLED"
# キャンセル要求が例外/失敗（注文は生きている可能性があり要人手確認）
CANCEL_FAILED = "CANCEL_FAILED"
# 発注が証券会社に拒否された
REJECTED = "REJECTED"
# 状態不明（起動同期でAPIに見つからない等。誤って CANCELLED 扱いしない）
UNKNOWN = "UNKNOWN"

# 未約定（建玉・余力の引当対象として扱うべき）状態
OPEN_STATUSES = frozenset({PENDING, PARTIALLY_FILLED, CANCEL_REQUESTED})

# 人手確認が必要な異常状態（残っている間は新規発注を抑止する）
UNRESOLVED_STATUSES = frozenset({CANCEL_FAILED, UNKNOWN})
