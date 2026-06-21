"""
流動性フィルタ（レビュー P0-6）。

薄商いの銘柄に新規BUYを入れると、約定しない/不利な価格でしか約定しない/
退出時に売り抜けられない（ギャップ・スリッページ拡大）リスクが高い。
日足OHLCVから直近の平均売買代金（= 終値 × 出来高の平均）を見て、一定額に満たない
銘柄は新規買いを見送る。設定（config.yaml の `liquidity`）で閾値を0にすると無効。

板気配ベースのスプレッド判定はリアルタイム板（live/dry_run の get_board）が必要なため
別途 check_spread() に分離している（ペーパー/バックテストでは板が無いため使わない）。
"""
from typing import Optional

import pandas as pd


def average_turnover(df: pd.DataFrame, window: int = 20) -> float:
    """直近 window 日の平均売買代金（円）= mean(close × volume)。

    データが window 未満の場合は、ある分だけで平均する（0件なら0.0）。
    """
    if df is None or df.empty or "close" not in df or "volume" not in df:
        return 0.0
    tail = df.tail(window)
    turnover = (tail["close"] * tail["volume"]).dropna()
    if turnover.empty:
        return 0.0
    return float(turnover.mean())


def check_liquidity(symbol: str, df: pd.DataFrame, conf: dict) -> tuple[bool, str]:
    """新規BUYの流動性チェック。(ok, reason) を返す。

    conf（config.yaml の `liquidity` セクション）:
      - min_avg_turnover_yen: 直近平均売買代金の下限（円）。0/未設定で無効。
      - avg_window: 平均を取る日数（既定20）。
    """
    min_turnover = float((conf or {}).get("min_avg_turnover_yen", 0) or 0)
    if min_turnover <= 0:
        return True, ""
    window = int((conf or {}).get("avg_window", 20) or 20)
    turnover = average_turnover(df, window)
    if turnover < min_turnover:
        return False, (
            f"流動性不足: {symbol} 直近{window}日平均売買代金 "
            f"{turnover:,.0f}円 < 下限 {min_turnover:,.0f}円"
        )
    return True, ""


def check_spread(board: Optional[dict], conf: dict) -> tuple[bool, str]:
    """板気配のスプレッド（売り気配/買い気配の乖離率）チェック。(ok, reason) を返す。

    live/dry_run で get_board() から得た板を渡す。チェック自体が無効
    （max_spread_ratio<=0）なら常に通す。有効化している場合、板情報や最良気配
    (Buy1/Sell1) が取得できないときは fail-closed でブロックする（取引停止・
    急変で気配が消えている異常時ほど見送るべき場面であり、素通りさせると
    流動性ガードの意図に反するため。レビュー再指摘 Medium）。
      - max_spread_ratio: 許容スプレッド率（(ask-bid)/mid）。0/未設定で無効。
    """
    max_ratio = float((conf or {}).get("max_spread_ratio", 0) or 0)
    if max_ratio <= 0:
        return True, ""
    if not board:
        return False, "スプレッド判定不可: 板情報が取得できません"
    bid = (board.get("Buy1") or {}).get("Price")
    ask = (board.get("Sell1") or {}).get("Price")
    if not bid or not ask or bid <= 0 or ask <= 0:
        return False, "スプレッド判定不可: 気配が取得できません（取引停止/急変の可能性）"
    mid = (bid + ask) / 2
    spread = (ask - bid) / mid if mid > 0 else 0.0
    if spread > max_ratio:
        return False, f"スプレッド過大: {spread:.2%} > 上限 {max_ratio:.2%}"
    return True, ""
