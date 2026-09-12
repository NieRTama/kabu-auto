"""退出ポリシー — 売買規則の唯一の実装。

ラベル生成・バックテスト・実運用の3者がこのモジュールだけを使って
「いつ・なぜ退出するか」を決める。規則が1箇所にあるため、退出条件を
変えると3者が同時に追随する。従来はこの規則が labeling.py・engine.py・
risk/manager.py の3箇所に別々の実装として散らばっており、片方を直しても
他方とずれていた（レビューF04・F05）。

**約定価格はここで決めない。** 退出の「意図」（発動理由と基準価格）までを返し、
実際にいくらで約定したかは src/backtest/execution.py（過去検証）または
実運用アダプタが決める。判断と約定を同じ関数が返すと、将来の日足をまとめて
受け取る関数が過去検証専用になり、その時点までの足しか渡せない実運用の
インターフェースにならないため。

退出条件は src/risk/manager.py:evaluate_exit() と同じ構造にする。
**固定利確線は置かない**（運用に存在しないため）。
"""
from dataclasses import dataclass, replace
from datetime import date
from typing import Optional

from src.core import config as cfg

# 退出理由
STOP_LINE = "STOP_LINE"      # ブレークイーブン発動前の損切り線に到達
TRAILING = "TRAILING"        # 発動後の（引き上がった）基準線に到達
SIGNAL_SELL = "SIGNAL_SELL"  # ルール由来の売りシグナル
TIME_LIMIT = "TIME_LIMIT"    # 最大保有営業日数に到達

# 日足内でピークをいつ反映するか
PEAK_BASIS_PREVIOUS = "previous"          # 既定（悲観）: 前営業日終了時点のピークで当日の線を固定
PEAK_BASIS_SAME_SESSION = "same_session"  # 楽観: 当日の高値を即座に反映


@dataclass(frozen=True)
class HoldingState:
    """保有の状態。

    取得単価と数量だけでは、追加購入・部分決済・再起動を跨いだピーク価格や
    ストップ発動状態を表せない。実運用は src/data/database.py:162 の
    Position.peak_price としてまさにこの値を永続化している。

    `armed`（ブレークイーブン発動済みか）はフィールドとして持たない。
    実運用（risk/manager.py:574）も永続化せず peak_price から毎回導出しており、
    保存すると乖離しうるため is_armed() で導出する。
    """
    symbol: str
    entry_at: date
    avg_cost: float
    quantity: int
    peak_price: float      # 保有開始以降の最高値
    sessions_held: int     # 経過営業日数


@dataclass(frozen=True)
class Observation:
    """その時点で観測できた1営業日ぶんの情報。

    将来の足は含めない。実運用ではその日の板から、過去検証では日足から作る。
    score は combined_score（売りシグナル判定用）。ラベル生成で売りシグナルを
    考慮しない場合は None を渡す。
    """
    session: date
    open: float
    high: float
    low: float
    close: float
    score: Optional[float] = None


@dataclass(frozen=True)
class ExitIntent:
    """退出の意図。約定価格ではない。

    trigger_price は発動の基準となった価格であり、実際にいくらで約定したかは
    execution.py が決める（ギャップダウン時は基準価格では約定できない）。
    order_type は "STOP"（当日中に基準価格へ到達したとみなす）または
    "MARKET"（翌営業日の寄りで成行）。
    """
    reason: str
    trigger_price: Optional[float]
    order_type: str


@dataclass(frozen=True)
class PolicyConfig:
    """退出判定に使う設定。

    policy.py は config.yaml を直接読まない（純粋関数に保つため）。
    設定からの構築は config_from_settings() に閉じる。
    """
    stop_loss_pct: float           # trading.stop_loss_pct（負の値）
    breakeven_trigger_pct: float   # trading.breakeven_trigger_pct（0で無効）
    trailing_stop_pct: float       # trading.trailing_stop_pct（0で無効）
    sell_threshold: float          # strategy.sell_threshold
    max_holding_sessions: int      # strategy.tb_max_holding


def config_from_settings() -> PolicyConfig:
    """config.yaml から PolicyConfig を作る（唯一の読み出し口）。

    退出まわりの値は trading 節、判定閾値と保有期間は strategy 節にある。
    どちらから読んだ値かをここで固定し、呼び出し側が節を意識しないようにする。
    """
    trading = cfg.get_section("trading")
    strategy = cfg.get_section("strategy")
    return PolicyConfig(
        stop_loss_pct=trading.get("stop_loss_pct", -0.05),
        breakeven_trigger_pct=trading.get("breakeven_trigger_pct", 0.0),
        trailing_stop_pct=trading.get("trailing_stop_pct", 0.0),
        sell_threshold=strategy.get("sell_threshold", -0.25),
        max_holding_sessions=strategy.get("tb_max_holding", 10),
    )


def is_armed(state: HoldingState, conf: PolicyConfig) -> bool:
    """ブレークイーブンが発動済みか（peak_price から導出する）。"""
    if conf.breakeven_trigger_pct <= 0 or state.avg_cost <= 0:
        return False
    peak_gain_pct = (state.peak_price - state.avg_cost) / state.avg_cost
    return peak_gain_pct >= conf.breakeven_trigger_pct


def stop_line(state: HoldingState, conf: PolicyConfig) -> float:
    """この時点の基準線（損切り・ブレークイーブン・トレーリングの最も高い方）。

    src/risk/manager.py:572-578 と同じ式:
      1. 基準線 = 取得単価 × (1 + stop_loss_pct)
      2. ピーク時の含み益率が breakeven_trigger_pct 以上なら取得単価まで引き上げ
      3. さらに trailing_stop_pct > 0 なら ピーク×(1-trailing) とも比べて高い方
    """
    line = state.avg_cost * (1 + conf.stop_loss_pct)
    if is_armed(state, conf):
        line = max(line, state.avg_cost)
        if conf.trailing_stop_pct > 0:
            line = max(line, state.peak_price * (1 - conf.trailing_stop_pct))
    return line
