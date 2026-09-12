"""イベント表の生成 — 1行が1つの売買候補に対応する。

学習・検証・バックテストが共有する「何を予測するのか」をこのモジュールが確定させる。
候補は**バージョン固定のルールだけ**で生成し、MLの予測は使わない。scores に
今回学習するML自身を含めると、ラベルを作るためにそのMLの予測が必要になる循環が
生じるため（spec §6）。MLは当面エントリー候補の選別に限定する。

退出は src/strategy/policy.py に、約定とコストの控除は
src/backtest/execution.py に委ねる。本モジュールはそれらを組み立てて
「この候補はコスト控除後に得だったか」を確定させるだけで、学習も評価も行わない
（分割・重み再計算・学習は段階Cの validation.py が担当する）。

**未成熟・未約定・欠損は別ステータスにする。** 従来の labeling.py は
max_holding に満たない末尾のイベントにも最終リターンの符号でラベルを付けており、
未来1行しかないサンプルにラベル1が付いていた（レビューF05）。
"""
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest import execution
from src.core import config as cfg
from src.strategy import policy
from src.strategy.indicators import FEATURE_COLS, build_feature_frame
from src.strategy.signal import compute_rule_score

# ─── 状態区分 ──────────────────────────────────────────────────────────
# 未成熟・未約定・欠損を「損失0」に潰さないため、ラベルとは別に持つ。
STATUS_RESOLVED = "resolved"                  # 退出まで決着した（ラベルを付ける唯一の状態）
STATUS_IMMATURE = "immature"                  # 足が尽きて決着しなかった
STATUS_UNFILLED = "unfilled"                  # 約定できなかった（翌営業日の足が無い）
STATUS_INVALID_FEATURES = "invalid_features"  # 特徴量が揃っていない

# ─── 再現用の版 ────────────────────────────────────────────────────────
# 定義を変えたらここを上げる。dataset_id（内容ハッシュ）と併せて、
# 「コード変更による差」と「データ改訂による差」を分離するために持つ。
FEATURE_VERSION = "f1"                      # indicators.FEATURE_COLS の定義
STRATEGY_VERSION = "rule_only_v1"           # 候補生成はルールのみ
EXECUTION_MODEL_VERSION = "t1_open_v1"      # Tの引けで判断しT+1の寄りで執行
LABEL_VERSION = "l1"                        # トリプルバリアの定義そのもの

# 純収益率は数量に依存しない（買い代金・売り代金・手数料が同じ係数で伸縮し、
# 比を取ると約分される）。Fill の型を満たすための名目値として持つ。
NOMINAL_QUANTITY = 100

META_COLUMNS = [
    "event_id",
    "label_contract_id",        # ラベルの作られ方（退出ポリシー＋コスト＋版）
    "symbol",
    "decision_at",              # 売買判断の時点（Tの引け）
    "feature_as_of",            # 特徴量に使った情報の最終時点
    "entry_at",                 # 想定した執行時点（T+1の寄り）
    "label_end_at",             # ラベル確定に使った最後の時点
    "status",
    "label",                    # status==resolved のときのみ 0/1。他は NaN
    "net_return",               # コスト控除後の純収益率
    "exit_reason",
    "entry_price",
    "exit_price",
    "sessions_held",
    "rule_score",
    "feature_version",
    "strategy_version",
    "execution_model_version",
]

EVENT_COLUMNS = META_COLUMNS + list(FEATURE_COLS)


def make_event_id(symbol: str, decision_at: date) -> str:
    """イベントの一意識別子。

    銘柄と判断セッションの組で一意になる。内容ハッシュ（dataset_id）を
    安定させるため、乱数やタイムスタンプを混ぜず決定的に作る。

    **これは「どの判断を指すか」であって「ラベルがどう作られたか」ではない。**
    同じ銘柄・同じ日でも、退出ポリシーやコストを変えれば実績ラベルは別物になる。
    実績の保存キーには make_label_contract_id() と組で使うこと
    （外部レビューR07）。
    """
    return f"{symbol}:{pd.Timestamp(decision_at).strftime('%Y%m%d')}"


def make_label_contract_id(policy_conf: "policy.PolicyConfig",
                           costs: "execution.CostConfig", *,
                           peak_basis: str = "previous") -> str:
    """ラベルの作られ方を一意に決める契約ID（SHA256の先頭12桁）。

    同じ (symbol, decision_at) でも、損切り幅・トレーリング・売り閾値・
    最大保有期間・スリッページ・手数料・執行モデルが違えば `label` と
    `net_return` は別の値になる。実績を `event_id` だけで保存すると、
    別のコストで評価をやり直したときに**過去runの実績を上書きしてしまい、
    保存済みの指標が後から変わる**（外部レビューR07）。

    `dataset_id`（内容ハッシュ）ではなくパラメータのハッシュにする理由:
    dataset_id は銘柄を1つ増やしただけでも変わるため、shadow運用のように
    データが継ぎ足されていく用途では過去の実績と結び付かなくなる。
    契約IDは「ラベルの定義」だけを表し、対象データの増減では変わらない。

    順序が確定した JSON にしてからハッシュする。dict の反復順や float の
    既定表現に依存すると、同じ設定で違うIDが出る。
    """
    payload = {
        "label_version": LABEL_VERSION,
        "execution_model_version": EXECUTION_MODEL_VERSION,
        "peak_basis": peak_basis,
        "stop_loss_pct": round(float(policy_conf.stop_loss_pct), 8),
        "breakeven_trigger_pct": round(float(policy_conf.breakeven_trigger_pct), 8),
        "trailing_stop_pct": round(float(policy_conf.trailing_stop_pct), 8),
        "sell_threshold": round(float(policy_conf.sell_threshold), 8),
        "max_holding_sessions": int(policy_conf.max_holding_sessions),
        "slippage_pct": round(float(costs.slippage_pct), 8),
        "commission_pct": round(float(costs.commission_pct), 8),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def rule_scores(feat: pd.DataFrame) -> pd.Series:
    """各セッションのルールスコアを返す（feat の index を保持する）。

    signal.compute_rule_score() は df.iloc[-1] と df.iloc[-2] しか見ないため、
    2行の窓を渡せばそのセッションのスコアが求まる。ルールを二重に書かず
    既存実装をそのまま再利用する。前日が無い先頭セッションは NaN。
    """
    values = [float("nan")]
    for i in range(1, len(feat)):
        values.append(compute_rule_score(feat.iloc[i - 1:i + 1]))
    return pd.Series(values, index=feat.index, name="rule_score")


def find_candidates(feat: pd.DataFrame, buy_threshold: float) -> list[int]:
    """買い候補となるセッションの位置インデックスを返す。

    **MLの予測を使わない。** ラベルを作るためにMLの予測が要る循環を断つため、
    候補生成はバージョン固定のルールだけで行う（spec §6）。
    特徴量が揃っていないセッションは候補にしない。
    """
    scores = rule_scores(feat)
    valid = feat["feature_valid"] if "feature_valid" in feat.columns else pd.Series(
        True, index=feat.index)
    out = []
    for i in range(len(feat)):
        score = scores.iloc[i]
        if pd.isna(score) or not bool(valid.iloc[i]):
            continue
        if score >= buy_threshold:
            out.append(i)
    return out


@dataclass(frozen=True)
class EventOutcome:
    """1候補をシミュレートした結果。

    label は status == STATUS_RESOLVED のときだけ 0/1 を持つ。
    未成熟・未約定・欠損では None にして、損失0として扱わない（spec §6）。
    """
    status: str
    entry_at: Optional[date]
    label_end_at: Optional[date]
    label: Optional[int]
    net_return: Optional[float]
    exit_reason: Optional[str]
    entry_price: Optional[float]
    exit_price: Optional[float]
    sessions_held: int


def _observation(feat: pd.DataFrame, i: int) -> policy.Observation:
    row = feat.iloc[i]
    return policy.Observation(
        session=feat.index[i].date(),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        score=None,  # 退出の売りシグナルは段階Dで結線する（ラベルは執行と退出だけで決める）
    )


def _unresolved(status: str, entry_at: Optional[date] = None,
                sessions_held: int = 0) -> EventOutcome:
    return EventOutcome(
        status=status, entry_at=entry_at, label_end_at=None, label=None,
        net_return=None, exit_reason=None, entry_price=None, exit_price=None,
        sessions_held=sessions_held,
    )


def simulate_event(feat: pd.DataFrame, i: int,
                   policy_conf: policy.PolicyConfig,
                   costs: execution.CostConfig, *,
                   peak_basis: str = policy.PEAK_BASIS_PREVIOUS) -> EventOutcome:
    """位置 i のセッションを判断時点として、執行から退出までをシミュレートする。

    並びは「i の引けで判断 → i+1 の寄りで約定 → i+1 以降の足へ退出ポリシーを
    逐次適用」。エントリー当日（i+1）も退出判定の対象にする。実運用も朝に買った
    その日のうちに損切り監視が走るため。

    ラベルは**コスト控除後の純収益が正なら1、それ以外は0**とする。
    0は0側に含める。決着しなかった候補にはラベルを付けず、別ステータスにする。
    """
    if "feature_valid" in feat.columns and not bool(feat["feature_valid"].iloc[i]):
        return _unresolved(STATUS_INVALID_FEATURES)

    entry_idx = i + 1
    if entry_idx >= len(feat):
        return _unresolved(STATUS_UNFILLED)

    entry_bar = _observation(feat, entry_idx)
    entry = execution.entry_fill(entry_bar, NOMINAL_QUANTITY, costs)

    state = policy.HoldingState(
        symbol=str(feat.attrs.get("symbol", "")),
        entry_at=entry.at,
        avg_cost=entry.price,
        quantity=entry.quantity,
        peak_price=entry.price,
        sessions_held=0,
    )
    observations = [_observation(feat, k) for k in range(entry_idx, len(feat))]
    final, intent, _ = policy.run_session_series(
        state, observations, policy_conf, peak_basis=peak_basis)

    if intent is None:
        # 足が尽きて決着しなかった。未来1行しかないサンプルにラベルを付けない
        return _unresolved(STATUS_IMMATURE, entry_at=entry.at,
                           sessions_held=final.sessions_held)

    # 意図が出たのは entry_idx から数えて sessions_held 本目の足
    intent_idx = entry_idx + final.sessions_held - 1
    next_idx = intent_idx + 1
    exit_bar = _observation(feat, intent_idx)
    next_bar = _observation(feat, next_idx) if next_idx < len(feat) else None

    fill = execution.exit_fill(intent, exit_bar, next_bar, NOMINAL_QUANTITY, costs)
    if fill is None:
        return _unresolved(STATUS_UNFILLED, entry_at=entry.at,
                           sessions_held=final.sessions_held)

    ret = execution.net_return(entry, fill, costs)
    return EventOutcome(
        status=STATUS_RESOLVED,
        entry_at=entry.at,
        label_end_at=fill.at,
        label=1 if ret > 0 else 0,
        net_return=ret,
        exit_reason=intent.reason,
        entry_price=entry.price,
        exit_price=fill.price,
        sessions_held=final.sessions_held,
    )


def build_events(symbol: str, ohlcv: pd.DataFrame,
                 policy_conf: policy.PolicyConfig,
                 costs: execution.CostConfig, *,
                 buy_threshold: Optional[float] = None,
                 peak_basis: str = policy.PEAK_BASIS_PREVIOUS) -> pd.DataFrame:
    """単一銘柄のOHLCVからイベント表を作る（1行=1候補）。

    ohlcv は単一銘柄の時系列（日付インデックス・重複なし・昇順）であること。
    複数銘柄は build_events_multi() を使う。
    """
    if buy_threshold is None:
        buy_threshold = cfg.get_section("strategy").get("buy_threshold", 0.25)

    feat = build_feature_frame(ohlcv)
    feat.attrs["symbol"] = symbol
    scores = rule_scores(feat)
    candidates = find_candidates(feat, buy_threshold)

    # ラベル契約IDは銘柄・日付に依存しないので、ループの外で一度だけ作る
    label_contract_id = make_label_contract_id(
        policy_conf, costs, peak_basis=peak_basis)

    rows = []
    for i in candidates:
        outcome = simulate_event(feat, i, policy_conf, costs, peak_basis=peak_basis)
        decision_at = feat.index[i].date()
        row = {
            "event_id": make_event_id(symbol, decision_at),
            "label_contract_id": label_contract_id,
            "symbol": symbol,
            "decision_at": decision_at,
            # 特徴量は判断セッションまでの情報だけで作られるため同じ時点になる
            "feature_as_of": decision_at,
            "entry_at": outcome.entry_at,
            "label_end_at": outcome.label_end_at,
            "status": outcome.status,
            "label": outcome.label,
            "net_return": outcome.net_return,
            "exit_reason": outcome.exit_reason,
            "entry_price": outcome.entry_price,
            "exit_price": outcome.exit_price,
            "sessions_held": outcome.sessions_held,
            "rule_score": float(scores.iloc[i]),
            "feature_version": FEATURE_VERSION,
            "strategy_version": STRATEGY_VERSION,
            "execution_model_version": EXECUTION_MODEL_VERSION,
        }
        for col in FEATURE_COLS:
            row[col] = float(feat[col].iloc[i])
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)[EVENT_COLUMNS]


def build_events_multi(ohlcv_by_symbol: dict, policy_conf: policy.PolicyConfig,
                       costs: execution.CostConfig, **kwargs) -> pd.DataFrame:
    """複数銘柄のイベント表を作って連結する。

    **銘柄ごとに build_events() を呼んでから連結する。** 生のOHLCVを連結して
    から処理すると、移動平均・RSI・退出判定が銘柄境界をまたいで壊れる
    （ml_model.train_multi() と同じ理由）。
    データ不足などで失敗した銘柄はスキップし、他の銘柄を止めない。
    """
    parts = []
    for symbol, ohlcv in ohlcv_by_symbol.items():
        try:
            events = build_events(symbol, ohlcv, policy_conf, costs, **kwargs)
        except Exception as e:
            logger.warning(f"イベント表の生成をスキップ: {symbol} {e}")
            continue
        if len(events) > 0:
            parts.append(events)

    if not parts:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.concat(parts, ignore_index=True)[EVENT_COLUMNS]
