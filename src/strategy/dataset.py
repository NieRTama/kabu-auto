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

from src.strategy.indicators import FEATURE_COLS

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
