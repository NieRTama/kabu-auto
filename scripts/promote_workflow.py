"""候補モデルの学習→評価→昇格を人手で回すCLI。

段階A〜Eで `v2_training.train_v2()`（候補の保存）・
`evaluation.run_evaluation()`（評価記録の保存）・`promotion.promote()`
（現行の切替）は揃ったが、この3つを繋ぐ運用手順がどこにも無かった。
本スクリプトがその手順を1本のCLIにする。

**このスクリプトをスケジューラ（src/core/scheduler.py）へ登録しては
ならない。** 自動昇格を実装しないというのが段階Eの契約
（src/strategy/promotion.py のdocstring）であり、本スクリプトは人間が
手で叩くときだけ動く。判断ロジックもここには置かない。昇格可否は
`promotion.check_promotable()` が唯一の判断者である。

使い方:

    python -m scripts.promote_workflow train
    python -m scripts.promote_workflow evaluate <model_id>
    python -m scripts.promote_workflow promote <model_id> <evaluation_run_id> \\
        --reason "..." --decided-by "..."

共通オプション（--config / --base-dir）はサブコマンド名より**前**に書く:

    python -m scripts.promote_workflow --base-dir models train
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    # Windowsのcp932環境で日本語出力が UnicodeEncodeError で落ちるのを防ぐ
    # （scripts/gen_graph_notes.py と同じ対処）
    sys.stdout.reconfigure(encoding="utf-8")


# ModelMetrics.trigger は String(20)（src/data/database.py:404）。
# 週次の "weekly_schedule" とも legacy の "manual" とも区別できる名前にする。
TRIGGER_MANUAL = "manual_workflow"

EXIT_OK = 0
EXIT_ERROR = 1


def _bootstrap(config_path: str = "config.yaml") -> str:
    """main.py と同じ順序で設定を読み、DBを初期化する。アクティブなプロファイル名を返す。

    **順序を変えてはならない。** `risk_profile_store.load()` は `cfg.load()`
    が読んだ `trading` / `strategy` 節をアクティブプロファイル（現状
    high_risk）の値でプロセス内メモリ上だけ上書きする。この適用を飛ばすと
    `policy.config_from_settings()` が本番と別の閾値を返し、
    `dataset.build_events_multi()` が作るラベルそのものが本番と別物になる
    （docs/kabu-auto-ml-real-data-comparison_20260921.md §1: 適用漏れで
    イベント総数が 5,484件 → 12,033件 と2.2倍ずれ、結論が覆った）。

    `risk_profile.load()` は `_persist()` を呼ばないので `risk_profile.json`
    自体は書き換わらない（読むだけ）。
    """
    from src.core import config as cfg
    from src.core import risk_profile as risk_profile_store
    from src.core import watchlist as watchlist_store
    from src.data import database as db

    cfg.load(config_path)
    watchlist_store.load("watchlists.json")
    profile = risk_profile_store.load("risk_profile.json")
    db.init()
    return profile


# ─── train ────────────────────────────────────────────────────────────────


def _collect_ohlcv(limit: Optional[int]) -> tuple:
    """ウォッチリスト全銘柄のOHLCVを集める。

    `TradingServices.ml_retrain()` の v2 分岐（src/services/trading.py:323-333）
    と同じ集め方にする。200本未満の銘柄は特徴量の助走が足りないので落とし、
    1銘柄の読み込み失敗で全体を止めない。

    戻り値: `({symbol: DataFrame}, スキップ理由の文字列リスト)`
    """
    from src.core import watchlist as watchlist_store
    from src.data import market_data

    ohlcv_by_symbol: dict = {}
    skipped: list = []
    for symbol in watchlist_store.get_all_codes():
        try:
            df = market_data.load_ohlcv(symbol, limit=limit)
        except Exception as e:
            skipped.append(f"{symbol}(読み込み失敗: {e})")
            continue
        if len(df) < 200:
            skipped.append(f"{symbol}({len(df)}本)")
            continue
        ohlcv_by_symbol[symbol] = df
    return ohlcv_by_symbol, skipped


def cmd_train(args) -> int:
    """ウォッチリスト全銘柄の実データで候補モデルを学習・保存する。

    **運用モデルは変更しない。** `train_v2()` は候補を保存して終わる。
    """
    from src.backtest import execution
    from src.core import config as cfg
    from src.strategy import policy
    from src.strategy import v2_training

    profile = _bootstrap(args.config)
    limit = None if args.ohlcv_limit == 0 else args.ohlcv_limit
    ohlcv_by_symbol, skipped = _collect_ohlcv(limit)
    if not ohlcv_by_symbol:
        print("学習できる銘柄がありません（200本以上のOHLCVを持つ銘柄が0件）")
        if skipped:
            print(f"スキップ: {', '.join(skipped)}")
        return EXIT_ERROR

    # policy_conf / costs は _bootstrap() の risk_profile 適用**後**に作る。
    # 順序を逆にすると config.yaml の素の値でラベルを作ってしまう。
    policy_conf = policy.config_from_settings()
    costs = execution.config_from_settings()
    window_sessions = cfg.get_section("backtest").get("retrain_window_sessions", None)

    print(f"リスクプロファイル: {profile}")
    print(f"対象銘柄: {len(ohlcv_by_symbol)}件 / OHLCV本数上限: "
          f"{'全期間' if limit is None else limit}")
    if skipped:
        print(f"スキップ{len(skipped)}件: {', '.join(skipped)}")
    print(f"policy_conf: {policy_conf}")
    print(f"costs: {costs}")
    print(f"window_sessions: {window_sessions}（Noneは拡大窓＝切らない）")

    result = v2_training.train_v2(
        ohlcv_by_symbol,
        policy_conf=policy_conf,
        costs=costs,
        window_sessions=window_sessions,
        trigger=TRIGGER_MANUAL,
        base_dir=args.base_dir,
    )

    print(f"dataset_id: {result.dataset_id}")
    print(f"イベント {result.n_events}件 / 決着 {result.n_resolved}件"
          f"（必要 {v2_training.MIN_RESOLVED_EVENTS}件）")
    if result.model_id is None:
        print(f"候補モデルは作られませんでした: {result.skipped_reason}")
        return EXIT_ERROR

    print(f"正例率: {result.positive_rate:.4f}")
    print(f"候補モデル: {result.model_id}")
    print("運用モデルは変更していません。次は評価です:")
    print(f"  python -m scripts.promote_workflow evaluate {result.model_id}")
    return EXIT_OK


# ─── evaluate ─────────────────────────────────────────────────────────────


def _mean_or_none(series) -> Optional[float]:
    """NaN を除いた平均。全て NaN なら None を返す。

    `roc_auc` / `average_precision` は検証側が片側クラスだと None になる
    （src/strategy/evaluation.py:429-430）。その fold を 0 とみなして平均すると
    成績を過小評価するため、除いて平均する。
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def cmd_evaluate(args) -> int:
    """候補モデルと同じ構成（CurrentLightGBM）でwalk-forward評価し、記録を残す。

    評価対象のイベント表は**学習時に保存されたもの**を読み直す。作り直すと
    設定やデータの更新で別のイベント表になり、候補が学習したものと違う
    ラベル契約で評価してしまう（check_promotable がラベル契約の不一致で拒否する）。
    """
    from src.strategy import dataset as ds
    from src.strategy import evaluation
    from src.strategy import model_store as ms

    _bootstrap(args.config)

    try:
        meta = ms.read_meta(args.model_id, base_dir=args.base_dir)
    except FileNotFoundError as e:
        print(f"候補モデルが見つかりません: {e}")
        return EXIT_ERROR

    path = ds.dataset_path(meta.dataset_id)
    if not path.exists():
        print(f"イベント表がありません: {path}"
              f"（model_id={args.model_id} の dataset_id={meta.dataset_id}）")
        print("先に train を実行してください（イベント表は学習時に保存されます）")
        return EXIT_ERROR
    events = ds.load_events(meta.dataset_id)

    contracts = sorted(set(events["label_contract_id"].dropna().astype(str)))
    print(f"候補: {args.model_id}")
    print(f"dataset_id: {meta.dataset_id} / イベント {len(events)}件")
    print(f"ラベル契約: events={contracts} model={meta.label_contract_id}")
    if meta.label_contract_id not in contracts:
        print("警告: ラベル契約が一致しません。この評価記録では昇格できません")

    # **候補の model_id をキーにする。** check_promotable() は
    # load_prediction_details(evaluation_run_id, model_id=model_id) で
    # この鍵と突き合わせる（src/strategy/promotion.py:106）。"current_lightgbm"
    # のような汎用名で保存すると予測明細が見つからず永久に昇格できない。
    # window_sessions / feature_cols はメタから取る（学習時の条件を再現する）。
    out = evaluation.run_evaluation(
        events,
        model_factories={args.model_id: evaluation.CurrentLightGBM},
        n_splits=args.n_splits,
        window_sessions=meta.training_window_sessions,
        feature_cols=list(meta.feature_cols),
        persist=True,
    )

    summary = out["summary"]
    if summary.empty:
        print("fold結果が0件でした（分割できるイベントがありません）")
        return EXIT_ERROR

    print("")
    print(summary.to_string(index=False))
    print("")
    print(f"AUC平均: {_mean_or_none(summary['roc_auc'])}")
    print(f"Brier平均: {_mean_or_none(summary['brier'])}")
    print(f"Brier vs 定数(平均): {_mean_or_none(summary['brier_vs_constant'])}")
    print("（Brier vs 定数が正なら定数モデルより良い。"
          "AUC 0.5 は予測力が無いのと区別できない）")

    reasons = out["degraded_reasons"]
    if reasons:
        print(f"degraded: {len(reasons)}件。この評価記録では昇格できません")
        for reason in reasons:
            print(f"  - {reason}")

    print("")
    print(f"evaluation_run_id: {out['evaluation_run_id']}")
    print("昇格する場合（昇格するかどうかは評価結果を見てから判断すること）:")
    print(f"  python -m scripts.promote_workflow promote {args.model_id} "
          f"{out['evaluation_run_id']} --reason \"...\" --decided-by \"...\"")
    return EXIT_OK


# ─── CLI ──────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promote_workflow",
        description="候補モデルの学習・評価・昇格（人手運用専用。"
                    "スケジューラへ登録しないこと）")
    parser.add_argument("--config", default="config.yaml",
                        help="設定ファイル（既定: config.yaml）")
    parser.add_argument("--base-dir", default="models",
                        help="モデルの保存先（既定: models）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser(
        "train", help="ウォッチリスト全銘柄で候補モデルを学習・保存する")
    p_train.add_argument(
        "--ohlcv-limit", type=int, default=0,
        help="銘柄あたりのOHLCV本数。0で全期間（既定: 0＝全期間。"
             "500等を指定すると直近N本に絞れるが、週次再学習の"
             "500本既定は意図しない切り詰め＝レビューF07の対象であり、"
             "本ワークフローでは踏襲しない）")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser(
        "evaluate", help="候補モデルをwalk-forwardで評価し評価記録を残す")
    p_eval.add_argument("model_id", help="train が表示した候補のmodel_id")
    p_eval.add_argument(
        "--n-splits", type=int, default=5,
        help="walk-forwardの分割数（既定: 5＝train_v2 と同じfold構造）")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
