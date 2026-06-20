"""
取引モードの定義と判定ヘルパー（Phase 3 / 7.3）。

4つのモードを持つ:
  - paper:     仮想取引。実APIへ発注せず、終値で即時シミュレートする（従来どおり）。
  - live:      本番。実際の資金で実APIへ発注する（従来どおり）。
  - dry_run:   実APIから板/余力は読むが、発注（sendorder）は一切しない。
               「ライブと同じ条件で何を発注しようとするか」を安全に観察する検証モード。
               発注予定は DRY_RUN ステータスの Trade として記録され、建玉・余力には影響しない。
  - semi_live: 計画注文をいったん承認キュー（OrderApproval）に積み、ダッシュボードで
               人が承認した分だけ実APIへ発注する半自動モード。
               損切り・緊急決済（退出）は安全のため承認を介さず即時発注する。

モード判定は文字列散在を避けてこのモジュールに集約する。
"""

PAPER = "paper"
LIVE = "live"
DRY_RUN = "dry_run"
SEMI_LIVE = "semi_live"

VALID_MODES = (PAPER, LIVE, DRY_RUN, SEMI_LIVE)


def is_valid(mode: str) -> bool:
    return mode in VALID_MODES


def is_paper(mode: str) -> bool:
    return mode == PAPER


def places_real_orders(mode: str) -> bool:
    """実際の資金で実APIへ発注しうるモードか（live / semi_live）。

    起動時の二重確認（CONFIRM_LIVE_TRADING）・トークン取得失敗時の fail-closed・
    ポート競合時の起動中断は、このいずれかに該当する場合に適用する。
    """
    return mode in (LIVE, SEMI_LIVE)


def reads_broker_api(mode: str) -> bool:
    """実ブローカーAPIを読むモードか（paper 以外すべて）。"""
    return mode != PAPER


def uses_morning_execution(mode: str) -> bool:
    """翌営業日朝の発注ジョブ（morning_execution）を実行するモードか。

    paper は signal_scan 内で当日終値で即時シミュレートするため morning は不要。
    live / dry_run / semi_live は前日シグナルを翌朝に発注（または検証/承認）する。
    """
    return mode != PAPER


def description(mode: str) -> str:
    return {
        PAPER: "ペーパー（仮想取引）",
        LIVE: "ライブ（本番・実発注）",
        DRY_RUN: "ドライラン（API読取のみ・発注しない）",
        SEMI_LIVE: "セミライブ（承認後に実発注）",
    }.get(mode, mode)
