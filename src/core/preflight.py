"""
ライブ起動前のプリフライトチェック（Phase 3 / 4.4）。

「予測精度より、システムが口座状態を誤読して誤った前提で発注することが最大のリスク」
という思想に基づき、実発注モード（live / semi_live）を起動する前に、口座状態を
正しく把握できる前提が揃っているかを確認する。1つでも致命的(critical)な前提が崩れて
いれば起動を中断し、誤発注を未然に防ぐ。

チェック項目:
  1. API疎通: /orders・/positions・/wallet が取得できるか（トークンは取得済み前提）
  2. 未解決注文: DBに UNKNOWN / CANCEL_FAILED が残っていないか（実口座と乖離した状態）
  3. ポート: ダッシュボードのポートが空いているか
  4. モード↔エンドポイント整合: 実発注モードが検証用ポート(18081)を向いていないか
  5. 取引停止スイッチ: kill switch が ON のまま起動していないか（警告）

戻り値はチェック結果の集合で、main 側が level=="critical" の失敗有無で起動可否を判断する。
このモジュールは sys.exit せず判定結果だけを返す（テスト容易性・関心の分離のため）。
"""
from typing import Optional

from loguru import logger
from sqlalchemy import func, select

from src.core import halt
from src.data.database import Trade, get_session
from src.execution import order_status as st

CRITICAL = "critical"
WARNING = "warning"


def _check(name: str, ok: bool, level: str, detail: str = "") -> dict:
    return {"name": name, "ok": ok, "level": level, "detail": detail}


def _check_api_connectivity(client) -> list[dict]:
    """/orders・/positions・/wallet の疎通を確認する。"""
    results = []
    for name, fn in (
        ("API疎通: /orders", client.get_orders),
        ("API疎通: /positions", client.get_positions),
        ("API疎通: /wallet", client.get_wallet),
    ):
        try:
            fn()
            results.append(_check(name, True, CRITICAL))
        except Exception as e:
            results.append(_check(name, False, CRITICAL, f"取得失敗: {e}"))
    return results


def _check_unresolved_orders() -> dict:
    """DBに人手確認待ちの異常注文（UNKNOWN/CANCEL_FAILED）が無いことを確認する。"""
    try:
        with get_session() as session:
            count = session.scalar(
                select(func.count(Trade.id)).where(
                    Trade.status.in_(tuple(st.UNRESOLVED_STATUSES))
                )
            ) or 0
    except Exception as e:
        return _check("未解決注文の確認", False, CRITICAL, f"DB照会失敗: {e}")
    if count:
        return _check(
            "未解決注文の確認", False, CRITICAL,
            f"UNKNOWN/CANCEL_FAILED が {count} 件残っています。"
            "/orders・/positions を確認し解消してから起動してください",
        )
    return _check("未解決注文の確認", True, CRITICAL)


def _check_port(dash_host: str, dash_port: int) -> dict:
    from src.core.netutil import is_port_available
    if is_port_available(dash_host, dash_port):
        return _check("ダッシュボードポート空き", True, CRITICAL)
    return _check(
        "ダッシュボードポート空き", False, CRITICAL,
        f"{dash_host}:{dash_port} が使用中です",
    )


def _check_endpoint_mode_consistency(mode: str, base_url: str) -> dict:
    """実発注モードが kabuステーションの検証用ポート(18081)を向いていないか確認する。

    kabuステーションは本番=18080・検証=18081 でポートが分かれている。実発注モードで
    検証ポートを向いていると「本番のつもりが検証環境」「検証のつもりが本番」という
    取り違えが起きるため、実発注モードでは 18081 を critical 失敗とする。
    """
    if "18081" in (base_url or ""):
        return _check(
            "モード↔エンドポイント整合", False, CRITICAL,
            f"実発注モード({mode})が検証用ポート(18081)を向いています: {base_url}。"
            "本番は 18080 を使用してください",
        )
    return _check("モード↔エンドポイント整合", True, CRITICAL)


def _check_halt() -> dict:
    if halt.is_halted():
        state = halt.get_state()
        return _check(
            "取引停止スイッチ", False, WARNING,
            f"kill switch が ON のまま起動しています（理由: {state.get('reason')}）。"
            "新規発注は抑止されます",
        )
    return _check("取引停止スイッチ", True, WARNING)


def run_preflight(client, mode: str, *, base_url: str,
                  dash_host: str, dash_port: int) -> dict:
    """プリフライトチェックを実行し {"ok": bool, "checks": [...]} を返す。

    ok は level==CRITICAL の失敗が1つも無ければ True。
    """
    checks: list[dict] = []
    checks.extend(_check_api_connectivity(client))
    checks.append(_check_unresolved_orders())
    checks.append(_check_port(dash_host, dash_port))
    checks.append(_check_endpoint_mode_consistency(mode, base_url))
    checks.append(_check_halt())

    ok = all(c["ok"] for c in checks if c["level"] == CRITICAL)
    return {"ok": ok, "checks": checks}


def log_results(result: dict) -> None:
    """プリフライト結果をログに出力する。"""
    logger.info("─── プリフライトチェック ───")
    for c in result["checks"]:
        if c["ok"]:
            logger.info(f"  [OK] {c['name']}")
        elif c["level"] == CRITICAL:
            logger.critical(f"  [NG] {c['name']}: {c['detail']}")
        else:
            logger.warning(f"  [警告] {c['name']}: {c['detail']}")
