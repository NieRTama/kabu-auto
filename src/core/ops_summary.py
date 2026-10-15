"""当日の運用概要テキストを組み立てる（Discord の summary コマンドと scripts/daily_ops_report.py が共用）。

full=True はファイル保存用（Claude Code 等での後日分析向けに省略しない）。
full=False は Discord 返信用（1900字に収まるよう ALERT・根拠を切り詰める）。
"""
import re
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path

from sqlalchemy import func, select

REPO = Path(__file__).resolve().parents[2]

# 認証が切れた／戻ったことを示す ALERT の文言（alerts.alert の呼び出し側と揃える）
_AUTH_DOWN = ("再ログインが必要", "ログイン待ち")
_AUTH_UP = ("認証が回復", "接続しました")


def _log_lines(name: str) -> list[str]:
    p = REPO / "log" / name
    return p.read_text(encoding="utf-8", errors="replace").splitlines() if p.exists() else []


def _alerts(warn: list[str]) -> list[tuple[str, str]]:
    """[(HH:MM:SS, 本文)]。"""
    out = []
    for line in warn:
        m = re.match(r"^\S+ (\d\d:\d\d:\d\d)\S* .*?\[ALERT\] (.*)$", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _error_kinds(warn: list[str]) -> Counter:
    """ERROR/CRITICAL を「発生元 + 文言（数字・コードを伏せる）」で分類する。"""
    c = Counter()
    for line in warn:
        if "| ERROR" not in line and "| CRITICAL" not in line:
            continue
        m = re.match(r"^.*?\| \w+\s*\| ([\w.]+):\S* - (.*)$", line)
        src, msg = (m.group(1).rsplit(".", 1)[-1], m.group(2)) if m else ("?", line)
        msg = re.sub(r"\d+", "#", msg)[:70]
        c[f"{src}: {msg}"] += 1
    return c


def _auth_downtime(alerts: list[tuple[str, str]]) -> list[str]:
    """認証切れ〜回復の区間。回復しないまま終わった区間は「継続中」とする。"""
    spans, start = [], None
    for t, msg in alerts:
        if start is None and any(k in msg for k in _AUTH_DOWN):
            start = t
        elif start is not None and any(k in msg for k in _AUTH_UP):
            mins = (datetime.strptime(t, "%H:%M:%S") - datetime.strptime(start, "%H:%M:%S")).seconds // 60
            spans.append(f"{start[:5]}〜{t[:5]}（{mins}分）")
            start = None
    if start is not None:
        spans.append(f"{start[:5]}〜（継続中）")
    return spans


def _db_sections(today: date, full: bool) -> list[str]:
    from src.data.database import OrderIntent, Position, Signal, Trade, get_session
    from src.data.market_data import latest_closes

    day0, day1 = datetime.combine(today, time.min), datetime.combine(today, time.max)
    out = []
    with get_session() as s:
        intents = s.scalars(select(OrderIntent).where(OrderIntent.created_at.between(day0, day1))
                            .order_by(OrderIntent.id)).all()
        trades = {t.intent_id: t for t in s.scalars(select(Trade).where(
            Trade.intent_id.in_([i.id for i in intents]))).all()} if intents else {}
        out.append(f"▼発注意図 {len(intents)}件（source/状態/約定/損益/根拠）")
        for i in intents:
            t = trades.get(i.id)
            fill = (f"約定{t.filled_quantity or 0}/{t.quantity}@{t.filled_price or 0:,.0f}"
                    f" {t.status}" if t else "Tradeなし")
            pnl = f" 損益{t.pnl:+,.0f}" if t and t.pnl is not None else ""
            why = (i.rationale or "").replace("\n", " ")
            why = why if full else why[:50]
            out.append(f"{i.created_at:%H:%M} {i.side} {i.symbol} {i.target_quantity}株 "
                       f"[{i.source}/{i.status}] {fill}{pnl} | {why}")
        if not intents:
            out.append("(なし)")

        # 本日執行対象 = 今日より前の最新バッチ（前夜生成→当日朝に執行する運用のため）
        batch = s.scalar(select(func.max(func.date(Signal.generated_at)))
                         .where(Signal.generated_at < day0))
        if batch:
            sigs = s.scalars(select(Signal).where(func.date(Signal.generated_at) == batch)).all()
            cnt = Counter(x.action for x in sigs)
            done = {i.symbol for i in intents}
            buys = sorted((x for x in sigs if x.action == "BUY"),
                          key=lambda x: x.combined_score or 0, reverse=True)
            out += ["", f"▼本日執行対象シグナル（{batch}生成）BUY{cnt['BUY']} SELL{cnt['SELL']} HOLD{cnt['HOLD']}"]
            for x in buys[: None if full else 5]:
                mark = "発注済" if x.symbol in done else "未発注"
                out.append(f"{x.symbol} score={x.combined_score:.3f}"
                           f"(rule={x.rule_score or 0:.3f}, ml={x.ml_score or 0:.3f}) {mark}")
            missed = [x.symbol for x in buys if x.symbol not in done]
            if missed:
                out.append(f"BUYなのに未発注: {', '.join(missed)}（資金・リスク上限・認証切れ等）")

        pos = s.scalars(select(Position).where(Position.quantity > 0)).all()
        closes = latest_closes([p.symbol for p in pos]) if pos else {}
        out += ["", f"▼保有 {len(pos)}銘柄（集計時点・終値基準）"]
        for p in pos:
            c = closes.get(p.symbol)
            u = f"{(c - p.avg_cost) * p.quantity:+,.0f}円 ({c / p.avg_cost - 1:+.1%})" if c else "価格なし"
            peak = f" 高値{p.peak_price:,.0f}" if p.peak_price else ""
            out.append(f"{p.symbol} {p.quantity}株 取得{p.avg_cost:,.0f} 終値{c or 0:,.0f}{peak} 含み{u} "
                       f"建{p.opened_at:%m/%d}")
    return out


def build(today: date, full: bool = True) -> str:
    from src.core import code_version
    from src.core import config as cfg
    from src.core import reference_capital as rc
    from src.core import trading_mode as tm
    from src.core.pnl_report import _format_period, build_report

    t = cfg.get_section("trading")
    mode = t.get("mode", "paper")
    basis = rc.percent_basis(mode, paper_initial_capital=float(t.get("paper_initial_capital", 500_000)))
    r = build_report(basis, today)

    warn = _log_lines(f"kabu_auto_warning_{today}.log")
    alerts = _alerts(warn)
    errors = _error_kinds(warn)
    starts = [l[11:16] for l in _log_lines(f"kabu_auto_{today}.log") if "kabu-auto 起動完了" in l]
    halted = '"halted": true' in (REPO / "data" / "trading_halt.json").read_text(encoding="utf-8")
    down = _auth_downtime(alerts)

    out = [f"【kabu-auto 運用概要 {today}】", f"モード: {tm.description(mode)}",
           f"コード: {code_version.describe() if code_version._state else (code_version._head_commit(REPO) or '不明') + '（HEAD）'}", "",
           _format_period(r["daily"]), _format_period(r["weekly"]), _format_period(r["monthly"]), "",
           f"本体起動: {len(starts)}回 {' '.join(starts)}",
           f"認証切れ: {' / '.join(down) or 'なし'}",
           f"取引停止中: {'はい' if halted else 'いいえ'}",
           f"ERROR/CRITICAL: {sum(errors.values())}件 / ALERT: {len(alerts)}件", ""]
    try:
        out += _db_sections(today, full)
    except Exception as e:  # DB集計に失敗してもログ由来の情報は出す
        out.append(f"(DB集計失敗: {e})")
    out += ["", "▼エラー種別（件数順）"]
    out += [f"{n}× {k}" for k, n in errors.most_common(None if full else 5)] or ["(なし)"]
    out += ["", "▼当日のALERT"]
    shown = alerts if full else alerts[-10:]
    out += [f"{t} {m if full else m[:110]}" for t, m in shown] or ["(なし)"]
    text = "\n".join(out)
    return text if full else text[:1900]
