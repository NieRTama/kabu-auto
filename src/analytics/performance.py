"""
パフォーマンス分析（Phase 5 / 7.8）。

確定済み取引（pnl が確定したTrade＝主に決済）から、勝率だけでは見えない
「期待値・プロフィットファクター・連勝連敗・銘柄/セクター別の偏り」を算出する。

DB や FastAPI に依存しない純関数として実装し、Trade ライクなオブジェクトのリストを
受け取る（テスト容易性のため）。各要素に必要な属性:
  pnl, side, symbol, sector, price, filled_price, quantity, filled_quantity, filled_at
"""
from typing import Optional


def _exit_value(t) -> Optional[float]:
    """決済の約定総額（成行は price=0・filled_price に実単価）。"""
    px = t.filled_price if getattr(t, "filled_price", None) else t.price
    if not px:
        return None
    return px * (getattr(t, "filled_quantity", None) or t.quantity or 0)


def _return_pct(t) -> Optional[float]:
    """損益と約定総額から取得原価を逆算してリターン率を求める。"""
    exit_val = _exit_value(t)
    if exit_val is None or t.pnl is None:
        return None
    cost_basis = exit_val - t.pnl
    if cost_basis and cost_basis > 0:
        return t.pnl / cost_basis
    return None


def _max_streak(signs: list[int], target: int) -> int:
    """符号列の中で target（+1=勝ち / -1=負け）が連続する最大長を返す。"""
    best = cur = 0
    for s in signs:
        if s == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _group_stats(trades: list) -> dict:
    """グループ（銘柄/セクター）内の集計（取引数・損益・勝率）。"""
    n = len(trades)
    wins = sum(1 for t in trades if (t.pnl or 0) > 0)
    losses = sum(1 for t in trades if (t.pnl or 0) < 0)
    decided = wins + losses
    return {
        "trades": n,
        "pnl": round(sum(t.pnl or 0 for t in trades), 0),
        "win_rate": round(wins / decided, 3) if decided else None,
    }


def _stats_from_returns(pnls: list, returns: list) -> dict:
    decided = sum(1 for p in pnls if p != 0)
    wins = sum(1 for p in pnls if p > 0)
    return {
        "trade_count": len(pnls),
        "win_rate": round(wins / decided, 3) if decided else None,
        "avg_return_pct": round(sum(returns) / len(returns), 4) if returns else None,
        "net_pnl": round(sum(pnls), 0),
    }


def compute_divergence(bt_trades: list, actual_trades: list) -> dict:
    """バックテストの取引と実取引（同一銘柄）を比較し、乖離を返す（7.7）。

    bt_trades: BacktestTradeRecord ライク（entry_price, exit_price, pnl）
    actual_trades: 損益確定済みの実 Trade ライク（pnl, price/filled_price, quantity 等）

    勝率・平均リターン率・純損益を双方で算出し、実取引−バックテストの差分を返す。
    予測（バックテスト）と実運用がどれだけ食い違っているかの把握に使う。
    """
    bt_pnls = [t.pnl for t in bt_trades if t.pnl is not None]
    bt_returns = []
    for t in bt_trades:
        ep, xp = getattr(t, "entry_price", None), getattr(t, "exit_price", None)
        if ep and xp and ep > 0:
            bt_returns.append((xp - ep) / ep)

    act_pnls = [t.pnl for t in actual_trades if t.pnl is not None]
    act_returns = [r for r in (_return_pct(t) for t in actual_trades) if r is not None]

    bt = _stats_from_returns(bt_pnls, bt_returns)
    actual = _stats_from_returns(act_pnls, act_returns)

    def _diff(a, b):
        return round(a - b, 4) if (a is not None and b is not None) else None

    return {
        "backtest": bt,
        "actual": actual,
        "divergence": {
            "win_rate_diff": _diff(actual["win_rate"], bt["win_rate"]),
            "avg_return_pct_diff": _diff(actual["avg_return_pct"], bt["avg_return_pct"]),
        },
        "note": ("実取引のサンプルが少ないため参考値です"
                 if actual["trade_count"] < 5 else ""),
    }


def compute_performance(trades: list) -> dict:
    """確定済み取引リストからパフォーマンス指標一式を算出する。

    `trades` は pnl が None でない取引（=損益確定済み）のみを想定する。
    """
    realized = [t for t in trades if t.pnl is not None]
    n = len(realized)
    if n == 0:
        return {
            "total_trades": 0, "win_count": 0, "loss_count": 0, "win_rate": None,
            "gross_profit": 0.0, "gross_loss": 0.0, "net_pnl": 0.0,
            "profit_factor": None, "avg_win": None, "avg_loss": None,
            "expectancy": None, "avg_return_pct": None,
            "largest_win": None, "largest_loss": None,
            "max_consecutive_wins": 0, "max_consecutive_losses": 0,
            "by_symbol": {}, "by_sector": {},
        }

    pnls = [t.pnl for t in realized]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_count, loss_count = len(wins), len(losses)
    decided = win_count + loss_count
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    avg_win = (gross_profit / win_count) if win_count else None
    avg_loss = (gross_loss / loss_count) if loss_count else None  # 正の平均損失額

    # 期待値: 1取引あたりの平均損益（勝率×平均利益 − 敗率×平均損失）
    expectancy = (sum(pnls) / decided) if decided else None

    # 時系列順（filled_at→id 相当の入力順）の符号列で連勝連敗を測る
    ordered = sorted(realized, key=lambda t: (getattr(t, "filled_at", None) or 0, getattr(t, "id", 0)))
    signs = [1 if (t.pnl or 0) > 0 else (-1 if (t.pnl or 0) < 0 else 0) for t in ordered]

    returns = [r for r in (_return_pct(t) for t in realized) if r is not None]

    by_symbol: dict = {}
    by_sector: dict = {}
    for t in realized:
        by_symbol.setdefault(t.symbol, []).append(t)
        by_sector.setdefault(getattr(t, "sector", None) or "(未設定)", []).append(t)

    return {
        "total_trades": n,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_count / decided, 3) if decided else None,
        "gross_profit": round(gross_profit, 0),
        "gross_loss": round(gross_loss, 0),
        "net_pnl": round(sum(pnls), 0),
        # プロフィットファクター = 総利益 / 総損失。損失ゼロなら None（無限大は返さない）
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else None,
        "avg_win": round(avg_win, 0) if avg_win is not None else None,
        "avg_loss": round(avg_loss, 0) if avg_loss is not None else None,
        "expectancy": round(expectancy, 0) if expectancy is not None else None,
        "avg_return_pct": round(sum(returns) / len(returns), 4) if returns else None,
        "largest_win": round(max(pnls), 0),
        "largest_loss": round(min(pnls), 0),
        "max_consecutive_wins": _max_streak(signs, 1),
        "max_consecutive_losses": _max_streak(signs, -1),
        "by_symbol": {k: _group_stats(v) for k, v in by_symbol.items()},
        "by_sector": {k: _group_stats(v) for k, v in by_sector.items()},
    }
