"""分割と漏れの遮断（src/strategy/validation.py）のテスト

全銘柄共通のカレンダー日付でfoldを切り、学習側へ未来が入る経路を
purge以外も含めて塞ぐ（spec §7）。モデルは扱わない。
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.strategy import dataset
from src.strategy import validation


def _events(rows: list[dict]) -> pd.DataFrame:
    """テスト用の最小イベント表。

    rows の各要素は {symbol, decision_at, label_end_at, label} を持つ。
    足りない列は既定値で埋める。
    """
    out = []
    for k, r in enumerate(rows):
        out.append({
            "event_id": f"{r['symbol']}:{r['decision_at']:%Y%m%d}",
            "symbol": r["symbol"],
            "decision_at": r["decision_at"],
            "entry_at": r.get("entry_at", r["decision_at"] + timedelta(days=1)),
            "label_end_at": r.get("label_end_at"),
            "status": r.get("status", dataset.STATUS_RESOLVED
                            if r.get("label") is not None else dataset.STATUS_IMMATURE),
            "label": r.get("label"),
            "net_return": r.get("net_return", 0.01),
            "f1": r.get("f1", float(k)),
            "f2": r.get("f2", float(k) * 2),
        })
    return pd.DataFrame(out)


def _daily_events(symbols: list[str], start: date, n_sessions: int,
                  holding: int = 2) -> pd.DataFrame:
    """各銘柄が n_sessions 日ぶん連続して候補になるイベント表"""
    rows = []
    for i in range(n_sessions):
        d = start + timedelta(days=i)
        for s in symbols:
            rows.append({
                "symbol": s,
                "decision_at": d,
                "label_end_at": d + timedelta(days=holding),
                "label": i % 2,
            })
    return _events(rows)


class TestSessionsOf:
    def test_returns_sorted_unique_sessions(self):
        events = _daily_events(["7203", "9984"], date(2026, 1, 5), 4)
        got = validation.sessions_of(events)
        assert got == [date(2026, 1, 5), date(2026, 1, 6),
                       date(2026, 1, 7), date(2026, 1, 8)]

    def test_is_shared_across_symbols(self):
        """銘柄ごとではなく全銘柄共通のセッション列になる"""
        rows = [
            {"symbol": "7203", "decision_at": date(2026, 1, 5), "label": 1},
            {"symbol": "9984", "decision_at": date(2026, 1, 6), "label": 0},
        ]
        assert validation.sessions_of(_events(rows)) == [
            date(2026, 1, 5), date(2026, 1, 6)]


class TestCalendarFolds:
    def test_produces_requested_number_of_folds(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 60)
        folds = validation.calendar_folds(events, n_splits=5)
        assert len(folds) == 5
        assert [f.index for f in folds] == [0, 1, 2, 3, 4]

    def test_training_always_precedes_validation(self):
        """学習期間は検証期間より前（片方向walk-forward）"""
        events = _daily_events(["7203"], date(2026, 1, 5), 60)
        for f in validation.calendar_folds(events, n_splits=5):
            assert f.train_end < f.val_start
            assert f.val_start <= f.val_end

    def test_training_window_expands(self):
        """拡大窓: foldが進むほど学習期間の終わりが後ろへ伸びる"""
        events = _daily_events(["7203"], date(2026, 1, 5), 60)
        folds = validation.calendar_folds(events, n_splits=5)
        ends = [f.train_end for f in folds]
        assert ends == sorted(ends)
        assert ends[0] < ends[-1]

    def test_validation_periods_do_not_overlap(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 60)
        folds = validation.calendar_folds(events, n_splits=5)
        for a, b in zip(folds, folds[1:]):
            assert a.val_end < b.val_start

    def test_boundaries_are_dates_not_row_positions(self):
        """銘柄数が変わってもfold境界の日付は変わらない（行番号ではない）"""
        one = validation.calendar_folds(
            _daily_events(["7203"], date(2026, 1, 5), 60), n_splits=5)
        many = validation.calendar_folds(
            _daily_events(["7203", "9984", "6758"], date(2026, 1, 5), 60), n_splits=5)
        assert [(f.train_end, f.val_start, f.val_end) for f in one] == \
               [(f.train_end, f.val_start, f.val_end) for f in many]

    def test_raises_when_too_few_sessions(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 3)
        with pytest.raises(ValueError, match="セッション"):
            validation.calendar_folds(events, n_splits=5)
