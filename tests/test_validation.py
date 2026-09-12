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


class TestSplitEventsBasics:
    def test_validation_is_the_fold_window(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=0)
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, val = validation.split_events(events, fold)
        assert val["decision_at"].min() >= fold.val_start
        assert val["decision_at"].max() <= fold.val_end

    def test_training_stops_at_the_cutoff(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=0)
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, val = validation.split_events(events, fold)
        assert train["decision_at"].max() <= fold.train_end

    def test_unresolved_events_are_excluded_from_both_sides(self):
        """ラベルの無いイベントは学習にも検証にも使わない"""
        rows = [
            {"symbol": "7203", "decision_at": date(2026, 1, 5),
             "label_end_at": date(2026, 1, 6), "label": 1},
            {"symbol": "7203", "decision_at": date(2026, 1, 6),
             "label_end_at": None, "label": None,
             "status": dataset.STATUS_IMMATURE},
            {"symbol": "7203", "decision_at": date(2026, 1, 7),
             "label_end_at": date(2026, 1, 8), "label": 0},
            {"symbol": "7203", "decision_at": date(2026, 1, 8),
             "label_end_at": date(2026, 1, 9), "label": 1},
        ]
        events = _events(rows)
        fold = validation.Fold(
            index=0, train_start=date(2026, 1, 5), train_end=date(2026, 1, 6),
            val_start=date(2026, 1, 7), val_end=date(2026, 1, 8))
        train, val = validation.split_events(events, fold)
        assert train["status"].eq(dataset.STATUS_RESOLVED).all()
        assert val["status"].eq(dataset.STATUS_RESOLVED).all()
        assert date(2026, 1, 6) not in set(train["decision_at"])


class TestPurge:
    def _overlapping_events(self):
        """1/6 に判断したイベントだけ、ラベルが検証期間（1/8〜）まで伸びる"""
        rows = [
            {"symbol": "7203", "decision_at": date(2026, 1, 5),
             "label_end_at": date(2026, 1, 6), "label": 1},
            {"symbol": "7203", "decision_at": date(2026, 1, 6),
             "label_end_at": date(2026, 1, 9), "label": 0},   # 検証期間へ食い込む
            {"symbol": "7203", "decision_at": date(2026, 1, 7),
             "label_end_at": date(2026, 1, 7), "label": 1},
            {"symbol": "7203", "decision_at": date(2026, 1, 8),
             "label_end_at": date(2026, 1, 9), "label": 0},
        ]
        return _events(rows), validation.Fold(
            index=0, train_start=date(2026, 1, 5), train_end=date(2026, 1, 7),
            val_start=date(2026, 1, 8), val_end=date(2026, 1, 9))

    def test_purges_events_whose_label_reaches_validation(self):
        """ラベルが検証開始日以降に確定する学習イベントを除外する"""
        events, fold = self._overlapping_events()
        train, _ = validation.split_events(events, fold)
        assert date(2026, 1, 6) not in set(train["decision_at"])

    def test_keeps_events_resolved_before_validation(self):
        events, fold = self._overlapping_events()
        train, _ = validation.split_events(events, fold)
        assert date(2026, 1, 5) in set(train["decision_at"])
        assert date(2026, 1, 7) in set(train["decision_at"])

    def test_no_training_label_resolves_after_the_cutoff(self):
        """不変条件: 学習側のどのラベルも学習締切までに確定している"""
        events, fold = self._overlapping_events()
        train, _ = validation.split_events(events, fold)
        assert (train["label_end_at"] <= fold.train_end).all()

    def test_excludes_label_resolved_after_cutoff_even_when_gap_before_validation(self):
        """判断は締切前でもラベル確定が締切後なら学習に使えない

        分割日は候補イベントのある営業日から作るため、train_end と val_start の
        間に候補の無い空白期間ができる。そこへラベル確定日が落ちると、
        「label_end_at < val_start」という条件では素通りしてしまう。
        学習締切の時点では誰も知り得ない情報なので除外されなければならない
        （外部レビューR06の反例をそのまま置く）。
        """
        rows = [
            # 1/5に判断し1/8に確定。train_end=1/5 の時点では結果が分からない
            {"symbol": "7203", "decision_at": date(2026, 1, 5),
             "label_end_at": date(2026, 1, 8), "label": 1},
            # 1/5に判断し1/5に確定。こちらは締切時点で観測できる
            {"symbol": "9984", "decision_at": date(2026, 1, 5),
             "label_end_at": date(2026, 1, 5), "label": 0},
            {"symbol": "7203", "decision_at": date(2026, 1, 12),
             "label_end_at": date(2026, 1, 12), "label": 1},
        ]
        events = _events(rows)
        fold = validation.Fold(
            index=0, train_start=date(2026, 1, 5), train_end=date(2026, 1, 5),
            val_start=date(2026, 1, 12), val_end=date(2026, 1, 12))

        train, _ = validation.split_events(events, fold)
        symbols = set(train["symbol"])
        assert "7203" not in symbols   # 締切後に確定するので除外
        assert "9984" in symbols       # 締切までに確定するので残る

    def test_purge_applies_to_every_generated_fold(self):
        events = _daily_events(["7203", "9984"], date(2026, 1, 5), 60, holding=5)
        for fold in validation.calendar_folds(events, n_splits=5):
            train, _ = validation.split_events(events, fold)
            if len(train) == 0:
                continue
            assert (train["label_end_at"] <= fold.train_end).all()


class TestFoldContract:
    def test_rejects_fold_whose_training_spans_the_validation_period(self):
        """検証期間を跨ぐ・後ろまで伸びる分割は作れない

        片方向walk-forward専用なので train_end < val_start が不変条件である。
        これを満たさないFoldを黙って受け取ると、purge・embargoの契約が
        定義できないまま「未来で学習して過去を検証する」分割が成立する
        （外部レビューR23）。
        """
        with pytest.raises(ValueError, match="train_end < val_start"):
            validation.Fold(
                index=0, train_start=date(2026, 1, 5), train_end=date(2026, 1, 12),
                val_start=date(2026, 1, 7), val_end=date(2026, 1, 8))

    def test_rejects_inverted_validation_period(self):
        with pytest.raises(ValueError, match="val_start <= val_end"):
            validation.Fold(
                index=0, train_start=date(2026, 1, 5), train_end=date(2026, 1, 6),
                val_start=date(2026, 1, 9), val_end=date(2026, 1, 8))

    def test_rejects_inverted_training_period(self):
        with pytest.raises(ValueError, match="train_start <= train_end"):
            validation.Fold(
                index=0, train_start=date(2026, 1, 7), train_end=date(2026, 1, 6),
                val_start=date(2026, 1, 9), val_end=date(2026, 1, 10))

    def test_every_generated_fold_satisfies_the_contract(self):
        """calendar_folds が作る分割は全てこの契約を満たす"""
        events = _daily_events(["7203", "9984"], date(2026, 1, 5), 60, holding=5)
        folds = validation.calendar_folds(events, n_splits=5)
        assert len(folds) > 0
        for fold in folds:
            assert fold.train_start <= fold.train_end < fold.val_start <= fold.val_end


class TestEmbargo:
    """embargo は「検証開始の直前に空白セッションを置く」方向にだけ存在する。

    `val_end` より後ろを外す向きは片方向walk-forwardでは到達不能なので
    実装しない（外部レビューR23）。
    """

    def _events_with_gap(self):
        rows = [
            {"symbol": "7203", "decision_at": date(2026, 1, 5),
             "label_end_at": date(2026, 1, 5), "label": 1},
            {"symbol": "7203", "decision_at": date(2026, 1, 6),
             "label_end_at": date(2026, 1, 6), "label": 0},
            {"symbol": "7203", "decision_at": date(2026, 1, 7),
             "label_end_at": date(2026, 1, 7), "label": 1},   # val_start直前
            {"symbol": "7203", "decision_at": date(2026, 1, 8),
             "label_end_at": date(2026, 1, 8), "label": 0},   # 検証側
        ]
        return _events(rows), validation.Fold(
            index=0, train_start=date(2026, 1, 5), train_end=date(2026, 1, 7),
            val_start=date(2026, 1, 8), val_end=date(2026, 1, 8))

    def test_default_is_disabled(self):
        events, fold = self._events_with_gap()
        train, _ = validation.split_events(events, fold)
        assert date(2026, 1, 7) in set(train["decision_at"])

    def test_excludes_the_sessions_immediately_before_validation(self):
        """embargo_sessions=1 なら val_start 直前の1営業日を学習から外す"""
        events, fold = self._events_with_gap()
        train, _ = validation.split_events(events, fold, embargo_sessions=1)
        decisions = set(train["decision_at"])
        assert date(2026, 1, 7) not in decisions   # 直前1営業日は外れる
        assert date(2026, 1, 6) in decisions       # その前は残る
        assert date(2026, 1, 5) in decisions

    def test_embargo_of_two_removes_two_sessions(self):
        events, fold = self._events_with_gap()
        train, _ = validation.split_events(events, fold, embargo_sessions=2)
        decisions = set(train["decision_at"])
        assert date(2026, 1, 7) not in decisions
        assert date(2026, 1, 6) not in decisions
        assert date(2026, 1, 5) in decisions

    def test_embargo_never_touches_the_validation_set(self):
        events, fold = self._events_with_gap()
        _, val_off = validation.split_events(events, fold)
        _, val_on = validation.split_events(events, fold, embargo_sessions=2)
        assert list(val_off["event_id"]) == list(val_on["event_id"])


class TestInnerFolds:
    def _outer(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 120, holding=1)
        fold = validation.calendar_folds(events, n_splits=5)[3]
        train, val = validation.split_events(events, fold)
        return events, fold, train, val

    def test_inner_folds_live_inside_outer_training(self):
        """内側foldは外側の学習期間の中だけで完結する"""
        _, outer, train, _ = self._outer()
        for inner in validation.inner_folds(train, n_splits=3):
            assert inner.train_start >= train["decision_at"].min()
            assert inner.val_end <= train["decision_at"].max()
            assert inner.val_end < outer.val_start

    def test_inner_folds_never_touch_outer_validation(self):
        """内側foldのどの期間も外側の検証期間に重ならない（外側は最終評価専用）"""
        _, outer, train, _ = self._outer()
        for inner in validation.inner_folds(train, n_splits=3):
            assert inner.val_start < outer.val_start
            assert inner.train_end < outer.val_start

    def test_inner_folds_are_forward_only(self):
        _, _, train, _ = self._outer()
        for inner in validation.inner_folds(train, n_splits=3):
            assert inner.train_end < inner.val_start

    def test_purge_applies_inside_inner_folds_too(self):
        """内側foldにも同じpurgeが効く"""
        _, _, train, _ = self._outer()
        for inner in validation.inner_folds(train, n_splits=3):
            inner_train, _ = validation.split_events(train, inner)
            if len(inner_train) == 0:
                continue
            assert (inner_train["label_end_at"] < inner.val_start).all()

    def test_raises_when_training_too_short(self):
        short = _daily_events(["7203"], date(2026, 1, 5), 2, holding=0)
        with pytest.raises(ValueError, match="セッション"):
            validation.inner_folds(short, n_splits=3)


class TestTrainingWeights:
    def test_length_matches_training_events(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=2)
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, _ = validation.split_events(events, fold)
        assert len(validation.training_weights(train)) == len(train)

    def test_weights_are_positive_for_resolved_events(self):
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=2)
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, _ = validation.split_events(events, fold)
        w = validation.training_weights(train)
        assert (w > 0).all()

    def test_recomputed_from_the_training_subset_only(self):
        """全期間で計算した重みとは値が異なる＝fold内で計算し直している。

        差が出るのは学習期間の**末尾**のイベント。先頭付近のイベントは、
        重なり相手が全期間にも部分集合にも等しく含まれるため値が一致する。
        末尾では後続の重なり相手が切り落とされ、重みが上がる。
        """
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=5)
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, _ = validation.split_events(events, fold)

        whole = dataset.uniqueness_weights(events)
        in_fold = validation.training_weights(train)

        last_id = train["event_id"].iloc[-1]
        pos = list(events["event_id"]).index(last_id)
        assert in_fold[-1] > whole[pos]

    def test_early_training_events_keep_the_same_weight(self):
        """逆に、重なり相手が全て学習側に残る先頭付近では値が一致する。

        これが成り立たないなら、重みの計算が集合の大きさ自体に依存している
        （＝相対的な重なりを測れていない）ことになる。
        """
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=5)
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, _ = validation.split_events(events, fold)

        whole = dataset.uniqueness_weights(events)
        in_fold = validation.training_weights(train)

        first_id = train["event_id"].iloc[0]
        pos = list(events["event_id"]).index(first_id)
        assert in_fold[0] == pytest.approx(whole[pos])

    def test_validation_side_does_not_affect_training_weights(self):
        """検証側イベントの終了時点を変えても学習側の重みは変わらない（spec §7）"""
        events = _daily_events(["7203"], date(2026, 1, 5), 60, holding=2)
        fold = validation.calendar_folds(events, n_splits=5)[2]
        train, _ = validation.split_events(events, fold)
        before = validation.training_weights(train)

        tampered = events.copy()
        in_val = tampered["decision_at"] >= fold.val_start
        tampered.loc[in_val, "label_end_at"] = date(2027, 1, 1)
        train_after, _ = validation.split_events(tampered, fold)
        after = validation.training_weights(train_after)

        assert list(train["event_id"]) == list(train_after["event_id"])
        assert before == pytest.approx(after)

    def test_empty_training_set_gives_empty_weights(self):
        empty = _daily_events(["7203"], date(2026, 1, 5), 10, holding=0).iloc[0:0]
        assert len(validation.training_weights(empty)) == 0
