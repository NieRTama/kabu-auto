"""shadow運用 — 現行と候補の判断を並行記録する。

同じ入力に対し現行と候補の判断を記録し、差が何によって生じたかを
**見送った候補の結果も含めて**残す。採った分だけでは差が測れない。

**候補は発注に繋がらない。** 本モジュールは発注系（src/execution）も
取引サービス（src/services/trading）も import しない。記録だけを行う。
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.core import clock
from src.strategy.evaluation import PURPOSE_SHADOW, save_predictions

# shadowの予測はどのfoldにも属さない
SHADOW_FOLD_INDEX = -1

AGREEMENT_BOTH_TAKE = "both_take"
AGREEMENT_BOTH_SKIP = "both_skip"
AGREEMENT_ONLY_CURRENT = "only_current"
AGREEMENT_ONLY_CANDIDATE = "only_candidate"


@dataclass(frozen=True)
class ShadowComparison:
    """1イベントについての、現行と候補の判断の突き合わせ。"""
    event_id: str
    current_probability: Optional[float]
    candidate_probability: float
    current_takes: bool
    candidate_takes: bool
    agreement: str


def _predict(model, features: pd.DataFrame) -> Optional[np.ndarray]:
    if model is None:
        return None
    return np.asarray(model.predict(features), dtype=float)


def compare(event_ids: list, features: pd.DataFrame, *,
            current, candidate, threshold: float) -> list:
    """同じ入力に現行と候補を当て、判断の一致・不一致を記録用に並べる。

    現行が無い（v2はモデル未昇格から始まる）場合は、現行側を「採らない」
    として扱い確率は None にする。
    """
    current_p = _predict(current, features)
    candidate_p = _predict(candidate, features)
    if candidate_p is None:
        raise ValueError("候補モデルは必須です")

    out = []
    for i, event_id in enumerate(event_ids):
        cur = float(current_p[i]) if current_p is not None else None
        cand = float(candidate_p[i])
        cur_takes = cur is not None and cur >= threshold
        cand_takes = cand >= threshold
        if cur_takes and cand_takes:
            agreement = AGREEMENT_BOTH_TAKE
        elif not cur_takes and not cand_takes:
            agreement = AGREEMENT_BOTH_SKIP
        elif cur_takes:
            agreement = AGREEMENT_ONLY_CURRENT
        else:
            agreement = AGREEMENT_ONLY_CANDIDATE
        out.append(ShadowComparison(
            event_id=str(event_id), current_probability=cur,
            candidate_probability=cand, current_takes=cur_takes,
            candidate_takes=cand_takes, agreement=agreement,
        ))
    return out


def record_shadow(comparisons: list, *, evaluation_run_id: str,
                  candidate_model_id: str, current_model_id: Optional[str],
                  threshold: float, label_contract_id: str) -> int:
    """**比較そのもの**を保存する。保存件数を返す。

    候補の確率だけを書くと、再起動後に「その時どちらを採り、なぜ見送ったか」
    を復元できない（外部レビューR21）。並行記録と呼べるのは、
    **両モデルID・両確率・判断閾値・採否・一致区分・入力契約**が
    同じ行に揃っているときだけである。関数名や一時的な戻り値では満たさない。

    現行が未昇格のときは `current_model_id=None` / `current_probability=None`
    で保存し、「現行が無かった」という状態として残す。行を作らないと
    「比較しなかった」のか「現行が無かった」のか後から区別できない。

    実績（PredictionOutcome）はここでは書かない。予測時点では確定して
    いないため、満期後に別途関連付ける（spec §7）。結合キーは
    `(label_contract_id, event_id)`。

    `save_predictions()` と同じく、同一 `evaluation_run_id` の既存比較行を
    削除してから挿入する（run単位の置換）。`shadow_comparisons` には
    `(evaluation_run_id, label_contract_id, event_id)` の UNIQUE索引があり、
    素の INSERT のままだと同じ run を2回目に実行した時点で必ず
    IntegrityError になる。その結果、先に commit 済みの Prediction 側だけが
    新しい内容に置き換わり、比較行側は古いまま残る恒久的な食い違いが
    起きていた（外部レビュー最終ブランチレビュー M1）。
    """
    from sqlalchemy import delete as sa_delete

    from src.data.database import ShadowComparisonRow, get_session

    if not comparisons:
        return 0

    # 候補の予測は従来どおり Prediction へ（指標の再計算に使う）
    preds = pd.DataFrame({
        "event_id": [c.event_id for c in comparisons],
        "label_contract_id": [label_contract_id] * len(comparisons),
        "raw_probability": [c.candidate_probability for c in comparisons],
        "calibrated_probability": [c.candidate_probability for c in comparisons],
        "fold_index": [SHADOW_FOLD_INDEX] * len(comparisons),
    })
    n = save_predictions(preds, evaluation_run_id, candidate_model_id,
                         purpose=PURPOSE_SHADOW)

    # 比較の内訳はこちらへ。これが「並行記録」の実体
    now = clock.now()
    with get_session() as session:
        session.execute(sa_delete(ShadowComparisonRow).where(
            ShadowComparisonRow.evaluation_run_id == evaluation_run_id))
        for c in comparisons:
            session.add(ShadowComparisonRow(
                evaluation_run_id=evaluation_run_id,
                event_id=c.event_id,
                label_contract_id=label_contract_id,
                current_model_id=current_model_id,
                candidate_model_id=candidate_model_id,
                current_probability=c.current_probability,
                candidate_probability=c.candidate_probability,
                threshold=float(threshold),
                current_takes=1 if c.current_takes else 0,
                candidate_takes=1 if c.candidate_takes else 0,
                agreement=c.agreement,
                recorded_at=now,
            ))
        session.commit()

    logger.info(
        f"shadow記録: run={evaluation_run_id} 現行={current_model_id} "
        f"候補={candidate_model_id} 閾値={threshold} {n}件")
    return n


def load_shadow_comparisons(evaluation_run_id: str) -> pd.DataFrame:
    """保存済みの比較を読み直す。

    この出力から `disagreement_summary()` を再計算できること
    （＝再起動後に当時の比較を復元できること）が並行記録の合格条件である。
    """
    from src.data.database import ShadowComparisonRow, get_session
    from sqlalchemy import select as sa_select

    with get_session() as session:
        rows = list(session.scalars(sa_select(ShadowComparisonRow).where(
            ShadowComparisonRow.evaluation_run_id == evaluation_run_id)).all())
    return pd.DataFrame([{
        "event_id": r.event_id,
        "label_contract_id": r.label_contract_id,
        "current_model_id": r.current_model_id,
        "candidate_model_id": r.candidate_model_id,
        "current_probability": r.current_probability,
        "candidate_probability": r.candidate_probability,
        "threshold": r.threshold,
        "current_takes": bool(r.current_takes),
        "candidate_takes": bool(r.candidate_takes),
        "agreement": r.agreement,
    } for r in rows])


def disagreement_summary(comparisons: list) -> dict:
    """判断の一致・不一致の内訳。

    「候補が現行と何件違ったか」だけでなく、どちらの向きに違ったかを分けて
    数える。片側にだけ寄っているなら、それは閾値の差であって能力の差では
    ないかもしれない。
    """
    counts = {
        AGREEMENT_BOTH_TAKE: 0, AGREEMENT_BOTH_SKIP: 0,
        AGREEMENT_ONLY_CURRENT: 0, AGREEMENT_ONLY_CANDIDATE: 0,
    }
    for c in comparisons:
        counts[c.agreement] += 1
    n = len(comparisons)
    agreed = counts[AGREEMENT_BOTH_TAKE] + counts[AGREEMENT_BOTH_SKIP]
    return {
        **counts, "n": n,
        "agreement_rate": (agreed / n) if n else None,
    }
