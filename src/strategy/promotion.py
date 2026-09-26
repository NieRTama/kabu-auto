"""昇格の契約 — 評価を通った候補だけが現行になる。

現行の ml_model は週次再学習が成功するとその戻り値をそのまま運用モデルへ
代入しており、成績に応じた合格判断がその経路に無かった（レビューF09）。

**自動昇格は実装しない。** 昇格は明示的な呼び出しでのみ起き、判断者と理由の
記録を必須にする。AUC等の採用数値をここで決める必要はない（spec §9）。
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

from src.strategy import model_store as ms


@dataclass(frozen=True)
class PromotionCheck:
    """昇格できるかの判定。できない理由は**全件**返す。

    1つ直せば通る、を繰り返さずに済むようにするため。
    """
    ok: bool
    blockers: list = field(default_factory=list)


def check_promotable(model_id: str, *, evaluation_run_id: Optional[str],
                     degraded: bool, base_dir: str,
                     expected_feature_cols: list) -> PromotionCheck:
    """昇格不可の条件を検査する（spec §9）。

    - degraded な実行: 推論例外が起きた実行・再学習が結線されていない
      診断実行の成績は比較に使えない。**呼び出し側の bool ではなく
      保存済みの実行記録から読む**（外部レビューR13）
    - 未評価: 何を根拠に昇格するのかが残らない
    - **実績が未確定**: 予測明細の行数だけでは「評価済み」と言えない。
      `load_prediction_details()` は実績が無くても行を返すので、
      ラベルが全て未確定の shadow 予測でも行数条件は通ってしまう
    - **shadow だけ**: 並行記録は現行と候補の比較であって評価ではない
    - 評価実行と候補モデルの食い違い、ラベル契約の食い違い
    - 特徴量定義の不一致: ラベルや特徴量が変わったモデルを黙って現行にすると、
      同じ数字が別の意味になる

    引数の `degraded` は**補助的な早期拒否**としてのみ使う。保存状態と
    食い違う場合は保存状態を優先する。
    """
    from src.strategy.evaluation import (
        PURPOSE_SHADOW, load_evaluation_run, load_prediction_details)

    blockers: list = []

    try:
        meta = ms.read_meta(model_id, base_dir=base_dir)
    except FileNotFoundError:
        return PromotionCheck(ok=False,
                              blockers=[f"候補として保存されていません: {model_id}"])

    # 既に現行のモデルを再昇格すると、previous_model_id が自分自身を指し、
    # 以後 rollback() が過去のモデルへ永久に戻れなくなる（外部レビュー
    # 最終ブランチレビュー M4）。自動昇格を実装しない契約に沿い、
    # 何もせず成功にはせず明示的に拒否する
    current = ms.read_current(base_dir=base_dir)
    if current is not None and current.model_id == model_id:
        blockers.append(f"既に現行のモデルです: {model_id}")

    if degraded:
        blockers.append("degraded な実行の成績は昇格の根拠にできません")

    if not evaluation_run_id:
        blockers.append("未評価です（evaluation_run_id がありません）")
    else:
        # 保存済みの実行記録を根拠にする。呼び出し側が渡した degraded を
        # そのまま信じない（外部レビューR13）
        run = load_evaluation_run(evaluation_run_id)
        if run is None:
            blockers.append(
                f"評価実行の記録がありません（evaluation_run_id={evaluation_run_id}）")
        else:
            if run.degraded:
                blockers.append(
                    "保存済みの実行記録が degraded です。その成績は昇格の根拠にできません")
            if run.label_contract_id and not meta.label_contract_id:
                # meta.label_contract_id が None（既定値のまま）だと、
                # 食い違い検査そのものが黙ってスキップされてしまう。
                # 学習パイプラインがこのフィールドを埋め忘れた場合、
                # ラベル契約の食い違いが永久に検出できなくなるため、
                # 欠落を沈黙ではなく拒否にする（外部レビュー最終ブランチ
                # レビュー 保留Ruling m5 の再確認・推奨事項）
                blockers.append(
                    "モデルにラベル契約IDが記録されていません"
                    "（meta.label_contract_id）。食い違いを検査できないため"
                    "昇格できません")
            elif (run.label_contract_id and meta.label_contract_id
                    and run.label_contract_id != meta.label_contract_id):
                blockers.append(
                    "ラベル契約が一致しません: "
                    f"実行={run.label_contract_id} モデル={meta.label_contract_id}")

        # `EvaluationRun.model_id` は同一 evaluation_run_id を複数モデルの
        # 評価で共有する場合、最後に保存した呼び出しの値で上書きされる
        # （save_evaluation_run は evaluation_run_id で upsert するため）。
        # そのため「候補モデル自身の予測明細が実在するか」を主根拠にし、
        # 明細が無いときに限って `EvaluationRun.model_id` を補助的に見て
        # 「別のモデルの実行では」と伝える。
        details = load_prediction_details(evaluation_run_id, model_id=model_id)
        if len(details) == 0:
            if run is not None and run.model_id and run.model_id != model_id:
                blockers.append(
                    f"評価実行が別のモデルのものです: 実行={run.model_id} 候補={model_id}")
            else:
                blockers.append(
                    f"予測明細がありません（evaluation_run_id={evaluation_run_id}）")
        else:
            # **行数だけでは「評価済み」と言えない。**
            # load_prediction_details は実績が無くても行を返すので、
            # ラベルが全て未確定の shadow 予測でもこの条件を通せてしまう
            # （外部レビューR13）。実績の確定を根拠にする。
            resolved = details["actual_label"].notna()
            if not resolved.any():
                blockers.append(
                    "実績が1件も確定していません（予測だけでは成績を測れません）")

            purposes = set(details["purpose"].dropna().astype(str))
            if purposes and purposes <= {PURPOSE_SHADOW}:
                blockers.append(
                    "shadow記録だけでは昇格できません"
                    "（並行記録は現行と候補の比較であって評価ではありません）")

            unresolved = int((~resolved).sum())
            if unresolved:
                logger.info(
                    f"昇格検査: 未確定の予測が{unresolved}件あります"
                    f"（確定{int(resolved.sum())}件で判断します）")

    if list(meta.feature_cols) != list(expected_feature_cols):
        blockers.append(
            f"特徴量定義が一致しません: モデル={list(meta.feature_cols)} "
            f"期待={list(expected_feature_cols)}"
        )

    return PromotionCheck(ok=not blockers, blockers=blockers)


# 昇格の状態。参照の切替とDB記録は別の永続化先なので、片方だけが進んだ
# 状態が起こりうる。それを**検出して決着できる**ようにする（外部レビューR11）。
PROMOTION_PENDING = "pending"       # 記録済み。参照の切替はまだ
PROMOTION_COMMITTED = "committed"   # 参照も切り替わった
PROMOTION_FAILED = "failed"         # 切替に失敗。現行は前のまま


def _close_promotion(promotion_id: int, state: str, *,
                     switched_at: Optional[datetime]) -> None:
    """昇格記録を確定させる。"""
    from src.data.database import ModelPromotion, get_session

    with get_session() as session:
        row = session.get(ModelPromotion, promotion_id)
        if row is None:
            logger.error(f"昇格記録が見つかりません: id={promotion_id}")
            return
        row.state = state
        row.switched_at = switched_at
        session.commit()


def recover_promotions(*, base_dir: str = "models") -> list:
    """起動時に、決着していない昇格を実体と突き合わせて閉じる。

    `promote()` は「記録(pending) → 参照切替 → 記録(committed)」の順で進む。
    2と3の間でプロセスが落ちると pending が残る。このとき参照ファイルが
    既に新モデルを指しているかどうかで、実際に切り替わったかが分かる。

      - 参照が pending の model_id を指している → 切替は完了していた。committed
      - 指していない → 切替前に落ちた。failed

    **自動で参照を書き換えない。** 実体に合わせて記録のほうを直すだけである。
    取引の履歴に関わる状態なので、勝手に「やり直す」ことはしない。
    決着した件数と内容を返し、呼び出し側がユーザーへ報告する。

    戻り値: `[{"promotion_id", "model_id", "resolved_to"}, ...]`
    """
    from sqlalchemy import select as sa_select

    from src.data.database import ModelPromotion, get_session

    ref = ms.read_current(base_dir=base_dir)
    current_id = ref.model_id if ref else None

    resolved = []
    with get_session() as session:
        rows = list(session.scalars(sa_select(ModelPromotion).where(
            ModelPromotion.state == PROMOTION_PENDING)).all())
        for row in rows:
            if current_id == row.model_id:
                row.state = PROMOTION_COMMITTED
                row.switched_at = ref.switched_at
                outcome = PROMOTION_COMMITTED
            else:
                row.state = PROMOTION_FAILED
                row.switched_at = None
                outcome = PROMOTION_FAILED
            resolved.append({"promotion_id": row.id, "model_id": row.model_id,
                             "resolved_to": outcome})
        if rows:
            session.commit()

    for item in resolved:
        logger.warning(
            f"未決着の昇格を決着させました: {item['model_id']} "
            f"→ {item['resolved_to']}（参照の実体に合わせました）")
    return resolved


def rollback(*, decided_by: str, reason: str, base_dir: str = "models") -> Optional[int]:
    """1つ前のモデルへ戻し、`promote()` と同じ形式で `ModelPromotion` に記録する。

    `model_store.rollback()` は現行参照を切り替えるだけで、DBには何も残さない。
    そのため、ロールバック後は監査証跡（誰が・いつ・なぜ切り替えたか）と
    実際の現行モデルが食い違う（外部レビュー最終ブランチレビュー M3）。
    切り替える方向を問わず、判断者と理由を必須にする（自動昇格を作らない
    という契約は戻す方向にも及ぶ）。

    `promote()` と同じ2段階（pending → committed/failed）で記録する。
    戻り先が無ければ何もせず None を返す（記録も作らない）。
    """
    from src.data.database import ModelPromotion, get_session

    if not decided_by:
        raise ValueError("判断者（decided_by）は必須です")
    if not reason:
        raise ValueError("理由（reason）は必須です")

    current = ms.read_current(base_dir=base_dir)
    if current is None or current.previous_model_id is None:
        return None
    target_id = current.previous_model_id

    with get_session() as session:
        row = ModelPromotion(
            model_id=target_id, evaluation_run_id=None,
            decided_by=decided_by, reason=reason,
            previous_model_id=current.model_id,
            state=PROMOTION_PENDING, switched_at=None,
        )
        session.add(row)
        session.commit()
        promotion_id = row.id

    try:
        ref = ms.rollback(base_dir=base_dir)
    except BaseException:
        _close_promotion(promotion_id, PROMOTION_FAILED, switched_at=None)
        raise

    if ref is None:
        # 記録作成後・切替前の間に現行が別経路で変わっていた（並行変更）。
        # 切替は起きていないので failed として閉じる
        _close_promotion(promotion_id, PROMOTION_FAILED, switched_at=None)
        return None

    _close_promotion(promotion_id, PROMOTION_COMMITTED, switched_at=ref.switched_at)
    logger.warning(
        f"モデルをロールバック: {current.model_id} → {ref.model_id}"
        f"（判断者={decided_by}）")
    return promotion_id


def promote(model_id: str, *, evaluation_run_id: Optional[str],
            decided_by: str, reason: str, degraded: bool,
            base_dir: str, expected_feature_cols: list) -> int:
    """候補を現行へ昇格する。ModelPromotion.id を返す。

    **検査に通らなければ現行を一切変更せずに例外を投げる。** 途中状態を残さない。
    判断者と理由は必須（自動昇格を作らないため）。
    """
    from src.data.database import ModelPromotion, get_session

    if not decided_by:
        raise ValueError("判断者（decided_by）は必須です")
    if not reason:
        raise ValueError("理由（reason）は必須です")

    check = check_promotable(
        model_id, evaluation_run_id=evaluation_run_id, degraded=degraded,
        base_dir=base_dir, expected_feature_cols=expected_feature_cols)
    if not check.ok:
        raise ValueError("昇格できません: " + " / ".join(check.blockers))

    previous = ms.read_current(base_dir=base_dir)
    previous_id = previous.model_id if previous else None

    # ── 2段階で切り替える（外部レビューR11）──────────────────────────
    #
    # 参照の切替を先に行い、その後で履歴をcommitすると、DB書込みの失敗や
    # 両処理の間でのプロセス停止によって「参照は新モデルなのに昇格記録が
    # 無い」状態が残る。順序を逆にしても「記録はあるのに参照が古い」という
    # 逆向きの不整合が残るだけで、解決しない。
    #
    # 参照ファイル単体の原子的置換（os.replace）は、DBを含む取引の
    # 原子性ではない。**片方だけが進んだ状態を検出して決着できる**ように、
    # 昇格の意図を先に永続化し、切替後に確定させる。
    #
    #   1. pending として記録（switched_at は None）
    #   2. 参照を切り替える（os.replace）
    #   3. committed へ更新
    #
    # 2と3の間で落ちた場合、起動時に pending が残る。
    # `recover_promotions()` が参照の実体と突き合わせて決着させる。

    with get_session() as session:
        row = ModelPromotion(
            model_id=model_id, evaluation_run_id=evaluation_run_id,
            decided_by=decided_by, reason=reason,
            previous_model_id=previous_id,
            state=PROMOTION_PENDING, switched_at=None,
        )
        session.add(row)
        session.commit()
        promotion_id = row.id

    try:
        ref = ms.set_current(model_id, base_dir=base_dir)
    except BaseException:
        # 切替に失敗した。現行は前のまま。意図を失敗として閉じる
        _close_promotion(promotion_id, PROMOTION_FAILED, switched_at=None)
        raise

    _close_promotion(promotion_id, PROMOTION_COMMITTED,
                     switched_at=ref.switched_at)

    logger.warning(
        f"モデルを昇格: {previous_id} → {model_id}（判断者={decided_by}）")
    return promotion_id
