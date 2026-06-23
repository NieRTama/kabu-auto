"""
機械学習モデル（LightGBM）の学習・推論

モデルは表形式の金融データで実績のあるLightGBMを採用（2025-2026の各種比較研究でも
日足スイングのような小規模サンプルでは深層学習より優位とされる）。
ラベリングはトリプルバリア法（labeling.py）を用い、ラベル期間の重なりに対する
サンプル一意性重みを付与してルックアヘッド・過学習を抑制する。
"""
import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.core import clock
from src.core import config as cfg
from src.strategy.indicators import active_feature_cols, build_features
from src.strategy.labeling import build_training_set

MODEL_PATH = Path("models/lgb_model.pkl")


def _meta_path() -> Path:
    """モデルのメタ情報（SHA256ハッシュ・学習日時等）のサイドカーパス。"""
    return MODEL_PATH.with_suffix(".meta.json")


def _sha256_file(path: Path) -> str:
    """ファイルのSHA256を計算する（大きいモデルでもメモリを食わないよう逐次読み）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_model_meta(n_samples: int, n_estimators: int,
                      cv_mean: float, cv_std: float) -> None:
    """保存済みモデルのSHA256とメタ情報をサイドカーJSONに書き出す。

    pickle はロード時に任意コードを実行しうるため、配置後にハッシュを記録し、
    次回ロード時に改ざん・破損を検出できるようにする（レビュー Security）。
    """
    meta = {
        "sha256": _sha256_file(MODEL_PATH),
        "trained_at": clock.now().isoformat(),
        "n_samples": n_samples,
        "n_estimators": n_estimators,
        "cv_mean_accuracy": round(cv_mean, 4),
        "cv_std_accuracy": round(cv_std, 4),
        # 学習時に実際に使った特徴量セットを記録する。ロード時にこの集合と
        # active_feature_cols() を照合し、フラグ切替によるシェイプ不一致を検知する。
        "features": list(active_feature_cols()),
        "lightgbm_version": lgb.__version__,
    }
    _meta_path().write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"MLモデルメタ保存: sha256={meta['sha256'][:12]}… trained_at={meta['trained_at']}")


def _read_model_meta() -> Optional[dict]:
    path = _meta_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"モデルメタ読み込み失敗: {e}")
        return None


def train(df: pd.DataFrame, trigger: Optional[str] = None,
          save: bool = True) -> lgb.LGBMClassifier:
    """単一銘柄のOHLCVからトリプルバリアラベル＋サンプル重みでLightGBMを学習し保存する。

    trigger が指定された場合（"weekly_schedule" / "manual"）、
    CV精度・特徴量重要度をDBに保存する。
    バックテスト内での学習時は trigger=None で呼び出してDBに記録しない。

    注意: df は単一銘柄の時系列であること。複数銘柄を学習する場合は
    train_multi() を使う（build_features の rolling/ewm や トリプルバリア法の
    「N日後」判定は単一時系列前提のため、複数銘柄を結合したdfを渡すと
    銘柄境界をまたいで指標・ラベルが破壊される）。
    """
    X, y, weights = build_training_set(df)
    return _fit(X, y, weights, trigger=trigger, save=save)


def train_multi(dfs: list[pd.DataFrame], trigger: Optional[str] = None,
                 save: bool = True) -> lgb.LGBMClassifier:
    """複数銘柄のOHLCVから学習する。

    各dfごとに個別に build_training_set() で特徴量・ラベル・サンプル重みを
    作成してから連結する（生のOHLCVをconcatしてから処理すると、移動平均/RSI/
    トリプルバリア法が銘柄境界をまたいで壊れるため、これは行わない）。
    """
    X_parts, y_parts, w_parts = [], [], []
    for df in dfs:
        try:
            X, y, w = build_training_set(df)
        except ValueError as e:
            logger.warning(f"学習データ生成をスキップ: {e}")
            continue
        if len(X) == 0:
            continue
        X_parts.append(X)
        y_parts.append(y)
        w_parts.append(w)

    if not X_parts:
        raise ValueError("学習データが不足しています: 0件")

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    weights = np.concatenate(w_parts)
    return _fit(X, y, weights, trigger=trigger, save=save)


def _fit(X: pd.DataFrame, y: pd.Series, weights: np.ndarray,
         trigger: Optional[str] = None, save: bool = True) -> lgb.LGBMClassifier:
    """特徴量・ラベル・サンプル重みからLightGBMを学習し保存する（train/train_multi共通部）。

    TimeSeriesSplitによるCVは、train_multi() 経由（複数銘柄連結）の場合は
    「銘柄ごとの時系列順序」は保たれるが「全体としての時系列順序」は厳密ではない
    （銘柄Aの全期間→銘柄Bの全期間の順に連結されるため）。CV精度はモデル選択の
    近似指標として扱い、厳密な銘柄別時系列CVは将来の改善課題とする。
    """
    if len(X) < 100:
        raise ValueError(f"学習データが不足しています: {len(X)}件")

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    auc_scores = []
    brier_scores = []
    best_n_list = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_tr = weights[train_idx]
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                num_leaves=31, random_state=42, verbose=-1)
        m.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(20, verbose=False),
                         lgb.log_evaluation(period=-1)])
        preds = m.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
        best_n_list.append(m.best_iteration_ or 200)
        # 精度だけでは収益性・確率の信頼性を測れないため、確率予測の質も評価する
        # （AUC=順位付けの良さ / Brier=確率校正の良さ。レビュー ML）。
        # 検証foldが片側クラスのみだとAUCは未定義になるためスキップ。
        proba = m.predict_proba(X_val)[:, 1]
        if len(np.unique(y_val)) > 1:
            auc_scores.append(roc_auc_score(y_val, proba))
        brier_scores.append(brier_score_loss(y_val, proba))

    best_n_estimators = int(np.mean(best_n_list))
    cv_mean = float(np.mean(scores))
    cv_std = float(np.std(scores))
    cv_auc = float(np.mean(auc_scores)) if auc_scores else None
    cv_brier = float(np.mean(brier_scores)) if brier_scores else None

    auc_str = f"{cv_auc:.3f}" if cv_auc is not None else "—"
    brier_str = f"{cv_brier:.3f}" if cv_brier is not None else "—"
    logger.info(
        f"MLモデルCV精度: {cv_mean:.3f} (+/-{cv_std:.3f}) AUC={auc_str} Brier={brier_str}"
    )

    # CV後に全データで最終モデルを学習（サンプル重み込み）
    model = lgb.LGBMClassifier(n_estimators=best_n_estimators, learning_rate=0.05,
                                num_leaves=31, random_state=42, verbose=-1)
    model.fit(X, y, sample_weight=weights)
    logger.info("MLモデル学習完了（全データ・トリプルバリア法）")

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        _write_model_meta(
            n_samples=len(X), n_estimators=best_n_estimators,
            cv_mean=cv_mean, cv_std=cv_std)

    if trigger is not None:
        # zip はサイレントに短い方で切り詰めるため、列数と重要度数の一致を明示検査する
        cols = active_feature_cols()
        importances = model.feature_importances_.tolist()
        if len(cols) != len(importances):
            raise ValueError(
                f"特徴量数の不一致: active_feature_cols={len(cols)} "
                f"feature_importances={len(importances)}"
            )
        _save_metrics(
            cv_mean=cv_mean,
            cv_std=cv_std,
            cv_auc=cv_auc,
            cv_brier=cv_brier,
            n_samples=len(X),
            n_estimators=best_n_estimators,
            feature_importances=dict(zip(cols, importances)),
            trigger=trigger,
        )

    return model


def _save_metrics(
    cv_mean: float, cv_std: float, n_samples: int,
    n_estimators: int, feature_importances: dict, trigger: str,
    cv_auc: Optional[float] = None, cv_brier: Optional[float] = None,
) -> None:
    from src.data.database import ModelMetrics, get_session
    with get_session() as session:
        session.add(ModelMetrics(
            trained_at=clock.now(),
            cv_mean_accuracy=round(cv_mean, 4),
            cv_std_accuracy=round(cv_std, 4),
            cv_auc=round(cv_auc, 4) if cv_auc is not None else None,
            cv_brier=round(cv_brier, 4) if cv_brier is not None else None,
            n_samples=n_samples,
            n_estimators=n_estimators,
            feature_importances_json=json.dumps(feature_importances),
            trigger=trigger,
        ))
        session.commit()
    logger.info(f"MLモデルメトリクス保存完了 (trigger={trigger})")


def load() -> Optional[lgb.LGBMClassifier]:
    """学習済みモデルを読み込む。

    pickle.load は任意コード実行のリスクがあるため、ロード前にファイルのSHA256を
    メタ情報（_meta_path）と照合する（レビュー Security）。
    - メタのハッシュと不一致 → 改ざん/破損とみなしロード中止（fail-closed）。
    - メタが無い旧モデル → 既定では fail-closed でロードを拒否する（攻撃者がモデルを
      改ざんし、メタファイルも削除すれば検証自体を無効化できてしまうため。レビュー
      再指摘 Critical）。`strategy.allow_unverified_model_load: true` を明示設定した
      場合のみ、警告のうえロードして現在のハッシュをメタに記録する（移行用の救済弁）。
    起動時にハッシュと学習日時をログへ出す（監査・運用可視化）。
    """
    if not MODEL_PATH.exists():
        return None
    try:
        digest = _sha256_file(MODEL_PATH)
    except OSError as e:
        logger.error(f"MLモデル読み込み失敗（ハッシュ計算）: {e}")
        return None

    from src.core.alerts import alert
    meta = _read_model_meta()
    if meta and meta.get("sha256"):
        if meta["sha256"] != digest:
            logger.critical(
                "MLモデルのSHA256が記録と不一致です（改ざん/破損の可能性）。"
                f"ロードを中止しました。期待={meta['sha256'][:12]}… 実際={digest[:12]}…"
            )
            alert("MLモデル検証失敗",
                  "models/lgb_model.pkl のSHA256が記録と一致しません。ロードを中止しました。")
            return None
        logger.info(
            f"MLモデルロード: sha256={digest[:12]}… "
            f"trained_at={meta.get('trained_at', '不明')} "
            f"CV精度={meta.get('cv_mean_accuracy', '?')}"
        )
    else:
        allow_unverified = bool(
            cfg.get_section("strategy").get("allow_unverified_model_load", False))
        if not allow_unverified:
            logger.critical(
                "MLモデルのメタ情報（SHA256記録）がありません。改ざん検知できないため"
                "ロードを拒否しました（fail-closed）。意図的な初回移行であれば "
                "config.yaml の strategy.allow_unverified_model_load: true を設定して"
                "ください（設定後はこのモデルのハッシュが正として記録されます）"
            )
            alert("MLモデル検証失敗（メタ無し）",
                  "models/lgb_model.pkl にハッシュ記録がなく、ロードを拒否しました。"
                  "意図的な移行なら strategy.allow_unverified_model_load を設定してください")
            return None
        logger.warning(
            f"MLモデルのメタ情報がありません（hash未検証でロード）。sha256={digest[:12]}… を記録します"
        )

    # ── 特徴量セットの整合検証（SHA256はファイル完全性のみ、feature不一致は別途検知）──
    # メタに記録した学習時の特徴量と、現在の active_feature_cols() が食い違う場合、
    # use_news_features フラグの切替とモデル再学習が非同期になっている（desync）。
    # そのままでは predict 時にシェイプ不一致や誤推論を招くため fail-closed する。
    if meta and meta.get("features") is not None:
        trained_features = list(meta["features"])
        active = list(active_feature_cols())
        if trained_features != active:
            from src.core.alerts import alert
            logger.critical(
                "MLモデルの特徴量セットが現在の設定と不一致です（フラグとモデルのdesync）。"
                "ロードを中止しました。use_news_features を切り替えた場合は再学習が必要です。"
                f" 学習時={trained_features} / 現在={active}"
            )
            alert("MLモデル特徴量不一致",
                  "学習済みモデルの特徴量セットが現在の use_news_features 設定と一致しません。"
                  "ロードを中止しました。フラグ切替後は再学習してください。")
            return None

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # 旧モデル（メタ無し・明示許可あり）は今回のハッシュを記録し、次回から検証対象にする
    if not meta:
        try:
            _meta_path().write_text(
                json.dumps({"sha256": digest, "trained_at": None,
                            "note": "legacy model; hash recorded at first load"},
                           ensure_ascii=False, indent=2),
                encoding="utf-8")
        except OSError as e:
            logger.warning(f"モデルメタの記録に失敗: {e}")
    return model


def predict_proba(model: lgb.LGBMClassifier, df: pd.DataFrame,
                  news_df: Optional[pd.DataFrame] = None) -> float:
    """最新の特徴量で上昇確率を返す（0.0〜1.0）。
    df に既に特徴量列がある場合は再計算しない。

    列はモデル自身が学習した特徴量（model.feature_name_）で選択する。これにより
    フラグ切替で active_feature_cols() がモデルとずれた場合に、黙ってスライスして
    誤推論するのではなく KeyError を送出して上位（signal.generate）で検知できる。
    """
    cols = active_feature_cols()
    if cols[0] not in df.columns:
        df = build_features(df, news_df=news_df)
    if df.empty or len(df) < 2:
        return 0.5
    # モデルが学習した特徴量順で選択する（学習時と推論時の列順・列集合を厳密一致させる）
    model_cols = getattr(model, "feature_name_", None) or cols
    latest = df[model_cols].iloc[[-1]]
    proba = model.predict_proba(latest)[0][1]
    return float(proba)
