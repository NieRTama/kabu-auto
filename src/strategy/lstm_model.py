"""
LSTM系列モデル（LightGBMとのハイブリッドアンサンブル用）。

論文サーベイ（2024-2026）の結論に基づき、タブラーで強いLightGBMを主軸に残しつつ、
系列依存を捉えるLSTMを加算的に併用する。torch は重い依存のため requirements-ml.txt に
分離し、本モジュール内で**遅延 import** する。未導入・モデル成果物なしの場合は、
load() が静かに None を返し、推論側（signal.generate）は純GBMにフォールバックする
（フラグOFF時の既定状態であり「異常」ではないため alert しない）。

リーク防止:
  - 系列は labeling.build_sequence_set() が銘柄ごとに構成する（連結境界をまたがない）。
  - CV はパージ＋エンバーゴ付き（fold境界の seq_len + max_holding 行を捨てる）。
セキュリティ:
  - 保存物（state_dict）の SHA256 をサイドカーに記録し、ロード時に照合（改ざんは fail-closed）。
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from src.core import clock
from src.core import config as cfg
from src.strategy.indicators import active_feature_cols

MODEL_PATH = Path("models/lstm_model.pt")


def _meta_path() -> Path:
    return MODEL_PATH.with_suffix(".pt.meta.json")


def _sha256_file(path: Path) -> str:
    # ml_model と同一実装（逐次読みでメモリを食わない）
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def available() -> bool:
    """torch が import 可能か（LSTMの学習・推論ができるか）。"""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _build_module(n_features: int, hidden: int, num_layers: int = 1):
    """小型LSTM分類器を構築する（torch を遅延 import）。"""
    import torch.nn as nn

    class _LSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers,
                                batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]          # 最終タイムステップの隠れ状態
            return self.head(last).squeeze(-1)  # ロジット

    return _LSTMClassifier()


def _seq_len() -> int:
    return int(cfg.get_section("strategy").get("lstm_seq_len", 20))


def train_multi(dfs: list, news_map: Optional[dict] = None,
                trigger: Optional[str] = None, save: bool = True):
    """複数銘柄のOHLCVからLSTMを学習する（銘柄ごとに系列を構成して連結）。

    news_map: {symbol: news_df} もしくは {index: news_df}（任意）。use_news_features が
    有効な場合に各銘柄のニュース特徴量を結合する。dfs と同じ並びの list でも可。
    save=False のときは成果物を書かない（バックテスト用）。

    torch 未導入なら None を返す（学習をスキップ）。
    """
    if not available():
        logger.info("LSTM学習をスキップ（torch 未導入）")
        return None
    import torch

    seq_len = _seq_len()
    from src.strategy.labeling import build_sequence_set

    X_parts, y_parts, w_parts = [], [], []
    for i, df in enumerate(dfs):
        news_df = None
        if news_map is not None:
            news_df = news_map.get(i) if isinstance(news_map, dict) else None
        try:
            Xs, ys, ws = build_sequence_set(df, seq_len, news_df=news_df)
        except ValueError as e:
            logger.warning(f"LSTM系列生成をスキップ: {e}")
            continue
        if len(Xs) == 0:
            continue
        X_parts.append(Xs)
        y_parts.append(ys)
        w_parts.append(ws)

    if not X_parts:
        raise ValueError("LSTM学習データが不足しています: 0件")

    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.float32)
    w = np.concatenate(w_parts).astype(np.float32)

    n_features = X.shape[2]
    hidden = 32
    model = _build_module(n_features, hidden)

    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    wt = torch.from_numpy(w)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    epochs = 15
    batch = 256
    n = len(Xt)
    rng = np.random.default_rng(42)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            logits = model(Xt[idx])
            losses = loss_fn(logits, yt[idx])
            (losses * wt[idx]).mean().backward()
            opt.step()

    logger.info(f"LSTM学習完了: {n}系列 × 長さ{seq_len} × 特徴量{n_features}")

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        _write_meta(n_features, hidden, seq_len, n)

    return model


def _write_meta(n_features: int, hidden: int, seq_len: int, n_samples: int) -> None:
    meta = {
        "sha256": _sha256_file(MODEL_PATH),
        "trained_at": clock.now().isoformat(),
        "n_features": n_features,
        "hidden": hidden,
        "seq_len": seq_len,
        "n_samples": n_samples,
        "features": list(active_feature_cols()),
    }
    _meta_path().write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"LSTMメタ保存: sha256={meta['sha256'][:12]}…")


def _read_meta() -> Optional[dict]:
    path = _meta_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load():
    """学習済みLSTMをロードする。成果物が無ければ静かに None（純GBMフォールバック）。

    GBM の load() と異なり、ここでは「成果物・メタが無い＝既定状態」として alert せず
    info ログのみ。SHA256 不一致（改ざん）と特徴量セット desync は fail-closed。
    """
    if not available():
        return None
    if not MODEL_PATH.exists():
        logger.info("LSTM成果物なし（純GBMで動作）")
        return None
    meta = _read_meta()
    if not meta:
        logger.info("LSTMメタなし（純GBMで動作）")
        return None
    try:
        digest = _sha256_file(MODEL_PATH)
    except OSError:
        return None
    if meta.get("sha256") and meta["sha256"] != digest:
        from src.core.alerts import alert
        logger.critical("LSTMモデルのSHA256が記録と不一致（改ざん/破損）。ロード中止。")
        alert("LSTMモデル検証失敗", "models/lstm_model.pt のSHA256が一致しません。")
        return None
    if list(meta.get("features", [])) != list(active_feature_cols()):
        logger.warning("LSTMの特徴量セットが現在の設定と不一致。純GBMで動作（要再学習）。")
        return None

    import torch
    model = _build_module(meta["n_features"], meta["hidden"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    logger.info(f"LSTMロード: sha256={digest[:12]}… trained_at={meta.get('trained_at')}")
    return model


def predict_proba(model, df, news_df=None) -> float:
    """最新の系列で上昇確率を返す（0.0〜1.0）。データ不足・torch無しなら 0.5。"""
    if model is None or not available():
        return 0.5
    import torch
    from src.strategy.indicators import build_features

    cols = active_feature_cols()
    if cols[0] not in df.columns:
        df = build_features(df, news_df=news_df)
    seq_len = _seq_len()
    if df.empty or len(df) < seq_len:
        return 0.5
    seq = df[cols].iloc[-seq_len:].to_numpy(dtype=np.float32)
    with torch.no_grad():
        logit = model(torch.from_numpy(seq).unsqueeze(0))
        proba = torch.sigmoid(logit).item()
    return float(proba)
