"""稼働中のプロセスがどのコードで動いているかを記録し、手元の変更が未反映なら知らせる。

2026-09-30 に入れた修正が、9/28 起動のプロセスでは動かないまま10日間眠り、
10/9 の再起動で初めて有効になって 10/13 に不具合として表に出た。設定ファイルと
同じく「直したら、稼働中のプロセスが読み直したか」を人の記憶に頼っていたのが原因。

判定は「起動時刻より後に更新されたコード・設定があるか」で行う（gitのコミット
比較だけでは、未コミットの編集を見逃すため）。表示用に起動時のコミットも記録する。
"""
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[2]
_state: dict = {}


def _head_commit(repo: Path) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def capture(repo: Path = _REPO) -> None:
    """起動時に1回呼ぶ。起動時刻と、その時点のコミットを記録する。"""
    _state.clear()
    _state.update(repo=Path(repo), started_at=time.time(), commit=_head_commit(Path(repo)))


def describe() -> str:
    """稼働中のコードの説明（例: 'abc1234（起動 10/13 22:19）'）。"""
    if not _state:
        return "不明"
    started = datetime.fromtimestamp(_state["started_at"]).strftime("%m/%d %H:%M")
    return f"{_state.get('commit') or 'コミット不明'}（起動 {started}）"


def stale_notice() -> Optional[str]:
    """起動後に更新されたコード・設定があれば注意文を返す。無ければ（判定不能も）None。"""
    if not _state:
        return None
    repo: Path = _state["repo"]
    started_at: float = _state["started_at"]
    candidates = [repo / "main.py", repo / "config.yaml", *(repo / "src").rglob("*.py")]
    changed = sorted(
        p.relative_to(repo).as_posix() for p in candidates
        if p.exists() and p.stat().st_mtime > started_at
    )
    if not changed:
        return None
    shown = ", ".join(changed[:3]) + (f" ほか{len(changed) - 3}件" if len(changed) > 3 else "")
    return f"⚠ 未反映の変更があります（起動後に更新: {shown}）。再起動で反映されます"
