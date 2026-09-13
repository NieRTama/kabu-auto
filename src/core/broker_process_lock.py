"""KabuS.exe（kabuステーション）の起動・再起動操作に対する、プロセス全体で共有する排他制御。

## 背景

`broker_launcher.py`（5分毎の生存監視による自動起動）と `broker_full_login.py`
（完全自動ログイン、平日06:45の定時実行）は、それぞれ独立に KabuS.exe の起動/再起動を
行う。両モジュールがそれぞれ独自の `threading.Lock` と独自のクールダウン／実行中状態を
持っていたため、両方の自動実行フラグ（`auto_launch_broker` / `broker_full_login_enabled`）
を同時に有効化すると、互いの存在を知らない2つの状態管理が、それぞれ安全なつもりで
同じ KabuS.exe を同時に起動/再起動しうるという設計上の隙間があった
（2026-09-13、統合テストで発見）。

**ロックオブジェクトを共有するだけでは解決しない**（最初の対応（`lock` のみ共有）で
見つかった問題、2026-09-13）。`broker_full_login.run()` は外部呼び出し
（最大180秒）の間ロックを手放す設計のため、その間に `broker_launcher.launch()` が
同じロックを取得できてしまい、「実行中かどうか」を判定する状態がモジュールごとに
別々のままでは検知できず、実際に再現テストで二重起動が成立した。

このため、「今まさに起動/再起動の操作が進行中か」（`in_progress`）と「直近の操作が
完了した時刻」（`last_operation_at`、クールダウン判定の起点）という、
**物理的に1つのKabuS.exeプロセスに対する状態**は、どちらのモジュールが起こした
操作かに関わらずこのモジュールで共有する。（1日あたりの試行回数上限は、
モジュールごとに別の予算として意図的に分離したまま残す——launcher と
full_login は性質の異なる別の安全弁であり、統合する理由が無い。）

## 使い方

KabuS.exe の起動・再起動を行うすべてのモジュールは、`lock` に加えて
`in_progress` / `last_operation_at` を使うこと。新しい独自の状態を作らない
（それが今回の問題の原因だった）。読み書きは必ず `lock` 保持中に行う。

`in_progress` / `last_operation_at` は可変オブジェクトではない（bool / float）ため、
`x = broker_process_lock.in_progress` のようにローカル変数へ束縛して書き換えても
元の値は更新されない。値を変えるときは必ず `broker_process_lock.in_progress = ...`
と、このモジュールの属性として直接代入すること。
"""
import threading

lock = threading.Lock()

# 直近の起動/再起動操作が完了した時刻（time.monotonic() 系列）。
# クールダウン判定の起点として両モジュールで共有する。
last_operation_at: float = -float('inf')

# 起動/再起動の操作が現在進行中かどうか。`lock` 保持中に読み書きする。
in_progress: bool = False


def reset() -> None:
    """テスト用に状態を初期化する。"""
    global last_operation_at, in_progress
    with lock:
        last_operation_at = -float('inf')
        in_progress = False
