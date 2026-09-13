"""KabuS.exe（kabuステーション）の起動・再起動操作に対する、プロセス全体で共有する排他ロック。

## 背景

`broker_launcher.py`（5分毎の生存監視による自動起動）と `broker_full_login.py`
（完全自動ログイン、平日06:45の定時実行）は、それぞれ独立に KabuS.exe の起動/再起動を
行う。両モジュールがそれぞれ独自の `threading.Lock` を持っていたため、両方の自動実行
フラグ（`auto_launch_broker` / `broker_full_login_enabled`）を同時に有効化すると、
互いの存在を知らない2つのロックが、それぞれ安全なつもりで同じ KabuS.exe を同時に
起動/再起動しうるという設計上の隙間があった（2026-09-13、統合テストで発見。
現状は両方 false の段階導入中で実害はまだ無い）。

## 使い方

KabuS.exe の起動・再起動を行うすべてのモジュールは、この `lock` を使うこと。
新しいロックを作らない（それが今回の問題の原因だった）。
"""
import threading

lock = threading.Lock()
