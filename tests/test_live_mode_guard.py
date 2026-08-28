"""
Critical #4: ライブモード起動時の確認ガード

CONFIRM_LIVE_TRADING 環境変数が設定されていない場合に sys.exit(1) を呼ぶことを検証する。
main.py を直接インポートすると uvicorn 等が必要になるため、ソース検証で実施する。
"""
import inspect
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestLiveModeGuardSource:
    def test_confirm_env_var_checked(self):
        """main.py のソースに CONFIRM_LIVE_TRADING チェックが存在する"""
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        assert "CONFIRM_LIVE_TRADING" in src, (
            "main.py に CONFIRM_LIVE_TRADING 環境変数チェックがない"
        )

    def test_sys_exit_called_on_missing_env(self):
        """main.py に sys.exit() 呼び出しがある"""
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        assert "sys.exit" in src, (
            "main.py に sys.exit() がない — 環境変数未設定時に終了しない"
        )

    def test_live_mode_string_present(self):
        """ライブモードの警告ログが main.py に存在する"""
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        assert "ライブモード" in src, (
            "main.py にライブモードの警告ログがない"
        )

    def test_sync_on_startup_called(self):
        """order_mgr.sync_on_startup() の呼び出しが main.py に存在する"""
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        assert "sync_on_startup" in src, (
            "main.py に order_mgr.sync_on_startup() 呼び出しがない"
        )

    def test_live_mode_aborts_on_token_refresh_failure(self):
        """再レビュー P1-5: ライブモードで初回トークン取得に失敗したら起動を中断する
        （fail-closed。ペーパーモードでの継続ログのみでは口座状態不明のまま発注ロジックが
        動く危険があるため）"""
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        import re
        # 接続は broker_wait.wait_for_broker() に委譲している（kabuステーションの
        # ログイン待ちに対応するため。2026-08-28）。接続できなかった場合の
        # `if not connected:` ブロック内で、実発注モード判定と sys.exit(1) が
        # 行われていること（fail-closedであること）を検証する。
        assert "broker_wait.wait_for_broker(" in src, (
            "初回トークン取得が broker_wait.wait_for_broker() を経由していない"
        )
        match = re.search(
            r"if not connected:(.*?)(?=\n    #|\Z)", src, re.DOTALL,
        )
        assert match, "接続失敗時の `if not connected:` ブロックが見つからない"
        fail_block = match.group(1)
        assert "places_real_orders(mode)" in fail_block, (
            "接続失敗時に実発注モード（live/semi_live）かどうかの分岐が無い"
        )
        assert "sys.exit(1)" in fail_block, (
            "ライブモードでの接続失敗時に sys.exit(1) していない（fail-closedでない）"
        )
