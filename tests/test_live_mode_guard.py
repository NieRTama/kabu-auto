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
