"""
ダッシュボードのポート競合検知のテスト（再レビュー P1-6対応）

is_port_available() の単体テストと、main.py がライブモードで
ポート競合時に起動中断することのソース検証（main.py は uvicorn 等の
重い依存をトップレベルでimportするため、他の起動系テストと同様にソース検証で行う）。
"""
import socket

from src.core.netutil import is_port_available


class TestIsPortAvailable:
    def test_free_port_is_available(self):
        # OS に空きポートを割らせてから一旦閉じる
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        assert is_port_available("127.0.0.1", free_port) is True

    def test_occupied_port_is_not_available(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            port = occupied.getsockname()[1]
            assert is_port_available("127.0.0.1", port) is False


class TestMainPyPortConflictSource:
    def test_checks_port_before_starting_dashboard(self):
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        assert "is_port_available" in src, (
            "main.py がダッシュボード起動前にポート空き確認をしていない"
        )

    def test_live_mode_aborts_on_port_conflict(self):
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        import re
        match = re.search(
            r"if not is_port_available\(.*?\):(.*?)(?=\n    dash_thread)",
            src, re.DOTALL,
        )
        assert match, "is_port_available の判定ブロックが見つからない"
        block = match.group(1)
        assert "live" in block and "sys.exit(1)" in block, (
            "ライブモードでポート競合時に sys.exit(1) していない"
        )
