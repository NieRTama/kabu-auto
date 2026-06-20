"""
main.py における多重起動防止ロックの結線テスト（再レビュー P1-1）

main.py は uvicorn 等の重い依存をトップレベルでimportするため、他の起動系テスト
（test_port_conflict.py 等）と同様にソース検証で行う。
"""
import re


def _read_main() -> str:
    with open("main.py", encoding="utf-8") as f:
        return f.read()


class TestMainPyProcessLockWiring:
    def test_acquires_lock_at_startup(self):
        src = _read_main()
        assert "process_lock.acquire(" in src

    def test_non_paper_mode_aborts_on_lock_conflict(self):
        src = _read_main()
        match = re.search(
            r"if lock_ok:(.*?)(?=\n    # ─── 実発注モード)",
            src, re.DOTALL,
        )
        assert match, "process_lock.acquire の結果分岐ブロックが見つからない"
        block = match.group(1)
        assert "paper" in block and "sys.exit(1)" in block, (
            "paper以外のモードで多重起動検知時に sys.exit(1) していない"
        )

    def test_releases_lock_via_atexit(self):
        """ロック解放は atexit に登録され、sys.exit(1)・例外を含むあらゆる終了経路を
        カバーする（フォローレビュー対応）。KeyboardInterrupt ハンドラ内の明示呼び出しは廃止。"""
        src = _read_main()
        assert "atexit.register(process_lock.release)" in src

    def test_respects_allow_multiple_paper_instances_config(self):
        src = _read_main()
        assert "allow_multiple_paper_instances" in src

    def test_dry_run_production_read_warning_in_banner(self):
        src = _read_main()
        assert "dry_run" in src and "本番口座データ" in src
