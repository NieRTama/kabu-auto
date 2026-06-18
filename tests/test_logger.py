"""
ログ出力の日付付きファイル名・保持期間設定・レベル分割のテスト

要件: ログは1日単位で区切り、ファイル名末尾に日付を入れ、15日周期でローテーション。
さらに INFO以下（DEBUG/INFO）と WARNING以上（WARNING/ERROR/CRITICAL）を別ファイルに分ける。
"""
from pathlib import Path
from unittest.mock import patch

import src.core.logger as log_setup
from loguru import logger as loguru_logger


class TestDatedLogPath:
    def test_inserts_date_token_before_suffix(self):
        result = log_setup.dated_log_path("data/kabu_auto.log")
        assert result.replace("\\", "/") == "data/kabu_auto_{time:YYYY-MM-DD}.log"

    def test_keeps_directory_and_extension(self):
        result = Path(log_setup.dated_log_path("logs/app.log"))
        assert result.parent.name == "logs"
        assert result.suffix == ".log"
        assert "{time:YYYY-MM-DD}" in result.name

    def test_handles_path_without_directory(self):
        result = log_setup.dated_log_path("kabu_auto.log")
        assert result == "kabu_auto_{time:YYYY-MM-DD}.log"

    def test_tag_is_inserted_before_date(self):
        result = log_setup.dated_log_path("data/kabu_auto.log", tag="warning")
        assert result.replace("\\", "/") == "data/kabu_auto_warning_{time:YYYY-MM-DD}.log"

    def test_no_tag_matches_default_behavior(self):
        assert log_setup.dated_log_path("data/kabu_auto.log", tag="") == \
            log_setup.dated_log_path("data/kabu_auto.log")


class TestSetupUsesConfig:
    def _capture_file_sinks(self, conf):
        """logger.add() への全呼び出しのうち、ファイルシンク（文字列パス）を記録する。"""
        calls = []

        def fake_add(sink, **kwargs):
            if isinstance(sink, str):
                calls.append((sink, kwargs))
            return 1

        with patch.object(log_setup.cfg, "get_section", lambda s: conf.get(s, {})):
            with patch.object(log_setup.logger, "add", side_effect=fake_add), \
                 patch.object(log_setup.logger, "remove"), \
                 patch.object(log_setup.logger, "info"):
                log_setup.setup()
        return calls

    def test_creates_two_file_sinks_info_and_warning(self, tmp_path):
        """INFO以下用とWARNING以上用、2つのファイルシンクが作られる"""
        conf = {"logging": {"level": "INFO", "file": str(tmp_path / "kabu_auto.log")}}
        calls = self._capture_file_sinks(conf)
        assert len(calls) == 2
        names = [Path(c[0]).name for c in calls]
        assert any("warning" in n for n in names)
        assert any("warning" not in n for n in names)

    def test_info_sink_has_filter_excluding_warning(self, tmp_path):
        conf = {"logging": {"level": "INFO", "file": str(tmp_path / "kabu_auto.log")}}
        calls = self._capture_file_sinks(conf)
        info_call = next(c for c in calls if "warning" not in Path(c[0]).name)
        assert "filter" in info_call[1]
        filter_fn = info_call[1]["filter"]
        assert filter_fn({"level": loguru_logger.level("INFO")}) is True
        assert filter_fn({"level": loguru_logger.level("WARNING")}) is False
        assert filter_fn({"level": loguru_logger.level("ERROR")}) is False

    def test_warning_sink_level_is_warning(self, tmp_path):
        conf = {"logging": {"level": "INFO", "file": str(tmp_path / "kabu_auto.log")}}
        calls = self._capture_file_sinks(conf)
        warn_call = next(c for c in calls if "warning" in Path(c[0]).name)
        assert warn_call[1]["level"] == "WARNING"

    def test_both_sinks_use_dated_path_and_default_15day_retention(self, tmp_path):
        conf = {"logging": {"level": "INFO", "file": str(tmp_path / "kabu_auto.log")}}
        calls = self._capture_file_sinks(conf)
        for sink, kwargs in calls:
            assert "{time:YYYY-MM-DD}" in sink
            assert kwargs["retention"] == "15 days"
            assert kwargs["rotation"] == "00:00"

    def test_setup_creates_log_directory(self, tmp_path):
        """設定されたログ格納フォルダ（例: log/）が無ければ作成される"""
        log_dir = tmp_path / "log"
        assert not log_dir.exists()
        conf = {"logging": {"level": "INFO", "file": str(log_dir / "kabu_auto.log")}}
        with patch.object(log_setup.cfg, "get_section", lambda s: conf.get(s, {})):
            with patch.object(log_setup.logger, "add", return_value=1), \
                 patch.object(log_setup.logger, "remove"), \
                 patch.object(log_setup.logger, "info"):
                log_setup.setup()
        assert log_dir.is_dir()

    def test_setup_respects_custom_retention(self, tmp_path):
        conf = {"logging": {"level": "INFO", "file": str(tmp_path / "x.log"), "retention": "7 days"}}
        calls = self._capture_file_sinks(conf)
        for _, kwargs in calls:
            assert kwargs["retention"] == "7 days"


class TestActualLevelSplitBehavior:
    """実際に loguru を初期化し、レベルごとに正しいファイルへ振り分けられることを確認する"""

    def test_info_and_warning_go_to_separate_files(self, tmp_path):
        conf = {"logging": {"level": "INFO", "file": str(tmp_path / "kabu_auto.log")}}
        with patch.object(log_setup.cfg, "get_section", lambda s: conf.get(s, {})):
            log_setup.setup()
        try:
            loguru_logger.info("info message")
            loguru_logger.warning("warning message")
            loguru_logger.error("error message")
            loguru_logger.complete()

            info_files = list(tmp_path.glob("kabu_auto_2*.log"))
            warn_files = list(tmp_path.glob("kabu_auto_warning_2*.log"))
            assert len(info_files) == 1
            assert len(warn_files) == 1

            info_content = info_files[0].read_text(encoding="utf-8")
            warn_content = warn_files[0].read_text(encoding="utf-8")

            assert "info message" in info_content
            assert "warning message" not in info_content
            assert "error message" not in info_content

            assert "warning message" in warn_content
            assert "error message" in warn_content
            assert "info message" not in warn_content
        finally:
            loguru_logger.remove()
