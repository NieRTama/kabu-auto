"""
ログ出力の日付付きファイル名・保持期間設定のテスト

要件: ログは1日単位で区切り、ファイル名末尾に日付を入れ、15日周期でローテーション。
"""
from pathlib import Path
from unittest.mock import patch

import src.core.logger as log_setup


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


class TestSetupUsesConfig:
    def test_setup_uses_dated_path_and_default_15day_retention(self, tmp_path):
        captured = {}

        def fake_add(sink, **kwargs):
            # ファイルシンク（stderr ではない）の呼び出しを捕捉する
            if isinstance(sink, str):
                captured["sink"] = sink
                captured["kwargs"] = kwargs
            return 1

        conf = {"logging": {"level": "INFO", "file": str(tmp_path / "kabu_auto.log")}}
        with patch.object(log_setup.cfg, "get_section", lambda s: conf.get(s, {})):
            with patch.object(log_setup.logger, "add", side_effect=fake_add), \
                 patch.object(log_setup.logger, "remove"), \
                 patch.object(log_setup.logger, "info"):
                log_setup.setup()

        assert "{time:YYYY-MM-DD}" in captured["sink"]
        assert captured["kwargs"]["retention"] == "15 days"
        assert captured["kwargs"]["rotation"] == "00:00"

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
        captured = {}

        def fake_add(sink, **kwargs):
            if isinstance(sink, str):
                captured["kwargs"] = kwargs
            return 1

        conf = {"logging": {"level": "INFO", "file": str(tmp_path / "x.log"), "retention": "7 days"}}
        with patch.object(log_setup.cfg, "get_section", lambda s: conf.get(s, {})):
            with patch.object(log_setup.logger, "add", side_effect=fake_add), \
                 patch.object(log_setup.logger, "remove"), \
                 patch.object(log_setup.logger, "info"):
                log_setup.setup()

        assert captured["kwargs"]["retention"] == "7 days"
