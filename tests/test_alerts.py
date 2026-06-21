"""
通知プロバイダ抽象化（LINE Notify → Discord Webhook移行）のテスト

LINE Notifyは2025年3月31日にサービス終了済みのため、Discord Webhookを
唯一の現行プロバイダとして実装した。将来Slack/Telegram等を追加する際も
alert()の呼び出し側を変更せずに済むことを保証するための回帰テストを含む。
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

import src.core.alerts as alerts_mod
from src.core.alerts import DiscordWebhookProvider, alert, build_providers


class TestDiscordWebhookProviderSend:
    def test_posts_correct_payload(self):
        provider = DiscordWebhookProvider("https://discord.example/webhook/abc")
        with patch.object(alerts_mod.requests, "post") as mock_post:
            mock_post.return_value = MagicMock(raise_for_status=MagicMock())
            provider.send("hello")
        mock_post.assert_called_once_with(
            "https://discord.example/webhook/abc",
            json={"content": "hello"},
            timeout=10,
        )

    def test_short_message_not_truncated(self):
        provider = DiscordWebhookProvider("https://discord.example/webhook/abc")
        with patch.object(alerts_mod.requests, "post") as mock_post:
            mock_post.return_value = MagicMock(raise_for_status=MagicMock())
            provider.send("x" * 100)
        sent = mock_post.call_args.kwargs["json"]["content"]
        assert sent == "x" * 100

    def test_long_message_truncated_within_limit(self):
        provider = DiscordWebhookProvider("https://discord.example/webhook/abc")
        with patch.object(alerts_mod.requests, "post") as mock_post:
            mock_post.return_value = MagicMock(raise_for_status=MagicMock())
            provider.send("x" * 3000)
        sent = mock_post.call_args.kwargs["json"]["content"]
        assert len(sent) <= alerts_mod.DISCORD_MAX_CONTENT_LENGTH
        assert sent.endswith(alerts_mod._TRUNCATION_SUFFIX)

    def test_http_error_propagates(self):
        provider = DiscordWebhookProvider("https://discord.example/webhook/abc")
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        with patch.object(alerts_mod.requests, "post", return_value=resp):
            with pytest.raises(requests.exceptions.HTTPError):
                provider.send("hello")

    def test_network_error_propagates(self):
        provider = DiscordWebhookProvider("https://discord.example/webhook/abc")
        with patch.object(alerts_mod.requests, "post",
                          side_effect=requests.exceptions.ConnectionError("down")):
            with pytest.raises(requests.exceptions.ConnectionError):
                provider.send("hello")


class TestBuildProviders:
    def test_empty_when_unconfigured(self):
        with patch.object(alerts_mod.cfg, "get_section", return_value={}):
            assert build_providers() == []

    def test_includes_discord_when_configured(self):
        with patch.object(alerts_mod.cfg, "get_section",
                          return_value={"discord_webhook_url": "https://discord.example/x"}):
            providers = build_providers()
        assert len(providers) == 1
        assert isinstance(providers[0], DiscordWebhookProvider)


class TestAlert:
    def test_no_providers_logs_only_no_exception(self):
        with patch.object(alerts_mod, "build_providers", return_value=[]):
            alert("タイトル", "本文")  # 例外が出ないこと

    def test_sends_to_configured_provider_with_formatted_text(self):
        provider = MagicMock()
        provider.name = "discord"
        with patch.object(alerts_mod, "build_providers", return_value=[provider]):
            alert("タイトル", "本文")
        provider.send.assert_called_once_with("【kabu-auto】タイトル\n本文")

    def test_one_provider_failure_does_not_block_others(self):
        failing = MagicMock()
        failing.name = "discord"
        failing.send.side_effect = RuntimeError("boom")
        succeeding = MagicMock()
        succeeding.name = "other"
        with patch.object(alerts_mod, "build_providers", return_value=[failing, succeeding]):
            alert("タイトル", "本文")  # 例外を外に投げない
        failing.send.assert_called_once()
        succeeding.send.assert_called_once()

    def test_all_providers_failing_does_not_raise(self):
        failing = MagicMock()
        failing.name = "discord"
        failing.send.side_effect = RuntimeError("boom")
        with patch.object(alerts_mod, "build_providers", return_value=[failing]):
            alert("タイトル", "本文")  # 取引処理を壊さないため例外を外に投げない


class TestEnvOverride:
    def test_discord_webhook_url_env_overrides_config(self, monkeypatch, tmp_path):
        import src.core.config as cfg

        config_path = tmp_path / "config.yaml"
        config_path.write_text("alerts:\n  discord_webhook_url: \"\"\n", encoding="utf-8")
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/from-env")
        cfg.load(str(config_path))
        assert cfg.get_section("alerts")["discord_webhook_url"] == "https://discord.example/from-env"
