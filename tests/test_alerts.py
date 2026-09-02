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
        # 先頭に重要度の記号が付く（引数を省略した従来の呼び出しは critical = 🔴）。
        # 記号は通知一覧で対応要否を見分けるためのもので、送信先や送信可否は変えない
        # （詳細は tests/test_alert_levels.py）。
        provider.send.assert_called_once_with("🔴【kabu-auto】タイトル\n本文")

    def test_one_provider_failure_does_not_block_others(self):
        failing = MagicMock()
        failing.name = "discord"
        failing.send.side_effect = RuntimeError("boom")
        succeeding = MagicMock()
        succeeding.name = "other"
        with patch.object(alerts_mod, "build_providers", return_value=[failing, succeeding]), \
             patch.object(alerts_mod.time, "sleep"):
            alert("タイトル", "本文")  # 例外を外に投げない
        # 失敗側は再送を尽くしたうえで諦める
        assert failing.send.call_count == alerts_mod.ALERT_SEND_ATTEMPTS
        succeeding.send.assert_called_once()

    def test_all_providers_failing_does_not_raise(self):
        failing = MagicMock()
        failing.name = "discord"
        failing.send.side_effect = RuntimeError("boom")
        with patch.object(alerts_mod, "build_providers", return_value=[failing]), \
             patch.object(alerts_mod.time, "sleep"):
            alert("タイトル", "本文")  # 取引処理を壊さないため例外を外に投げない


class TestSendRetry:
    """通知の再送（2026-09-02 の 8:45 ハートビート消失が起点）。

    DNS解決失敗 (getaddrinfo failed) で送信できず、再送も無くそのまま消えた。
    「⚪が来ないこと自体が異常」という運用前提は、送信失敗を握り潰すと崩れる。
    """

    def _provider(self, side_effect=None):
        p = MagicMock()
        p.name = "discord"
        p.send.side_effect = side_effect
        return p

    def test_transient_failure_then_success(self):
        """一時的な失敗は再送で回復し、失敗として記録しない"""
        p = self._provider(side_effect=[requests.ConnectionError("getaddrinfo failed"), None])
        with patch.object(alerts_mod.time, "sleep") as slept:
            assert alerts_mod._send_one(p, "本文") is True
        assert p.send.call_count == 2
        slept.assert_called_once_with(alerts_mod.ALERT_RETRY_BASE_DELAY)

    def test_gives_up_after_attempts(self):
        p = self._provider(side_effect=requests.ConnectionError("getaddrinfo failed"))
        with patch.object(alerts_mod.time, "sleep"):
            assert alerts_mod._send_one(p, "本文") is False
        assert p.send.call_count == alerts_mod.ALERT_SEND_ATTEMPTS

    def test_backoff_is_exponential(self):
        p = self._provider(side_effect=requests.ConnectionError("x"))
        with patch.object(alerts_mod.time, "sleep") as slept:
            alerts_mod._send_one(p, "本文", attempts=3)
        assert [c.args[0] for c in slept.call_args_list] == [
            alerts_mod.ALERT_RETRY_BASE_DELAY, alerts_mod.ALERT_RETRY_BASE_DELAY * 2,
        ]

    def _http_error(self, status: int) -> requests.HTTPError:
        resp = MagicMock()
        resp.status_code = status
        return requests.HTTPError(f"{status}", response=resp)

    def test_invalid_webhook_is_not_retried(self):
        """404（Webhook削除・URL誤り）は何度送っても直らないので即あきらめる"""
        p = self._provider(side_effect=self._http_error(404))
        with patch.object(alerts_mod.time, "sleep") as slept:
            assert alerts_mod._send_one(p, "本文") is False
        p.send.assert_called_once()
        slept.assert_not_called()

    def test_server_error_is_retried(self):
        """Discord側の一時障害は再送する"""
        p = self._provider(side_effect=[self._http_error(503), None])
        with patch.object(alerts_mod.time, "sleep"):
            assert alerts_mod._send_one(p, "本文") is True
        assert p.send.call_count == 2

    def test_rate_limit_is_retried(self):
        p = self._provider(side_effect=[self._http_error(429), None])
        with patch.object(alerts_mod.time, "sleep"):
            assert alerts_mod._send_one(p, "本文") is True
        assert p.send.call_count == 2


class TestEnvOverride:
    def test_discord_webhook_url_env_overrides_config(self, monkeypatch, tmp_path):
        import src.core.config as cfg

        config_path = tmp_path / "config.yaml"
        config_path.write_text("alerts:\n  discord_webhook_url: \"\"\n", encoding="utf-8")
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/from-env")
        cfg.load(str(config_path))
        assert cfg.get_section("alerts")["discord_webhook_url"] == "https://discord.example/from-env"
