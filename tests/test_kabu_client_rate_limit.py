"""
kabuステーションAPIの429（Code=4001006 API実行回数エラー）に対するリトライのテスト。

2026-09-11、ダッシュボードのポーリングと朝/後場の売買判定ループが同時にREST APIを
叩き、kabuステーションAPIの実行回数制限（429）に達して警告が15分で495件発生した。
このうち一部は「後場買い発注失敗: 7453/4755」のように**発注機会そのものを失う**
実害にもつながっていた（get_board の429がそのまま例外としてループに伝播していた）。

対処: 429だけは一時的な輻輳とみなし、短いバックオフを挟んで数回リトライする
（401/500など429以外はリトライ対象にしない＝原因の異なるエラーを誤って隠さない）。
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

import src.api.kabu_client as mod
from src.core import broker_auth


@pytest.fixture(autouse=True)
def _reset():
    broker_auth.reset()
    yield
    broker_auth.reset()


def _client():
    with patch.object(mod, "cfg") as cfg_mock:
        cfg_mock.get_section.return_value = {"base_url": "http://localhost:18080/kabusapi"}
        return mod.KabuClient()


def _response(status: int, payload=None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.reason = {200: "OK", 429: "Too Many Requests"}.get(status, "")
    resp.text = ""
    resp.json.return_value = payload if payload is not None else {}
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status} Client Error")
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestRateLimitRetry:
    def test_retries_after_429_and_succeeds(self):
        """429を1回受けても、直後に成功すればリトライして結果を返すこと"""
        c = _client()
        responses = [_response(429), _response(200, {"CurrentPrice": 174.6})]
        with patch.object(mod.requests, "request", side_effect=responses):
            with patch.object(mod.time, "sleep") as sleep_mock:
                result = c.get_board("9432")
        assert result["CurrentPrice"] == 174.6
        assert sleep_mock.called, "リトライ前にバックオフ待機すること"

    def test_gives_up_after_max_retries_and_raises(self):
        """429が上限回数を超えて続く場合は、諦めて例外を出すこと（無限リトライしない）"""
        c = _client()
        with patch.object(mod.requests, "request", return_value=_response(429)):
            with patch.object(mod.time, "sleep"):
                with pytest.raises(requests.HTTPError):
                    c.get_board("9432")

    def test_non_429_error_is_not_retried(self):
        """401等429以外は即座に例外にする（原因の異なるエラーを429対策で隠さない）"""
        c = _client()
        call_count = []

        def _request(*args, **kwargs):
            call_count.append(1)
            return _response(401)

        with patch.object(mod.requests, "request", side_effect=_request):
            with pytest.raises(requests.HTTPError):
                c.get_board("9432")
        assert len(call_count) == 1, "429以外はリトライせず1回で終わること"

    def test_429_still_marks_expired_state_untouched(self):
        """429はレート制限であって認証切れではないため、broker_authの状態を変えないこと"""
        c = _client()
        with patch.object(mod.requests, "request", return_value=_response(429)):
            with patch.object(mod.time, "sleep"):
                with pytest.raises(requests.HTTPError):
                    c.get_board("9432")
        assert broker_auth.is_expired() is False
