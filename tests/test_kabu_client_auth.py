"""
REST呼び出しの401を「ログイン認証切れ」として記録することのテスト

2026-09-02 の事故が起点。8:30のトークン更新は成功したあと kabuステーション側の
セッションが切れ、以後497回の401を出し続けたが **誰も認証切れと認識しなかった**。
broker_auth の状態が「有効」に固定されるため、

  - auth_recovery（5分毎）は is_expired() が False なので即 return
  - Discord の reconnect も「認証は有効です」と返すだけ
  - 🔴通知も出ない

となり、kabuステーションにログインし直しても復帰せず、プロセス再起動でしか
直せない状態だった。9:05 の売り注文はこの間に401で失敗している。

「トークン更新の成否」ではなく「実際のAPI応答」で認証状態を判断させる。
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
    resp.json.return_value = payload if payload is not None else {}
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status} Client Error")
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestUnauthorizedMarksExpired:
    """401を見たら認証切れとして記録する（今回の事故の本丸）"""

    def test_get_board_401_marks_expired(self):
        c = _client()
        with patch.object(mod.requests, "request", return_value=_response(401)):
            with pytest.raises(requests.HTTPError):
                c.get_board("9432")
        assert broker_auth.is_expired() is True

    def test_get_positions_401_marks_expired(self):
        """建玉照合は15秒毎に走り、今回もっとも多く401を出した経路"""
        c = _client()
        with patch.object(mod.requests, "request", return_value=_response(401)):
            with pytest.raises(requests.HTTPError):
                c.get_positions()
        assert broker_auth.is_expired() is True

    def test_send_order_401_marks_expired(self):
        """発注が401で弾かれたなら、以後の新規発注は止めなければならない"""
        c = _client()
        with patch.object(mod.requests, "request", return_value=_response(401)):
            with pytest.raises(requests.HTTPError):
                c.send_order({"Symbol": "9432"})
        assert broker_auth.is_expired() is True

    def test_wallet_401_marks_expired(self):
        c = _client()
        with patch.object(mod.requests, "request", return_value=_response(401)):
            with pytest.raises(requests.HTTPError):
                c.get_wallet()
        assert broker_auth.is_expired() is True


class TestOtherOutcomesDoNotMarkExpired:
    """401以外を認証切れと混同しない（誤って発注を止めない）"""

    def test_success_leaves_state_valid(self):
        c = _client()
        with patch.object(mod.requests, "request", return_value=_response(200, {"CurrentPrice": 174.6})):
            assert c.get_board("9432")["CurrentPrice"] == 174.6
        assert broker_auth.is_expired() is False

    def test_server_error_is_not_auth_expiry(self):
        """500はブローカー側の障害。再ログインしても直らないので別扱いにする"""
        c = _client()
        with patch.object(mod.requests, "request", return_value=_response(500)):
            with pytest.raises(requests.HTTPError):
                c.get_positions()
        assert broker_auth.is_expired() is False

    def test_connection_error_is_not_auth_expiry(self):
        """kabuステーション未起動（接続不可）は broker_wait / preflight の担当"""
        c = _client()
        with patch.object(mod.requests, "request",
                          side_effect=requests.ConnectionError("refused")):
            with pytest.raises(requests.ConnectionError):
                c.get_positions()
        assert broker_auth.is_expired() is False


class TestRequestsAreRouted:
    """認証付き呼び出しが共通経路を通ること（新しいメソッドが素通りしないように）"""

    # 共通経路を通さなくてよいメソッドと、その理由。
    #   _request      : 共通経路そのもの
    #   refresh_token : 認証前の /token を叩くため401は「認証切れ」ではなく
    #                   パスワード誤り等。失敗時の扱いは main.py 側が持つ
    _EXEMPT = {"_request", "refresh_token"}

    def test_no_direct_requests_call_in_rest_methods(self):
        """`requests.get(` 等の直呼びが認証付きメソッドに残っていないこと。

        直呼びが1つでも残ると、その経路の401だけ検知できない穴になる。
        メソッドを列挙せず全走査するのは、**新しく追加したメソッドが素通りする**
        のを防ぐため（同種の修正を1か所だけ直して他を見落とす失敗の再発防止）。
        """
        import inspect
        leaked = []
        for name, fn in inspect.getmembers(mod.KabuClient, predicate=inspect.isfunction):
            if name in self._EXEMPT:
                continue
            src = inspect.getsource(fn)
            for verb in ("requests.get(", "requests.post(", "requests.put(",
                         "requests.delete(", "requests.patch("):
                if verb in src:
                    leaked.append(f"{name}() の {verb}")
        assert not leaked, "認証付き呼び出しが共通経路を通っていません: " + ", ".join(leaked)

    def test_query_params_are_forwarded(self):
        """共通化でパラメータが落ちていないこと"""
        c = _client()
        with patch.object(mod.requests, "request", return_value=_response(200, [])) as req:
            c.get_orders({"product": 1})
        assert req.call_args.kwargs["params"] == {"product": 1}

    def test_order_payload_is_forwarded(self):
        c = _client()
        with patch.object(mod.requests, "request", return_value=_response(200, {"OK": 0})) as req:
            c.send_order({"Symbol": "9432", "Qty": 100})
        assert req.call_args.kwargs["json"] == {"Symbol": "9432", "Qty": 100}

    def test_auth_header_is_sent(self):
        c = _client()
        c._token = "tok"
        with patch.object(mod.requests, "request", return_value=_response(200, {})) as req:
            c.get_positions()
        assert req.call_args.kwargs["headers"] == {"X-API-KEY": "tok"}


class TestRecoveryPath:
    """ログインし直せば復帰できること（今回はここが繋がっていなかった）"""

    def test_refresh_token_clears_expired_state(self):
        """認証切れ → 再ログイン → トークン取得成功 で発注が再開できる。

        auth_recovery が refresh_token() を呼び、成功したら mark_valid() する。
        本テストは「401で切れた状態から回復できる」経路が塞がっていないことを守る。
        """
        c = _client()
        with patch.object(mod.requests, "request", return_value=_response(401)):
            with pytest.raises(requests.HTTPError):
                c.get_positions()
        assert broker_auth.is_expired() is True

        with patch.object(mod.requests, "post", return_value=_response(200, {"Token": "new"})):
            assert c.refresh_token() == "new"
        broker_auth.mark_valid()
        assert broker_auth.is_expired() is False
