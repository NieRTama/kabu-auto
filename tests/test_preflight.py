"""
ライブ起動前プリフライトチェックのテスト（Phase 3 / 4.4）
"""
import socket
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import src.core.preflight as pf
from src.execution import order_status as st


@pytest.fixture
def no_halt(tmp_path):
    from src.core import halt
    halt.load(str(tmp_path / "nohalt.json"))
    yield
    halt.load(str(tmp_path / "nohalt.json"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _good_client():
    c = MagicMock()
    c.get_orders.return_value = []
    c.get_positions.return_value = []
    c.get_wallet.return_value = {}
    return c


@contextmanager
def _patch_db(unresolved=0):
    session = MagicMock()
    session.scalar.return_value = unresolved

    @contextmanager
    def ctx():
        yield session

    with patch.object(pf, "get_session", ctx):
        yield


class TestPreflight:
    def test_all_pass(self, no_halt):
        with _patch_db(unresolved=0):
            result = pf.run_preflight(
                _good_client(), "live",
                base_url="http://localhost:18080/kabusapi",
                dash_host="127.0.0.1", dash_port=_free_port(),
            )
        assert result["ok"] is True

    def test_api_failure_is_critical(self, no_halt):
        client = _good_client()
        client.get_wallet.side_effect = RuntimeError("接続不可")
        with _patch_db(unresolved=0):
            result = pf.run_preflight(
                client, "live",
                base_url="http://localhost:18080/kabusapi",
                dash_host="127.0.0.1", dash_port=_free_port(),
            )
        assert result["ok"] is False
        wallet = next(c for c in result["checks"] if "wallet" in c["name"])
        assert wallet["ok"] is False and wallet["level"] == pf.CRITICAL

    def test_unresolved_orders_block(self, no_halt):
        with _patch_db(unresolved=2):
            result = pf.run_preflight(
                _good_client(), "live",
                base_url="http://localhost:18080/kabusapi",
                dash_host="127.0.0.1", dash_port=_free_port(),
            )
        assert result["ok"] is False
        chk = next(c for c in result["checks"] if c["name"] == "未解決注文の確認")
        assert chk["ok"] is False

    def test_occupied_port_blocks(self, no_halt):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occ:
            occ.bind(("127.0.0.1", 0))
            occ.listen(1)
            port = occ.getsockname()[1]
            with _patch_db(unresolved=0):
                result = pf.run_preflight(
                    _good_client(), "live",
                    base_url="http://localhost:18080/kabusapi",
                    dash_host="127.0.0.1", dash_port=port,
                )
        assert result["ok"] is False

    def test_test_port_18081_is_critical_for_live(self, no_halt):
        with _patch_db(unresolved=0):
            result = pf.run_preflight(
                _good_client(), "live",
                base_url="http://localhost:18081/kabusapi",
                dash_host="127.0.0.1", dash_port=_free_port(),
            )
        assert result["ok"] is False
        chk = next(c for c in result["checks"] if c["name"] == "モード↔エンドポイント整合")
        assert chk["ok"] is False

    def test_halt_on_is_warning_not_blocking(self, no_halt):
        from src.core import halt
        halt.engage("テスト")
        try:
            with _patch_db(unresolved=0):
                result = pf.run_preflight(
                    _good_client(), "live",
                    base_url="http://localhost:18080/kabusapi",
                    dash_host="127.0.0.1", dash_port=_free_port(),
                )
        finally:
            halt.release()
        # kill switch は warning なので ok は True のまま（致命ではない）
        assert result["ok"] is True
        chk = next(c for c in result["checks"] if c["name"] == "取引停止スイッチ")
        assert chk["ok"] is False and chk["level"] == pf.WARNING


class TestDryRunProductionDataWarning:
    """再レビュー P1-3: dry_run が本番用エンドポイント(18081以外)から口座データを
    読んでいることを大きく警告する（致命的ではないので起動は止めない）。"""

    def test_dry_run_against_production_endpoint_warns(self, no_halt):
        with _patch_db(unresolved=0):
            result = pf.run_preflight(
                _good_client(), "dry_run",
                base_url="http://localhost:18080/kabusapi",
                dash_host="127.0.0.1", dash_port=_free_port(),
            )
        assert result["ok"] is True  # warningなので起動は止めない
        chk = next(c for c in result["checks"] if "dry_run" in c["name"])
        assert chk["ok"] is False and chk["level"] == pf.WARNING

    def test_dry_run_against_test_endpoint_does_not_warn(self, no_halt):
        with _patch_db(unresolved=0):
            result = pf.run_preflight(
                _good_client(), "dry_run",
                base_url="http://localhost:18081/kabusapi",
                dash_host="127.0.0.1", dash_port=_free_port(),
            )
        chk = next(c for c in result["checks"] if "dry_run" in c["name"])
        assert chk["ok"] is True

    def test_live_mode_does_not_trigger_dry_run_warning(self, no_halt):
        with _patch_db(unresolved=0):
            result = pf.run_preflight(
                _good_client(), "live",
                base_url="http://localhost:18080/kabusapi",
                dash_host="127.0.0.1", dash_port=_free_port(),
            )
        chk = next(c for c in result["checks"] if "dry_run" in c["name"])
        assert chk["ok"] is True


class TestMainPyPreflightWiring:
    def test_main_calls_preflight(self):
        with open("main.py", encoding="utf-8") as f:
            src = f.read()
        assert "run_preflight" in src
        assert "places_real_orders" in src
