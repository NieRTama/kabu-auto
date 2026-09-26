"""
RiskManager.get_current_prices() の短期キャッシュのテスト。

2026-09-11、ダッシュボードの /api/positions と /api/pnl_enhanced_summary が
それぞれ独立に get_current_prices() → price_fn（=client.get_board のREST呼び出し）
を呼んでおり、フロントエンドがこの2つを10秒ごとに Promise.all で同時実行するため、
同じ保有銘柄に対して10秒ごとに最低2重のAPI呼び出しが発生していた。これに朝/後場の
売買判定ループの get_board 呼び出しが重なり、kabuステーションAPIの実行回数制限
（Code=4001006）に達して429が多発し、直近15分で495件の警告が出た。

対処: 短いTTL（数秒）のキャッシュを price_fn の手前に挟み、同一銘柄への
重複問い合わせを吸収する。売買発注の価格決定（trading.py が直接 get_board を
呼ぶ経路）はこのキャッシュを経由しないため影響を受けない。
"""
from unittest.mock import patch

from src.risk.manager import RiskManager


class TestGetCurrentPricesCache:
    def test_second_call_within_ttl_does_not_call_price_fn_again(self, isolated_db):
        """TTL内の2回目の呼び出しはキャッシュを使い、price_fnを再度呼ばない
        （ダッシュボードの2エンドポイントが同時に叩く重複を吸収する）"""
        calls = []
        risk = RiskManager(price_fn=lambda syms: calls.append(list(syms)) or {"9432": 172.7})

        with patch("src.risk.manager.time.monotonic", return_value=1000.0):
            first = risk.get_current_prices(["9432"])
            second = risk.get_current_prices(["9432"])

        assert first == {"9432": 172.7}
        assert second == {"9432": 172.7}
        assert len(calls) == 1, "TTL内の2回目はキャッシュから返し、price_fnを呼ばないこと"

    def test_call_after_ttl_expires_calls_price_fn_again(self, isolated_db):
        """TTLを過ぎたら再度price_fnを呼ぶ（リアルタイム性を失わないこと）"""
        calls = []
        risk = RiskManager(price_fn=lambda syms: calls.append(list(syms)) or {"9432": 172.7})

        with patch("src.risk.manager.time.monotonic", return_value=1000.0):
            risk.get_current_prices(["9432"])
        with patch("src.risk.manager.time.monotonic", return_value=1000.0 + RiskManager._PRICE_CACHE_TTL_SEC + 1):
            risk.get_current_prices(["9432"])

        assert len(calls) == 2, "TTL経過後はキャッシュを使わず再取得すること"

    def test_uncached_symbol_is_fetched_while_cached_one_is_reused(self, isolated_db):
        """一部銘柄だけキャッシュ済みの場合、未キャッシュの銘柄だけを問い合わせる"""
        calls = []

        def price_fn(syms):
            calls.append(list(syms))
            return {s: 100.0 for s in syms}

        risk = RiskManager(price_fn=price_fn)

        with patch("src.risk.manager.time.monotonic", return_value=1000.0):
            risk.get_current_prices(["9432"])
            result = risk.get_current_prices(["9432", "9434"])

        assert result == {"9432": 100.0, "9434": 100.0}
        assert calls[-1] == ["9434"], "既にキャッシュ済みの9432を再問い合わせしないこと"

    def test_no_price_fn_behaves_exactly_like_before(self, isolated_db):
        """price_fn未注入（既定None）ならキャッシュ機構が介入せず、従来どおり
        OHLCV終値のみを使う（回帰防止）"""
        risk = RiskManager()
        assert risk.get_current_prices(["9432"]) == {}
