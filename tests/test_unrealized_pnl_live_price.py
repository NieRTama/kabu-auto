"""
含み損益（未実現）がブローカーのリアルタイム現在値ではなく、前々営業日までの
OHLCV終値で計算されていた問題のテスト。

2026-09-09、実際の保有3銘柄（9432/9434/3387）で実測:
  OHLCV終値ベース（当時の latest_closes()）: 合計 -440円（含み損と表示）
  ブローカーのリアルタイム板（CurrentPrice）: 合計 +7,100円（実際は含み益）

原因は2つ重なっている。
  ① yfinanceの `end` 引数は排他的なため、当日16:00のデータ更新でも
     「前日」までのローソク足しか入らない。さらに祝日・データ配信遅延を挟むと
     「前々営業日」まで遡ることもある
  ② それでも RiskManager.unrealized_pnl() は常に OHLCV終値だけを見ており、
     kabu-autoは既にkabuステーションAPIに接続していて板情報を持っているのに
     それを一切使っていなかった

「手動購入だから」ではなく、保有銘柄全体に一律で効く問題である
（ユーザーからの報告を受けて調査し、この結論に至った）。

対処: RiskManager にリアルタイム価格を取得する関数(price_fn)を注入できるように
する。price_fn が使える銘柄はそちらを優先し、取得できない銘柄（paper運用・
API障害・price_fn未注入）だけ従来通りOHLCV終値へフォールバックする。
"""
from datetime import date

import pandas as pd
import pytest

import src.core.config as cfg
import src.data.database as db
from src.data.database import Position, get_session
from src.data.market_data import upsert_ohlcv
from src.risk.manager import RiskManager


@pytest.fixture
def isolated_db(tmp_path):
    cfg.load("config.yaml")
    cfg.get_section("data")["db_path"] = str(tmp_path / "test.db")
    db.init()
    try:
        yield tmp_path
    finally:
        db._engine = None
        db._Session = None


def _add_position(symbol, quantity, avg_cost, sector=""):
    with get_session() as session:
        session.add(Position(symbol=symbol, quantity=quantity, avg_cost=avg_cost, sector=sector))
        session.commit()


def _set_ohlcv_close(symbol: str, close: float, d: date = date(2026, 1, 1)) -> None:
    df = pd.DataFrame(
        {"open": [close], "high": [close], "low": [close], "close": [close], "volume": [1000]},
        index=pd.to_datetime([d]),
    )
    df.index.name = "date"
    upsert_ohlcv(symbol, df)


class TestGetCurrentPricesPrefersLive:
    def test_uses_price_fn_result_when_available(self, isolated_db):
        _add_position("9432", 100, 160.0)
        _set_ohlcv_close("9432", 168.8)  # 古い終値（低め）
        risk = RiskManager(price_fn=lambda syms: {"9432": 172.7})  # リアルタイム現在値

        prices = risk.get_current_prices(["9432"])

        assert prices["9432"] == 172.7, "OHLCV終値ではなくリアルタイム価格を使うこと"

    def test_falls_back_to_ohlcv_when_price_fn_misses_symbol(self, isolated_db):
        """price_fnが一部銘柄を返せなくても（未対応銘柄・部分失敗）、その銘柄だけ終値で補う"""
        _set_ohlcv_close("9999", 500.0)
        risk = RiskManager(price_fn=lambda syms: {})  # 何も返せない

        prices = risk.get_current_prices(["9999"])

        assert prices["9999"] == 500.0

    def test_falls_back_entirely_when_price_fn_raises(self, isolated_db):
        """price_fn がAPI障害等で例外を出しても、含み損益計算全体を巻き込んで壊さない"""
        _set_ohlcv_close("9999", 500.0)

        def boom(_symbols):
            raise ConnectionError("kabuステーションAPI応答なし")

        risk = RiskManager(price_fn=boom)

        prices = risk.get_current_prices(["9999"])

        assert prices["9999"] == 500.0

    def test_no_price_fn_behaves_exactly_like_before(self, isolated_db):
        """price_fn未注入（既定None）なら従来どおりOHLCV終値のみを使う（回帰防止）"""
        _set_ohlcv_close("7203", 2500.0)
        risk = RiskManager()

        prices = risk.get_current_prices(["7203"])

        assert prices["7203"] == 2500.0

    def test_empty_symbol_list_short_circuits(self, isolated_db):
        called = []
        risk = RiskManager(price_fn=lambda syms: called.append(syms) or {})
        assert risk.get_current_prices([]) == {}
        assert called == [], "空リストでpiece_fnを呼ぶ必要は無い"


class TestUnrealizedPnlUsesLivePrice:
    """実測した事故のシナリオを再現する: OHLCV終値だと含み損、実勢だと含み益"""

    def _setup_three_positions(self):
        # 実際の保有銘柄・実際の数値（2026-09-09実測）
        _add_position("9432", 100, 160.0)
        _set_ohlcv_close("9432", 168.8)
        _add_position("9434", 100, 239.8)
        _set_ohlcv_close("9434", 238.6)
        _add_position("3387", 300, 765.0)
        _set_ohlcv_close("3387", 761.0)

    def test_ohlcv_only_shows_a_loss_matching_the_reported_bug(self, isolated_db):
        """price_fn無し（旧来の挙動）だと、実測どおり合計がマイナスになること"""
        self._setup_three_positions()
        risk = RiskManager()

        total = risk.unrealized_pnl()

        assert total == pytest.approx(-440.0)

    def test_live_price_reveals_the_true_gain(self, isolated_db):
        """price_fnでリアルタイム価格を渡すと、実測どおり合計がプラスになること"""
        self._setup_three_positions()
        live_prices = {"9432": 172.7, "9434": 241.1, "3387": 784.0}
        risk = RiskManager(price_fn=lambda syms: {s: live_prices[s] for s in syms})

        total = risk.unrealized_pnl()

        assert total == pytest.approx(7100.0)

    def test_manual_purchase_is_not_special_cased(self, isolated_db):
        """『手動購入だから』という特別扱いは存在しない。sourceに関わらず
        avg_cost が正しく入っていれば同じロジックで計算されること"""
        # Position.avg_cost は発注経路（手動/自動）を一切区別しないカラムであり、
        # ここでは avg_cost が入ってさえいれば price_fn 経由の現在値と正しく
        # 突き合わされることを確認する（発注元は Trade/OrderIntent 側の関心事）。
        _add_position("9434", 100, 239.8)
        risk = RiskManager(price_fn=lambda syms: {"9434": 241.1})

        total = risk.unrealized_pnl()

        assert total == pytest.approx(130.0)  # (241.1-239.8)*100


class TestCurrentTotalDrawdownUsesLivePrice:
    """kill switch のドローダウン判定も、ライブ価格を使うことで実損失を正しく反映する"""

    def test_drawdown_reflects_live_price_not_stale_close(self, isolated_db):
        _add_position("9432", 100, 200.0)
        _set_ohlcv_close("9432", 199.0)  # 前々営業日終値: 含み損100円のように見える
        # 実際は今、さらに大きく下がっている（現在値180円）
        risk = RiskManager(price_fn=lambda syms: {"9432": 180.0})

        dd = risk.current_total_drawdown()

        assert dd == pytest.approx(2000.0)  # (200-180)*100 の含み損


class TestMainPyWiring:
    """main.py の結線（ソース検証）。main.py は uvicorn 等の重い依存をトップレベルで
    importするため、他の結線テストと同様にソーステキスト検証で行う。"""

    def _main_src(self) -> str:
        with open("main.py", encoding="utf-8") as f:
            return f.read()

    def test_risk_manager_receives_price_fn(self):
        src = self._main_src()
        assert "RiskManager(price_fn=" in src, (
            "RiskManagerにリアルタイム価格取得関数が注入されていない"
        )

    def test_price_fn_uses_broker_board_not_ohlcv(self):
        src = self._main_src()
        i = src.index("_live_prices")
        body = src[i:i + 800]
        assert "client.get_board" in body, (
            "現在値取得がブローカーの板(get_board)を使っていない"
        )

    def test_paper_mode_does_not_use_live_board(self):
        """paperはkabuステーションが起動していなくても動く設計を壊さない
        （板を叩くとpaper運用が壊れる）。"""
        src = self._main_src()
        i = src.index("RiskManager(price_fn=")
        line = src[i:i + 120]
        assert "tm.is_paper(mode)" in line
