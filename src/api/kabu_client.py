"""
kabuステーションAPI クライアント
kabuステーションアプリが localhost:18080 で待受していることが前提。
"""
import threading
import time
from typing import Callable, Optional
import json

import requests
import websocket
from loguru import logger

from src.core import broker_auth
from src.core import config as cfg
from src.core import liveness


class KabuClient:
    # 再接続ループのログ間引き設定（_ws_log 参照）。
    # 最初の N 回は毎回記録し、それ以降は INTERVAL 秒に1回へ落とす。
    _WS_LOG_FIRST_N = 3
    _WS_LOG_INTERVAL_SEC = 300

    def __init__(self):
        conf = cfg.get_section("kabu_station")
        self._base_url = conf.get("base_url", "http://localhost:18080/kabusapi")
        self._password = conf.get("password", "")
        self._token: Optional[str] = None
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._on_price: Optional[Callable] = None
        self._on_order_event: Optional[Callable] = None
        self._ws_reconnect = True
        self._ws_reconnect_delay = 2
        # 再接続ループのログ間引き用。kind → (連続回数, 最後に出力した時刻)（_ws_log 参照）
        self._ws_log_state: dict = {}

    # ─── 認証 ───────────────────────────────────────────

    def refresh_token(self) -> str:
        """APIトークンを取得・更新する（毎朝8:30に呼び出す）"""
        url = f"{self._base_url}/token"
        payload = {"APIPassword": self._password}
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        self._token = resp.json()["Token"]
        logger.info("kabuステーション APIトークン更新完了")
        return self._token

    @property
    def _headers(self) -> dict:
        return {"X-API-KEY": self._token or ""}

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """認証付きREST呼び出しの共通経路。401を「ログイン認証切れ」として記録する。

        トークン更新（毎朝8:30）が成功しても、その後に kabuステーション側の
        セッションが切れることがある。ここで401を拾わないと broker_auth の状態が
        「有効」に固定され、**自動復帰・reconnect・🔴通知のすべてが無効化**される。

        2026-09-02 に実際に発生: 8:30のトークン更新は成功 → 9:00にセッション断 →
        以後497回の401を出しながら誰も認証切れと認識せず、9:05の売り注文が失敗した。
        ログインしても復帰せず、プロセス再起動でしか直せない状態だった。
        """
        resp = requests.request(
            method, f"{self._base_url}{path}", headers=self._headers, timeout=10, **kwargs
        )
        if resp.status_code == 401:
            broker_auth.mark_expired(f"{method} {path} が401を返しました")
        resp.raise_for_status()
        return resp

    # ─── REST API ────────────────────────────────────────

    def get_board(self, symbol: str, exchange: int = 1) -> dict:
        """銘柄の板情報・現在値を取得"""
        data = self._request("GET", f"/board/{symbol}@{exchange}").json()
        # 「動いている形跡」を記録する。場中にこれが途絶えたら health_check が
        # サイレント故障として検知する（例外を出さずに止まる故障への防御）。
        liveness.mark_alive(now=time.monotonic())
        return data

    def get_symbol(self, symbol: str, exchange: int = 1) -> dict:
        """銘柄情報を取得"""
        return self._request("GET", f"/symbol/{symbol}@{exchange}").json()

    def get_positions(self) -> list:
        """現在の保有ポジションを取得"""
        return self._request("GET", "/positions").json()

    def get_orders(self, query: Optional[dict] = None) -> list:
        """注文一覧を取得"""
        return self._request("GET", "/orders", params=query).json()

    def get_wallet(self) -> dict:
        """余力（現金残高）を取得"""
        return self._request("GET", "/wallet/cash").json()

    def send_order(self, order: dict) -> dict:
        """注文を発注する"""
        result = self._request("POST", "/sendorder", json=order).json()
        logger.info(f"注文送信: {order.get('Symbol')} 結果={result}")
        return result

    def cancel_order(self, order_id: str) -> dict:
        """注文をキャンセルする"""
        payload = {"OrderID": order_id, "Password": self._password}
        resp = self._request("PUT", "/cancelorder", json=payload)
        logger.info(f"注文キャンセル: OrderID={order_id}")
        return resp.json()

    def register_push(self, symbols: list) -> None:
        """WebSocketプッシュ配信に銘柄を登録する"""
        payload = {"Symbols": [{"Symbol": s, "Exchange": 1} for s in symbols]}
        self._request("PUT", "/register", json=payload)

    def unregister_push(self, symbols: list) -> None:
        payload = {"Symbols": [{"Symbol": s, "Exchange": 1} for s in symbols]}
        self._request("PUT", "/unregister", json=payload)

    def unregister_all(self) -> None:
        self._request("PUT", "/unregister/all")

    # ─── WebSocket ───────────────────────────────────────

    def start_websocket(
        self,
        on_price: Optional[Callable] = None,
        on_order_event: Optional[Callable] = None,
    ) -> None:
        """WebSocketを開始し価格・約定イベントを受信する"""
        self._on_price = on_price
        self._on_order_event = on_order_event
        self._ws_reconnect = True
        self._ws_thread = threading.Thread(target=self._ws_run_loop, daemon=True)
        self._ws_thread.start()

    def stop_websocket(self) -> None:
        self._ws_reconnect = False
        if self._ws:
            self._ws.close()

    def _ws_run_loop(self) -> None:
        ws_url = self._base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = ws_url.replace("/kabusapi", "") + "/kabusapi/websocket"
        while self._ws_reconnect:
            try:
                self._ws_log("WebSocket接続中...")
                self._ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close,
                    on_open=self._on_ws_open,
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                logger.error(f"WebSocketエラー: {e}")
            if self._ws_reconnect:
                self._ws_log(f"WebSocket再接続まで {self._ws_reconnect_delay}秒待機...")
                time.sleep(self._ws_reconnect_delay)
                self._ws_reconnect_delay = min(self._ws_reconnect_delay * 2, 60)

    def _ws_log(self, message: str, *, level: str = "info",
                kind: str = "connect") -> None:
        """再接続ループのログを間引いて出力する。

        kabuステーション未起動・場外時間帯は接続と切断を延々繰り返すため、毎回記録すると
        ログが埋まり本物の異常が埋もれる。連続失敗中は最初の数回だけ出し、以後は
        5分に1回へ落とす（接続が確立すると _on_ws_open がカウンタを戻す）。

        `kind` ごとに独立して数える。接続試行・エラー・切断は1回の失敗で
        すべて発生するため、状態を共有すると「最初の数回」が種類の数だけ目減りする。

        **WARNING も間引く。** 2026-09-04、kabuステーションが47分間停止した際に
        `_on_ws_error` が間引き無しで WARNING を **522件** 出し、当日の警告614件の
        大半を占めた。昨日入れた警告率監視（15分50件）を単独で振り切る量で、
        「未知の障害を捉える」はずの監視が既知の接続断で埋まってしまう。
        """
        count, last_at = self._ws_log_state.get(kind, (0, 0.0))
        count += 1
        now = time.monotonic()
        if count <= self._WS_LOG_FIRST_N or now - last_at >= self._WS_LOG_INTERVAL_SEC:
            last_at = now
            getattr(logger, level)(message)
        self._ws_log_state[kind] = (count, last_at)

    def _on_ws_open(self, ws) -> None:
        logger.info("WebSocket接続確立")
        self._ws_reconnect_delay = 2
        # 接続できたらログ間引きのカウンタを戻す（次に切れたときは再び数回は記録する）
        self._ws_log_state.clear()

    def _on_ws_message(self, ws, message: str) -> None:
        try:
            data = json.loads(message)
            if "Symbol" in data and self._on_price:
                self._on_price(data)
            elif "OrderEvent" in data and self._on_order_event:
                self._on_order_event(data)
        except Exception as e:
            logger.error(f"WebSocketメッセージ処理エラー: {e}")

    def _on_ws_error(self, ws, error) -> None:
        self._ws_log(f"WebSocketエラー: {error}", level="warning", kind="error")

    def _on_ws_close(self, ws, close_status_code, close_msg) -> None:
        self._ws_log(f"WebSocket切断: {close_status_code} {close_msg}", kind="close")
