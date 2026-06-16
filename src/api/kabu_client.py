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

from src.core import config as cfg


class KabuClient:
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

    # ─── REST API ────────────────────────────────────────

    def get_board(self, symbol: str, exchange: int = 1) -> dict:
        """銘柄の板情報・現在値を取得"""
        url = f"{self._base_url}/board/{symbol}@{exchange}"
        resp = requests.get(url, headers=self._headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_symbol(self, symbol: str, exchange: int = 1) -> dict:
        """銘柄情報を取得"""
        url = f"{self._base_url}/symbol/{symbol}@{exchange}"
        resp = requests.get(url, headers=self._headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_positions(self) -> list:
        """現在の保有ポジションを取得"""
        url = f"{self._base_url}/positions"
        resp = requests.get(url, headers=self._headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_orders(self, query: Optional[dict] = None) -> list:
        """注文一覧を取得"""
        url = f"{self._base_url}/orders"
        resp = requests.get(url, headers=self._headers, params=query, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_wallet(self) -> dict:
        """余力（現金残高）を取得"""
        url = f"{self._base_url}/wallet/cash"
        resp = requests.get(url, headers=self._headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def send_order(self, order: dict) -> dict:
        """注文を発注する"""
        url = f"{self._base_url}/sendorder"
        resp = requests.post(url, headers=self._headers, json=order, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        logger.info(f"注文送信: {order.get('Symbol')} 結果={result}")
        return result

    def cancel_order(self, order_id: str) -> dict:
        """注文をキャンセルする"""
        url = f"{self._base_url}/cancelorder"
        payload = {"OrderID": order_id, "Password": self._password}
        resp = requests.put(url, headers=self._headers, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info(f"注文キャンセル: OrderID={order_id}")
        return resp.json()

    def register_push(self, symbols: list) -> None:
        """WebSocketプッシュ配信に銘柄を登録する"""
        url = f"{self._base_url}/register"
        payload = {"Symbols": [{"Symbol": s, "Exchange": 1} for s in symbols]}
        resp = requests.put(url, headers=self._headers, json=payload, timeout=10)
        resp.raise_for_status()

    def unregister_push(self, symbols: list) -> None:
        url = f"{self._base_url}/unregister"
        payload = {"Symbols": [{"Symbol": s, "Exchange": 1} for s in symbols]}
        resp = requests.put(url, headers=self._headers, json=payload, timeout=10)
        resp.raise_for_status()

    def unregister_all(self) -> None:
        url = f"{self._base_url}/unregister/all"
        resp = requests.put(url, headers=self._headers, timeout=10)
        resp.raise_for_status()

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
        delay = 2
        while self._ws_reconnect:
            try:
                logger.info("WebSocket接続中...")
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
                logger.info(f"WebSocket再接続まで {delay}秒待機...")
                time.sleep(delay)
                delay = min(delay * 2, 60)

    def _on_ws_open(self, ws) -> None:
        logger.info("WebSocket接続確立")

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
        logger.warning(f"WebSocketエラー: {error}")

    def _on_ws_close(self, ws, close_status_code, close_msg) -> None:
        logger.info(f"WebSocket切断: {close_status_code} {close_msg}")
