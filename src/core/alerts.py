"""異常時アラート（LINE Notify）"""
import requests
from loguru import logger

from src.core import config as cfg


def send_line(message: str) -> None:
    """LINE Notifyでメッセージを送信する"""
    token = cfg.get_section("alerts").get("line_notify_token", "")
    if not token:
        return
    try:
        requests.post(
            "https://notify-api.line.me/api/notify",
            headers={"Authorization": f"Bearer {token}"},
            data={"message": f"\n{message}"},
            timeout=10,
        )
    except Exception as e:
        logger.error(f"LINE通知失敗: {e}")


def alert(title: str, message: str) -> None:
    logger.warning(f"[ALERT] {title}: {message}")
    send_line(f"【kabu-auto】{title}\n{message}")
