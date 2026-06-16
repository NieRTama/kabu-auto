"""
APSchedulerによるジョブスケジューラ
- 毎朝8:30: APIトークン更新
- 毎日16:00: データ更新・バックアップ・ML再学習（週次）
- 土曜: メンテナンス時間帯の回避
"""
from datetime import datetime

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

TZ = pytz.timezone("Asia/Tokyo")


class TradingScheduler:
    def __init__(self):
        self._scheduler = BackgroundScheduler(timezone=TZ)
        self._registered_callbacks: dict = {}

    def register(self, name: str, callback) -> None:
        self._registered_callbacks[name] = callback

    def start(self) -> None:
        cb = self._registered_callbacks

        if "token_refresh" in cb:
            self._scheduler.add_job(
                cb["token_refresh"], "cron",
                hour=8, minute=30, id="token_refresh",
            )
        if "data_update" in cb:
            self._scheduler.add_job(
                cb["data_update"], "cron",
                hour=16, minute=0, id="data_update",
            )
        if "db_backup" in cb:
            self._scheduler.add_job(
                cb["db_backup"], "cron",
                hour=17, minute=0, id="db_backup",
            )
        if "ml_retrain" in cb:
            # 週次: 毎週日曜 02:00 に再学習
            self._scheduler.add_job(
                cb["ml_retrain"], "cron",
                day_of_week="sun", hour=2, minute=0, id="ml_retrain",
            )
        if "stop_loss_check" in cb:
            # 市場時間中 5分ごとに損切りチェック（平日 9:00-15:30）
            self._scheduler.add_job(
                cb["stop_loss_check"], "cron",
                day_of_week="mon-fri",
                hour="9-15", minute="*/5",
                id="stop_loss_check",
            )
        if "signal_scan" in cb:
            # 後場終了後 15:35 に翌日の銘柄スクリーニング
            self._scheduler.add_job(
                cb["signal_scan"], "cron",
                day_of_week="mon-fri",
                hour=15, minute=35, id="signal_scan",
            )
        if "morning_execution" in cb:
            # 9:05 に前日シグナルを元に発注（ライブモードのみ実行される）
            self._scheduler.add_job(
                cb["morning_execution"], "cron",
                day_of_week="mon-fri",
                hour=9, minute=5, id="morning_execution",
            )

        self._scheduler.start()
        logger.info("スケジューラ起動完了")

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        logger.info("スケジューラ停止")

    @staticmethod
    def is_market_open() -> bool:
        """現在が東証の取引時間内かチェック（土日・メンテ除外）"""
        now = datetime.now(TZ)
        if now.weekday() >= 5:  # 土日
            return False
        now_hm = (now.hour, now.minute)
        morning = (9, 0) <= now_hm < (11, 30)
        afternoon = (12, 30) <= now_hm <= (15, 30)
        return morning or afternoon

    @staticmethod
    def is_maintenance_window() -> bool:
        """土曜深夜のメンテナンス時間帯かチェック"""
        now = datetime.now(TZ)
        return now.weekday() == 5 and 1 <= now.hour <= 5
