"""
APSchedulerによるジョブスケジューラ
- 毎朝8:25: 日次リスクカウンタリセット
- 毎朝8:30: APIトークン更新
- 毎日16:00: データ更新
- 毎日16:05: ニュース取得・センチメント採点（data_update後・signal_scan前）
- 毎日16:20: シグナルスキャン（data_updateの完了を待つため16:00より後ろに設定）
- 毎日17:00: DBバックアップ
- 毎週日曜2:00: MLモデル再学習（週次）
- 15秒ごと: 注文状態の照合（reconcile_orders。市場時間中のみコールバック側で実働）
- 土曜: メンテナンス時間帯の回避
"""
from datetime import datetime
from typing import Optional

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

        if "risk_reset" in cb:
            # 取引開始前に日次カウンタ（注文数・損失額）をリセットする
            self._scheduler.add_job(
                cb["risk_reset"], "cron",
                day_of_week="mon-fri", hour=8, minute=25, id="risk_reset",
            )
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
        if "news_update" in cb:
            # data_update（16:00）後・signal_scan（16:20）前にニュースを取得・採点する。
            # signal_scan が当日分のニュース特徴量を使えるよう、必ずこの間の時刻にすること。
            self._scheduler.add_job(
                cb["news_update"], "cron",
                day_of_week="mon-fri",
                hour=16, minute=5, id="news_update",
            )
        if "signal_scan" in cb:
            # 後場終了後、当日分のOHLCV更新（data_update 16:00）が完了してから
            # 翌日の銘柄スクリーニングを行う。data_updateより前に動かすと前日終値で
            # シグナルを生成してしまうため、必ず data_update より後の時刻にすること。
            self._scheduler.add_job(
                cb["signal_scan"], "cron",
                day_of_week="mon-fri",
                hour=16, minute=20, id="signal_scan",
            )
        if "morning_execution" in cb:
            # 9:05 に前日シグナルを元に発注（ライブモードのみ実行される）
            self._scheduler.add_job(
                cb["morning_execution"], "cron",
                day_of_week="mon-fri",
                hour=9, minute=5, id="morning_execution",
            )
        if "health_check" in cb:
            # 運用異常（未解決注文・損失上限接近・kill switch）の定期検知（平日 8:00-23:00 / 15分毎）。
            # 市場時間に限定しないのは、場が引けた後でも未解決注文は要対応のため。
            self._scheduler.add_job(
                cb["health_check"], "cron",
                day_of_week="mon-fri", hour="8-23", minute="*/15",
                id="health_check",
            )
        if "reconcile_orders" in cb:
            # WebSocketイベントの取り逃し・切断・再起動を跨いでDB↔ブローカーの
            # 注文状態ズレを定期的に検知・補正する（市場時間外はコールバック側でスキップ）
            self._scheduler.add_job(
                cb["reconcile_orders"], "interval",
                seconds=15, id="reconcile_orders",
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

    @staticmethod
    def is_near_close(minutes: int, now: Optional[datetime] = None) -> bool:
        """大引け(15:30)まで minutes 分以内かを返す（新規BUY見送り判定用。P0-6）。

        minutes<=0 なら常に False（無効）。場が引けた後（15:30以降）も「新規は出さない」
        意図に合わせて True を返す。平日かつ後場の引け間際のみ意味を持つ。
        """
        if minutes <= 0:
            return False
        now = now or datetime.now(TZ)
        if now.weekday() >= 5:
            return False
        close_min = 15 * 60 + 30  # 15:30
        now_min = now.hour * 60 + now.minute
        # 後場（12:30以降）で、引けまで minutes 分以内、または引け後
        return now_min >= close_min - minutes and now_min >= 12 * 60 + 30
