import sys
from pathlib import Path

from loguru import logger

from src.core import config as cfg


def dated_log_path(log_file: str) -> str:
    """設定されたログパスのファイル名末尾に日付を入れたパターンを返す。

    例: data/kabu_auto.log -> data/kabu_auto_{time:YYYY-MM-DD}.log
    loguru は {time} を含むパスに対し、ファイル生成（＝日次ローテーション）のたびに
    その日の日付でファイル名を確定する。
    """
    p = Path(log_file)
    return str(p.with_name(f"{p.stem}_{{time:YYYY-MM-DD}}{p.suffix}"))


def setup() -> None:
    conf = cfg.get_section("logging")
    level = conf.get("level", "INFO")
    log_file = conf.get("file", "data/kabu_auto.log")
    retention = conf.get("retention", "15 days")

    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    # ログは1日単位で区切り（毎日0時にローテーション）、ファイル名末尾に日付を付与する。
    # retention で古いログを自動削除する（既定: 15日周期）。
    logger.add(
        dated_log_path(log_file),
        level=level,
        rotation="00:00",
        retention=retention,
        encoding="utf-8",
    )
    logger.info("Logger initialized")
