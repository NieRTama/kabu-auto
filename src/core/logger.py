import sys
from pathlib import Path

from loguru import logger

from src.core import config as cfg

_WARN_LEVEL = "WARNING"


def dated_log_path(log_file: str, tag: str = "") -> str:
    """設定されたログパスのファイル名末尾に日付（と任意のタグ）を入れたパターンを返す。

    例: dated_log_path("data/kabu_auto.log") -> "data/kabu_auto_{time:YYYY-MM-DD}.log"
        dated_log_path("data/kabu_auto.log", "warning") -> "data/kabu_auto_warning_{time:YYYY-MM-DD}.log"
    loguru は {time} を含むパスに対し、ファイル生成（＝日次ローテーション）のたびに
    その日の日付でファイル名を確定する。
    """
    p = Path(log_file)
    name_tag = f"_{tag}" if tag else ""
    return str(p.with_name(f"{p.stem}{name_tag}_{{time:YYYY-MM-DD}}{p.suffix}"))


def setup() -> None:
    conf = cfg.get_section("logging")
    level = conf.get("level", "INFO")
    log_file = conf.get("file", "data/kabu_auto.log")
    retention = conf.get("retention", "15 days")

    # ログ格納フォルダ（例: log/）が無ければ作成する
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    # ログは1日単位で区切り（毎日0時にローテーション）、ファイル名末尾に日付を付与する。
    # retention で古いログを自動削除する（既定: 15日周期）。
    # INFO以下（DEBUG/INFO）と WARNING以上（WARNING/ERROR/CRITICAL）を別ファイルに分ける。
    logger.add(
        dated_log_path(log_file),
        level=level,
        rotation="00:00",
        retention=retention,
        encoding="utf-8",
        filter=lambda record: record["level"].no < logger.level(_WARN_LEVEL).no,
    )
    logger.add(
        dated_log_path(log_file, tag="warning"),
        level=_WARN_LEVEL,
        rotation="00:00",
        retention=retention,
        encoding="utf-8",
    )
    logger.info("Logger initialized")
