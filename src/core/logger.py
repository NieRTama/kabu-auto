import sys
from loguru import logger
from src.core import config as cfg


def setup() -> None:
    conf = cfg.get_section("logging")
    level = conf.get("level", "INFO")
    log_file = conf.get("file", "data/kabu_auto.log")

    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(log_file, level=level, rotation="1 day", retention="30 days",
               encoding="utf-8")
    logger.info("Logger initialized")
