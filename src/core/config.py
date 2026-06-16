from pathlib import Path
import yaml
from loguru import logger


_config: dict = {}
_config_path: Path = Path("config.yaml")


def load(path: str = "config.yaml") -> dict:
    global _config, _config_path
    _config_path = Path(path)
    with open(_config_path, encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    logger.info(f"Config loaded from {_config_path}")
    return _config


def get() -> dict:
    if not _config:
        load()
    return _config


def get_section(section: str) -> dict:
    return get().get(section, {})
