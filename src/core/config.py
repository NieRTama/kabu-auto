import os
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
    _apply_env_overrides(_config)
    logger.info(f"Config loaded from {_config_path}")
    return _config


def _apply_env_overrides(config: dict) -> None:
    """機密情報は環境変数（.env）で上書きできるようにし、config.yaml への平文保存を避ける"""
    env_password = os.environ.get("KABU_API_PASSWORD")
    if env_password:
        config.setdefault("kabu_station", {})["password"] = env_password


def get() -> dict:
    if not _config:
        load()
    return _config


def get_section(section: str) -> dict:
    return get().get(section, {})
