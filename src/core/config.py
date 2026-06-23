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
    # config.yaml に平文パスワードが書かれていたら警告（.env での管理を強制したい）
    if (_config.get("kabu_station") or {}).get("password"):
        logger.warning(
            "config.yaml に kabu_station.password が記載されています。"
            "漏洩防止のため削除し、.env の KABU_API_PASSWORD を使用してください。"
        )
    _apply_env_overrides(_config)
    logger.info(f"Config loaded from {_config_path}")
    return _config


def _apply_env_overrides(config: dict) -> None:
    """機密情報は環境変数（.env）で上書きできるようにし、config.yaml への平文保存を避ける"""
    env_password = os.environ.get("KABU_API_PASSWORD")
    if env_password:
        config.setdefault("kabu_station", {})["password"] = env_password
    env_discord_webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if env_discord_webhook:
        config.setdefault("alerts", {})["discord_webhook_url"] = env_discord_webhook
    env_dash_token = os.environ.get("KABU_DASHBOARD_TOKEN")
    if env_dash_token:
        config.setdefault("dashboard", {})["api_token"] = env_dash_token
    env_discord_report_webhook = os.environ.get("DISCORD_REPORT_WEBHOOK_URL")
    if env_discord_report_webhook:
        config.setdefault("discord_report", {})["webhook_url"] = env_discord_report_webhook
    _apply_x_env_overrides(config)


def _apply_x_env_overrides(config: dict) -> None:
    """X（旧Twitter）APIの4つの鍵を環境変数から上書きする（config.yamlへの平文保存を避ける）。"""
    mapping = {
        "X_API_KEY": "api_key",
        "X_API_SECRET": "api_secret",
        "X_ACCESS_TOKEN": "access_token",
        "X_ACCESS_TOKEN_SECRET": "access_token_secret",
    }
    for env_name, key in mapping.items():
        value = os.environ.get(env_name)
        if value:
            config.setdefault("x", {})[key] = value


def get() -> dict:
    if not _config:
        load()
    return _config


def get_section(section: str) -> dict:
    return get().get(section, {})


def get_api_password() -> str:
    """kabuステーションAPIパスワードを取得する唯一の入口（レビュー Security）。

    環境変数 KABU_API_PASSWORD を最優先し、無ければ config.yaml の値を使う
    （平文を config に残さない運用を推奨）。値そのものは絶対にログへ出さない。
    """
    return os.environ.get("KABU_API_PASSWORD") or get_section("kabu_station").get("password", "")


def require_api_password() -> str:
    """APIパスワードを取得し、未設定なら例外を投げる（live/semi_live の発注前チェック用）。"""
    pw = get_api_password()
    if not pw:
        raise RuntimeError(
            "kabuステーションAPIパスワードが未設定です。.env の KABU_API_PASSWORD を設定してください"
        )
    return pw
