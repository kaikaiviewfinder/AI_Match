import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_name: str = "default") -> dict:
    base = _load_yaml(_PROJECT_ROOT / "config" / "default.yaml")
    if config_name != "default":
        override = _load_yaml(_PROJECT_ROOT / "config" / f"{config_name}.yaml")
        if override:
            base = _deep_merge(base, override)
    base = _resolve_env_vars(base)
    base["_project_root"] = str(_PROJECT_ROOT)
    return base


def _load_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_env_vars(cfg: dict) -> dict:
    if isinstance(cfg, dict):
        return {k: _resolve_env_vars(v) for k, v in cfg.items()}
    if isinstance(cfg, str) and cfg.startswith("${") and cfg.endswith("}"):
        return os.getenv(cfg[2:-1], cfg)
    return cfg


def get_path(cfg: dict, *keys) -> str:
    base = cfg.get("_project_root", ".")
    val = cfg
    for k in keys:
        val = val[k]
    if os.path.isabs(val):
        return val
    return os.path.join(base, val)
