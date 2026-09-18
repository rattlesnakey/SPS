from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_config(cfg: dict, model_key: str) -> dict:
    if model_key not in cfg["models"]:
        raise KeyError(f"Unknown model key: {model_key}")
    out = dict(cfg["models"][model_key])
    out["key"] = model_key
    out["dtype"] = cfg["runtime"]["dtype"]
    out["enable_thinking"] = bool(cfg["runtime"]["enable_thinking"])
    return out


def output_dir(cfg: dict, model_key: str) -> Path:
    return Path(cfg["paths"]["output_root"]) / model_key
