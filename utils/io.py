import json
from pathlib import Path
import logging

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def setup_logger(log_path: Path):
    logger = logging.getLogger("run_all")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    sh = logging.StreamHandler()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def save_config_snapshot(out_path: Path, cfg: dict):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
