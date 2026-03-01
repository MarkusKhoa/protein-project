"""Reproducibility utilities: seed setup and run metadata."""
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic when possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get torch device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def log_run_metadata(save_dir: Path, config: dict, metrics: dict | None = None) -> Path:
    """Write run metadata to save_dir."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
    }
    if metrics:
        meta["metrics"] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in metrics.items()}

    meta_path = save_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path
