"""
SentinelCold — Reproducibility Utilities
Seed setting, hash tracking, and model-artifact fingerprinting.
"""

import hashlib
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Fix all random-number generators for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (import guarded — may not be installed in all envs)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute the hex-digest hash of a file's contents."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_hash(df, algorithm: str = "sha256") -> str:
    """Compute a hash over the entire DataFrame contents (values + columns)."""
    import pandas as pd
    h = hashlib.new(algorithm)
    h.update(pd.util.hash_pandas_object(df).values.tobytes())
    h.update(",".join(df.columns).encode())
    return h.hexdigest()


def model_artifact_info(model, model_path: Optional[Path] = None) -> dict:
    """Return a metadata dict suitable for audit logs."""
    info = {
        "model_class": type(model).__name__,
    }
    if model_path and model_path.exists():
        info["model_hash"] = file_hash(model_path)
        info["model_path"] = str(model_path)
    return info
