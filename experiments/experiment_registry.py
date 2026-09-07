"""
SentinelCold — Experiment Registry
CSV-based tracking of all experiments with query capabilities.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger("sentinelcold.registry")


class ExperimentRegistry:
    """Track all experiments in a CSV file."""

    COLUMNS = [
        "exp_id", "imputation", "balancing", "feature_set", "model",
        "threshold", "ae_variant",
        "f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std",
        "recall_mean", "recall_std", "precision_mean", "precision_std",
        "pr_auc_mean", "pr_auc_std", "mcc_mean",
        "runtime_s", "timestamp",
    ]

    def __init__(self, registry_path: Path):
        self.path = registry_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            pd.DataFrame(columns=self.COLUMNS).to_csv(self.path, index=False)

    def register(self, config: dict, aggregated: dict, runtime: float) -> str:
        """Add an experiment result. Returns the experiment ID."""
        df = pd.read_csv(self.path)
        exp_id = f"exp_{len(df) + 1:03d}"

        row = {
            "exp_id": exp_id,
            "imputation": config.get("imputation", ""),
            "balancing": config.get("balancing", ""),
            "feature_set": config.get("feature_set", ""),
            "model": config.get("model", ""),
            "threshold": config.get("threshold", ""),
            "ae_variant": config.get("ae_variant", "none"),
            "runtime_s": round(runtime, 2),
            "timestamp": datetime.now().isoformat(),
        }
        for metric in ["f1", "roc_auc", "recall", "precision", "pr_auc", "mcc"]:
            row[f"{metric}_mean"] = aggregated.get(f"{metric}_mean", "")
            row[f"{metric}_std"] = aggregated.get(f"{metric}_std", "")

        new_row = pd.DataFrame([row])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.path, index=False)

        log.info("Registered experiment %s", exp_id)
        return exp_id

    def load(self) -> pd.DataFrame:
        """Load the full registry as a DataFrame."""
        return pd.read_csv(self.path)

    def best_by(self, metric: str = "f1_mean") -> pd.Series:
        """Return the best experiment by a given metric."""
        df = self.load()
        if metric not in df.columns:
            raise ValueError(f"Unknown metric: '{metric}'")
        idx = df[metric].astype(float).idxmax()
        return df.iloc[idx]

    def filter(self, **kwargs) -> pd.DataFrame:
        """Filter experiments by column values."""
        df = self.load()
        for col, val in kwargs.items():
            if col in df.columns:
                df = df[df[col].astype(str).str.contains(str(val), na=False)]
        return df

    def summary(self) -> str:
        """Print a summary of all experiments."""
        df = self.load()
        return df.to_string(index=False)
