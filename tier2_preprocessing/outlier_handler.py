"""
SentinelCold — Tier 2: Outlier Handler
IQR-based outlier handling with two modes:
  - 'tree' : passthrough (outliers retained — trees are robust)
  - 'linear': winsorize to IQR fences (for KNN, LogReg, etc.)

No record is ever deleted solely on outlier status.
"""

from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import tier_logger

log = tier_logger(2)


class OutlierHandler:
    """
    Compute IQR-based outlier bounds per numeric feature on the training fold,
    then either pass through (tree) or winsorize (linear).
    """

    def __init__(
        self,
        mode: str = "tree",
        iqr_multiplier: float = 1.5,
        exclude_cols: Optional[list[str]] = None,
    ):
        """
        Parameters
        ----------
        mode : str
            'tree' — outliers kept as-is (for gradient-boosted ensembles).
            'linear' — outliers capped at IQR fence (for linear/distance models).
        iqr_multiplier : float
            Multiplier for IQR fences. Default 1.5 (standard Tukey fences).
        exclude_cols : list[str] | None
            Columns to skip (e.g., target, id, categoricals).
        """
        if mode not in ("tree", "linear"):
            raise ValueError(f"mode must be 'tree' or 'linear', got '{mode}'")
        self.mode = mode
        self.iqr_mult = iqr_multiplier
        self.exclude = set(exclude_cols or [])
        self.bounds_: dict[str, tuple[float, float]] = {}
        self.outlier_rates_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame) -> "OutlierHandler":
        """Compute IQR fences on the training fold."""
        numeric_cols = [
            c for c in X.select_dtypes(include=[np.number]).columns
            if c not in self.exclude
        ]
        for col in numeric_cols:
            q1 = float(X[col].quantile(0.25))
            q3 = float(X[col].quantile(0.75))
            iqr = q3 - q1
            lower = q1 - self.iqr_mult * iqr
            upper = q3 + self.iqr_mult * iqr
            self.bounds_[col] = (lower, upper)

            n_outliers = int(((X[col] < lower) | (X[col] > upper)).sum())
            rate = n_outliers / len(X) if len(X) > 0 else 0.0
            self.outlier_rates_[col] = round(rate * 100, 2)

        log.info(
            "Outlier bounds computed for %d features (mode=%s)",
            len(self.bounds_), self.mode,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier handling based on mode."""
        if self.mode == "tree":
            # Passthrough — tree models handle outliers natively
            log.info("Outlier mode='tree' — passing through all values")
            return X

        # mode == 'linear' → winsorize
        X = X.copy()
        total_clipped = 0
        for col, (lower, upper) in self.bounds_.items():
            if col not in X.columns:
                continue
            n_before = int(((X[col] < lower) | (X[col] > upper)).sum())
            X[col] = X[col].clip(lower=lower, upper=upper)
            total_clipped += n_before

        log.info(
            "Winsorized %d total outlier values across %d features",
            total_clipped, len(self.bounds_),
        )
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def get_report(self) -> pd.DataFrame:
        """Return a summary DataFrame of outlier rates per feature."""
        rows = []
        for col, rate in sorted(self.outlier_rates_.items(), key=lambda x: -x[1]):
            lower, upper = self.bounds_[col]
            rows.append({
                "feature": col,
                "lower_fence": round(lower, 4),
                "upper_fence": round(upper, 4),
                "outlier_rate_pct": rate,
            })
        return pd.DataFrame(rows)
