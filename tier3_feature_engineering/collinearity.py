"""
SentinelCold — Tier 3: Multicollinearity Management (VIF Filtering)
Applied only to linear / distance-based auxiliary models, not to tree ensembles.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils.logger import tier_logger

log = tier_logger(3)


class VIFFilter:
    """
    Variance Inflation Factor filter for numeric features.
    Iteratively drops the highest-VIF feature until all remaining
    features are below the threshold.
    """

    def __init__(
        self,
        threshold: float = 10.0,
        exclude_cols: list[str] | None = None,
    ):
        self.threshold = threshold
        self.exclude = set(exclude_cols or [])
        self.retained_: list[str] = []
        self.dropped_: list[str] = []
        self.vif_report_: pd.DataFrame = pd.DataFrame()

    def fit(self, X: pd.DataFrame) -> "VIFFilter":
        """Compute VIF iteratively, dropping the worst offender each round."""
        numeric_cols = [
            c for c in X.select_dtypes(include=[np.number]).columns
            if c not in self.exclude
        ]
        remaining = list(numeric_cols)
        dropped = []

        while True:
            if len(remaining) < 2:
                break
            # Compute VIF for remaining features (drop NaNs for VIF computation)
            sub = X[remaining].dropna()
            if len(sub) < len(remaining) + 1:
                break

            vifs = {}
            for i, col in enumerate(remaining):
                try:
                    vifs[col] = variance_inflation_factor(sub.values, i)
                except Exception:
                    vifs[col] = np.inf

            max_col = max(vifs, key=vifs.get)
            max_vif = vifs[max_col]

            if max_vif <= self.threshold:
                break

            remaining.remove(max_col)
            dropped.append((max_col, round(max_vif, 2)))
            log.info("VIF drop: '%s' (VIF=%.2f)", max_col, max_vif)

        self.retained_ = remaining
        self.dropped_ = dropped

        # Final VIF report
        if len(remaining) >= 2:
            sub = X[remaining].dropna()
            rows = []
            for i, col in enumerate(remaining):
                try:
                    v = variance_inflation_factor(sub.values, i)
                except Exception:
                    v = np.nan
                rows.append({"feature": col, "vif": round(v, 2)})
            self.vif_report_ = pd.DataFrame(rows).sort_values("vif", ascending=False)

        log.info(
            "VIF filtering done — %d retained, %d dropped (threshold=%.1f)",
            len(self.retained_), len(self.dropped_), self.threshold,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the retained columns (+ any excluded columns)."""
        keep = [c for c in X.columns if c in self.retained_ or c in self.exclude]
        # Preserve non-numeric columns too
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        keep = list(dict.fromkeys(keep + non_numeric))  # deduplicate, preserve order
        return X[keep]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
