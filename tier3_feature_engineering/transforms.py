"""
SentinelCold — Tier 3: Feature Transforms
All engineered features from PRD §4.2 plus alternative transforms.
Z-score scaler fitted on training fold only.
"""

from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import tier_logger

log = tier_logger(3)


class FeatureEngineer:
    """
    Computes all engineered features.  Call ``fit`` on the training fold
    (to learn z-score parameters) and ``transform`` on any fold.
    """

    def __init__(
        self,
        temp_range_weight: float = 0.4,
        sensor_gap_weight: float = 0.6,
    ):
        self.w_temp = temp_range_weight
        self.w_gap = sensor_gap_weight

        # z-score params (fitted on train)
        self._z_means: dict[str, float] = {}
        self._z_stds: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit(self, X: pd.DataFrame) -> "FeatureEngineer":
        """Learn z-score parameters from training data."""
        # We need temp_range_c to exist first for z-score computation
        temp_range = X["temp_max_c"] - X["temp_min_c"]
        self._z_means["temp_range_c"] = float(temp_range.mean())
        self._z_stds["temp_range_c"] = float(temp_range.std())

        self._z_means["sensor_gap_hours"] = float(X["sensor_gap_hours"].mean())
        self._z_stds["sensor_gap_hours"] = float(X["sensor_gap_hours"].std())

        log.info(
            "FeatureEngineer fit: z(temp_range) μ=%.4f σ=%.4f, z(sensor_gap) μ=%.4f σ=%.4f",
            self._z_means["temp_range_c"], self._z_stds["temp_range_c"],
            self._z_means["sensor_gap_hours"], self._z_stds["sensor_gap_hours"],
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transforms."""
        X = X.copy()

        # --- PRD features (always computed) --------------------------------
        X["temp_range_c"] = X["temp_max_c"] - X["temp_min_c"]

        # excursion_intensity: 0.4 × z(temp_range) + 0.6 × z(sensor_gap)
        z_tr = self._zscore(X["temp_range_c"], "temp_range_c")
        z_sg = self._zscore(X["sensor_gap_hours"], "sensor_gap_hours")
        X["excursion_intensity"] = self.w_temp * z_tr + self.w_gap * z_sg

        # vibration_rate: vibration_index / transit_days
        X["vibration_rate"] = X["vibration_index"] / X["transit_days"].replace(0, np.nan)
        X["vibration_rate"] = X["vibration_rate"].fillna(0)

        # rh_range: rh_max - rh_mean  (following EDA code, not PRD's rh_baseline)
        X["rh_range"] = X["rh_max"] - X["rh_mean"]

        # door_open_rate: computed but documented as negative result
        X["door_open_rate"] = X["door_opens"] / X["transit_days"].replace(0, np.nan)
        X["door_open_rate"] = X["door_open_rate"].fillna(0)

        # --- Alternative features (FS-D: log transforms) -------------------
        for col in ["transit_days", "sensor_gap_hours", "door_opens"]:
            X[f"log1p_{col}"] = np.log1p(X[col])

        # --- Alternative features (FS-E: polynomial interactions) ----------
        X["transit_x_temp_range"] = X["transit_days"] * X["temp_range_c"]
        X["door_x_sensor_gap"] = X["door_opens"] * X["sensor_gap_hours"]
        X["rh_range_x_temp_range"] = X["rh_range"] * X["temp_range_c"]

        log.info("Feature engineering applied — %d new columns added", 10)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _zscore(self, series: pd.Series, key: str) -> pd.Series:
        mu = self._z_means.get(key, 0.0)
        sigma = self._z_stds.get(key, 1.0)
        if sigma == 0:
            sigma = 1.0
        return (series - mu) / sigma


def compute_feature_correlations(
    df: pd.DataFrame,
    target_col: str = "silent_failure",
) -> pd.DataFrame:
    """
    Compute Pearson correlation of every numeric feature against the target.
    Returns a sorted DataFrame for comparison tables.
    """
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric:
        numeric.remove(target_col)

    corrs = []
    for col in numeric:
        r = df[col].corr(df[target_col])
        corrs.append({"feature": col, "pearson_r": round(r, 4), "abs_r": round(abs(r), 4)})

    return (
        pd.DataFrame(corrs)
        .sort_values("abs_r", ascending=False)
        .reset_index(drop=True)
    )
