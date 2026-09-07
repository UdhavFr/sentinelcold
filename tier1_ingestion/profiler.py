"""
SentinelCold — Tier 1: Data Profiler
Statistical profiling, missingness analysis, target audit, and range checks.
Generates a machine-readable JSON profiling report.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import tier_logger

log = tier_logger(1)


class DataProfiler:
    """
    Produces a comprehensive profile of the raw dataset that feeds
    downstream tier-2 preprocessing decisions.
    """

    def __init__(
        self,
        target_col: str = "silent_failure",
        id_col: str = "shipment_id",
        missingness_threshold: float = 0.05,
        expected_prevalence: float = 0.1968,
        prevalence_drift_tol: float = 0.05,
    ):
        self.target_col = target_col
        self.id_col = id_col
        self.miss_thresh = missingness_threshold
        self.expected_prev = expected_prevalence
        self.drift_tol = prevalence_drift_tol
        self.report: dict = {}

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

    def profile(self, df: pd.DataFrame) -> dict:
        """Run the full profiling suite and return a report dict."""
        self.report = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
        }
        self._profile_missingness(df)
        self._profile_duplicates(df)
        self._profile_identifier(df)
        self._profile_target(df)
        self._profile_descriptive_stats(df)
        self._profile_range_checks(df)

        log.info(
            "Profiling complete — %d rows, %d cols, %d missing-flagged columns",
            self.report["n_rows"],
            self.report["n_cols"],
            len(self.report["missingness"]["flagged_columns"]),
        )
        return self.report

    def save_report(self, path: Path) -> None:
        """Persist the profiling report as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, default=str)
        log.info("Profiling report saved → %s", path)

    # --------------------------------------------------------------------- #
    #  Internal profiling methods                                            #
    # --------------------------------------------------------------------- #

    def _profile_missingness(self, df: pd.DataFrame) -> None:
        missing = df.isna().sum()
        missing_pct = (missing / len(df)).round(6)
        flagged = missing_pct[missing_pct > self.miss_thresh].index.tolist()

        detail = {}
        for col in df.columns:
            if missing[col] > 0:
                detail[col] = {
                    "missing_count": int(missing[col]),
                    "missing_pct": round(float(missing_pct[col]) * 100, 2),
                }
        self.report["missingness"] = {
            "total_missing_cells": int(missing.sum()),
            "columns_with_missing": detail,
            "flagged_columns": flagged,
            "threshold_pct": self.miss_thresh * 100,
        }
        if flagged:
            log.warning(
                "Columns exceeding %.1f%% missingness: %s",
                self.miss_thresh * 100,
                flagged,
            )

    def _profile_duplicates(self, df: pd.DataFrame) -> None:
        n_dup = int(df.duplicated().sum())
        self.report["duplicates"] = {"full_row_duplicates": n_dup}
        if n_dup > 0:
            log.warning("Found %d full-row duplicates", n_dup)
        else:
            log.info("No full-row duplicates found")

    def _profile_identifier(self, df: pd.DataFrame) -> None:
        if self.id_col in df.columns:
            is_unique = bool(df[self.id_col].is_unique)
            n_unique = int(df[self.id_col].nunique())
        else:
            is_unique = None
            n_unique = None
        self.report["identifier"] = {
            "column": self.id_col,
            "is_unique": is_unique,
            "n_unique": n_unique,
        }
        if is_unique is False:
            log.error("Identifier column '%s' is NOT unique!", self.id_col)

    def _profile_target(self, df: pd.DataFrame) -> None:
        if self.target_col not in df.columns:
            self.report["target"] = {"error": f"'{self.target_col}' not found"}
            return

        vc = df[self.target_col].value_counts()
        positive = int(vc.get(1, 0))
        negative = int(vc.get(0, 0))
        prevalence = round(positive / len(df), 4)
        drift = abs(prevalence - self.expected_prev)
        drift_alert = drift > self.drift_tol

        self.report["target"] = {
            "column": self.target_col,
            "n_positive": positive,
            "n_negative": negative,
            "prevalence": prevalence,
            "expected_prevalence": self.expected_prev,
            "drift_from_expected": round(drift, 4),
            "drift_alert": drift_alert,
        }
        if drift_alert:
            log.warning(
                "TARGET DRIFT ALERT — prevalence %.2f%% vs expected %.2f%% (Δ=%.2f pp)",
                prevalence * 100,
                self.expected_prev * 100,
                drift * 100,
            )
        else:
            log.info(
                "Target balance: %d positive / %d negative (%.2f%%)",
                positive, negative, prevalence * 100,
            )

    def _profile_descriptive_stats(self, df: pd.DataFrame) -> None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude id column if numeric
        numeric_cols = [c for c in numeric_cols if c != self.id_col]

        stats = {}
        for col in numeric_cols:
            s = df[col].dropna()
            stats[col] = {
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "25%": round(float(s.quantile(0.25)), 4),
                "50%": round(float(s.median()), 4),
                "75%": round(float(s.quantile(0.75)), 4),
                "max": round(float(s.max()), 4),
                "skew": round(float(s.skew()), 4),
                "kurtosis": round(float(s.kurtosis()), 4),
            }
        self.report["descriptive_stats"] = stats

    def _profile_range_checks(self, df: pd.DataFrame) -> None:
        """Check for physically impossible values."""
        violations = {}

        # fill_ratio must be in [0, 1]
        if "fill_ratio" in df.columns:
            bad = df[
                (df["fill_ratio"] < 0) | (df["fill_ratio"] > 1)
            ]
            if len(bad) > 0:
                violations["fill_ratio_out_of_range"] = len(bad)

        # rh_mean must be in [0, 1]
        if "rh_mean" in df.columns:
            bad = df[(df["rh_mean"] < 0) | (df["rh_mean"] > 1)]
            if len(bad) > 0:
                violations["rh_mean_out_of_range"] = len(bad)

        # temp_max >= temp_min
        if {"temp_max_c", "temp_min_c"}.issubset(df.columns):
            bad = df[df["temp_max_c"] < df["temp_min_c"]]
            if len(bad) > 0:
                violations["temp_max_lt_min"] = len(bad)

        self.report["range_violations"] = violations
        if violations:
            log.warning("Range violations detected: %s", violations)
        else:
            log.info("All range checks passed")
