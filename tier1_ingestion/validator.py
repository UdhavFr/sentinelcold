"""
SentinelCold — Tier 1: Data Validator (Orchestrator)
Ties together schema validation and profiling to produce the
Validated Tabular Dataset output contract for Tier 2.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from tier1_ingestion.schema import validate_batch
from tier1_ingestion.profiler import DataProfiler
from utils.logger import tier_logger

log = tier_logger(1)


class DataValidator:
    """
    End-to-end Tier 1 orchestrator.

    1. Load CSV
    2. Validate every row against the Pydantic schema
    3. Run statistical profiling
    4. Emit a ValidatedTabularDataset (DataFrame + metadata)
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            The ``tier1`` section of the master config.
        """
        self.target_col = config.get("target_column", "silent_failure")
        self.id_col = config.get("id_column", "shipment_id")
        self.miss_thresh = config.get("missingness_threshold", 0.05)
        self.expected_prev = config.get("expected_positive_prevalence", 0.1968)
        self.drift_tol = config.get("prevalence_drift_tolerance", 0.05)

    # ------------------------------------------------------------------ #

    def validate(
        self,
        data_path: Path,
        report_path: Optional[Path] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Run the full Tier 1 pipeline.

        Returns
        -------
        df_valid : pd.DataFrame
            Validated, profiled DataFrame (no schema-failing rows).
        metadata : dict
            Profiling report + quarantine summary.
        """
        log.info("═══ TIER 1 — Ingestion & Data Profiling ═══")

        # 1. Load raw data
        log.info("Loading data from %s", data_path)
        df_raw = pd.read_csv(data_path)
        log.info("Raw shape: %s", df_raw.shape)

        # 2. Schema validation (row-by-row Pydantic check)
        log.info("Running Pydantic schema validation …")
        records = df_raw.to_dict(orient="records")
        valid_recs, quarantined = validate_batch(records)
        log.info(
            "Schema validation: %d valid, %d quarantined",
            len(valid_recs),
            len(quarantined),
        )

        df_valid = pd.DataFrame(valid_recs)

        # 3. Statistical profiling
        profiler = DataProfiler(
            target_col=self.target_col,
            id_col=self.id_col,
            missingness_threshold=self.miss_thresh,
            expected_prevalence=self.expected_prev,
            prevalence_drift_tol=self.drift_tol,
        )
        profile_report = profiler.profile(df_valid)

        if report_path:
            profiler.save_report(report_path)

        # 4. Build metadata
        metadata = {
            "profile": profile_report,
            "quarantined_count": len(quarantined),
            "quarantined_sample": quarantined[:5] if quarantined else [],
        }

        log.info("Tier 1 complete — emitting ValidatedTabularDataset (%d rows)", len(df_valid))
        return df_valid, metadata
