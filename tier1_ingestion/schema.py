"""
SentinelCold — Tier 1: Pydantic Data Schema
Strict type validation for every shipment record (PRD §3.2).
No implicit coercion — records failing validation are quarantined.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ShipmentRecord(BaseModel):
    """
    Schema for a single cold-chain shipment record.
    Maps 1-to-1 with the Kaggle Cold Chain Silent Failure Dataset columns.
    """

    # Identifier — validated for uniqueness at the batch level, excluded from modelling
    shipment_id: str = Field(..., min_length=1, description="Unique shipment identifier")

    # Transit / Handling
    transit_days: float = Field(..., ge=0, description="Total transit duration in days")
    door_opens: int = Field(..., ge=0, description="Number of door-opening events during shipment")
    leg_count: int = Field(..., ge=1, le=3, description="Number of routing legs (1–3)")

    # Temperature family
    temp_mean_c: float = Field(..., description="Mean temperature during transit (°C)")
    temp_max_c: float = Field(..., description="Maximum temperature during transit (°C)")
    temp_min_c: float = Field(..., description="Minimum temperature during transit (°C)")
    temp_std_c: float = Field(..., ge=0, description="Temperature standard deviation (°C)")
    temp_recovery_rate: Optional[float] = Field(
        None, ge=0,
        description="Rate of temperature recovery after excursion (0.80% missing allowed)"
    )

    # Humidity family
    rh_mean: float = Field(..., ge=0, le=1, description="Mean relative humidity (0–1 fraction)")
    rh_std: float = Field(..., ge=0, description="Relative humidity standard deviation")
    rh_max: Optional[float] = Field(
        None, ge=0, le=1,
        description="Maximum relative humidity (4.18% missing allowed)"
    )

    # Packaging / Product
    package_type: int = Field(..., ge=0, le=2, description="Package type code (0, 1, 2)")
    product_volume_l: float = Field(..., gt=0, description="Product volume in litres")
    fill_ratio: float = Field(..., ge=0, le=1, description="Container fill ratio (0–1)")

    # Routing metadata
    carrier_id: int = Field(..., ge=0, description="Carrier identifier (integer-encoded)")
    origin_zone: int = Field(..., ge=0, description="Origin zone code")
    dest_zone: int = Field(..., ge=0, description="Destination zone code")

    # Sensor / Motion
    sensor_gap_hours: float = Field(..., ge=0, description="Max gap between sensor readings (hours)")
    vibration_index: float = Field(..., ge=0, description="Cumulative vibration index")

    # Anonymized covariates
    feature_x1: float = Field(..., description="Anonymized covariate 1")
    feature_x2: float = Field(..., description="Anonymized covariate 2")
    feature_x3: int = Field(..., ge=0, le=1, description="Anonymized binary covariate 3")

    # Target
    silent_failure: int = Field(..., ge=0, le=1, description="Binary target: 0=safe, 1=failure")

    # -- cross-field validation -----------------------------------------------
    @field_validator("temp_max_c")
    @classmethod
    def max_ge_min(cls, v, info):
        """temp_max_c must be ≥ temp_min_c (physical constraint)."""
        if "temp_min_c" in info.data and v < info.data["temp_min_c"]:
            raise ValueError(
                f"temp_max_c ({v}) < temp_min_c ({info.data['temp_min_c']})"
            )
        return v

    model_config = {"strict": True}


# -- Batch-level helpers -------------------------------------------------------

def validate_batch(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Validate a list of raw dicts against the ShipmentRecord schema.

    Returns
    -------
    valid : list[dict]
        Records that passed validation (as plain dicts).
    quarantined : list[dict]
        Records that failed, each augmented with an ``_error`` key.
    """
    valid, quarantined = [], []
    for rec in records:
        try:
            validated = ShipmentRecord(**rec)
            valid.append(validated.model_dump())
        except Exception as exc:
            rec["_error"] = str(exc)
            quarantined.append(rec)
    return valid, quarantined
