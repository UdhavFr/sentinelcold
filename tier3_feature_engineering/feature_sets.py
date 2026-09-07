"""
SentinelCold — Tier 3: Feature Set Definitions
Six interchangeable feature sets for ablation experiments.

FS-A  Raw Only          — baseline (no engineering)
FS-B  PRD Full          — raw + 4 PRD-specified engineered features
FS-C  PRD + door_rate   — FS-B + door_open_rate (testing the "negative result")
FS-D  PRD + Log         — FS-B + log1p transforms for right-skewed features
FS-E  PRD + Polynomial  — FS-B + 2nd-order interaction terms
FS-F  Kitchen Sink      — everything
"""

# -- Raw feature groups --------------------------------------------------------
_ID = ["shipment_id"]
_TARGET = ["silent_failure"]

_RAW_TRANSIT = ["transit_days", "door_opens", "leg_count"]
_RAW_TEMP = ["temp_mean_c", "temp_max_c", "temp_min_c", "temp_std_c", "temp_recovery_rate"]
_RAW_HUMIDITY = ["rh_mean", "rh_std", "rh_max"]
_RAW_PACKAGE = ["package_type", "product_volume_l", "fill_ratio"]
_RAW_ROUTING = ["carrier_id", "origin_zone", "dest_zone"]
_RAW_SENSOR = ["sensor_gap_hours", "vibration_index"]
_RAW_ANON = ["feature_x1", "feature_x2", "feature_x3"]

_ALL_RAW = (
    _RAW_TRANSIT + _RAW_TEMP + _RAW_HUMIDITY
    + _RAW_PACKAGE + _RAW_ROUTING + _RAW_SENSOR + _RAW_ANON
)

# -- Engineered feature groups -------------------------------------------------
_ENG_PRD = ["temp_range_c", "excursion_intensity", "vibration_rate", "rh_range"]
_ENG_DOOR_RATE = ["door_open_rate"]
_ENG_LOG = ["log1p_transit_days", "log1p_sensor_gap_hours", "log1p_door_opens"]
_ENG_POLY = ["transit_x_temp_range", "door_x_sensor_gap", "rh_range_x_temp_range"]

# -- Named feature sets --------------------------------------------------------
FEATURE_SETS: dict[str, list[str]] = {
    "FS-A": _ALL_RAW,
    "FS-B": _ALL_RAW + _ENG_PRD,
    "FS-C": _ALL_RAW + _ENG_PRD + _ENG_DOOR_RATE,
    "FS-D": _ALL_RAW + _ENG_PRD + _ENG_LOG,
    "FS-E": _ALL_RAW + _ENG_PRD + _ENG_POLY,
    "FS-F": _ALL_RAW + _ENG_PRD + _ENG_DOOR_RATE + _ENG_LOG + _ENG_POLY,
}

FEATURE_SET_DESCRIPTIONS: dict[str, str] = {
    "FS-A": "Raw features only — baseline to quantify feature engineering value",
    "FS-B": "PRD-specified set: raw + temp_range, excursion_intensity, vibration_rate, rh_range",
    "FS-C": "FS-B + door_open_rate (testing documented negative result — trees may still benefit)",
    "FS-D": "FS-B + log1p transforms for right-skewed features (transit_days, sensor_gap, door_opens)",
    "FS-E": "FS-B + 2nd-order polynomial interactions (transit×temp_range, door×sensor_gap, rh×temp)",
    "FS-F": "Kitchen sink — all raw + all engineered features; maximum information for tree models",
}

CATEGORICAL_FEATURES = ["carrier_id", "origin_zone", "dest_zone", "package_type", "leg_count", "feature_x3"]


def get_feature_set(name: str) -> list[str]:
    """Return the list of column names for a given feature set."""
    if name not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set '{name}'. Options: {list(FEATURE_SETS.keys())}"
        )
    return FEATURE_SETS[name]


def list_feature_sets() -> dict[str, str]:
    """Return {name: description} for all defined feature sets."""
    return dict(FEATURE_SET_DESCRIPTIONS)
