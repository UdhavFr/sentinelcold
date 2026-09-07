"""
SentinelCold — Tier 3: Feature Registry
Central metadata for every feature, used by Tier 5 explainability
to generate human-readable explanations.
"""

FEATURE_REGISTRY: dict[str, dict] = {
    # --- Transit / Handling ---
    "transit_days":       {"family": "transit",   "type": "raw",        "description": "Total transit duration (days)"},
    "door_opens":         {"family": "transit",   "type": "raw",        "description": "Number of door-opening events"},
    "leg_count":          {"family": "transit",   "type": "raw",        "description": "Number of routing legs (1–3)"},
    # --- Temperature ---
    "temp_mean_c":        {"family": "temperature", "type": "raw",      "description": "Mean temperature (°C)"},
    "temp_max_c":         {"family": "temperature", "type": "raw",      "description": "Maximum temperature (°C)"},
    "temp_min_c":         {"family": "temperature", "type": "raw",      "description": "Minimum temperature (°C)"},
    "temp_std_c":         {"family": "temperature", "type": "raw",      "description": "Temperature std deviation (°C)"},
    "temp_recovery_rate": {"family": "temperature", "type": "raw",      "description": "Temp recovery rate after excursion"},
    "temp_range_c":       {"family": "temperature", "type": "engineered","description": "Temperature range (max − min)"},
    # --- Humidity ---
    "rh_mean":            {"family": "humidity",  "type": "raw",        "description": "Mean relative humidity (0–1)"},
    "rh_std":             {"family": "humidity",  "type": "raw",        "description": "Humidity std deviation"},
    "rh_max":             {"family": "humidity",  "type": "raw",        "description": "Maximum relative humidity (0–1)"},
    "rh_range":           {"family": "humidity",  "type": "engineered", "description": "Humidity range (max − mean)"},
    # --- Packaging ---
    "package_type":       {"family": "packaging", "type": "raw",        "description": "Package type code (0/1/2)"},
    "product_volume_l":   {"family": "packaging", "type": "raw",        "description": "Product volume (litres)"},
    "fill_ratio":         {"family": "packaging", "type": "raw",        "description": "Container fill ratio (0–1)"},
    # --- Routing ---
    "carrier_id":         {"family": "routing",   "type": "raw",        "description": "Carrier identifier"},
    "origin_zone":        {"family": "routing",   "type": "raw",        "description": "Origin zone code"},
    "dest_zone":          {"family": "routing",   "type": "raw",        "description": "Destination zone code"},
    # --- Sensor / Motion ---
    "sensor_gap_hours":   {"family": "sensor",    "type": "raw",        "description": "Max gap between sensor reads (hours)"},
    "vibration_index":    {"family": "sensor",    "type": "raw",        "description": "Cumulative vibration index"},
    "vibration_rate":     {"family": "sensor",    "type": "engineered", "description": "Vibration intensity (index / days)"},
    # --- Anonymized ---
    "feature_x1":         {"family": "anonymized","type": "raw",        "description": "Anonymized covariate 1"},
    "feature_x2":         {"family": "anonymized","type": "raw",        "description": "Anonymized covariate 2"},
    "feature_x3":         {"family": "anonymized","type": "raw",        "description": "Anonymized binary covariate 3"},
    # --- Composite ---
    "excursion_intensity":{"family": "composite", "type": "engineered", "description": "Composite excursion severity (0.4×z_temp_range + 0.6×z_sensor_gap)"},
    # --- Negative result (documented) ---
    "door_open_rate":     {"family": "transit",   "type": "engineered", "description": "Door opens / transit days (NEGATIVE RESULT — reduces signal vs raw)"},
    # --- Log transforms ---
    "log1p_transit_days":     {"family": "transit", "type": "engineered", "description": "log1p(transit_days) for skew reduction"},
    "log1p_sensor_gap_hours": {"family": "sensor",  "type": "engineered", "description": "log1p(sensor_gap_hours) for skew reduction"},
    "log1p_door_opens":       {"family": "transit", "type": "engineered", "description": "log1p(door_opens) for skew reduction"},
    # --- Polynomial interactions ---
    "transit_x_temp_range":   {"family": "interaction", "type": "engineered", "description": "transit_days × temp_range_c"},
    "door_x_sensor_gap":      {"family": "interaction", "type": "engineered", "description": "door_opens × sensor_gap_hours"},
    "rh_range_x_temp_range":  {"family": "interaction", "type": "engineered", "description": "rh_range × temp_range_c"},
}


def get_description(feature: str) -> str:
    """Return human-readable description for a feature."""
    entry = FEATURE_REGISTRY.get(feature)
    return entry["description"] if entry else feature


def get_family(feature: str) -> str:
    """Return the feature family (e.g., 'temperature', 'humidity')."""
    entry = FEATURE_REGISTRY.get(feature)
    return entry["family"] if entry else "unknown"
