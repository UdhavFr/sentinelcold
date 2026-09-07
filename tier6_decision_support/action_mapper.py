"""
SentinelCold — Tier 6: SHAP-to-Action Mapper
Maps dominant SHAP attributions to causally-grounded recommended responses
(PRD §7.2).
"""

import numpy as np
from utils.logger import tier_logger

log = tier_logger(6)


# --- Action rules (PRD §7.2) ------------------------------------------------

ACTION_RULES = [
    {
        "id": "thermal_intervention",
        "dominant_features": {"excursion_intensity", "temp_range_c"},
        "min_risk_tier": "High",
        "action": "Reroute / Re-cool",
        "rationale": "Cumulative unmonitored thermal exposure is the primary driver",
    },
    {
        "id": "physical_handling",
        "dominant_features": {"vibration_index", "vibration_rate"},
        "min_risk_tier": "High",
        "action": "QA Flag — Physical Handling Stress",
        "rationale": "Route to physical inspection for packaging integrity",
    },
    {
        "id": "door_discipline",
        "dominant_features": {"door_opens"},
        "min_risk_tier": "Medium",
        "action": "QA Flag — Handling Protocol Review",
        "rationale": "Flag origin/handling facility for door-discipline audit",
    },
    {
        "id": "humidity_exposure",
        "dominant_features": {"rh_range", "rh_max"},
        "min_risk_tier": "Medium",
        "action": "QA Flag — Humidity Exposure",
        "rationale": "Route to potency/stability re-test prior to release",
    },
    {
        "id": "sensor_audit",
        "dominant_features": {"sensor_gap_hours"},
        "min_risk_tier": "Medium",
        "action": "Sensor / Telemetry Audit",
        "rationale": "Flag hardware/connectivity fault rather than genuine product failure",
    },
]

RISK_TIER_RANK = {"Low": 0, "Medium": 1, "High": 2}


class ActionMapper:
    """
    Inspect dominant SHAP attributions to select a causally-grounded
    recommended response, not just a generic risk label.
    """

    def __init__(self, dominance_threshold: float = 0.15):
        self.threshold = dominance_threshold

    def map(
        self,
        risk_tier: str,
        shap_contributions: list[dict],
        anomaly_score: float | None = None,
        tabular_probability: float | None = None,
    ) -> dict:
        """
        Determine the recommended action for a single shipment.

        Parameters
        ----------
        risk_tier : str
            'Low', 'Medium', or 'High'.
        shap_contributions : list[dict]
            Sorted list of {feature, shap_value, direction}.
        anomaly_score : float | None
            Autoencoder reconstruction error (if available).
        tabular_probability : float | None
            Tabular ensemble failure probability.
        """
        # Identify dominant features (|SHAP| > threshold)
        dominant = set()
        for c in shap_contributions:
            if abs(c["shap_value"]) > self.threshold:
                dominant.add(c["feature"])

        # Check for AE / tabular discordance
        if (
            anomaly_score is not None
            and tabular_probability is not None
            and anomaly_score > 0.8
            and tabular_probability < 0.5
        ):
            return {
                "action": "Escalate for Manual Review",
                "rationale": "Discordance between temporal and tabular models",
                "rule_id": "ae_discordance",
                "dominant_features": list(dominant),
            }

        # Match against rules
        tier_rank = RISK_TIER_RANK.get(risk_tier, 0)
        for rule in ACTION_RULES:
            rule_rank = RISK_TIER_RANK.get(rule["min_risk_tier"], 0)
            if tier_rank >= rule_rank and dominant & rule["dominant_features"]:
                return {
                    "action": rule["action"],
                    "rationale": rule["rationale"],
                    "rule_id": rule["id"],
                    "dominant_features": list(dominant & rule["dominant_features"]),
                }

        # Diffuse attribution — no single feature dominates
        if not dominant:
            return {
                "action": "Monitor / Log Only",
                "rationale": "Probability elevated by combination of weak signals; "
                             "insufficient causal concentration to justify intervention",
                "rule_id": "diffuse",
                "dominant_features": [],
            }

        # Fallback
        return {
            "action": "Monitor / Log Only",
            "rationale": "No matching action rule; default to monitoring",
            "rule_id": "default",
            "dominant_features": list(dominant),
        }
