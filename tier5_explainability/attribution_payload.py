"""
SentinelCold — Tier 5: Attribution Payload
Structured JSON output per shipment (PRD §6.3 schema).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import tier_logger

log = tier_logger(5)


def build_attribution_payload(
    shipment_id: str,
    failure_probability: float,
    risk_tier: str,
    shap_explanation: dict,
    anomaly_score: Optional[float] = None,
    model_version: str = "unknown",
    threshold_used: float = 0.5,
    lime_agreement: Optional[float] = None,
) -> dict:
    """
    Construct the per-shipment Attribution Payload as defined in PRD §6.3.
    """
    top_features = []
    for contrib in shap_explanation.get("top_contributing_features", [])[:5]:
        top_features.append({
            "feature": contrib["feature"],
            "shap_value": contrib["shap_value"],
            "direction": contrib["direction"],
        })

    payload = {
        "shipment_id": shipment_id,
        "failure_probability": round(failure_probability, 4),
        "risk_tier": risk_tier,
        "top_contributing_features": top_features,
        "anomaly_reconstruction_score": round(anomaly_score, 4) if anomaly_score is not None else None,
        "model_version": model_version,
        "threshold_used": round(threshold_used, 4),
        "explanation_method": "TreeSHAP",
        "corroboration_method": "LIME",
        "corroboration_agreement": round(lime_agreement, 4) if lime_agreement is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return payload


def save_payloads(payloads: list[dict], path: Path) -> None:
    """Save a batch of attribution payloads as JSON-lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in payloads:
            f.write(json.dumps(p, default=str) + "\n")
    log.info("Saved %d attribution payloads → %s", len(payloads), path)
