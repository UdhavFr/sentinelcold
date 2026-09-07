"""
SentinelCold — Tier 6: Risk Tiering Engine
Assigns Low / Medium / High risk tiers based on calibrated probability
and the asymmetric-cost-optimized threshold.
"""

import numpy as np
import pandas as pd

from utils.logger import tier_logger

log = tier_logger(6)


class RiskTieringEngine:
    """
    Risk tier assignment per PRD §7.1:
      Low:    p < θ*
      Medium: θ* ≤ p < θ_high
      High:   p ≥ θ_high
    where θ_high = θ* + offset (default 0.35).
    """

    def __init__(self, theta_star: float, theta_high_offset: float = 0.35):
        self.theta_star = theta_star
        self.theta_high = theta_star + theta_high_offset
        log.info("Risk tiers: Low < %.3f, Medium [%.3f, %.3f), High ≥ %.3f",
                 self.theta_star, self.theta_star, self.theta_high, self.theta_high)

    def assign(self, probabilities: np.ndarray) -> np.ndarray:
        """Return an array of tier labels: 'Low', 'Medium', 'High'."""
        tiers = np.where(
            probabilities >= self.theta_high, "High",
            np.where(probabilities >= self.theta_star, "Medium", "Low")
        )
        return tiers

    def distribution(self, tiers: np.ndarray) -> dict:
        """Count and percentage per tier."""
        unique, counts = np.unique(tiers, return_counts=True)
        total = len(tiers)
        dist = {}
        for tier, count in zip(unique, counts):
            dist[tier] = {"count": int(count), "pct": round(count / total * 100, 2)}
        log.info("Risk distribution: %s", dist)
        return dist
