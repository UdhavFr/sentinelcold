"""
SentinelCold — Tier 4 Baseline: K-Means Clustering
Unsupervised baseline — documents why geometric clustering cannot
recover the failure/non-failure boundary.
Expected: Silhouette ≈ 0.274, ARI ≈ 0.2142.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from utils.logger import tier_logger

log = tier_logger(4)


class KMeansBaseline:
    """K-Means (K=2) unsupervised clustering baseline."""

    def __init__(self, n_clusters: int = 2, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.silhouette_: float = 0.0
        self.ari_: float = 0.0
        self.labels_: np.ndarray | None = None

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        log.info("═══ K-Means Baseline (K=%d) ═══", self.n_clusters)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))

        km = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.labels_ = km.fit_predict(X_scaled)
        self.silhouette_ = round(float(silhouette_score(X_scaled, self.labels_)), 4)
        self.ari_ = round(float(adjusted_rand_score(y, self.labels_)), 4)

        log.info("Silhouette Score: %.4f", self.silhouette_)
        log.info("Adjusted Rand Index: %.4f", self.ari_)

        return {
            "n_clusters": self.n_clusters,
            "silhouette_score": self.silhouette_,
            "adjusted_rand_index": self.ari_,
        }

    def rejection_rationale(self) -> str:
        return (
            "K-Means (K=2) is rejected because:\n"
            f"Silhouette Score = {self.silhouette_} (weak cluster structure).\n"
            f"ARI = {self.ari_} (poor alignment with true labels).\n"
            "The failure/non-failure boundary is not globally separable via "
            "geometric clustering — classes are only locally/interactively separable."
        )
