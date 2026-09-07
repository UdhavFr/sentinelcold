"""
SentinelCold — Tier 4 Baseline: Enhanced Isolation Forest (eiForest)
Benchmark against Xie et al. (2025).
Uses scikit-learn IsolationForest as a proxy — documented as approximation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from utils.logger import tier_logger

log = tier_logger(4)


class EIForestBaseline:
    """
    Isolation Forest baseline (proxy for Xie et al. eiForest).

    The original eiForest uses a cross-grouping factor (r=0.6) and
    subsampling enhancements not available in scikit-learn.
    This proxy uses the standard IsolationForest with contamination
    set to the known positive prevalence.
    """

    def __init__(
        self,
        n_estimators_list: list[int] | None = None,
        contamination: float = 0.197,
        random_state: int = 42,
    ):
        self.n_estimators_list = n_estimators_list or [100, 200, 500]
        self.contamination = contamination
        self.random_state = random_state
        self.results_: list[dict] = []
        self.best_: dict = {}

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        log.info("═══ Isolation Forest Baseline (proxy for eiForest) ═══")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))

        for n_est in self.n_estimators_list:
            iforest = IsolationForest(
                n_estimators=n_est,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
            # IsolationForest: -1 = anomaly, 1 = normal
            raw_pred = iforest.fit_predict(X_scaled)
            # Convert: anomaly (-1) → 1 (failure), normal (1) → 0
            y_pred = (raw_pred == -1).astype(int)
            # Decision scores for AUC (more negative = more anomalous)
            scores = -iforest.decision_function(X_scaled)

            row = {
                "n_estimators": n_est,
                "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y, scores)), 4),
            }
            self.results_.append(row)
            log.info(
                "n_estimators=%3d → P=%.4f R=%.4f F1=%.4f AUC=%.4f",
                n_est, row["precision"], row["recall"], row["f1"], row["roc_auc"],
            )

        results_df = pd.DataFrame(self.results_)
        best_idx = results_df["f1"].idxmax()
        self.best_ = results_df.iloc[best_idx].to_dict()

        log.info(
            "Best eiForest (proxy): n_est=%d → F1=%.4f, AUC=%.4f",
            self.best_["n_estimators"], self.best_["f1"], self.best_["roc_auc"],
        )
        return results_df

    def benchmark_note(self) -> str:
        return (
            "NOTE: This is a scikit-learn IsolationForest proxy, not the full "
            "Xie et al. (2025) eiForest with cross-grouping factor r=0.6. "
            "The original eiForest reports F1=0.8639, AUC=0.9064 on proprietary "
            "vehicle-mounted sensor data. Direct comparison should note this "
            "methodological difference."
        )
