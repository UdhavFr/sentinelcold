"""
SentinelCold — Tier 4 Baseline: KNN Classifier
Supervised baseline — documents why KNN is rejected for production.
Expected: Best F1 ≈ 0.4922 (k=7), Best ROC-AUC ≈ 0.8729 (k=31).
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

from utils.logger import tier_logger

log = tier_logger(4)


class KNNBaseline:
    """
    Grid-search over k values with standardized, winsorized features.
    Reports per-k metrics for the paper comparison table.
    """

    def __init__(
        self,
        k_values: list[int] | None = None,
        cv_splits: int = 5,
        random_state: int = 42,
    ):
        self.k_values = k_values or [3, 5, 7, 11, 15, 21, 31]
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.results_: list[dict] = []
        self.best_f1_: dict = {}
        self.best_auc_: dict = {}

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Run stratified k-fold CV for each k value."""
        log.info("═══ KNN Baseline — evaluating k ∈ %s ═══", self.k_values)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))

        skf = StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        scoring = {
            "f1": "f1",
            "roc_auc": "roc_auc",
            "recall": "recall",
            "precision": "precision",
        }

        for k in self.k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            cv_results = cross_validate(
                knn, X_scaled, y, cv=skf, scoring=scoring, return_train_score=False
            )
            row = {
                "k": k,
                "f1_mean": round(float(cv_results["test_f1"].mean()), 4),
                "f1_std": round(float(cv_results["test_f1"].std()), 4),
                "roc_auc_mean": round(float(cv_results["test_roc_auc"].mean()), 4),
                "roc_auc_std": round(float(cv_results["test_roc_auc"].std()), 4),
                "recall_mean": round(float(cv_results["test_recall"].mean()), 4),
                "precision_mean": round(float(cv_results["test_precision"].mean()), 4),
            }
            self.results_.append(row)
            log.info("k=%2d → F1=%.4f±%.4f  AUC=%.4f±%.4f", k, row["f1_mean"], row["f1_std"], row["roc_auc_mean"], row["roc_auc_std"])

        results_df = pd.DataFrame(self.results_)
        best_f1_idx = results_df["f1_mean"].idxmax()
        best_auc_idx = results_df["roc_auc_mean"].idxmax()
        self.best_f1_ = results_df.iloc[best_f1_idx].to_dict()
        self.best_auc_ = results_df.iloc[best_auc_idx].to_dict()

        log.info(
            "Best F1: k=%d → %.4f | Best AUC: k=%d → %.4f",
            self.best_f1_["k"], self.best_f1_["f1_mean"],
            self.best_auc_["k"], self.best_auc_["roc_auc_mean"],
        )
        return results_df

    def rejection_rationale(self) -> str:
        return (
            "KNN is rejected for production because:\n"
            "1. O(N) query-time cost is incompatible with <100ms inference latency.\n"
            "2. Inherent sensitivity to class imbalance in local neighborhood vote.\n"
            "3. No native feature-level explainability (required by Tier 5).\n"
            f"Best F1={self.best_f1_.get('f1_mean', 'N/A')} at k={self.best_f1_.get('k', 'N/A')}"
        )
