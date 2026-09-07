"""
SentinelCold — Tier 5: LIME Engine
Model-agnostic cross-check on a sampled subset of high-risk predictions.
Computes SHAP-LIME top-3 feature agreement rate.
"""

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from typing import Callable, Optional

from utils.logger import tier_logger

log = tier_logger(5)


class LIMEEngine:
    """LIME tabular explainer for corroboration of SHAP attributions."""

    def __init__(
        self,
        predict_fn: Callable,
        feature_names: list[str],
        categorical_features: list[int] | None = None,
        n_samples: int = 5000,
        n_features: int = 10,
    ):
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.cat_features = categorical_features or []
        self.n_samples = n_samples
        self.n_features = n_features
        self.explainer_: Optional[LimeTabularExplainer] = None

    def fit(self, X_train: np.ndarray) -> "LIMEEngine":
        """Initialize the LIME explainer with training data statistics."""
        self.explainer_ = LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            categorical_features=self.cat_features,
            class_names=["Safe", "Failure"],
            mode="classification",
            discretize_continuous=True,
        )
        log.info("LIME explainer initialized with %d training samples", len(X_train))
        return self

    def explain_instance(self, x: np.ndarray) -> dict:
        """Get LIME explanation for a single instance."""
        exp = self.explainer_.explain_instance(
            x,
            self.predict_fn,
            num_features=self.n_features,
            num_samples=self.n_samples,
        )
        contributions = []
        for feature_desc, weight in exp.as_list():
            contributions.append({
                "feature_description": feature_desc,
                "weight": round(float(weight), 6),
                "direction": "increases_risk" if weight > 0 else "decreases_risk",
            })
        return {
            "intercept": round(float(exp.intercept[1]), 6),
            "local_prediction": round(float(exp.local_pred[0]), 6) if hasattr(exp, 'local_pred') else None,
            "contributions": contributions,
        }

    def compute_agreement(
        self,
        X: pd.DataFrame,
        shap_explanations: list[dict],
        sample_indices: list[int],
        top_k: int = 3,
    ) -> dict:
        """
        Compare LIME top-k features with SHAP top-k features.
        Returns agreement rate and per-sample breakdown.
        """
        agreements = []
        for i, idx in enumerate(sample_indices):
            lime_exp = self.explain_instance(X.iloc[idx].values)

            # Extract top-k feature names from LIME
            lime_top = set()
            for contrib in lime_exp["contributions"][:top_k]:
                # LIME feature descriptions may include ranges; extract base name
                desc = contrib["feature_description"]
                for fn in self.feature_names:
                    if fn in desc:
                        lime_top.add(fn)
                        break

            # Extract top-k from SHAP
            shap_top = set()
            for contrib in shap_explanations[i]["top_contributing_features"][:top_k]:
                shap_top.add(contrib["feature"])

            overlap = len(lime_top & shap_top) / max(top_k, 1)
            agreements.append({
                "sample_index": idx,
                "shap_top": list(shap_top),
                "lime_top": list(lime_top),
                "overlap_ratio": round(overlap, 4),
            })

        overall = np.mean([a["overlap_ratio"] for a in agreements])
        result = {
            "top_k": top_k,
            "n_samples": len(sample_indices),
            "mean_agreement": round(float(overall), 4),
            "per_sample": agreements,
        }
        log.info(
            "SHAP-LIME agreement: %.1f%% (top-%d, %d samples)",
            overall * 100, top_k, len(sample_indices),
        )
        return result
