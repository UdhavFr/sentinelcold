"""
SentinelCold — Tier 5: SHAP Engine
Global and local TreeSHAP explanations for gradient-boosted tree ensembles.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from utils.logger import tier_logger

log = tier_logger(5)


class SHAPEngine:
    """
    TreeSHAP-based explainability engine.
    Produces global feature importance and per-shipment local attributions.
    """

    def __init__(self, model, feature_names: list[str], max_display: int = 15):
        self.model = model
        self.feature_names = feature_names
        self.max_display = max_display
        self.explainer_: Optional[shap.TreeExplainer] = None
        self.shap_values_: Optional[np.ndarray] = None
        self.expected_value_: float = 0.0

    def fit(self, X: pd.DataFrame) -> "SHAPEngine":
        """Compute SHAP values on the provided dataset (typically validation set)."""
        log.info("Computing TreeSHAP values for %d samples …", len(X))
        self.explainer_ = shap.TreeExplainer(self.model)
        self.shap_values_ = self.explainer_.shap_values(X)

        # Handle multi-output (some models return list)
        if isinstance(self.shap_values_, list):
            self.shap_values_ = self.shap_values_[1]  # positive-class SHAP

        self.expected_value_ = float(
            self.explainer_.expected_value[1]
            if isinstance(self.explainer_.expected_value, (list, np.ndarray))
            else self.explainer_.expected_value
        )
        log.info("SHAP values computed — shape %s", self.shap_values_.shape)
        return self

    # ------------------------------------------------------------------ #
    #  Global explainability                                               #
    # ------------------------------------------------------------------ #

    def global_importance(self) -> pd.DataFrame:
        """Ranked feature importance by mean |SHAP|."""
        mean_abs = np.abs(self.shap_values_).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": np.round(mean_abs, 6),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return df

    def plot_global_bar(self, save_path: Optional[Path] = None) -> None:
        """SHAP global feature importance bar chart."""
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values_,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=self.max_display,
            show=False,
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("SHAP global bar → %s", save_path)
        plt.close()

    def plot_beeswarm(self, X: pd.DataFrame, save_path: Optional[Path] = None) -> None:
        """SHAP beeswarm / summary plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values_,
            X,
            feature_names=self.feature_names,
            max_display=self.max_display,
            show=False,
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("SHAP beeswarm → %s", save_path)
        plt.close()

    def plot_dependence(self, feature: str, X: pd.DataFrame, save_path: Optional[Path] = None) -> None:
        """SHAP dependence plot for a single feature."""
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.dependence_plot(
            feature,
            self.shap_values_,
            X,
            feature_names=self.feature_names,
            show=False,
            ax=ax,
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("SHAP dependence (%s) → %s", feature, save_path)
        plt.close()

    # ------------------------------------------------------------------ #
    #  Local (per-shipment) explainability                                 #
    # ------------------------------------------------------------------ #

    def local_explanation(self, idx: int) -> dict:
        """
        Get SHAP attribution for a single sample.
        Returns a dict with feature contributions sorted by |SHAP|.
        """
        sv = self.shap_values_[idx]
        contributions = []
        for f, v in zip(self.feature_names, sv):
            contributions.append({
                "feature": f,
                "shap_value": round(float(v), 6),
                "direction": "increases_risk" if v > 0 else "decreases_risk",
            })
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return {
            "sample_index": idx,
            "expected_value": round(self.expected_value_, 6),
            "top_contributing_features": contributions[:5],
            "all_contributions": contributions,
        }

    def plot_waterfall(self, X: pd.DataFrame, idx: int, save_path: Optional[Path] = None) -> None:
        """SHAP waterfall plot for a single sample."""
        fig = plt.figure(figsize=(10, 6))
        explanation = shap.Explanation(
            values=self.shap_values_[idx],
            base_values=self.expected_value_,
            data=X.iloc[idx].values,
            feature_names=self.feature_names,
        )
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("SHAP waterfall (idx=%d) → %s", idx, save_path)
        plt.close()

    def plot_force(self, X: pd.DataFrame, idx: int, save_path: Optional[Path] = None) -> None:
        """SHAP force plot for a single sample."""
        shap.initjs()
        force = shap.force_plot(
            self.expected_value_,
            self.shap_values_[idx],
            X.iloc[idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False,
        )
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("SHAP force (idx=%d) → %s", idx, save_path)
        plt.close()
