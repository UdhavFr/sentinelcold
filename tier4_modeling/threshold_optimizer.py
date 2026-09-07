"""
SentinelCold — Tier 4: Threshold Optimization
Three strategies for classification threshold selection.

THR-A  asymmetric_cost  — PRD default: minimize cost subject to Recall ≥ 0.95
THR-B  f1_max           — Maximize F1-score
THR-C  youden_j         — Maximize Youden's J (Sensitivity + Specificity − 1)
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, recall_score, precision_score, confusion_matrix,
    roc_curve,
)

from utils.logger import tier_logger

log = tier_logger(4)


class BaseThresholdOptimizer(ABC):
    """Common interface for threshold optimization strategies."""

    def __init__(self):
        self.optimal_threshold_: float = 0.5
        self.sweep_results_: pd.DataFrame = pd.DataFrame()

    @abstractmethod
    def optimize(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class AsymmetricCostOptimizer(BaseThresholdOptimizer):
    """
    PRD §5.3: argmin_θ E[cost(θ)] subject to Recall(θ) ≥ recall_floor.
    Cost matrix: TN=0, FP=fp_cost, FN=fn_cost, TP=tp_cost.
    """

    def __init__(
        self,
        fn_cost: float = 20.0,
        fp_cost: float = 1.0,
        tp_cost: float = 2.0,
        tn_cost: float = 0.0,
        recall_floor: float = 0.95,
        theta_range: tuple[float, float] = (0.05, 0.50),
        theta_step: float = 0.01,
    ):
        super().__init__()
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.recall_floor = recall_floor
        self.theta_range = theta_range
        self.theta_step = theta_step

    def optimize(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        thresholds = np.arange(self.theta_range[0], self.theta_range[1] + self.theta_step, self.theta_step)
        rows = []

        for theta in thresholds:
            y_pred = (y_proba >= theta).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            cost = self.tn_cost * tn + self.fp_cost * fp + self.fn_cost * fn + self.tp_cost * tp
            rec = recall_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            rows.append({
                "threshold": round(theta, 3),
                "recall": round(rec, 4),
                "precision": round(prec, 4),
                "f1": round(f1, 4),
                "cost": round(cost, 2),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            })

        self.sweep_results_ = pd.DataFrame(rows)

        # Filter by recall floor, then pick lowest cost (highest precision)
        qualifying = self.sweep_results_[self.sweep_results_["recall"] >= self.recall_floor]
        if len(qualifying) == 0:
            # Relax: pick highest-recall threshold
            log.warning("No threshold meets recall floor %.2f — picking max-recall", self.recall_floor)
            best_idx = self.sweep_results_["recall"].idxmax()
        else:
            best_idx = qualifying["cost"].idxmin()

        self.optimal_threshold_ = self.sweep_results_.loc[best_idx, "threshold"]
        best = self.sweep_results_.loc[best_idx]
        log.info(
            "Asymmetric cost optimizer: θ*=%.3f (Recall=%.4f, Precision=%.4f, F1=%.4f, Cost=%.1f)",
            self.optimal_threshold_, best["recall"], best["precision"], best["f1"], best["cost"],
        )
        return self.optimal_threshold_


class F1MaximizingOptimizer(BaseThresholdOptimizer):
    """Select θ that maximizes F1-score."""

    def optimize(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        thresholds = np.arange(0.05, 0.95, 0.01)
        rows = []
        for theta in thresholds:
            y_pred = (y_proba >= theta).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rows.append({"threshold": round(theta, 3), "f1": round(f1, 4), "recall": round(rec, 4), "precision": round(prec, 4)})

        self.sweep_results_ = pd.DataFrame(rows)
        best_idx = self.sweep_results_["f1"].idxmax()
        self.optimal_threshold_ = self.sweep_results_.loc[best_idx, "threshold"]
        best = self.sweep_results_.loc[best_idx]
        log.info("F1-max optimizer: θ*=%.3f (F1=%.4f, Recall=%.4f, Precision=%.4f)", self.optimal_threshold_, best["f1"], best["recall"], best["precision"])
        return self.optimal_threshold_


class YoudenJOptimizer(BaseThresholdOptimizer):
    """Select θ maximizing Youden's J = Sensitivity + Specificity − 1."""

    def optimize(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr  # = Sensitivity + Specificity - 1
        best_idx = np.argmax(j_scores)
        self.optimal_threshold_ = float(thresholds[best_idx])

        # Build sweep for consistency
        rows = []
        for i in range(len(thresholds)):
            rows.append({"threshold": round(float(thresholds[i]), 4), "tpr": round(float(tpr[i]), 4), "fpr": round(float(fpr[i]), 4), "youden_j": round(float(j_scores[i]), 4)})
        self.sweep_results_ = pd.DataFrame(rows)

        log.info("Youden's J optimizer: θ*=%.3f (J=%.4f)", self.optimal_threshold_, j_scores[best_idx])
        return self.optimal_threshold_


class ThresholdOptimizerFactory:
    """Create a threshold optimizer by strategy name."""

    _REGISTRY = {
        "asymmetric_cost": AsymmetricCostOptimizer,
        "f1_max": F1MaximizingOptimizer,
        "youden_j": YoudenJOptimizer,
    }

    @classmethod
    def create(cls, strategy: str, config: dict = None) -> BaseThresholdOptimizer:
        config = config or {}
        if strategy not in cls._REGISTRY:
            raise ValueError(f"Unknown threshold strategy: '{strategy}'")

        if strategy == "asymmetric_cost":
            ac_config = config.get("asymmetric_cost", {})
            return AsymmetricCostOptimizer(
                fn_cost=ac_config.get("cost_fn", 20),
                fp_cost=ac_config.get("cost_fp", 1),
                tp_cost=ac_config.get("cost_tp", 2),
                tn_cost=ac_config.get("cost_tn", 0),
                recall_floor=ac_config.get("recall_floor", 0.95),
                theta_range=tuple(ac_config.get("theta_range", [0.05, 0.50])),
                theta_step=ac_config.get("theta_step", 0.01),
            )
        return cls._REGISTRY[strategy]()

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._REGISTRY.keys())
