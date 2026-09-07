"""
SentinelCold — Evaluation: Metrics
Comprehensive metric computation with confidence intervals.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    recall_score, precision_score, confusion_matrix,
    matthews_corrcoef, log_loss, brier_score_loss,
    classification_report,
)
from typing import Optional

from utils.logger import get_logger

log = get_logger("sentinelcold.eval")


def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    label: str = "",
) -> dict:
    """Compute all classification metrics at the given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "label": label,
        "threshold": round(threshold, 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0, 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "log_loss": round(float(log_loss(y_true, y_proba)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return metrics


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Compute mean ± std across CV folds."""
    df = pd.DataFrame(fold_metrics)
    agg = {}
    numeric_cols = ["f1", "roc_auc", "pr_auc", "recall", "precision", "specificity", "mcc", "log_loss", "brier_score"]
    for col in numeric_cols:
        if col in df.columns:
            agg[f"{col}_mean"] = round(float(df[col].mean()), 4)
            agg[f"{col}_std"] = round(float(df[col].std()), 4)
    return agg


def bootstrap_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric_fn,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for any metric."""
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_t = y_true[idx]
        y_p = y_proba[idx]
        if len(np.unique(y_t)) < 2:
            continue
        try:
            y_pred = (y_p >= threshold).astype(int)
            score = metric_fn(y_t, y_pred)
            scores.append(score)
        except Exception:
            pass

    scores = np.array(scores)
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    mean = np.mean(scores)
    return round(float(mean), 4), round(float(lower), 4), round(float(upper), 4)


def validate_targets(metrics: dict, targets: dict | None = None) -> dict:
    """Check if metrics meet PRD performance targets."""
    targets = targets or {
        "f1": 0.90,
        "roc_auc": 0.92,
        "recall": 0.95,
        "precision": 0.55,
    }
    results = {}
    for metric, target in targets.items():
        key = f"{metric}_mean" if f"{metric}_mean" in metrics else metric
        actual = metrics.get(key, 0)
        passed = actual >= target
        results[metric] = {
            "target": target,
            "actual": actual,
            "passed": passed,
        }
        status = "✓ PASS" if passed else "✗ FAIL"
        log.info("  %s %s: %.4f (target ≥ %.4f)", status, metric, actual, target)
    return results
