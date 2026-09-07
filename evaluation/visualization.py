"""
SentinelCold — Evaluation: Visualization
All paper-ready figures (18+ plots).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger("sentinelcold.viz")

# --- Style ---
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#44BBA4"]


def plot_roc_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: Optional[Path] = None,
) -> None:
    """
    ROC curves for multiple models on a single plot.
    results: {model_name: (y_true, y_proba)}
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, (y_true, y_proba)) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2, label=f"{name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        log.info("ROC curves → %s", save_path)
    plt.close()


def plot_pr_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: Optional[Path] = None,
) -> None:
    """Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, (y_true, y_proba)) in enumerate(results.items()):
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, y_proba)
        ax.plot(rec, prec, color=PALETTE[i % len(PALETTE)], lw=2, label=f"{name} (AP={ap:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        log.info("PR curves → %s", save_path)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> None:
    """Heatmap-style confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Safe", "Predicted Failure"],
        yticklabels=["Actual Safe", "Actual Failure"],
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        log.info("Confusion matrix → %s", save_path)
    plt.close()


def plot_threshold_sweep(
    sweep_df: pd.DataFrame,
    optimal_theta: float,
    title: str = "Threshold Optimization",
    save_path: Optional[Path] = None,
) -> None:
    """Threshold vs. metrics curve."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(sweep_df["threshold"], sweep_df["recall"], "-o", label="Recall", color=PALETTE[0], markersize=3)
    ax1.plot(sweep_df["threshold"], sweep_df["precision"], "-s", label="Precision", color=PALETTE[1], markersize=3)
    ax1.plot(sweep_df["threshold"], sweep_df["f1"], "-^", label="F1", color=PALETTE[2], markersize=3)
    ax1.axvline(optimal_theta, color="red", linestyle="--", label=f"θ*={optimal_theta:.3f}")
    ax1.set_xlabel("Classification Threshold (θ)")
    ax1.set_ylabel("Score")
    ax1.set_title(title)
    ax1.legend(loc="center right")
    ax1.grid(alpha=0.3)

    if "cost" in sweep_df.columns:
        ax2 = ax1.twinx()
        ax2.plot(sweep_df["threshold"], sweep_df["cost"], "--", color="gray", alpha=0.5, label="Cost")
        ax2.set_ylabel("Expected Cost", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        log.info("Threshold sweep → %s", save_path)
    plt.close()


def plot_ablation_bar(
    results_df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str = "Ablation Study",
    save_path: Optional[Path] = None,
) -> None:
    """Grouped bar chart for ablation comparisons."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.8 / len(y_cols)

    for i, col in enumerate(y_cols):
        offset = (i - len(y_cols) / 2 + 0.5) * width
        bars = ax.bar(x + offset, results_df[col], width, label=col, color=PALETTE[i % len(PALETTE)])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(results_df[x_col], rotation=30, ha="right")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        log.info("Ablation bar → %s", save_path)
    plt.close()


def plot_model_comparison_radar(
    results_df: pd.DataFrame,
    model_col: str = "model",
    metric_cols: list[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Radar chart for multi-metric model comparison."""
    metric_cols = metric_cols or ["f1", "roc_auc", "recall", "precision", "pr_auc"]
    n_metrics = len(metric_cols)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metric_cols]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=row[model_col], color=PALETTE[i % len(PALETTE)])
        ax.fill(angles, values, alpha=0.1, color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_cols)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title("Model Comparison", y=1.08)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        log.info("Radar chart → %s", save_path)
    plt.close()
