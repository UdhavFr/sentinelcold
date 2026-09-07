"""
SentinelCold — Tier 4: Ensemble Methods
Soft-voting ensemble and Stacking ensemble.
"""

import numpy as np
import pandas as pd
from typing import Optional

from utils.logger import tier_logger

log = tier_logger(4)


class SoftVotingEnsemble:
    """
    Average predicted probabilities from multiple base models.
    Weights can be equal (default) or optimized.
    """

    def __init__(self, base_models: list, weights: list[float] | None = None):
        self.base_models = base_models
        self.weights = weights or [1.0 / len(base_models)] * len(base_models)

    @property
    def name(self) -> str:
        return "Soft-Voting Ensemble"

    def fit(self, X: pd.DataFrame, y: pd.Series, *args, **kwargs) -> "SoftVotingEnsemble":
        for model in self.base_models:
            model.fit(X, y, *args, **kwargs)
        log.info("Soft-voting ensemble fitted — %d base models", len(self.base_models))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = []
        for model in self.base_models:
            p = model.predict_proba(X)
            probas.append(p)
        # Weighted average
        weights = np.array(self.weights)
        weights = weights / weights.sum()
        avg = np.zeros_like(probas[0])
        for w, p in zip(weights, probas):
            avg += w * p
        return avg

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class StackingEnsemble:
    """
    Two-level stacking:
      Level-0: base models generate out-of-fold predictions
      Level-1: logistic regression meta-learner
    """

    def __init__(self, base_models: list, n_folds: int = 5, random_state: int = 42):
        self.base_models = base_models
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_learner_ = None

    @property
    def name(self) -> str:
        return "Stacking Ensemble"

    def fit(self, X: pd.DataFrame, y: pd.Series, *args, **kwargs) -> "StackingEnsemble":
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Generate out-of-fold predictions for each base model
        oof_preds = np.zeros((len(X), len(self.base_models)))

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y.iloc[train_idx]

            for m_idx, model_template in enumerate(self.base_models):
                # Clone and fit on fold
                import copy
                model = copy.deepcopy(model_template)
                model.fit(X_tr, y_tr)
                oof_preds[val_idx, m_idx] = model.predict_proba(X_vl)

        # Fit meta-learner on OOF predictions
        self.meta_learner_ = LogisticRegression(
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.meta_learner_.fit(oof_preds, y)

        # Refit all base models on full data for inference
        for model in self.base_models:
            model.fit(X, y, *args, **kwargs)

        log.info("Stacking ensemble fitted — %d base models, LR meta-learner", len(self.base_models))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base_preds = np.column_stack([
            model.predict_proba(X) for model in self.base_models
        ])
        return self.meta_learner_.predict_proba(base_preds)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
