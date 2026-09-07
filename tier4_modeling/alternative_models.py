"""
SentinelCold — Tier 4: Alternative Models
Random Forest (bagging baseline) and Logistic Regression (linear baseline).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import Optional

from utils.logger import tier_logger

log = tier_logger(4)


class RandomForestModel:
    """Random Forest — bagging vs. boosting comparison point."""

    def __init__(self, config: dict):
        self.config = config
        self.model_: Optional[RandomForestClassifier] = None

    @property
    def name(self) -> str:
        return "Random Forest"

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "RandomForestModel":
        self.model_ = RandomForestClassifier(
            n_estimators=self.config.get("n_estimators", 500),
            max_depth=self.config.get("max_depth", None),
            class_weight=self.config.get("class_weight", "balanced_subsample"),
            random_state=self.config.get("random_state", 42),
            n_jobs=-1,
        )
        self.model_.fit(X, y)
        log.info("Random Forest fitted (%d trees)", self.model_.n_estimators)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_model(self):
        return self.model_


class LogisticRegressionModel:
    """
    Linear baseline — quantifies how much non-linearity the trees capture.
    Uses VIF-filtered, winsorized features.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_: Optional[LogisticRegression] = None
        self.scaler_ = None
        self.best_C_: float = 1.0

    @property
    def name(self) -> str:
        return "Logistic Regression"

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "LogisticRegressionModel":
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GridSearchCV

        self.scaler_ = StandardScaler()
        X_numeric = X.select_dtypes(include=[np.number])
        X_scaled = self.scaler_.fit_transform(X_numeric)

        C_values = self.config.get("C", [0.001, 0.01, 0.1, 1.0, 10.0])
        base_lr = LogisticRegression(
            class_weight=self.config.get("class_weight", "balanced"),
            max_iter=self.config.get("max_iter", 1000),
            solver=self.config.get("solver", "lbfgs"),
            random_state=self.config.get("random_state", 42),
        )
        grid = GridSearchCV(base_lr, {"C": C_values}, cv=3, scoring="f1", n_jobs=-1)
        grid.fit(X_scaled, y)

        self.model_ = grid.best_estimator_
        self.best_C_ = grid.best_params_["C"]
        log.info("Logistic Regression fitted (best C=%.4f, F1=%.4f)", self.best_C_, grid.best_score_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_numeric = X.select_dtypes(include=[np.number])
        X_scaled = self.scaler_.transform(X_numeric)
        return self.model_.predict_proba(X_scaled)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_model(self):
        return self.model_
