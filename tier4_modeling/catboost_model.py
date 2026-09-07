"""
SentinelCold — Tier 4: CatBoost Model Wrapper
Tertiary classifier — ordered boosting, native categorical handling.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from typing import Optional

from utils.logger import tier_logger

log = tier_logger(4)


class CatBoostModel:
    """CatBoost classifier with optional Optuna tuning."""

    def __init__(self, config: dict, tuned: bool = False, n_trials: int = 100, cat_features: list[str] = None):
        self.config = config
        self.tuned = tuned
        self.n_trials = n_trials
        self.cat_features = cat_features or []
        self.model_: Optional[CatBoostClassifier] = None
        self.best_params_: dict = {}

    @property
    def name(self) -> str:
        return "CatBoost (tuned)" if self.tuned else "CatBoost (default)"

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> "CatBoostModel":
        cat_idx = [X.columns.get_loc(c) for c in self.cat_features if c in X.columns]

        if self.tuned:
            self._tune(X, y, X_val, y_val, cat_idx)
        else:
            params = self.config.get("default", {})
            self.model_ = CatBoostClassifier(**params, cat_features=cat_idx)
            eval_set = (X_val, y_val) if X_val is not None else None
            self.model_.fit(X, y, eval_set=eval_set, verbose=0)
            self.best_params_ = params

        log.info("%s fitted", self.name)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_model(self):
        return self.model_

    def _tune(self, X, y, X_val, y_val, cat_idx):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        search = self.config.get("tuning", {}).get("search_space", {})

        def objective(trial):
            params = {
                "depth": trial.suggest_int("depth", *search.get("depth", [4, 10])),
                "learning_rate": trial.suggest_float("learning_rate", *search.get("learning_rate", [0.01, 0.3]), log=True),
                "iterations": trial.suggest_int("iterations", *search.get("iterations", [100, 1000])),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *search.get("l2_leaf_reg", [1.0, 10.0])),
                "bagging_temperature": trial.suggest_float("bagging_temperature", *search.get("bagging_temperature", [0.0, 5.0])),
                "random_strength": trial.suggest_float("random_strength", *search.get("random_strength", [0.0, 5.0])),
                "auto_class_weights": "Balanced",
                "random_seed": 42,
                "thread_count": -1,
                "verbose": 0,
            }
            model = CatBoostClassifier(**params, cat_features=cat_idx)
            if X_val is not None:
                model.fit(X, y, eval_set=(X_val, y_val), verbose=0)
                from sklearn.metrics import f1_score, recall_score
                preds = model.predict(X_val)
                f1 = f1_score(y_val, preds)
                rec = recall_score(y_val, preds)
                return f1 - (10.0 if rec < 0.95 else 0.0)
            else:
                from sklearn.model_selection import cross_validate
                scores = cross_validate(model, X, y, cv=3, scoring=["f1", "recall"])
                f1 = scores["test_f1"].mean()
                rec = scores["test_recall"].mean()
                return f1 - (10.0 if rec < 0.95 else 0.0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_params_ = study.best_params
        self.best_params_["auto_class_weights"] = "Balanced"
        self.best_params_["random_seed"] = 42
        self.best_params_["verbose"] = 0

        self.model_ = CatBoostClassifier(**self.best_params_, cat_features=cat_idx)
        self.model_.fit(X, y, verbose=0)
        log.info("Optuna best: F1=%.4f", study.best_value)
