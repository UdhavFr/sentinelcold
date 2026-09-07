"""
SentinelCold — Tier 4: LightGBM Model Wrapper
Secondary classifier — leaf-wise growth, native categorical handling.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Optional

from utils.logger import tier_logger

log = tier_logger(4)


class LightGBMModel:
    """LightGBM classifier with optional Optuna tuning."""

    def __init__(self, config: dict, tuned: bool = False, n_trials: int = 100, cat_features: list[str] = None):
        self.config = config
        self.tuned = tuned
        self.n_trials = n_trials
        self.cat_features = cat_features or []
        self.model_: Optional[lgb.LGBMClassifier] = None
        self.best_params_: dict = {}

    @property
    def name(self) -> str:
        return "LightGBM (tuned)" if self.tuned else "LightGBM (default)"

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> "LightGBMModel":
        # Ensure categorical columns are of 'category' dtype
        X = self._set_cat_dtype(X)
        if X_val is not None:
            X_val = self._set_cat_dtype(X_val)

        if self.tuned:
            self._tune(X, y, X_val, y_val)
        else:
            params = self.config.get("default", {})
            self.model_ = lgb.LGBMClassifier(**params)
            callbacks = [lgb.log_evaluation(period=0)]
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.model_.fit(X, y, eval_set=eval_set, callbacks=callbacks)
            self.best_params_ = params

        log.info("%s fitted", self.name)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = self._set_cat_dtype(X)
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_model(self):
        return self.model_

    def _set_cat_dtype(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cat_features:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X

    def _tune(self, X, y, X_val, y_val):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        search = self.config.get("tuning", {}).get("search_space", {})

        def objective(trial):
            params = {
                "num_leaves": trial.suggest_int("num_leaves", *search.get("num_leaves", [15, 63])),
                "learning_rate": trial.suggest_float("learning_rate", *search.get("learning_rate", [0.01, 0.3]), log=True),
                "n_estimators": trial.suggest_int("n_estimators", *search.get("n_estimators", [100, 1000])),
                "subsample": trial.suggest_float("subsample", *search.get("subsample", [0.6, 1.0])),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *search.get("colsample_bytree", [0.6, 1.0])),
                "min_child_samples": trial.suggest_int("min_child_samples", *search.get("min_child_samples", [5, 50])),
                "reg_alpha": trial.suggest_float("reg_alpha", *search.get("reg_alpha", [0.0, 10.0])),
                "reg_lambda": trial.suggest_float("reg_lambda", *search.get("reg_lambda", [0.0, 10.0])),
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            model = lgb.LGBMClassifier(**params)
            if X_val is not None:
                model.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(period=0)])
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
        self.best_params_["class_weight"] = "balanced"
        self.best_params_["random_state"] = 42
        self.best_params_["verbose"] = -1

        self.model_ = lgb.LGBMClassifier(**self.best_params_)
        self.model_.fit(X, y, callbacks=[lgb.log_evaluation(period=0)])
        log.info("Optuna best: F1=%.4f", study.best_value)
