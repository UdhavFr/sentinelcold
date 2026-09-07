"""
SentinelCold — Tier 4: XGBoost Model Wrapper
Primary classifier — default config + Optuna hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional

from utils.logger import tier_logger

log = tier_logger(4)


class XGBoostModel:
    """XGBoost classifier with optional Optuna hyperparameter optimization."""

    def __init__(self, config: dict, tuned: bool = False, n_trials: int = 100):
        self.config = config
        self.tuned = tuned
        self.n_trials = n_trials
        self.model_: Optional[xgb.XGBClassifier] = None
        self.best_params_: dict = {}

    @property
    def name(self) -> str:
        return "XGBoost (tuned)" if self.tuned else "XGBoost (default)"

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> "XGBoostModel":
        if self.tuned:
            self._tune(X, y, X_val, y_val)
        else:
            params = self.config.get("default", {})
            self.model_ = xgb.XGBClassifier(**params)
            self.model_.fit(
                X, y,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                verbose=False,
            )
            self.best_params_ = params
        log.info("%s fitted with params: %s", self.name, self.best_params_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_model(self):
        return self.model_

    def _tune(self, X, y, X_val, y_val):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        search = self.config.get("tuning", {}).get("search_space", {})

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", *search.get("max_depth", [3, 10])),
                "learning_rate": trial.suggest_float("learning_rate", *search.get("learning_rate", [0.01, 0.3]), log=True),
                "n_estimators": trial.suggest_int("n_estimators", *search.get("n_estimators", [100, 1000])),
                "subsample": trial.suggest_float("subsample", *search.get("subsample", [0.6, 1.0])),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *search.get("colsample_bytree", [0.6, 1.0])),
                "min_child_weight": trial.suggest_int("min_child_weight", *search.get("min_child_weight", [1, 10])),
                "gamma": trial.suggest_float("gamma", *search.get("gamma", [0.0, 5.0])),
                "reg_alpha": trial.suggest_float("reg_alpha", *search.get("reg_alpha", [0.0, 10.0])),
                "reg_lambda": trial.suggest_float("reg_lambda", *search.get("reg_lambda", [0.0, 10.0])),
                "scale_pos_weight": self.config["default"].get("scale_pos_weight", 4.08),
                "eval_metric": "logloss",
                "use_label_encoder": False,
                "random_state": self.config["default"].get("random_state", 42),
                "n_jobs": -1,
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X, y, eval_set=[(X_val, y_val)] if X_val is not None else None, verbose=False)

            if X_val is not None:
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
        self.best_params_["scale_pos_weight"] = self.config["default"].get("scale_pos_weight", 4.08)
        self.best_params_["eval_metric"] = "logloss"
        self.best_params_["use_label_encoder"] = False
        self.best_params_["random_state"] = self.config["default"].get("random_state", 42)

        self.model_ = xgb.XGBClassifier(**self.best_params_)
        self.model_.fit(X, y, verbose=False)
        log.info("Optuna best trial: F1=%.4f, params=%s", study.best_value, self.best_params_)
