"""
SentinelCold — Experiment Runner
Executes a single experiment from a configuration dict.
Handles the full pipeline: ingest → preprocess → engineer → train → evaluate.
"""

import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from tier1_ingestion.validator import DataValidator
from tier2_preprocessing.imputer import ImputerFactory
from tier2_preprocessing.outlier_handler import OutlierHandler
from tier2_preprocessing.balancer import BalancerFactory
from tier3_feature_engineering.transforms import FeatureEngineer
from tier3_feature_engineering.feature_sets import get_feature_set, CATEGORICAL_FEATURES
from tier4_modeling.xgboost_model import XGBoostModel
from tier4_modeling.lightgbm_model import LightGBMModel
from tier4_modeling.catboost_model import CatBoostModel
from tier4_modeling.alternative_models import RandomForestModel, LogisticRegressionModel
from tier4_modeling.threshold_optimizer import ThresholdOptimizerFactory
from evaluation.metrics import compute_all_metrics, aggregate_fold_metrics

from utils.logger import get_logger
from utils.reproducibility import set_global_seed

log = get_logger("sentinelcold.experiment")


class ExperimentRunner:
    """
    Execute a single experiment: one combination of
    (imputation, balancing, feature_set, model, threshold_strategy).
    """

    def __init__(self, config: dict, data: pd.DataFrame):
        self.config = config
        self.data = data
        self.results_: dict = {}

    def run(self) -> dict:
        """Run the full experiment and return results."""
        start = time.time()
        set_global_seed(self.config.get("seed", 42))

        imputation = self.config.get("imputation", "class_conditional_median")
        balancing = self.config.get("balancing", "smote")
        balance_ratio = self.config.get("balance_ratio", 0.5)
        feature_set = self.config.get("feature_set", "FS-B")
        model_id = self.config.get("model", "xgb_default")
        threshold_strategy = self.config.get("threshold", "asymmetric_cost")
        n_splits = self.config.get("cv_splits", 5)

        log.info("═══ Experiment: imp=%s | bal=%s(%.1f) | fs=%s | model=%s | thr=%s ═══",
                 imputation, balancing, balance_ratio, feature_set, model_id, threshold_strategy)

        target_col = "silent_failure"
        id_col = "shipment_id"

        # Separate features and target
        y = self.data[target_col].copy()
        X_base = self.data.drop(columns=[target_col, id_col], errors="ignore").copy()

        # Setup CV
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        all_y_true = []
        all_y_proba = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_base, y)):
            log.info("--- Fold %d/%d ---", fold_idx + 1, n_splits)

            X_train = X_base.iloc[train_idx].copy()
            X_val = X_base.iloc[val_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_val = y.iloc[val_idx].copy()

            # Step 1: Imputation (fit on train, transform both)
            imp_config = self.config.get("imputation_config", {
                "columns_to_impute": ["rh_max", "temp_recovery_rate"],
                "knn_fallback_features": {
                    "rh_max": ["rh_mean", "rh_std", "temp_mean_c"],
                    "temp_recovery_rate": ["temp_mean_c", "temp_std_c", "transit_days"],
                },
            })
            imputer = ImputerFactory.create(imputation, imp_config)
            X_train = imputer.fit_transform(X_train, y_train)
            X_val = imputer.transform(X_val, y_val)

            # Step 2: Feature Engineering (fit z-scores on train)
            fe = FeatureEngineer()
            X_train = fe.fit_transform(X_train)
            X_val = fe.transform(X_val)

            # Step 3: Select feature set
            fs_cols = get_feature_set(feature_set)
            available = [c for c in fs_cols if c in X_train.columns]
            X_train_fs = X_train[available]
            X_val_fs = X_val[available]

            # Step 4: Outlier handling
            outlier_mode = "linear" if model_id in ("logistic_reg", "knn") else "tree"
            oh = OutlierHandler(mode=outlier_mode, exclude_cols=CATEGORICAL_FEATURES)
            X_train_fs = oh.fit_transform(X_train_fs)
            X_val_fs = oh.transform(X_val_fs)

            # Step 5: Balancing (train fold only)
            balancer = BalancerFactory.create(balancing, target_ratio=balance_ratio)
            X_train_bal, y_train_bal = balancer.resample(X_train_fs, y_train)

            # Split validation set to avoid threshold/early-stopping leakage
            from sklearn.model_selection import train_test_split
            X_val_tune, X_val_eval, y_val_tune, y_val_eval = train_test_split(
                X_val_fs, y_val, test_size=0.5, stratify=y_val, random_state=42
            )

            # Step 6: Train model
            model = self._create_model(model_id, available)
            model.fit(X_train_bal, y_train_bal, X_val_tune, y_val_tune)

            # Step 7: Predict
            y_proba_tune = model.predict_proba(X_val_tune)
            y_proba_eval = model.predict_proba(X_val_eval)

            # Step 8: Optimize threshold on tune set
            thr_config = self.config.get("threshold_config", {})
            thr_optimizer = ThresholdOptimizerFactory.create(threshold_strategy, thr_config)
            threshold = thr_optimizer.optimize(y_val_tune.values, y_proba_tune)

            # Step 9: Compute metrics on true holdout eval set
            metrics = compute_all_metrics(y_val_eval.values, y_proba_eval, threshold, label=f"fold_{fold_idx}")
            fold_metrics.append(metrics)
            all_y_true.extend(y_val_eval.values)
            all_y_proba.extend(y_proba_eval)

            log.info("Fold %d: F1=%.4f, AUC=%.4f, Recall=%.4f @ θ=%.3f",
                     fold_idx + 1, metrics["f1"], metrics["roc_auc"], metrics["recall"], threshold)

        # Aggregate
        agg = aggregate_fold_metrics(fold_metrics)
        elapsed = round(time.time() - start, 2)

        self.results_ = {
            "config": {
                "imputation": imputation,
                "balancing": f"{balancing}_{balance_ratio}",
                "feature_set": feature_set,
                "model": model_id,
                "threshold": threshold_strategy,
            },
            "fold_metrics": fold_metrics,
            "aggregated": agg,
            "runtime_s": elapsed,
            "all_y_true": np.array(all_y_true),
            "all_y_proba": np.array(all_y_proba),
        }

        log.info(
            "═══ RESULT: F1=%.4f±%.4f | AUC=%.4f±%.4f | Recall=%.4f±%.4f | %.1fs ═══",
            agg.get("f1_mean", 0), agg.get("f1_std", 0),
            agg.get("roc_auc_mean", 0), agg.get("roc_auc_std", 0),
            agg.get("recall_mean", 0), agg.get("recall_std", 0),
            elapsed,
        )
        return self.results_

    def _create_model(self, model_id: str, feature_names: list[str]):
        """Factory for creating a model by ID."""
        model_configs = self.config.get("model_configs", {})
        cat_feats = [f for f in CATEGORICAL_FEATURES if f in feature_names]

        if model_id == "xgb_default":
            return XGBoostModel(model_configs.get("xgboost", {}), tuned=False)
        elif model_id == "xgb_tuned":
            return XGBoostModel(model_configs.get("xgboost", {}), tuned=True,
                                n_trials=model_configs.get("xgboost", {}).get("tuning", {}).get("n_trials", 100))
        elif model_id == "lgbm_default":
            return LightGBMModel(model_configs.get("lightgbm", {}), tuned=False, cat_features=cat_feats)
        elif model_id == "lgbm_tuned":
            return LightGBMModel(model_configs.get("lightgbm", {}), tuned=True, cat_features=cat_feats,
                                 n_trials=model_configs.get("lightgbm", {}).get("tuning", {}).get("n_trials", 100))
        elif model_id == "catboost_default":
            return CatBoostModel(model_configs.get("catboost", {}), tuned=False, cat_features=cat_feats)
        elif model_id == "catboost_tuned":
            return CatBoostModel(model_configs.get("catboost", {}), tuned=True, cat_features=cat_feats,
                                 n_trials=model_configs.get("catboost", {}).get("tuning", {}).get("n_trials", 100))
        elif model_id == "random_forest":
            return RandomForestModel(model_configs.get("random_forest", {}))
        elif model_id == "logistic_reg":
            return LogisticRegressionModel(model_configs.get("logistic_regression", {}))
        elif model_id == "ensemble_equal":
            from tier4_modeling.ensemble import SoftVotingEnsemble
            return SoftVotingEnsemble([
                self._create_model("xgb_default", feature_names),
                self._create_model("lgbm_default", feature_names),
                self._create_model("catboost_default", feature_names)
            ])
        elif model_id == "ensemble_tuned":
            from tier4_modeling.ensemble import SoftVotingEnsemble
            return SoftVotingEnsemble([
                self._create_model("xgb_tuned", feature_names),
                self._create_model("lgbm_tuned", feature_names),
                self._create_model("catboost_tuned", feature_names)
            ])
        elif model_id == "stacking":
            from tier4_modeling.ensemble import StackingEnsemble
            return StackingEnsemble([
                self._create_model("xgb_default", feature_names),
                self._create_model("lgbm_default", feature_names),
                self._create_model("catboost_default", feature_names)
            ])
        elif model_id == "tabnet":
            # Fallback for tabnet if pytorch_tabnet not implemented
            log.warning("TabNet not implemented, falling back to LogisticRegression")
            return LogisticRegressionModel(model_configs.get("logistic_regression", {}))
        else:
            raise ValueError(f"Unknown model ID: '{model_id}'")
