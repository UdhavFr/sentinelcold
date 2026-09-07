"""
SentinelCold — Tier 2: Multi-Strategy Imputation
Five interchangeable imputation strategies behind a common interface.
Each strategy fits on the training fold only to prevent data leakage.

Strategies
----------
IMP-A  class_conditional_median   PRD default — separate medians per target class
IMP-B  global_median              Naive baseline — single global median
IMP-C  knn                        KNNImputer (k=5)
IMP-D  mice                       IterativeImputer with BayesianRidge
IMP-E  missforest                 IterativeImputer with RandomForest estimator
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor

from utils.logger import tier_logger

log = tier_logger(2)


# ========================================================================== #
#  Base class                                                                 #
# ========================================================================== #

class BaseImputer(ABC):
    """Common interface for all imputation strategies."""

    def __init__(self, columns: list[str]):
        self.columns = columns
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseImputer":
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        ...

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X, y)

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ========================================================================== #
#  IMP-A: Class-Conditional Median Imputation                                #
# ========================================================================== #

class ClassConditionalMedianImputer(BaseImputer):
    """
    For each target column, compute separate medians conditioned on y=0 and y=1,
    then fill missing values with the matching class median.
    At inference (y unknown) a lightweight KNN regressor is used as fallback.
    """

    def __init__(
        self,
        columns: list[str],
        target_col: str = "silent_failure",
        knn_fallback_features: Optional[dict[str, list[str]]] = None,
        knn_k: int = 5,
    ):
        super().__init__(columns)
        self.target_col = target_col
        self.knn_fallback_features = knn_fallback_features or {}
        self.knn_k = knn_k
        self.class_medians_: dict[str, dict[int, float]] = {}
        self.fallback_models_: dict[str, KNeighborsRegressor] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ClassConditionalMedianImputer":
        if y is None and self.target_col in X.columns:
            y = X[self.target_col]
        if y is None:
            raise ValueError("y (target) must be provided for class-conditional imputation")

        for col in self.columns:
            if col not in X.columns:
                continue
            medians = {}
            for cls in [0, 1]:
                mask = y == cls
                medians[cls] = float(X.loc[mask, col].median())
            self.class_medians_[col] = medians
            log.info(
                "CCMI fit [%s]: median_0=%.4f, median_1=%.4f",
                col, medians[0], medians[1],
            )

            # Fit KNN fallback for inference time
            if col in self.knn_fallback_features:
                feat_cols = self.knn_fallback_features[col]
                valid_mask = X[col].notna() & X[feat_cols].notna().all(axis=1)
                if valid_mask.sum() > self.knn_k:
                    knn = KNeighborsRegressor(n_neighbors=self.knn_k)
                    knn.fit(X.loc[valid_mask, feat_cols], X.loc[valid_mask, col])
                    self.fallback_models_[col] = knn

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        X = X.copy()
        if y is None and self.target_col in X.columns:
            y = X[self.target_col]

        for col in self.columns:
            if col not in X.columns:
                continue
            missing = X[col].isna()
            if not missing.any():
                continue

            if y is not None:
                # Training / evaluation: use class-conditional medians
                for cls in [0, 1]:
                    fill_mask = missing & (y == cls)
                    X.loc[fill_mask, col] = self.class_medians_[col][cls]
            elif col in self.fallback_models_:
                # Inference: use KNN fallback
                feat_cols = self.knn_fallback_features[col]
                pred_mask = missing & X[feat_cols].notna().all(axis=1)
                if pred_mask.any():
                    X.loc[pred_mask, col] = self.fallback_models_[col].predict(
                        X.loc[pred_mask, feat_cols]
                    )
            # Last resort: global class-0 median (majority)
            X[col] = X[col].fillna(self.class_medians_[col][0])

            n_filled = int(missing.sum())
            log.info("Imputed %d values in '%s'", n_filled, col)

        return X


# ========================================================================== #
#  IMP-B: Global Median Imputation                                           #
# ========================================================================== #

class GlobalMedianImputer(BaseImputer):
    """Naive baseline — single global median per column."""

    def __init__(self, columns: list[str]):
        super().__init__(columns)
        self.medians_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "GlobalMedianImputer":
        for col in self.columns:
            if col in X.columns:
                self.medians_[col] = float(X[col].median())
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns and col in self.medians_:
                n_miss = int(X[col].isna().sum())
                X[col] = X[col].fillna(self.medians_[col])
                if n_miss > 0:
                    log.info("Global median imputed %d values in '%s' (median=%.4f)", n_miss, col, self.medians_[col])
        return X


# ========================================================================== #
#  IMP-C: KNN Imputation                                                     #
# ========================================================================== #

class KNNImputerWrapper(BaseImputer):
    """sklearn KNNImputer — captures local structure via k nearest neighbors."""

    def __init__(self, columns: list[str], n_neighbors: int = 5):
        super().__init__(columns)
        self.n_neighbors = n_neighbors
        self._imputer: Optional[KNNImputer] = None
        self._all_numeric_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "KNNImputerWrapper":
        self._all_numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self._imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self._imputer.fit(X[self._all_numeric_cols])
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        X = X.copy()
        imputed = self._imputer.transform(X[self._all_numeric_cols])
        X[self._all_numeric_cols] = imputed
        return X


# ========================================================================== #
#  IMP-D: MICE (Iterative Imputer with BayesianRidge)                        #
# ========================================================================== #

class MICEImputer(BaseImputer):
    """Multiple Imputation by Chained Equations using BayesianRidge."""

    def __init__(self, columns: list[str], max_iter: int = 10, random_state: int = 42):
        super().__init__(columns)
        self.max_iter = max_iter
        self.random_state = random_state
        self._imputer: Optional[IterativeImputer] = None
        self._all_numeric_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MICEImputer":
        self._all_numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self._imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._imputer.fit(X[self._all_numeric_cols])
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        X = X.copy()
        imputed = self._imputer.transform(X[self._all_numeric_cols])
        X[self._all_numeric_cols] = imputed
        return X


# ========================================================================== #
#  IMP-E: MissForest (Iterative Imputer with RandomForest)                   #
# ========================================================================== #

class MissForestImputer(BaseImputer):
    """Iterative imputation using RandomForest estimator — handles non-linearity."""

    def __init__(self, columns: list[str], max_iter: int = 10, random_state: int = 42):
        super().__init__(columns)
        self.max_iter = max_iter
        self.random_state = random_state
        self._imputer: Optional[IterativeImputer] = None
        self._all_numeric_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MissForestImputer":
        self._all_numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self._imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._imputer.fit(X[self._all_numeric_cols])
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        X = X.copy()
        imputed = self._imputer.transform(X[self._all_numeric_cols])
        X[self._all_numeric_cols] = imputed
        return X


# ========================================================================== #
#  Factory                                                                    #
# ========================================================================== #

class ImputerFactory:
    """Create an imputer by strategy name."""

    _REGISTRY = {
        "class_conditional_median": ClassConditionalMedianImputer,
        "global_median": GlobalMedianImputer,
        "knn": KNNImputerWrapper,
        "mice": MICEImputer,
        "missforest": MissForestImputer,
    }

    @classmethod
    def create(cls, strategy: str, config: dict) -> BaseImputer:
        columns = config.get("columns_to_impute", ["rh_max", "temp_recovery_rate"])
        if strategy not in cls._REGISTRY:
            raise ValueError(f"Unknown imputation strategy: '{strategy}'. Options: {list(cls._REGISTRY.keys())}")

        if strategy == "class_conditional_median":
            return ClassConditionalMedianImputer(
                columns=columns,
                target_col=config.get("target_col", "silent_failure"),
                knn_fallback_features=config.get("knn_fallback_features"),
                knn_k=config.get("knn_neighbors", 5),
            )
        elif strategy == "knn":
            return KNNImputerWrapper(columns=columns, n_neighbors=config.get("knn_neighbors", 5))
        elif strategy == "mice":
            return MICEImputer(
                columns=columns,
                max_iter=config.get("mice_max_iter", 10),
                random_state=config.get("random_state", 42),
            )
        elif strategy == "missforest":
            return MissForestImputer(
                columns=columns,
                max_iter=config.get("mice_max_iter", 10),
                random_state=config.get("random_state", 42),
            )
        elif strategy == "global_median":
            return GlobalMedianImputer(columns=columns)
        else:
            klass = cls._REGISTRY[strategy]
            return klass(columns=columns)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._REGISTRY.keys())
