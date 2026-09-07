"""
SentinelCold — Tier 2: Multi-Strategy Class Balancer
Four interchangeable balancing strategies behind a factory.

Applied ONLY to the training fold — validation/test folds always
retain the true 19.68 % prevalence.

Strategies
----------
BAL-A  smote        Standard Synthetic Minority Oversampling
BAL-B  adasyn       Adaptive Synthetic — more samples near decision boundary
BAL-C  class_weight No resampling; returns class weights for model-native handling
BAL-D  smote_tomek  SMOTE + Tomek-link cleaning of noisy boundary samples
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from utils.logger import tier_logger

log = tier_logger(2)


# ========================================================================== #
#  Base class                                                                 #
# ========================================================================== #

class BaseBalancer(ABC):
    """Common interface for all balancing strategies."""

    def __init__(self, target_ratio: float = 0.5, random_state: int = 42):
        """
        Parameters
        ----------
        target_ratio : float
            Desired minority / majority ratio after resampling.
            0.5 → 1:2,  1.0 → 1:1.
        """
        self.target_ratio = target_ratio
        self.random_state = random_state

    @abstractmethod
    def resample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Return resampled (X, y). Must NOT modify the original arrays."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ========================================================================== #
#  BAL-A: SMOTE                                                               #
# ========================================================================== #

class SMOTEBalancer(BaseBalancer):
    """Standard SMOTE oversampling."""

    def __init__(self, target_ratio: float = 0.5, k_neighbors: int = 5, random_state: int = 42):
        super().__init__(target_ratio, random_state)
        self.k_neighbors = k_neighbors

    def resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        from imblearn.over_sampling import SMOTE

        sm = SMOTE(
            sampling_strategy=self.target_ratio,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )
        X_res, y_res = sm.fit_resample(X, y)
        n_synthetic = len(X_res) - len(X)
        log.info(
            "SMOTE: generated %d synthetic minority samples (ratio=%.2f)",
            n_synthetic, self.target_ratio,
        )
        self._quality_check(X, X_res, y, y_res)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)

    def _quality_check(self, X_orig, X_res, y_orig, y_res):
        """KS-test: compare synthetic minority distribution vs. real minority."""
        minority_orig = X_orig[y_orig == 1]
        minority_new = pd.DataFrame(X_res, columns=X_orig.columns)[
            pd.Series(y_res) == 1
        ].iloc[len(minority_orig):]  # only the new synthetics

        if len(minority_new) == 0:
            return

        for col in X_orig.select_dtypes(include=[np.number]).columns[:5]:
            stat, p = sp_stats.ks_2samp(
                minority_orig[col].dropna(), minority_new[col].dropna()
            )
            if p < 0.05:
                log.warning(
                    "KS-test WARNING for '%s': stat=%.4f, p=%.4f (synthetic may diverge)",
                    col, stat, p,
                )


# ========================================================================== #
#  BAL-B: ADASYN                                                              #
# ========================================================================== #

class ADASYNBalancer(BaseBalancer):
    """Adaptive synthetic oversampling — focuses on harder-to-learn boundary samples."""

    def __init__(self, target_ratio: float = 0.5, random_state: int = 42):
        super().__init__(target_ratio, random_state)

    def resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        from imblearn.over_sampling import ADASYN

        ada = ADASYN(
            sampling_strategy=self.target_ratio,
            random_state=self.random_state,
        )
        try:
            X_res, y_res = ada.fit_resample(X, y)
            n_synthetic = len(X_res) - len(X)
            log.info("ADASYN: generated %d synthetic samples (ratio=%.2f)", n_synthetic, self.target_ratio)
        except ValueError as e:
            # ADASYN can fail if not enough neighbors; fall back to SMOTE
            log.warning("ADASYN failed (%s) — falling back to SMOTE", e)
            fallback = SMOTEBalancer(self.target_ratio, random_state=self.random_state)
            return fallback.resample(X, y)

        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


# ========================================================================== #
#  BAL-C: Class Weight (no resampling)                                        #
# ========================================================================== #

class ClassWeightBalancer(BaseBalancer):
    """
    No resampling — returns original data unchanged.
    The model is expected to use class_weight / scale_pos_weight natively.
    The computed weights are stored in `self.class_weights_` for downstream use.
    """

    def __init__(self, target_ratio: float = 0.5, random_state: int = 42):
        super().__init__(target_ratio, random_state)
        self.class_weights_: dict[int, float] = {}
        self.scale_pos_weight_: float = 1.0

    def resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        self.scale_pos_weight_ = n_neg / max(n_pos, 1)
        self.class_weights_ = {0: 1.0, 1: self.scale_pos_weight_}
        log.info(
            "ClassWeight: scale_pos_weight=%.2f (no resampling, %d pos / %d neg)",
            self.scale_pos_weight_, n_pos, n_neg,
        )
        return X.copy(), y.copy()


# ========================================================================== #
#  BAL-D: SMOTE + Tomek Links                                                #
# ========================================================================== #

class SMOTETomekBalancer(BaseBalancer):
    """SMOTE oversampling followed by Tomek-link undersampling to clean boundary."""

    def __init__(self, target_ratio: float = 0.5, random_state: int = 42):
        super().__init__(target_ratio, random_state)

    def resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        from imblearn.combine import SMOTETomek

        smt = SMOTETomek(
            sampling_strategy=self.target_ratio,
            random_state=self.random_state,
        )
        X_res, y_res = smt.fit_resample(X, y)
        log.info(
            "SMOTETomek: %d → %d samples (ratio=%.2f)",
            len(X), len(X_res), self.target_ratio,
        )
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


# ========================================================================== #
#  Factory                                                                    #
# ========================================================================== #

class BalancerFactory:
    """Create a balancer by strategy name."""

    _REGISTRY = {
        "smote": SMOTEBalancer,
        "adasyn": ADASYNBalancer,
        "class_weight": ClassWeightBalancer,
        "smote_tomek": SMOTETomekBalancer,
    }

    @classmethod
    def create(
        cls,
        strategy: str,
        target_ratio: float = 0.5,
        random_state: int = 42,
        **kwargs,
    ) -> BaseBalancer:
        if strategy not in cls._REGISTRY:
            raise ValueError(
                f"Unknown balancing strategy: '{strategy}'. Options: {list(cls._REGISTRY.keys())}"
            )
        klass = cls._REGISTRY[strategy]
        return klass(target_ratio=target_ratio, random_state=random_state, **kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._REGISTRY.keys())
