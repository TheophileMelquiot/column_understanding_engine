"""XGBoost strong-baseline for column semantic classification."""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from column_engine.models.base import BaseColumnClassifier


class XGBoostColumnClassifier(BaseColumnClassifier):
    """XGBoost-based column classifier (strong baseline)."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        use_label_encoder: bool = False,
    ) -> None:
        self._model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="mlogloss",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
