"""Logistic Regression baseline for column semantic classification."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from column_engine.models.base import BaseColumnClassifier


class LogisticColumnClassifier(BaseColumnClassifier):
    """Logistic Regression baseline.

    Uses :class:`sklearn.linear_model.LogisticRegression` with standard
    scaling.
    """

    def __init__(self, max_iter: int = 1000, C: float = 1.0) -> None:
        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            solver="lbfgs",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)
