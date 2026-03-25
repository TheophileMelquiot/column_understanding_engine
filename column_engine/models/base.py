"""Base model interface for column semantic classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class BaseColumnClassifier(ABC):
    """Abstract base class for column-type classifiers."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on feature matrix *X* and labels *y*."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class-probability estimates (shape ``[n_samples, n_classes]``)."""

    def classify_columns(
        self,
        feature_dicts: List[Dict[str, Any]],
        label_names: List[str],
    ) -> List[Dict[str, Any]]:
        """High-level API: given a list of feature dicts, return predictions.

        Parameters
        ----------
        feature_dicts : list[dict]
            One dict per column (output of :func:`extract_column_features`).
        label_names : list[str]
            Ordered list of class names.

        Returns
        -------
        list[dict]
            One dict per column with keys ``predicted_label``, ``confidence``,
            and ``type``.
        """
        X = self._dicts_to_matrix(feature_dicts)
        preds = self.predict(X)
        probas = self.predict_proba(X)
        results: List[Dict[str, Any]] = []
        for i, fd in enumerate(feature_dicts):
            pred_idx = int(preds[i])
            results.append({
                "name": fd.get("header_text", f"col_{i}"),
                "predicted_label": label_names[pred_idx],
                "type": _infer_type(fd),
                "confidence": float(np.max(probas[i])),
            })
        return results

    @staticmethod
    def _dicts_to_matrix(dicts: List[Dict[str, Any]]) -> np.ndarray:
        """Convert feature dicts to a numeric matrix.

        Keeps only scalar (int / float) features.
        """
        if not dicts:
            return np.empty((0, 0))
        keys = [k for k, v in dicts[0].items() if isinstance(v, (int, float))]
        rows = [[d[k] for k in keys] for d in dicts]
        return np.array(rows, dtype=np.float32)


def _infer_type(features: Dict[str, Any]) -> str:
    """Heuristic type inference from feature dict."""
    if features.get("is_email", 0) > 0.5:
        return "email"
    if features.get("is_date", 0) > 0.5:
        return "date"
    if features.get("is_uuid", 0) > 0.5:
        return "identifier"
    if features.get("is_phone", 0) > 0.5:
        return "phone"
    if features.get("is_price", 0) > 0.5:
        return "currency"
    if features.get("null_ratio", 1) < 0.5 and features.get("std", 0) > 0:
        return "numeric"
    return "categorical"
