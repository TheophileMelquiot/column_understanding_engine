"""Step 6 — Evaluation.

Provides metrics computation, model benchmarking, and ablation study
utilities for column semantic classification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from column_engine.models.base import BaseColumnClassifier


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like, optional
        Predicted class probabilities (required for ROC-AUC).

    Returns
    -------
    dict[str, float]
        Dictionary with ``accuracy``, ``f1_macro``, and optionally
        ``roc_auc_ovr``.
    """
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except ValueError:
            pass
    return metrics


def benchmark_models(
    models: Dict[str, BaseColumnClassifier],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate multiple models.

    Parameters
    ----------
    models : dict[str, BaseColumnClassifier]
        Named model instances.
    X_train, y_train : arrays
        Training data.
    X_test, y_test : arrays
        Test data.

    Returns
    -------
    dict[str, dict[str, float]]
        ``model_name -> metrics_dict``.
    """
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        results[name] = compute_metrics(y_test, y_pred, y_proba)
    return results


def ablation_study(
    model: BaseColumnClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_groups: Dict[str, List[int]],
) -> Dict[str, Dict[str, float]]:
    """Run an ablation study by removing one feature group at a time.

    Parameters
    ----------
    model : BaseColumnClassifier
        Model to evaluate (a new fit is done each time).
    X_train, y_train, X_test, y_test : arrays
        Training and test data.
    feature_groups : dict[str, list[int]]
        Mapping ``group_name -> list_of_column_indices`` in *X*.

    Returns
    -------
    dict[str, dict[str, float]]
        ``"without_{group}" -> metrics_dict`` for each group, plus a
        ``"full"`` entry with all features.
    """
    results: Dict[str, Dict[str, float]] = {}

    # Full model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    results["full"] = compute_metrics(y_test, y_pred, y_proba)

    # Ablation
    for group_name, indices in feature_groups.items():
        mask = np.ones(X_train.shape[1], dtype=bool)
        mask[indices] = False
        X_tr_abl = X_train[:, mask]
        X_te_abl = X_test[:, mask]
        model.fit(X_tr_abl, y_train)
        y_pred_abl = model.predict(X_te_abl)
        y_proba_abl = model.predict_proba(X_te_abl)
        results[f"without_{group_name}"] = compute_metrics(
            y_test, y_pred_abl, y_proba_abl
        )

    return results
