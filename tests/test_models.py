"""Tests for Step 5 — Models and Step 6 — Evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from column_engine.models.base import BaseColumnClassifier
from column_engine.models.logistic import LogisticColumnClassifier
from column_engine.models.xgboost_model import XGBoostColumnClassifier
from column_engine.models.deep_model import DeepColumnClassifier
from column_engine.evaluation import compute_metrics, benchmark_models


def _make_dataset(n_samples=100, n_features=10, n_classes=3, seed=42):
    """Generate a synthetic classification dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.randint(0, n_classes, size=n_samples)
    return X, y


class TestLogisticModel:
    def test_fit_predict(self):
        X, y = _make_dataset()
        model = LogisticColumnClassifier()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
        proba = model.predict_proba(X)
        assert proba.shape == (100, 3)


class TestXGBoostModel:
    def test_fit_predict(self):
        X, y = _make_dataset()
        model = XGBoostColumnClassifier(n_estimators=10)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)


class TestDeepModel:
    def test_fit_predict(self):
        embedding_dim = 8
        n_tabular = 5
        n_features = embedding_dim + n_tabular
        X, y = _make_dataset(n_features=n_features, n_samples=50)
        model = DeepColumnClassifier(
            embedding_dim=embedding_dim,
            n_classes=3,
            epochs=5,
            hidden_dim=16,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)
        proba = model.predict_proba(X)
        assert proba.shape == (50, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestMetrics:
    def test_compute_metrics(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 2])
        metrics = compute_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert metrics["accuracy"] == pytest.approx(0.8)

    def test_benchmark(self):
        X, y = _make_dataset(n_samples=60)
        X_train, y_train = X[:40], y[:40]
        X_test, y_test = X[40:], y[40:]
        models = {
            "logistic": LogisticColumnClassifier(),
        }
        results = benchmark_models(models, X_train, y_train, X_test, y_test)
        assert "logistic" in results
        assert "accuracy" in results["logistic"]
