"""Tests for Step 4 — Feature Engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from column_engine.feature_engineering import (
    compute_statistical_features,
    compute_pattern_features,
    compute_textual_features,
    extract_column_features,
    extract_all_features,
)


class TestStatisticalFeatures:
    def test_numeric_column(self):
        col = pd.Series([10, 20, 30, 40, 50])
        features = compute_statistical_features(col)
        assert features["mean"] == pytest.approx(30.0)
        assert features["n_unique"] == 5
        assert features["null_ratio"] == 0.0

    def test_with_nulls(self):
        col = pd.Series([1, 2, None, None, 5])
        features = compute_statistical_features(col)
        assert features["null_ratio"] == pytest.approx(0.4)

    def test_entropy_uniform(self):
        col = pd.Series(["a", "b", "c", "d"])
        features = compute_statistical_features(col)
        assert features["entropy"] == pytest.approx(2.0)


class TestPatternFeatures:
    def test_email_detection(self):
        col = pd.Series(["alice@example.com", "bob@test.org", "charlie@mail.net"])
        features = compute_pattern_features(col)
        assert features["is_email"] == pytest.approx(1.0)

    def test_date_detection(self):
        col = pd.Series(["2024-01-15", "2024-02-20", "2024-03-10"])
        features = compute_pattern_features(col)
        assert features["is_date"] == pytest.approx(1.0)

    def test_uuid_detection(self):
        col = pd.Series([
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        ])
        features = compute_pattern_features(col)
        assert features["is_uuid"] == pytest.approx(1.0)

    def test_non_matching(self):
        col = pd.Series(["hello", "world", "foo"])
        features = compute_pattern_features(col)
        assert features["is_email"] == 0.0
        assert features["is_date"] == 0.0


class TestTextualFeatures:
    def test_zero_embeddings_default(self):
        col = pd.Series(["a", "b", "c"])
        features = compute_textual_features(col, "test_header")
        assert features["header_text"] == "test_header"
        assert features["header_embedding"].shape == (768,)
        assert np.all(features["header_embedding"] == 0)

    def test_custom_embedding_fn(self):
        col = pd.Series(["hello", "world"])
        mock_emb = np.ones(768, dtype=np.float32)
        features = compute_textual_features(
            col, "test_header", embedding_fn=lambda _: mock_emb
        )
        assert np.all(features["header_embedding"] == 1.0)
        assert np.all(features["values_embedding"] == 1.0)


class TestExtractAllFeatures:
    def test_basic(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        features = extract_all_features(df)
        assert len(features) == 2
        assert features[0]["header_text"] == "A"
        assert features[1]["header_text"] == "B"
