"""Step 4 — Feature Engineering.

Extracts three families of features from each column of a cleaned DataFrame:

* **Statistical features**: mean, std, n_unique, entropy, null_ratio
* **Pattern features**: regex-based detectors (email, phone, price, date, uuid)
* **Textual features**: header and sampled-value embeddings via a pre-trained
  transformer (DistilBERT by default).

The public API (:func:`extract_column_features`) returns a flat feature
dictionary; :func:`extract_all_features` processes every column of a
DataFrame.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
_PHONE_RE = re.compile(
    r"^[\+]?[\d\s\-\.\(\)]{7,20}$"
)
_PRICE_RE = re.compile(
    r"^[\$€£¥₹]?\s?\d[\d\s,]*\.?\d*\s?[\$€£¥₹]?$"
)
_DATE_RE = re.compile(
    r"^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}$"
)
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# ---------------------------------------------------------------------------
# Statistical features
# ---------------------------------------------------------------------------


def _entropy(series: pd.Series) -> float:
    """Shannon entropy of value counts (base-2)."""
    counts = series.dropna().value_counts(normalize=True)
    if counts.empty:
        return 0.0
    return float(-sum(p * math.log2(p) for p in counts if p > 0))


def compute_statistical_features(col: pd.Series) -> Dict[str, float]:
    """Return a dictionary of statistical features for *col*."""
    numeric = pd.to_numeric(col, errors="coerce")
    n_total = len(col)
    return {
        "mean": float(numeric.mean()) if not numeric.isna().all() else 0.0,
        "std": float(numeric.std()) if not numeric.isna().all() else 0.0,
        "n_unique": int(col.nunique()),
        "entropy": _entropy(col),
        "null_ratio": float(col.isna().sum() / max(n_total, 1)),
    }


# ---------------------------------------------------------------------------
# Pattern features
# ---------------------------------------------------------------------------


def _match_ratio(col: pd.Series, pattern: re.Pattern[str]) -> float:
    """Fraction of non-null string values matching *pattern*."""
    str_vals = col.dropna().astype(str)
    if str_vals.empty:
        return 0.0
    matches = str_vals.apply(lambda v: bool(pattern.match(v)))
    return float(matches.mean())


def compute_pattern_features(col: pd.Series) -> Dict[str, float]:
    """Return regex-based pattern feature dictionary."""
    return {
        "is_email": _match_ratio(col, _EMAIL_RE),
        "is_phone": _match_ratio(col, _PHONE_RE),
        "is_price": _match_ratio(col, _PRICE_RE),
        "is_date": _match_ratio(col, _DATE_RE),
        "is_uuid": _match_ratio(col, _UUID_RE),
    }


# ---------------------------------------------------------------------------
# Textual features (lightweight — embedding-ready)
# ---------------------------------------------------------------------------


def _sample_values(col: pd.Series, n: int = 5) -> str:
    """Return a string of up to *n* sampled non-null values."""
    non_null = col.dropna().astype(str)
    if non_null.empty:
        return ""
    sampled = non_null.sample(min(n, len(non_null)), random_state=42)
    return " ".join(sampled.tolist())


def compute_textual_features(
    col: pd.Series,
    header: str,
    embedding_fn: Optional[Any] = None,
    embedding_dim: int = 768,
) -> Dict[str, Any]:
    """Return textual features.

    If *embedding_fn* is provided it should accept a ``str`` and return a
    1-D numpy array of size *embedding_dim*.  Otherwise a zero-vector
    placeholder is used (embeddings are computed later in the pipeline).
    """
    sample_text = _sample_values(col)
    if embedding_fn is not None:
        header_emb = embedding_fn(header)
        values_emb = embedding_fn(sample_text)
    else:
        header_emb = np.zeros(embedding_dim, dtype=np.float32)
        values_emb = np.zeros(embedding_dim, dtype=np.float32)
    return {
        "header_text": header,
        "sample_text": sample_text,
        "header_embedding": header_emb,
        "values_embedding": values_emb,
    }


# ---------------------------------------------------------------------------
# Combined feature extraction
# ---------------------------------------------------------------------------


def extract_column_features(
    col: pd.Series,
    header: str,
    embedding_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Extract all features for a single column.

    Returns a dictionary containing statistical, pattern, and textual
    features.
    """
    features: Dict[str, Any] = {}
    features.update(compute_statistical_features(col))
    features.update(compute_pattern_features(col))
    features.update(compute_textual_features(col, header, embedding_fn))
    return features


def extract_all_features(
    df: pd.DataFrame,
    embedding_fn: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Extract features for every column in *df*.

    Returns a list of feature dictionaries, one per column.
    """
    all_features: List[Dict[str, Any]] = []
    for col_name in df.columns:
        features = extract_column_features(df[col_name], str(col_name), embedding_fn)
        all_features.append(features)
    return all_features
