"""End-to-end pipeline for the Column Understanding Engine.

Connects table detection → header reconstruction → data cleaning →
feature engineering → (optional) classification into a single callable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import openpyxl
import pandas as pd

from column_engine.table_detection import detect_tables
from column_engine.data_cleaning import clean_tables_from_sheet
from column_engine.feature_engineering import extract_all_features
from column_engine.models.base import BaseColumnClassifier


def process_excel(
    filepath: str,
    sheet_name: Optional[str] = None,
    min_cells: int = 4,
    noise_threshold: float = 0.8,
    embedding_fn: Optional[Any] = None,
    classifier: Optional[BaseColumnClassifier] = None,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the full Column Understanding Engine pipeline on an Excel file.

    Parameters
    ----------
    filepath : str
        Path to a ``.xlsx`` file.
    sheet_name : str, optional
        Process only this sheet; all sheets if *None*.
    min_cells : int
        Minimum cell count for a connected component to be kept.
    noise_threshold : float
        Fraction of empty cells above which a row is removed.
    embedding_fn : callable, optional
        ``str -> np.ndarray`` function for text embeddings.
    classifier : BaseColumnClassifier, optional
        Trained classifier for column labelling.
    label_names : list[str], optional
        Ordered list of class names (required when *classifier* is given).

    Returns
    -------
    dict
        ``{ sheet_name: { table_key: { "dataframe": DataFrame,
        "columns": [...] } } }``
    """
    wb = openpyxl.load_workbook(filepath, data_only=True)
    sheets = [sheet_name] if sheet_name else wb.sheetnames
    output: Dict[str, Any] = {}

    for sname in sheets:
        ws = wb[sname]
        regions = detect_tables(ws, min_cells=min_cells)
        tables = clean_tables_from_sheet(ws, regions, noise_threshold=noise_threshold)

        sheet_output: Dict[str, Any] = {}
        for tkey, df in tables.items():
            features = extract_all_features(df, embedding_fn=embedding_fn)
            col_info: List[Dict[str, Any]] = []
            if classifier is not None and label_names is not None:
                col_info = classifier.classify_columns(features, label_names)
            else:
                # Without a classifier, return extracted features only
                for feat in features:
                    col_info.append({
                        "name": feat.get("header_text", ""),
                        "type": _infer_type_from_features(feat),
                        "features": {
                            k: v
                            for k, v in feat.items()
                            if isinstance(v, (int, float, str))
                        },
                    })
            sheet_output[tkey] = {
                "dataframe": df,
                "columns": col_info,
            }
        output[sname] = sheet_output

    wb.close()
    return output


def _infer_type_from_features(features: Dict[str, Any]) -> str:
    """Simple heuristic type inference."""
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
