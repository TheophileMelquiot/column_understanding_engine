"""Step 3 — Data Cleaning.

For each detected table: identify header rows, remove empty/noisy rows,
normalize values, and produce a clean pandas DataFrame.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openpyxl.worksheet.worksheet import Worksheet

from column_engine.header_reconstruction import (
    _get_merged_cell_map,
    _cell_value,
    detect_header_rows,
    reconstruct_headers,
)

Region = Tuple[int, int, int, int]


def _extract_raw_data(
    ws: Worksheet,
    region: Region,
    merged_map: Dict[Tuple[int, int], Any],
    skip_rows: int = 0,
) -> List[List[Any]]:
    """Extract raw cell values from a region, skipping the first *skip_rows* rows."""
    min_r, min_c, max_r, max_c = region
    data: List[List[Any]] = []
    for r in range(min_r + skip_rows, max_r + 1):
        row_data: List[Any] = []
        for c in range(min_c, max_c + 1):
            val = _cell_value(ws, r + 1, c + 1, merged_map)  # 1-based
            row_data.append(val)
        data.append(row_data)
    return data


def _normalize_value(val: Any) -> Any:
    """Normalize a single cell value.

    - Strips whitespace from strings.
    - Attempts numeric conversion for string representations of numbers.
    """
    if val is None:
        return np.nan
    if isinstance(val, str):
        val = val.strip()
        if val == "":
            return np.nan
        # Try numeric conversion
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
    return val


def _is_noisy_row(row: List[Any], threshold: float = 0.8) -> bool:
    """Return True if the row has more than *threshold* fraction of NaN/None values."""
    n_empty = sum(1 for v in row if v is None or (isinstance(v, float) and np.isnan(v)))
    return n_empty / max(len(row), 1) > threshold


def clean_table(
    ws: Worksheet,
    region: Region,
    n_header_rows: Optional[int] = None,
    noise_threshold: float = 0.8,
) -> pd.DataFrame:
    """Extract, clean, and return a DataFrame for one table region.

    Parameters
    ----------
    ws : Worksheet
        The openpyxl worksheet.
    region : Region
        0-indexed region ``(min_row, min_col, max_row, max_col)``.
    n_header_rows : int, optional
        Number of header rows; auto-detected if *None*.
    noise_threshold : float
        Fraction of empty cells above which a row is considered noise.

    Returns
    -------
    pd.DataFrame
    """
    merged_map = _get_merged_cell_map(ws)

    if n_header_rows is None:
        n_header_rows = detect_header_rows(ws, region, merged_map)
    if n_header_rows == 0:
        n_header_rows = 1

    headers = reconstruct_headers(ws, region, n_header_rows=n_header_rows)
    raw_data = _extract_raw_data(ws, region, merged_map, skip_rows=n_header_rows)

    # Normalize values
    normalized = [[_normalize_value(v) for v in row] for row in raw_data]

    # Filter noisy rows
    cleaned = [row for row in normalized if not _is_noisy_row(row, noise_threshold)]

    if not cleaned:
        return pd.DataFrame(columns=headers)

    df = pd.DataFrame(cleaned, columns=headers)

    # Drop fully-empty columns
    df.dropna(axis=1, how="all", inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df


def clean_tables_from_sheet(
    ws: Worksheet,
    regions: List[Region],
    noise_threshold: float = 0.8,
) -> Dict[str, pd.DataFrame]:
    """Clean all detected tables in a sheet.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are ``"table_1"``, ``"table_2"``, etc.
    """
    tables: Dict[str, pd.DataFrame] = {}
    for idx, region in enumerate(regions, start=1):
        tables[f"table_{idx}"] = clean_table(ws, region, noise_threshold=noise_threshold)
    return tables
