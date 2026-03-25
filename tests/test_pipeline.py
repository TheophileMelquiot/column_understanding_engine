"""Tests for the end-to-end pipeline."""

from __future__ import annotations

import pytest

from column_engine.pipeline import process_excel
from tests.conftest import create_simple_workbook, create_multi_table_workbook


@pytest.fixture()
def simple_xlsx(tmp_path):
    return create_simple_workbook(str(tmp_path / "simple.xlsx"))


@pytest.fixture()
def multi_xlsx(tmp_path):
    return create_multi_table_workbook(str(tmp_path / "multi.xlsx"))


class TestPipeline:
    def test_simple_pipeline(self, simple_xlsx):
        result = process_excel(simple_xlsx)
        assert "Sheet1" in result
        tables = result["Sheet1"]
        assert len(tables) >= 1
        first_table = list(tables.values())[0]
        assert "dataframe" in first_table
        assert "columns" in first_table
        assert len(first_table["columns"]) > 0

    def test_multi_table_pipeline(self, multi_xlsx):
        result = process_excel(multi_xlsx)
        tables = result["Sheet1"]
        assert len(tables) == 2

    def test_column_info_structure(self, simple_xlsx):
        result = process_excel(simple_xlsx)
        tables = result["Sheet1"]
        first_table = list(tables.values())[0]
        for col_info in first_table["columns"]:
            assert "name" in col_info
            assert "type" in col_info
