import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from backend.engine.ingestion.service import IngestionService, DataValidator, DataIngestionError


def test_load_csv_valid():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8") as f:
        f.write("a,b,c\n1,2,3\n4,5,6\n")
        temp_path = f.name

    try:
        df = IngestionService.load_dataset(temp_path)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b", "c"]
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_load_csv_semicolon_delimited():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8") as f:
        f.write("col1;col2;col3\n10;20;hello\n30;40;world\n")
        temp_path = f.name

    try:
        df = IngestionService.load_dataset(temp_path)
        assert len(df) == 2
        assert len(df.columns) == 3
        assert "col1" in df.columns
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_validation_drops_missing_target_and_duplicates():
    df = pd.DataFrame({
        "feature1": [1, 2, 2, 3, 4],
        "feature2": ["a", "b", "b", "c", "d"],
        "target": [10, 20, 20, np.nan, 40],
    })

    clean_df, result = DataValidator.validate_and_clean(df, target_column="target")
    assert result.is_valid
    # Row 3 (NaN target) dropped, Row 2 (duplicate of row 1) dropped
    assert len(clean_df) == 3
    assert len(result.exclusions) >= 2


def test_validation_drops_constant_and_empty_columns():
    df = pd.DataFrame({
        "good_feature": [1, 2, 3, 4],
        "constant_feature": [99, 99, 99, 99],
        "all_null": [np.nan, np.nan, np.nan, np.nan],
        "target": [0, 1, 0, 1],
    })

    clean_df, result = DataValidator.validate_and_clean(df, target_column="target")
    assert result.is_valid
    assert "constant_feature" not in clean_df.columns
    assert "all_null" not in clean_df.columns
    assert "good_feature" in clean_df.columns
    assert len(result.exclusions) >= 2


def test_validation_missing_target_column():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    _, result = DataValidator.validate_and_clean(df, target_column="nonexistent")
    assert not result.is_valid
    assert any("does not exist" in err for err in result.errors)
