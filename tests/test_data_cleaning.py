import sys
import os
import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from src.data_cleaning import clean_data

# Ensure root path is added for local module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_clean_data_raises_on_missing_columns():
    """Ensure clean_data raises KeyError when expected columns are missing."""
    df = pd.DataFrame({"some_column": ["value"]})
    with pytest.raises(KeyError):
        clean_data(df.copy())


def test_clean_data_handles_empty_dataframe():
    """Verify that an empty DataFrame is returned unchanged."""
    df = pd.DataFrame(
        columns=["creation_time", "end_time", "time", "src_ip_country_code"]
    )
    cleaned = clean_data(df.copy())
    assert cleaned.empty, "Empty DataFrame should return empty after cleaning"


def test_clean_data_transforms_datetime_and_uppercase():
    """Ensure datetime columns are parsed and country codes uppercased."""
    df = pd.DataFrame(
        {
            "creation_time": ["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"],
            "end_time": ["2024-01-01T00:10:00Z", "2024-01-01T00:15:00Z"],
            "time": ["2024-01-01T00:20:00Z", "2024-01-01T00:25:00Z"],
            "src_ip_country_code": ["us", "ae"],
        }
    )
    cleaned = clean_data(df.copy())

# sourcery skip: no-loop-in-tests
    for col in ["creation_time", "end_time", "time"]:
        assert (
            is_datetime64_any_dtype(cleaned[col]) or is_datetime64tz_dtype(cleaned[col])
        ), f"Column {col} is not of datetime type"

    assert cleaned["src_ip_country_code"].str.isupper().all(), "Country codes are not uppercased"


def test_clean_data_removes_nulls_and_duplicates():
    """Ensure null values and duplicate rows are removed."""
    df = pd.DataFrame(
        {
            "creation_time": ["2024-01-01T00:00:00Z", None],
            "end_time": ["2024-01-01T00:10:00Z", "2024-01-01T00:15:00Z"],
            "time": ["2024-01-01T00:20:00Z", "2024-01-01T00:25:00Z"],
            "src_ip_country_code": ["us", "us"],
        }
    )
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # Add duplicate

    cleaned = clean_data(df.copy())

    assert not cleaned.isnull().values.any(), "Null values were not removed"
    assert not cleaned.duplicated().any(), "Duplicate rows were not removed"