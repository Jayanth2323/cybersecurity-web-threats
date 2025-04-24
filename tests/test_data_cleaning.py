import sys
import os
import pandas as pd
from src.data_cleaning import clean_data

# Ensure proper path resolution for module imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_clean_data_transforms_datetime_and_uppercase():
    """Test datetime conversion and uppercasing of country codes."""
    df = pd.DataFrame({
        "creation_time": ["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"],
        "end_time": ["2024-01-01T00:10:00Z", "2024-01-01T00:15:00Z"],
        "time": ["2024-01-01T00:20:00Z", "2024-01-01T00:25:00Z"],
        "src_ip_country_code": ["us", "ae"]
    })

    cleaned = clean_data(df.copy())

    # Assert datetime conversion
    assert pd.api.types.is_datetime64_any_dtype(cleaned["creation_time"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["end_time"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["time"])

    # Assert country codes are uppercased
    assert cleaned["src_ip_country_code"].str.isupper().all()


def test_clean_data_removes_nulls_and_duplicates():
    """Test that nulls and duplicates are removed."""
    df = pd.DataFrame({
        "creation_time": ["2024-01-01T00:00:00Z", None],
        "end_time": ["2024-01-01T00:10:00Z", "2024-01-01T00:15:00Z"],
        "time": ["2024-01-01T00:20:00Z", "2024-01-01T00:25:00Z"],
        "src_ip_country_code": ["us", "us"]
    })

    # Duplicate the first row intentionally
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    cleaned = clean_data(df.copy())

    # Assert no null values
    assert not cleaned.isnull().values.any()

    # Assert no duplicate rows
    assert not cleaned.duplicated().any()
