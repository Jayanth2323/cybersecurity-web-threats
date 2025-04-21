import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import pandas as pd
from src.data_cleaning import clean_data


def test_clean_data():
    df = pd.DataFrame(
        {
            "creation_time": ["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"],
            "end_time": ["2024-01-01T00:10:00Z", "2024-01-01T00:15:00Z"],
            "time": ["2024-01-01T00:20:00Z", "2024-01-01T00:25:00Z"],
            "src_ip_country_code": ["us", "ae"],
        }
    )

    cleaned = clean_data(df.copy())
    assert pd.api.types.is_datetime64_any_dtype(cleaned["creation_time"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["end_time"])
    assert cleaned["src_ip_country_code"].str.isupper().all()
