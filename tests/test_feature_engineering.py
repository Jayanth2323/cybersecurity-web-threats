import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import pandas as pd
from datetime import datetime
from src.feature_engineering import add_features


def test_add_features():
    df = pd.DataFrame(
        {
            "creation_time": [pd.Timestamp("2024-01-01 00:00:00")],
            "end_time": [pd.Timestamp("2024-01-01 00:01:00")],
            "bytes_in": [1000],
            "bytes_out": [2000],
        }
    )

    features = add_features(df.copy())
    assert "duration_seconds" in features.columns
    assert "avg_packet_size" in features.columns
    assert features["duration_seconds"].iloc[0] == 60
