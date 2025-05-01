import sys
import os
import pandas as pd
from pytest import approx
from src.feature_engineering import add_features

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


def test_add_features_basic_case():
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
    assert features["avg_packet_size"].iloc[0] == approx(50.0)


def test_add_features_zero_duration():
    df = pd.DataFrame(
        {
            "creation_time": [pd.Timestamp("2024-01-01 00:00:00")],
            "end_time": [pd.Timestamp("2024-01-01 00:00:00")],
            "bytes_in": [500],
            "bytes_out": [1500],
        }
    )

    features = add_features(df.copy())

    assert features["duration_seconds"].iloc[0] == 0
    assert features["avg_packet_size"].iloc[0] == 0


def test_add_features_missing_columns():
    df = pd.DataFrame(
        {
            "creation_time": [pd.Timestamp("2024-01-01 00:00:00")],
            "end_time": [pd.Timestamp("2024-01-01 00:01:00")],
            # bytes_in and bytes_out are missing
        }
    )

    try:
        add_features(df.copy())
    except KeyError as e:
        assert "bytes_in" in str(e) or "bytes_out" in str(e)
    else:
        assert False, "Expected KeyError due to missing input columns"
