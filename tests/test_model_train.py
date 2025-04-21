import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import pandas as pd
from src.model_train import train_anomaly_model


def test_train_anomaly_model():
    df = pd.DataFrame(
        {
            "bytes_in": [100, 200, 150],
            "bytes_out": [300, 400, 350],
            "duration_seconds": [60, 60, 60],
            "avg_packet_size": [6.67, 10, 8.33],
        }
    )

    result = train_anomaly_model(
        df.copy(),
        features=[
            "bytes_in",
            "bytes_out",
            "duration_seconds",
            "avg_packet_size",
        ],
    )

    assert "anomaly" in result.columns
    assert set(result["anomaly"].unique()).issubset({"Normal", "Suspicious"})
