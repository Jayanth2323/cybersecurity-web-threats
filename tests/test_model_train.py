import sys
import os
import pandas as pd
from src.model_train import train_anomaly_model

# Add root path for importing local modules
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_train_anomaly_model_output_labels():
    """Test that train_anomaly_model returns expected anomaly labels."""
    df = pd.DataFrame({
        "bytes_in": [100, 200, 150],
        "bytes_out": [300, 400, 350],
        "duration_seconds": [60, 60, 60],
        "avg_packet_size": [6.67, 10.0, 8.33]
    })

    result = train_anomaly_model(
        df.copy(),
        features=[
            "bytes_in",
            "bytes_out",
            "duration_seconds",
            "avg_packet_size"
        ]
    )

    # Ensure 'anomaly' column exists
    assert "anomaly" in result.columns, "Missing 'anomaly' column in output."

    # Ensure labels are only 'Normal' or 'Suspicious'
    allowed_labels = {"Normal", "Suspicious"}
    unique_labels = set(result["anomaly"].unique())
    assert unique_labels.issubset(allowed_labels)
    f"Unexpected labels: {unique_labels}"
