import sys
import os
import pandas as pd
from src.model_train import train_anomaly_model

# Add root path for importing local modules
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


def test_train_anomaly_model_output_labels():
    """Test that train_anomaly_model returns expected anomaly labels."""
    df = pd.DataFrame(
        {
            "bytes_in": [100, 200, 150],
            "bytes_out": [300, 400, 350],
            "duration_seconds": [60, 60, 60],
            "avg_packet_size": [6.67, 10.0, 8.33],
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

    # Validate row count consistency
    assert (
        result.shape[0] == df.shape[0]
    ), "Output row count does not match input."

    # Ensure 'anomaly' column exists
    assert (
        "anomaly" in result.columns
    ), "'anomaly' column not found in model output."

    # Ensure labels are only 'Normal' or 'Suspicious'
    allowed_labels = {"Normal", "Suspicious"}
    unique_labels = set(result["anomaly"].unique())
    assert unique_labels.issubset(
        allowed_labels
    ), f"Unexpected labels detected: {unique_labels}"

    # Check label data type
    assert result["anomaly"].dtype.name in {
        "object",
        "category",
    }, "Anomaly column should be of type object or category."
