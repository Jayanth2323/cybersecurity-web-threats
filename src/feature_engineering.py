import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add duration in seconds
    df["duration_seconds"] = (
        df["end_time"] - df["creation_time"]
    ).dt.total_seconds()
    df["duration_seconds"] = df["duration_seconds"].fillna(0)

    # Avoid division by zero in average packet size
    total_bytes = df["bytes_in"] + df["bytes_out"]
    safe_duration = df["duration_seconds"].replace(
        0, 1
    )  # Prevent division by zero
    df["avg_packet_size"] = total_bytes / safe_duration

    # Set avg_packet_size to 0 explicitly where original duration was 0
    df.loc[df["duration_seconds"] == 0, "avg_packet_size"] = 0

    return df
