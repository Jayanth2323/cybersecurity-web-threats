def add_features(df):
    df["duration_seconds"] = (
        df["end_time"] - df["creation_time"]
    ).dt.total_seconds()

    df["avg_packet_size"] = (df["bytes_in"] + df["bytes_out"]) / df[
        "duration_seconds"
    ]
    return df
