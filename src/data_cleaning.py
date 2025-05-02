# src/data_cleaning.py

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by:
    - Ensuring required columns exist
    - Converting datetime fields to timezone-aware datetime64[ns, UTC]
    - Uppercasing ISO country codes
    - Removing null values
    - Removing duplicates

    Parameters:
        df (pd.DataFrame): The raw input data

    Returns:
        pd.DataFrame: Cleaned and formatted data

    Raises:
        KeyError: If required columns are missing
    """
    required_columns = [
        "creation_time",
        "end_time",
        "time",
        "src_ip_country_code",
    ]
    if missing_columns := [
        col for col in required_columns if col not in df.columns
    ]:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # Handle empty dataframe early
    if df.empty:
        return df

    # Convert datetime fields to timezone-aware datetimes (UTC)
    for col in ["creation_time", "end_time", "time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Uppercase country code
    df["src_ip_country_code"] = df["src_ip_country_code"].str.upper()

    # Drop rows with any nulls and duplicates
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    return df
