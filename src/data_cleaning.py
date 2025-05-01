import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop nulls and duplicates
    df = df.dropna().drop_duplicates()

    # Safely convert time columns using .loc to avoid SettingWithCopyWarning
    datetime_cols = ["creation_time", "end_time", "time"]
    for col in datetime_cols:
        if col in df.columns:
            df.loc[:, col] = pd.to_datetime(df[col], errors="coerce")

    # Normalize country codes
    if "src_ip_country_code" in df.columns:
        df.loc[:, "src_ip_country_code"] = df[
            "src_ip_country_code"
        ].str.upper()

    return df
