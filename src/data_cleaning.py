import pandas as pd


def clean_data(df):
    df = df.drop_duplicates()
    df["creation_time"] = pd.to_datetime(df["creation_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df["time"] = pd.to_datetime(df["time"])
    df["src_ip_country_code"] = df["src_ip_country_code"].str.upper()
    return df
