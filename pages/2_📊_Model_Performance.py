import streamlit as st
import pandas as pd
from models.rf_model import train_rf_model


def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Failed to load data: {e}")


def main():
    file_path = "data/analyzed_output.csv"
    df = load_data(file_path)

    if df is not None:
        features = [
            "bytes_in",
            "bytes_out",
            "duration_seconds",
            "avg_packet_size",
        ]
        df = train_rf_model(df, features)

        st.title("Model Insights: Random Forest")
        st.write("Random Forest Model Features:")
        st.write(features)
        st.dataframe(df.head())


if __name__ == "__main__":
    main()
