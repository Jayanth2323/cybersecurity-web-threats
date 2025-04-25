import streamlit as st
import pandas as pd
import warnings
from models.rf_model import train_rf_model

# from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")


def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None


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

        try:
            df = train_rf_model(df, features)

            st.title("Model Insights: Random Forest")
            st.write("Random Forest Model Features:")
            st.write(features)
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Model training failed: {str(e)}")


if __name__ == "__main__":
    main()
