import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
# from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Standard library
from typing import Optional

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
DEFAULT_DATA_PATH = "data/CloudWatch_Traffic_Web_Attack.csv"

# Features aligned to the provided heatmap
FEATURES_TO_INCLUDE = [
    "bytes_in",
    "bytes_out",
    "response.code",
    "dst_port",
    "duration_seconds",
    "scaled_bytes_in",
    "scaled_bytes_out",
    "scaled_duration_seconds",
    "src_ip_country_code_AE",
    "src_ip_country_code_AT",
    "src_ip_country_code_CA",
    "src_ip_country_code_DE",
    "src_ip_country_code_IL",
    "src_ip_country_code_NL",
    "src_ip_country_code_US",
]


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Correlation Matrix",
        layout="wide",
    )


@st.cache_data
def load_and_prepare_data(
    file_path: str = DEFAULT_DATA_PATH,
) -> Optional[pd.DataFrame]:
    """Load dataset and prepare features for correlation analysis."""
    try:
        df = pd.read_csv(file_path)

        # Calculate duration in seconds
        df["creation_time"] = pd.to_datetime(df["creation_time"])
        df["end_time"] = pd.to_datetime(df["end_time"])
        df["duration_seconds"] = (
            df["end_time"] - df["creation_time"]
        ).dt.total_seconds()

        # One-hot encode country codes
        country_dummies = pd.get_dummies(
            df["src_ip_country_code"], prefix="src_ip_country_code"
        )
        df = pd.concat([df, country_dummies], axis=1)

        # Create scaled features
        scaler = MinMaxScaler()
        df["scaled_bytes_in"] = scaler.fit_transform(df[["bytes_in"]])
        df["scaled_bytes_out"] = scaler.fit_transform(df[["bytes_out"]])
        df["scaled_duration_seconds"] = scaler.fit_transform(
            df[["duration_seconds"]]
        )

        # Select and order final features
        final_df = df[FEATURES_TO_INCLUDE].copy()
        return final_df.rename(columns={"response.code": "response_code"})

    except Exception as e:
        st.error(f"Data processing failed: {str(e)}")
        return None


def create_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Generate and return the correlation heatmap."""
    st.subheader("Correlation Matrix Heatmap")
    st.markdown(f"**Features Included:** `{', '.join(df.columns)}`")

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=1,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Correlation Matrix Heatmap", fontsize=16, pad=20)
    plt.tight_layout()
    return fig


def main():
    configure_page()
    st.title("📊 Feature Correlation Analysis")
    st.markdown(
        """
        This heatmap offers insights into linear relationships between
        selected features, aiding in the discovery of data patterns,
        multicollinearity, and model improvement opportunities.
        """
    )

    df = load_and_prepare_data()
    if df is None:
        st.error("Cannot display heatmap without sufficient feature data.")
        return

    try:
        fig = create_correlation_heatmap(df)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Heatmap generation error: {str(e)}")


if __name__ == "__main__":
    main()
