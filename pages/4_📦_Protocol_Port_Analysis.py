# pages/4_📦_Protocol_Port_Analysis.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from typing import Tuple, Optional

# st.title("📦 Protocol & Port Analysis")

DEFAULT_DATA_PATH = "data/analyzed_output.csv"

warnings.filterwarnings("ignore")


@st.cache_data
def load_data(file_path: str = DEFAULT_DATA_PATH) -> Optional[pd.DataFrame]:
    """Load and validate the dataset."""
    try:
        df = pd.read_csv(file_path)
        required_cols = {"protocol", "dst_port", "anomaly"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            st.warning(f"Missing required columns: {missing}")
            return None
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None


def create_protocol_plot(df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """Create protocol count visualization."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(data=df, x="protocol", hue="anomaly", ax=ax)
    ax.set_title("Anomalies by Protocol")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig, ax


def create_port_plot(df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """Create destination port visualization."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(
        data=df, x="dst_port", hue="anomaly", multiple="stack", bins=40, ax=ax
    )
    ax.set_title("Destination Ports - Normal vs Suspicious")
    plt.tight_layout()
    return fig, ax


def main():
    st.title("📦 Protocol & Port Analysis")

    df = load_data()
    if df is None:
        st.error("Cannot proceed without valid data")
        return

    st.subheader("📦 Protocol vs Anomaly")
    try:
        fig1, _ = create_protocol_plot(df)
        st.pyplot(fig1)
    except Exception as e:
        st.error(f"Protocol plot failed: {str(e)}")

    st.subheader("📫 Destination Port Distribution")
    try:
        fig2, _ = create_port_plot(df)
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Port plot failed: {str(e)}")


if __name__ == "__main__":
    main()
