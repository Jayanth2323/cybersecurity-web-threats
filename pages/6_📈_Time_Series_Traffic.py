# pages/6_📈_Time_Series_Traffic.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

# Standard library
# from datetime import datetime


# Configuration
DEFAULT_DATA_PATH = "data/analyzed_output.csv"


@st.cache_data
def load_and_prepare_time_data(
    file_path: str = DEFAULT_DATA_PATH,
) -> Optional[pd.DataFrame]:
    """Load and prepare time series data."""
    try:
        df = pd.read_csv(file_path)

        # Convert and validate time column
        if "creation_time" not in df.columns:
            st.error("Missing required column: creation_time")
            return None

        df["creation_time"] = pd.to_datetime(
            df["creation_time"], errors="coerce"
        )
        if df["creation_time"].isnull().all():
            st.error("Invalid datetime format in creation_time column")
            return None

        # Validate anomaly column exists
        if "anomaly" not in df.columns:
            st.error("Missing required column: anomaly")
            return None

        return df
    except Exception as e:
        st.error(f"Time data processing failed: {str(e)}")
        return None


def create_time_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Create appropriate time bins based on data duration."""
    time_range = df["creation_time"].max() - df["creation_time"].min()

    if time_range > pd.Timedelta(days=7):
        # Weekly bins for long durations
        df["time_bin"] = (
            df["creation_time"].dt.to_period("W").dt.to_timestamp()
        )
    elif time_range > pd.Timedelta(days=1):
        # Daily bins for medium durations
        df["time_bin"] = df["creation_time"].dt.date
    else:
        # Hourly bins for short durations
        df["time_bin"] = df["creation_time"].dt.floor("H")

    return df


def create_time_series_plot(df: pd.DataFrame) -> plt.Figure:
    """Generate time series visualization."""
    time_df = (
        df.groupby(["time_bin", "anomaly"]).size().reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        data=time_df,
        x="time_bin",
        y="count",
        hue="anomaly",
        ax=ax,
        marker="o",
        palette={"normal": "blue", "suspicious": "red"},  # Explicit colors
    )

    ax.set_title("Network Anomalies Over Time", pad=15)
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Event Count")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Improve x-axis formatting
    if pd.api.types.is_datetime64_any_dtype(time_df["time_bin"]):
        plt.xticks(rotation=45)
        ax.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M")
        )

    plt.tight_layout()
    return fig


def main():
    st.title("📈 Time Series: Traffic Over Time")
    st.markdown(
        """Visualizing traffic patterns over time to identify spikes or
        trends in suspicious activity."""
    )

    df = load_and_prepare_time_data()
    if df is None:
        return

    try:
        _extracted_from_main_12(df)
    except Exception as e:
        st.error(f"Time series visualization failed: {str(e)}")


# TODO Rename this here and in `main`
def _extracted_from_main_12(df):
    df = create_time_bins(df)
    fig = create_time_series_plot(df)
    st.pyplot(fig)

    # Add summary statistics
    st.subheader("Summary Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Time Range Covered",
            f"{df['creation_time'].min().strftime('%Y-%m-%d')} to "
            f"{df['creation_time'].max().strftime('%Y-%m-%d')}",
        )

    with col2:
        anomaly_rate = (
            df["anomaly"].value_counts(normalize=True).get("suspicious", 0)
        )
        st.metric("Anomaly Rate", f"{anomaly_rate:.1%}")


if __name__ == "__main__":
    main()
