import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from typing import Optional

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
DEFAULT_DATA_PATH = "data/CloudWatch_Traffic_Web_Attack.csv"

# Features aligned to your original heatmap
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
    """Configure Streamlit page."""
    st.set_page_config(
        page_title="Correlation Matrix Heatmap",
        layout="wide",
    )


@st.cache_data
def load_and_prepare_data(
    file_path: str = DEFAULT_DATA_PATH,
) -> Optional[pd.DataFrame]:
    """Load and process the CSV data."""
    try:
        df = pd.read_csv(file_path)

        # Parse datetime columns
        df["creation_time"] = pd.to_datetime(df["creation_time"])
        df["end_time"] = pd.to_datetime(df["end_time"])

        # Compute session duration
        df["duration_seconds"] = (
            df["end_time"] - df["creation_time"]
        ).dt.total_seconds()

        # One-hot encode the 'src_ip_country_code'
        country_dummies = pd.get_dummies(
            df["src_ip_country_code"], prefix="src_ip_country_code"
        )
        df = pd.concat([df, country_dummies], axis=1)

        # MinMax Scaling for specific fields
        scaler = MinMaxScaler()
        df["scaled_bytes_in"] = scaler.fit_transform(df[["bytes_in"]])
        df["scaled_bytes_out"] = scaler.fit_transform(df[["bytes_out"]])
        df["scaled_duration_seconds"] = scaler.fit_transform(
            df[["duration_seconds"]]
        )
        # Select and prepare the final set of features
        final_df = df[FEATURES_TO_INCLUDE].copy()
        final_df = final_df.rename(columns={"response.code": "response_code"})

        return final_df

    except Exception as e:
        st.error(f"Data processing failed: {str(e)}")
        return None


def create_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Create the correlation matrix heatmap."""
    st.subheader("Correlation Matrix Heatmap")
    st.markdown(f"**Analyzing Features:** `{', '.join(df.columns)}`")

    # Calculate correlation matrix
    corr = df.corr()

    # Create heatmap figure
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=1,
        linecolor="white",
        cbar_kws={"shrink": 0.7},
        square=True,
        mask=corr.isnull(),
        ax=ax,
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Correlation Matrix Heatmap", fontsize=16, pad=20)
    plt.tight_layout()
    return fig


def main():
    configure_page()

    st.title("📊 Network Traffic Feature Correlation Matrix")
    st.markdown(
        """
        This heatmap helps uncover **feature interrelationships** and
        **linear associations** in the network traffic dataset.
        Great for **model refinement**, **feature selection**,
        and understanding **behavioral patterns**.
        """
    )

    df = load_and_prepare_data()
    if df is None:
        st.error("Unable to generate the correlation matrix.")
        return

    try:
        fig = create_correlation_heatmap(df)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Heatmap generation error: {str(e)}")


if __name__ == "__main__":
    main()
