import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Standard library
from typing import Optional

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
DEFAULT_DATA_PATH = "data/CloudWatch_Traffic_Web_Attack.csv"

# Features explicitly aligned to the provided heatmap
FEATURES_TO_INCLUDE = [
    "bytes_in",
    "bytes_out",
    "response_code",
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
        menu_items={
            "Get Help": "https://example.com/help",
            "Report a bug": "https://example.com/bug",
            "About": "# Network Traffic Analyzer",
        },
    )


@st.cache_data
def load_and_prepare_data(
    file_path: str = DEFAULT_DATA_PATH,
) -> Optional[pd.DataFrame]:
    """Load dataset and filter relevant features for correlation analysis."""
    try:
        df = pd.read_csv(file_path)
        # Ensure only selected columns are loaded if they exist
        selected_columns = [
            col for col in FEATURES_TO_INCLUDE if col in df.columns
        ]

        if len(selected_columns) < 2:
            st.warning(
                "Insufficient columns available for correlation analysis."
            )
            return None

        return df[selected_columns]
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None


def create_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Generate and return the correlation heatmap."""
    st.subheader("Correlation Matrix Heatmap")
    st.markdown(f"**Features Included:** `{', '.join(df.columns)}`")

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))  # Adjusted size for clarity
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",  # Match to the reference image
        center=0,
        linewidths=1,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )

    # Fine-tuned axis label rotation for readability
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
        selected features,aiding in the discovery of data patterns,
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
