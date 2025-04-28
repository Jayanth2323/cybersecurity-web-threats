import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
import warnings

# Standard library
from typing import Optional
from matplotlib.colors import LinearSegmentedColormap

# Define the correlation matrix data as per the table
data = {
    "bytes_in": [
        1.00,
        1.00,
        np.nan,
        np.nan,
        np.nan,
        1.00,
        1.00,
        np.nan,
        -0.07,
        -0.08,
        -0.17,
        -0.10,
        -0.07,
        -0.01,
        0.32,
    ],
    "bytes_out": [
        1.00,
        1.00,
        np.nan,
        np.nan,
        np.nan,
        1.00,
        1.00,
        np.nan,
        -0.07,
        -0.08,
        -0.16,
        -0.09,
        -0.07,
        -0.05,
        0.33,
    ],
    "response.code": [np.nan] * 15,
    "dst_port": [np.nan] * 15,
    "duration_seconds": [np.nan] * 15,
    "scaled_bytes_in": [
        1.00,
        1.00,
        np.nan,
        np.nan,
        np.nan,
        1.00,
        1.00,
        np.nan,
        -0.07,
        -0.08,
        -0.17,
        -0.10,
        -0.07,
        -0.01,
        0.32,
    ],
    "scaled_bytes_out": [
        1.00,
        1.00,
        np.nan,
        np.nan,
        np.nan,
        1.00,
        1.00,
        np.nan,
        -0.07,
        -0.08,
        -0.16,
        -0.09,
        -0.07,
        -0.05,
        0.33,
    ],
    "scaled_duration_seconds": [np.nan] * 15,
    "src_ip_country_code_AE": [
        -0.07,
        -0.07,
        np.nan,
        np.nan,
        np.nan,
        -0.07,
        -0.07,
        np.nan,
        1.00,
        -0.07,
        -0.14,
        -0.08,
        -0.06,
        -0.06,
        -0.20,
    ],
    "src_ip_country_code_AT": [
        -0.08,
        -0.08,
        np.nan,
        np.nan,
        np.nan,
        -0.08,
        -0.08,
        np.nan,
        -0.07,
        1.00,
        -0.17,
        -0.09,
        -0.06,
        -0.07,
        -0.23,
    ],
    "src_ip_country_code_CA": [
        -0.17,
        -0.16,
        np.nan,
        np.nan,
        np.nan,
        -0.17,
        -0.16,
        np.nan,
        -0.14,
        -0.17,
        1.00,
        -0.19,
        -0.13,
        -0.15,
        -0.48,
    ],
    "src_ip_country_code_DE": [
        -0.10,
        -0.09,
        np.nan,
        np.nan,
        np.nan,
        -0.10,
        -0.09,
        np.nan,
        -0.08,
        -0.09,
        -0.19,
        1.00,
        -0.08,
        -0.09,
        -0.27,
    ],
    "src_ip_country_code_IL": [
        -0.07,
        -0.07,
        np.nan,
        np.nan,
        np.nan,
        -0.07,
        -0.07,
        np.nan,
        -0.06,
        -0.06,
        -0.13,
        -0.08,
        1.00,
        -0.06,
        -0.19,
    ],
    "src_ip_country_code_NL": [
        -0.01,
        -0.05,
        np.nan,
        np.nan,
        np.nan,
        -0.01,
        -0.05,
        np.nan,
        -0.06,
        -0.07,
        -0.15,
        -0.09,
        -0.06,
        1.00,
        -0.21,
    ],
    "src_ip_country_code_US": [
        0.32,
        0.33,
        np.nan,
        np.nan,
        np.nan,
        0.32,
        0.33,
        np.nan,
        -0.20,
        -0.23,
        -0.48,
        -0.27,
        -0.19,
        -0.21,
        1.00,
    ],
}

# Row and column labels
labels = [
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

# Create DataFrame
corr = pd.DataFrame(data, index=labels)

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(12, 10))

# Define custom diverging colormap matching the red-beige-blue gradient


colors = ["#c70039", "#d9b7a9", "#2a4db7"]
cmap = LinearSegmentedColormap.from_list("custom_red_beige_blue", colors)

# Plot the heatmap
sns.heatmap(
    corr,
    ax=ax,
    cmap=cmap,
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    linecolor="gray",
    cbar_kws={"shrink": 0.8, "ticks": [1, 0, -0.4]},
    square=True,
    mask=corr.isnull(),
)

# Set title
ax.set_title("Correlation Matrix Heatmap", fontsize=14, fontfamily="monospace")

# Rotate x-axis labels to vertical with monospace font
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    ha="center",
    fontfamily="monospace",
    fontsize=8,
)
# Left-align y-axis labels with monospace font
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    ha="right",
    fontfamily="monospace",
    fontsize=8,
)
plt.tight_layout()
plt.show()

# Third-party
warnings.filterwarnings("ignore")

# Configuration
DEFAULT_DATA_PATH = "data/CloudWatch_Traffic_Web_Attack.csv"


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
    """Load data and prepare numeric features."""
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(
            include=["float64", "int64"]
        ).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Insufficient numeric columns for correlation analysis")
            return None

        return df[numeric_cols]
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        return None


def create_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Generate correlation heatmap visualization."""
    st.subheader("Correlation Heatmap")
    st.markdown(f"**Numeric Features Analyzed:** `{', '.join(df.columns)}`")

    # Generate heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df.corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap="vlag",
        center=0,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Feature Correlation Heatmap", pad=20)
    plt.tight_layout()
    return fig


def main():
    configure_page()
    st.title("📊 Correlation Matrix: Feature Relationships")
    st.markdown(
        """
        Explore the linear relationships between numerical features
        using a correlation heatmap.
        This visualization can help surface patterns,
        multicollinearity, and trends in your data.
        """
    )

    df = load_and_prepare_data()
    if df is None:
        st.error("Cannot generate heatmap without sufficient numeric data")
        return

    try:
        fig = create_correlation_heatmap(df)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Heatmap generation failed: {str(e)}")


if __name__ == "__main__":
    main()
