import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Standard library
from typing import Optional

# Third-party
warnings.filterwarnings("ignore")

# Configuration
DEFAULT_DATA_PATH = "data/analyzed_output.csv"


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
