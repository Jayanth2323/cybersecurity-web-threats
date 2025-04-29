import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# import plotly.figure_factory as ff
import warnings
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
from typing import Tuple

warnings.filterwarnings("ignore")

# Static CSV Load
DATA_PATH = "data/CloudWatch_Traffic_Web_Attack.csv"

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
    st.set_page_config(
        page_title="Network Traffic Correlation & Attack Analysis",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_data
def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)

    # Parse datetime
    df["creation_time"] = pd.to_datetime(df["creation_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    # Compute duration
    df["duration_seconds"] = (
        df["end_time"] - df["creation_time"]
    ).dt.total_seconds()

    # One-hot encode country codes
    country_dummies = pd.get_dummies(
        df["src_ip_country_code"], prefix="src_ip_country_code"
    )
    df = pd.concat([df, country_dummies], axis=1)

    # MinMax scaling
    scaler = MinMaxScaler()
    df["scaled_bytes_in"] = scaler.fit_transform(df[["bytes_in"]])
    df["scaled_bytes_out"] = scaler.fit_transform(df[["bytes_out"]])
    df["scaled_duration_seconds"] = scaler.fit_transform(
        df[["duration_seconds"]]
    )

    final_df = df[FEATURES_TO_INCLUDE].copy()
    final_df = final_df.rename(columns={"response.code": "response_code"})

    # def round_decimal(x):
    #     rounded = round(x, 2)
    #     return FormattedValue if rounded != 1.0 else f"{rounded:.2f}"
    def round_decimal(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else x

    decimal_cols = [
        "bytes_in",
        "bytes_out",
        "duration_seconds",
        "scaled_bytes_in",
        "scaled_bytes_out",
        "scaled_duration_seconds",
    ]
    final_df[decimal_cols] = final_df[decimal_cols].applymap(round_decimal)
    # final_df[decimal_cols] = final_df[decimal_cols].applymap(
    #     lambda x: round_decimal(x) if isinstance(x, (int(float))) else x
    # )
    # final_df[decimal_cols] = final_df[decimal_cols].applymap(
    #     lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
    # )

    return df, final_df


def create_correlation_heatmap(
    df: pd.DataFrame,
) -> Tuple[px.imshow, pd.DataFrame]:
    corr = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        annot_kws={"size": 12},  # Increase font size here
        cbar_kws={"shrink": 0.82},
    )

    # Show the plot in Streamlit
    st.pyplot(plt)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=[
            (1.0, "maroon"),
            (0.5, "lightblue"),
            (0.0, "blue"),
            (0.3, "lightblue"),  # Added for a smoother gradient
            (0.2, "lightblue"),
            (0.1, "lightblue"),
            (0.0, "blue"),
        ],
        title="Correlation Matrix Heatmap",
        aspect="auto",
        height=900,
        width=1200,
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=80, b=40),
        coloraxis_colorbar=dict(
            title="Correlation",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticks="outside",
        ),
        title=dict(font=dict(size=24)),
        xaxis_title=dict(font=dict(size=18)),
        yaxis_title=dict(font=dict(size=18)),
    )

    fig.update_traces(textfont=dict(size=12))

    return fig, corr


def create_country_detection_barplot(df: pd.DataFrame) -> px.bar:
    country_counts = df["src_ip_country_code"].value_counts().reset_index()
    country_counts.columns = ["Country Code", "Detections"]

    fig = px.bar(
        country_counts,
        x="Country Code",
        y="Detections",
        title="Detection Types by Country Code",
        color="Detections",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        xaxis_title="Country Code", yaxis_title="Detection Frequency"
    )

    return fig


def convert_df_to_csv(df: pd.DataFrame) -> BytesIO:
    output = BytesIO()
    df.to_csv(output, index=True)
    output.seek(0)
    return output


def main():
    configure_page()

    st.title("🌐 Network Traffic Analytics Dashboard")
    st.markdown(
        """
        📊 **Analyze CloudWatch Traffic:**
        - Explore feature correlations
        - Visualize detection patterns by country
        """
    )

    with st.spinner("Loading and processing data..."):
        try:
            raw_df, features_df = load_and_prepare_data()
        except Exception as e:
            st.error(f"🚨 Critical Error: {e}")
            st.stop()

    # Sidebar filters
    st.sidebar.header("🔎 Filter Options")

    selected_countries = st.sidebar.multiselect(
        "Select Country Codes",
        options=raw_df["src_ip_country_code"].unique(),
        default=list(raw_df["src_ip_country_code"].unique()),
    )

    duration_range = st.sidebar.slider(
        "Select Duration Range (seconds)",
        min_value=0,
        max_value=int(raw_df["duration_seconds"].max()),
        value=(0, int(raw_df["duration_seconds"].max())),
    )

    # Filter data based on user input
    filtered_df = raw_df[
        (raw_df["src_ip_country_code"].isin(selected_countries))
        & (
            raw_df["duration_seconds"].between(
                duration_range[0], duration_range[1]
            )
        )
    ]

    st.sidebar.success(f"Filtered dataset size: {filtered_df.shape[0]} rows")

    tabs = st.tabs(
        ["📈 Correlation Heatmap", "🌎 Detection Types", "📥 Download Matrix"]
    )

    with tabs[0]:
        st.subheader("Correlation Heatmap (Filtered)")
        corr_fig, corr_matrix = create_correlation_heatmap(features_df)
        st.plotly_chart(corr_fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Detection Frequency by Country (Filtered)")
        bar_fig = create_country_detection_barplot(filtered_df)
        st.plotly_chart(bar_fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Download Correlation Matrix")
        csv = convert_df_to_csv(corr_matrix)
        st.download_button(
            label="Download Correlation Matrix as CSV",
            data=csv,
            file_name="correlation_matrix.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
