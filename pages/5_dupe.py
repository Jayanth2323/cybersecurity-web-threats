import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO

warnings.filterwarnings("ignore")

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

CSV_PATH = "/mnt/data/CloudWatch_Traffic_Web_Attack.csv"  # Predefined file path

def configure_page():
    st.set_page_config(
        page_title="Network Traffic Correlation & Attack Analysis",
        layout="wide",
    )

@st.cache_data
def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # Parse datetime columns
    df["creation_time"] = pd.to_datetime(df["creation_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    # Compute session duration
    df["duration_seconds"] = (df["end_time"] - df["creation_time"]).dt.total_seconds()

    # One-hot encode country codes
    country_dummies = pd.get_dummies(df["src_ip_country_code"], prefix="src_ip_country_code")
    df = pd.concat([df, country_dummies], axis=1)

    # Feature Scaling
    scaler = MinMaxScaler()
    df["scaled_bytes_in"] = scaler.fit_transform(df[["bytes_in"]])
    df["scaled_bytes_out"] = scaler.fit_transform(df[["bytes_out"]])
    df["scaled_duration_seconds"] = scaler.fit_transform(df[["duration_seconds"]])

    final_df = df[FEATURES_TO_INCLUDE].copy()
    final_df = final_df.rename(columns={"response.code": "response_code"})

    return df, final_df

def create_correlation_heatmap(df: pd.DataFrame) -> (plt.Figure, pd.DataFrame):
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
        cbar_kws={"shrink": 0.7},
        square=True,
        mask=corr.isnull(),
        ax=ax,
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Correlation Matrix Heatmap", fontsize=16, pad=20)
    plt.tight_layout()

    return fig, corr

def create_country_detection_barplot(df: pd.DataFrame) -> plt.Figure:
    country_counts = df["src_ip_country_code"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=country_counts.index,
        y=country_counts.values,
        ax=ax,
        palette="Blues_d",
    )

    ax.set_title("Detection Types by Country Code", fontsize=16)
    ax.set_xlabel("Country Code", fontsize=12)
    ax.set_ylabel("Frequency of Detection Types", fontsize=12)
    ax.bar_label(ax.containers[0])
    plt.tight_layout()

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
        This dashboard automatically processes internal traffic data to deliver:
        - 📈 **Feature Correlation Analysis**
        - 🌎 **Geographical Attack Distribution**
        """
    )

    with st.spinner("Loading and processing data..."):
        try:
            raw_df, features_df = load_and_prepare_data(CSV_PATH)

            tab1, tab2, tab3 = st.tabs(["📈 Correlation Heatmap", "🌎 Country Detection", "📥 Download Matrix"])

            with tab1:
                st.subheader("📈 Correlation Matrix Heatmap")
                corr_fig, corr_matrix = create_correlation_heatmap(features_df)
                st.pyplot(corr_fig)

            with tab2:
                st.subheader("🌎 Detection Types by Country Code")
                barplot_fig = create_country_detection_barplot(raw_df)
                st.pyplot(barplot_fig)

            with tab3:
                st.subheader("📥 Download Correlation Matrix")
                csv = convert_df_to_csv(corr_matrix)
                st.download_button(
                    label="Download Correlation Matrix as CSV",
                    data=csv,
                    file_name="correlation_matrix.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"🚨 An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
