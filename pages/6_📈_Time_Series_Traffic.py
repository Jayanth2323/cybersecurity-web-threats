# pages/6_Time_Series_Traffic.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from io import BytesIO
import warnings
from st_aggrid import AgGrid, GridOptionsBuilder

# Configuration
DEFAULT_DATA_PATH = "data/analyzed_output.csv"
warnings.filterwarnings("ignore")


def configure_page():
    st.set_page_config(
        page_title="Time Series: Web Traffic Analysis",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_data
def load_data(path=DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["creation_time"] = pd.to_datetime(df["creation_time"], errors="coerce")
    df = df.dropna(subset=["creation_time", "bytes_in", "bytes_out"])
    df["anomaly"] = pd.Categorical(
        df["anomaly"], categories=["Normal", "Suspicious"], ordered=True
    )
    return df


def filter_data_by_date(df: pd.DataFrame):
    min_date = df["creation_time"].min().date()
    max_date = df["creation_time"].max().date()
    selected_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )
    mask = df["creation_time"].dt.date.between(*selected_range)
    return df[mask]


def plot_traffic(
    df: pd.DataFrame,
    view_mode: str,
    plot_type: str,
    show_anomalies: bool = True,
):
    df["time_bin"] = df["creation_time"].dt.floor("H")

    if view_mode == "Volume (Bytes)":
        grouped = (
            df.groupby("time_bin")
            .agg(
                bytes_in=("bytes_in", "sum"),
                bytes_out=("bytes_out", "sum"),
            )
            .reset_index()
        )
    else:
        grouped = df.groupby("time_bin").size().reset_index(name="count")

    fig, ax = plt.subplots(figsize=(14, 6))

    if view_mode == "Volume (Bytes)":
        if plot_type == "Line":
            ax.plot(
                grouped["time_bin"],
                grouped["bytes_in"],
                label="Bytes In",
                marker="o",
            )
            ax.plot(
                grouped["time_bin"],
                grouped["bytes_out"],
                label="Bytes Out",
                marker="o",
            )
        else:
            ax.bar(
                grouped["time_bin"],
                grouped["bytes_in"],
                width=0.02,
                label="Bytes In",
                alpha=0.7,
            )
            ax.bar(
                grouped["time_bin"],
                grouped["bytes_out"],
                width=0.02,
                bottom=grouped["bytes_in"],
                label="Bytes Out",
                alpha=0.7,
            )
    elif plot_type == "Line":
        ax.plot(
            grouped["time_bin"],
            grouped["count"],
            label="Record Count",
            marker="o",
        )
    else:
        ax.bar(
            grouped["time_bin"],
            grouped["count"],
            label="Record Count",
            width=0.02,
        )

    if show_anomalies:
        suspicious = (
            df[df["anomaly"] == "Suspicious"]
            .groupby("time_bin")
            .size()
            .reset_index(name="count")
        )
        for _, row in suspicious.iterrows():
            ax.axvline(
                x=row["time_bin"], color="red", linestyle="--", alpha=0.25
            )

    ax.set_title("Web Traffic Over Time", fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(
        "Volume (Bytes)" if view_mode == "Volume (Bytes)" else "Record Count",
        fontsize=12,
    )
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def download_plot(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    st.download_button(
        "📥 Download Plot as PNG", buffer, file_name="web_traffic_plot.png"
    )


def show_interactive_table(df: pd.DataFrame):
    st.markdown("### 📋 Explore Data Interactively")
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(filterable=True, sortable=True, resizable=True)
    grid_options = gb.build()
    AgGrid(
        df,
        gridOptions=grid_options,
        height=300,
        enable_enterprise_modules=True,
        fit_columns_on_grid_load=True,
    )


def main():
    configure_page()
    st.title("📈 Time Series: Web Traffic Analytics")

    df = load_data()
    if df.empty:
        st.warning("No data available.")
        return

    with st.sidebar:
        st.header("🔍 Filters & Visualization")
        df = filter_data_by_date(df)
        show_anomalies = st.checkbox("Show Anomaly Markers", value=True)
        view_mode = st.radio("Y-axis View", ["Volume (Bytes)", "Record Count"])
        plot_type = st.selectbox("Chart Type", ["Line", "Bar"])

    fig = plot_traffic(df, view_mode, plot_type, show_anomalies)
    st.pyplot(fig)
    download_plot(fig)

    show_interactive_table(df)


if __name__ == "__main__":
    main()
