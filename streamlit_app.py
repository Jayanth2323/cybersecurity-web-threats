import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io

# Configure Streamlit page
st.set_page_config(page_title="Threat Dashboard", layout="wide")

# Title
st.title("🔐 Suspicious Web Threat Interactions Dashboard")


# Load Data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/analyzed_output.csv")
    except FileNotFoundError:
        st.error("The data file could not be found. Please check the path.")
        return pd.DataFrame()


df = load_data()

if df.empty:
    st.warning("""No data loaded.
    Please ensure 'analyzed_output.csv' exists and has valid data.""")
    st.stop()

# Sidebar filters
st.sidebar.header("🔎 Filter Options")

status_options = df.get("anomaly", pd.Series()).dropna().unique().tolist()
status = st.sidebar.multiselect(
    "Anomaly Status",
    status_options,
    default=status_options)

country_options = df.get(
    "src_ip_country_code", pd.Series()).dropna().unique().tolist()
countries = st.sidebar.multiselect("Source Countries", country_options)

# Packet size slider
st.sidebar.subheader("📦 Avg Packet Size")
if "avg_packet_size" in df.columns:
    min_val = float(df["avg_packet_size"].min())
    max_val = float(df["avg_packet_size"].max())
    packet_range = st.sidebar.slider(
        "Select packet size range",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val)
    )
else:
    st.warning("Missing 'avg_packet_size' column in data.")
    st.stop()

# Apply filters
filtered_df = df[
    df["anomaly"].isin(status) &
    df["avg_packet_size"].between(packet_range[0], packet_range[1])
].copy()

if countries:
    filtered_df = filtered_df[
        filtered_df["src_ip_country_code"].isin(countries)
        ]

# Key Metrics
st.subheader("📊 Key Metrics")
col1, col2 = st.columns(2)
col1.metric("Total Records", len(df))
col2.metric("Suspicious", len(df[df['anomaly'] == 'Suspicious']))

# Suspicious Activities Map
st.subheader("🗺️ Suspicious Activities by Country")

country_counts = (
    filtered_df[filtered_df["anomaly"] == "Suspicious"]
    .groupby("src_ip_country_code")
    .size()
    .reset_index(name="count")
)

if not country_counts.empty:
    fig_map = px.choropleth(
        country_counts,
        locations="src_ip_country_code",
        locationmode="ISO-3",
        color="count",
        color_continuous_scale="Reds",
        title="Suspicious IPs by Country"
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No suspicious activity found for the selected filters.")

# Export Options
st.subheader("📤 Export Options")
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    filtered_df.to_excel(writer, index=False)
excel_buffer.seek(0)

st.download_button(
    label="Download Excel",
    data=excel_buffer,
    file_name="filtered_threats.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Table Display
st.subheader("🧾 Filtered Data View")
st.dataframe(filtered_df, use_container_width=True)

# Anomaly Scatterplot
st.subheader("📈 Anomaly Scatterplot")

if not filtered_df.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=filtered_df,
        x="bytes_in",
        y="bytes_out",
        hue="anomaly",
        palette="Set1",
        ax=ax
    )
    st.pyplot(fig)
else:
    st.info("No data available for scatterplot with the current filters.")
