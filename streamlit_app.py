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
    st.warning(
        """No data loaded. Please ensure
        'analyzed_output.csv' exists and has valid data.""")
    st.stop()

# Feature Columns
features = ["bytes_in", "bytes_out", "duration_seconds", "avg_packet_size"]

# Model selection
model_choice = st.sidebar.radio(
    "🧠 Select Model", ["Random Forest", "Neural Network"]
)

if model_choice == "Random Forest":
    from src.model_train import train_anomaly_model

    df = train_anomaly_model(df, features)
else:
    from models.nn_model import preprocess_and_train_nn
    from sklearn.preprocessing import LabelEncoder

    # Convert anomaly to binary (0, 1)
    df = df.copy()
    le = LabelEncoder()
    df["anomaly_binary"] = le.fit_transform(df["anomaly"])  # Suspicious = 1
    X = df[features]
    y = df["anomaly_binary"]
    nn_model, scaler = preprocess_and_train_nn(X, y)
    # Optional: Add predictions to df
    threat_scores = nn_model.predict(scaler.transform(X)).flatten()
    df["threat_score"] = threat_scores
    df["anomaly"] = df["threat_score"].apply(
        lambda x: "Suspicious" if x > 0.5 else "Normal"
    )

# Sidebar filters
st.sidebar.header("🔎 Filter Options")

status_options = df["anomaly"].dropna().unique().tolist()
status = st.sidebar.multiselect(
    "Anomaly Status", status_options, default=status_options
)

country_options = df["src_ip_country_code"].dropna().unique().tolist()
countries = st.sidebar.multiselect("Source Countries", country_options)

# Packet size slider
st.sidebar.subheader("📦 Avg Packet Size")
min_val = float(df["avg_packet_size"].min())
max_val = float(df["avg_packet_size"].max())
packet_range = st.sidebar.slider(
    "Select packet size range",
    min_value=min_val,
    max_value=max_val,
    value=(min_val, max_val),
)

# Apply filters
filtered_df = df[
    df["anomaly"].isin(status)
    & df["avg_packet_size"].between(packet_range[0], packet_range[1])
].copy()

if countries:
    filtered_df = filtered_df[
        filtered_df["src_ip_country_code"].isin(countries)
    ]

# Stats Panel
with st.expander("📊 Statistical Summary"):
    st.write(filtered_df.describe())

# Key Metrics
st.subheader("📊 Key Metrics")
col1, col2 = st.columns(2)
col1.metric("Total Records", len(df))
col2.metric("Suspicious", len(df[df["anomaly"] == "Suspicious"]))

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
        title="Suspicious IPs by Country",
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No suspicious activity found for the selected filters.")

# Export Options
st.subheader("📤 Export Options")
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    filtered_df.to_excel(writer, index=False)
excel_buffer.seek(0)

st.download_button(
    label="Download Excel",
    data=excel_buffer,
    file_name="filtered_threats.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
        ax=ax,)
    st.pyplot(fig)
else:
    st.info("No data available for scatterplot with the current filters.")
