import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

st.title("📊 Visual Analytics: Anomalies & Traffic Patterns")


@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/analyzed_output.csv")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error


df = load_data()

if df.empty:
    st.warning("No data loaded - some visualizations may not work")
else:
    # Filter for Suspicious only
    df_suspicious = df[df["anomaly"] == "Suspicious"]

    # 🌍 Choropleth Map
    st.subheader("🗺️ Suspicious Activities by Country")
    country_counts = (
        df_suspicious.groupby("src_ip_country_code")
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
            hover_name="src_ip_country_code",
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No suspicious activity data available.")

custom_palette = {
    "Normal": "#377eb8",      
    "Suspicious": "#e41a1c"   
}
# 📈 Scatterplot: Bytes In vs Bytes Out
st.subheader("📈 Anomaly Scatterplot: Bytes In vs Out")

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    data=df,
    x="bytes_in",
    y="bytes_out",
    hue="anomaly",
    palette=custom_palette,
    ax=ax
)
ax.set_xlabel("Bytes In")
ax.set_ylabel("Bytes Out")
ax.set_title("Anomaly Scatterplot")
st.pyplot(fig)

# 📊 Descriptive Statistics
st.subheader("📊 Statistical Summary")
st.write(df.describe())

# Additional statistics
st.subheader("📊 Additional Statistics")
st.write(df.groupby("anomaly").describe())

# Data distribution
st.subheader("📊 Data Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(
    data=df,
    x="bytes_in",
    hue="anomaly",
    palette=custom_palette,
    ax=ax
    )
ax.set_xlabel("Bytes In")
ax.set_ylabel("Frequency")
ax.set_title("Data Distribution")
st.pyplot(fig)
warnings.filterwarnings("ignore")
