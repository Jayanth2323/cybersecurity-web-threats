import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("📊 Visual Analytics: Anomalies & Traffic Patterns")


@st.cache_data
def load_data():
    return pd.read_csv("data/analyzed_output.csv")


df = load_data()

# Filter for Suspicious only
df_suspicious = df[df["anomaly"] == "Suspicious"]

# 🌍 Choropleth Map
st.subheader("🗺️ Suspicious Activities by Country")
country_counts = df_suspicious.groupby(
    "src_ip_country_code").size().reset_index(name="count")

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
    st.info("No suspicious activity data available.")

# 📈 Scatterplot: Bytes In vs Bytes Out
st.subheader("📈 Anomaly Scatterplot: Bytes In vs Out")

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    data=df, x="bytes_in", y="bytes_out", hue="anomaly", palette="Set1", ax=ax)
st.pyplot(fig)

# 📊 Descriptive Statistics
st.subheader("📊 Statistical Summary")
st.write(df.describe())
