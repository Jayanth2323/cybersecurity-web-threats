# pages/6_\ud83d\udcc8_Time_Series_Traffic.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("\ud83d\udcc8 Time Series: Traffic Over Time")


@st.cache_data
def load_data():
    df = pd.read_csv("data/analyzed_output.csv")
    df["creation_time"] = pd.to_datetime(df["creation_time"], errors="coerce")
    return df


df = load_data()

st.markdown(
    """Visualizing traffic patterns over time to identify spikes or
    trends in suspicious activity."""
)

# Group by day or hour depending on granularity
if df["creation_time"].dt.date.nunique() > 1:
    df["time_bin"] = df["creation_time"].dt.date
else:
    df["time_bin"] = df["creation_time"].dt.hour

time_df = df.groupby(["time_bin", "anomaly"]).size().reset_index(name="count")

fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=time_df, x="time_bin", y="count", hue="anomaly", ax=ax)
ax.set_title("Anomalies Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Number of Records")
st.pyplot(fig)
