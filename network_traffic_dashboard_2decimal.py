import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

DEFAULT_DATA_PATH = "analyzed_output.csv"

st.title("📈 Time Series: Traffic Over Time")

@st.cache_data
def load_data():
    df = pd.read_csv(DEFAULT_DATA_PATH)
    df["creation_time"] = pd.to_datetime(df["creation_time"], errors="coerce")
    # Ensure anomaly column is categorical with specific order
    df["anomaly"] = pd.Categorical(df["anomaly"], categories=["Normal", "Suspicious"], ordered=True)
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

# Sort by time_bin for proper plotting
time_df = time_df.sort_values("time_bin")

fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=time_df, x="time_bin", y="count", hue="anomaly", ax=ax, marker="o")
ax.set_title("Anomalies Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Number of Records")
ax.legend(title="Anomaly")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
