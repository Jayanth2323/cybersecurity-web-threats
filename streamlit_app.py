import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Threat Dashboard", layout="wide")

# Title
st.title("🔐 Suspicious Web Threat Interactions Dashboard")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("data/analyzed_output.csv")

df = load_data()

# Filters
st.sidebar.header("🔎 Filter Options")
status = st.sidebar.multiselect("Anomaly Status", df["anomaly"].unique(), default=df["anomaly"].unique())
countries = st.sidebar.multiselect("Source Countries", df["src_ip_country_code"].unique())

filtered_df = df[df["anomaly"].isin(status)]
if countries:
    filtered_df = filtered_df[filtered_df["src_ip_country_code"].isin(countries)]

# Display metrics
st.subheader("📊 Key Metrics")
col1, col2 = st.columns(2)
col1.metric("Total Records", len(df))
col2.metric("Suspicious", len(df[df['anomaly'] == 'Suspicious']))

# Table view
st.subheader("🧾 Filtered Data View")
st.dataframe(filtered_df, use_container_width=True)

# Visuals
st.subheader("📈 Anomaly Scatterplot")
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=filtered_df, x="bytes_in", y="bytes_out", hue="anomaly", palette="Set1", ax=ax)
st.pyplot(fig)
