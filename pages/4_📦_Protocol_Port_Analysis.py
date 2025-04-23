# pages/4_\ud83d\udce6_Protocol_Port_Analysis.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("\ud83d\udce6 Protocol & Port Analysis")


@st.cache_data
def load_data():
    return pd.read_csv("data/analyzed_output.csv")


df = load_data()

st.subheader("\ud83d\udd22 Protocol vs Anomaly")
if "protocol" in df.columns:
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.countplot(data=df, x="protocol", hue="anomaly", ax=ax1)
    ax1.set_title("Anomalies by Protocol")
    st.pyplot(fig1)
else:
    st.warning("Protocol column not found in dataset.")

st.subheader("\ud83d\udecb\ufe0f Destination Port Distribution")
if "dst_port" in df.columns:
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.histplot(
        data=df, x="dst_port", hue="anomaly",
        multiple="stack", bins=40, ax=ax2)
    ax2.set_title("Destination Ports - Normal vs Suspicious")
    st.pyplot(fig2)
else:
    st.warning("Destination port column not found.")
