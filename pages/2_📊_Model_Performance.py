import streamlit as st
import pandas as pd
from models.rf_model import train_rf_model

df = pd.read_csv("data/analyzed_output.csv")
features = ["bytes_in", "bytes_out", "duration_seconds", "avg_packet_size"]
df = train_rf_model(df, features)

st.title(" Model Insights: Random Forest")
st.write("Random Forest Model Features:")
st.write(features)
st.dataframe(df.head())
