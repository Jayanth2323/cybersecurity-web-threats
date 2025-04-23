# pages/5_\ud83d\udcca_Correlation_Heatmap.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("\ud83d\udcca Correlation Matrix: Feature Relationships")


@st.cache_data
def load_data():
    return pd.read_csv("data/analyzed_output.csv")


df = load_data()

st.markdown(
    """This heatmap helps identify
    linear relationships between numeric features."""
)

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

if len(numeric_cols) >= 2:
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
else:
    st.warning("Not enough numeric columns for correlation analysis.")
