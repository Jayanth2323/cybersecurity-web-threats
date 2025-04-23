import streamlit as st
import pandas as pd
from models.nn_model import preprocess_and_train_nn

# Load the dataset
df = pd.read_csv("data/analyzed_output.csv")

# Define the features to be used for training
features = ["bytes_in", "bytes_out", "duration_seconds", "avg_packet_size"]

# Convert the anomaly column to binary (0/1) for training
df["anomaly_binary"] = df["anomaly"].map(
    {"Suspicious": 1, "Not Suspicious": 0}
)

# Validate input data
if df.empty:
    st.error("No data loaded")
    st.stop()

# Validate feature columns
for feature in features:
    if feature not in df.columns:
        st.error(f"Missing feature column: {feature}")
        st.stop()

# Validate target column
if "anomaly_binary" not in df.columns:
    st.error("Missing target column: anomaly_binary")
    st.stop()

# Train the neural network model and get the scaler
try:
    model, scaler = preprocess_and_train_nn(df[features], df["anomaly_binary"])
    st.title("Neural Network Results")
    st.success("Model trained successfully!")
    st.write("Model:", model)
    st.write("Scaler:", scaler)
except Exception as e:
    st.error(f"Error training model: {e.__class__.__name__}: {str(e)}")
