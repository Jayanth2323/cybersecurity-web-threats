import streamlit as st
import pandas as pd
import shap
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Initialize SHAP
shap.initjs()


# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/analyzed_output.csv")
        df = df.dropna(
            subset=[
                "anomaly",
                "bytes_in",
                "bytes_out",
                "duration_seconds",
                "avg_packet_size",
            ]
        )
        le = LabelEncoder()
        df["anomaly_binary"] = le.fit_transform(df["anomaly"])
        return df
    except FileNotFoundError:
        st.error("Error: File 'data/analyzed_output.csv' not found.")
        return None


df = load_data()
if df is None:
    st.stop()

# Prepare data
features = ["bytes_in", "bytes_out", "duration_seconds", "avg_packet_size"]
X = df[features]
y = df["anomaly_binary"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
st.markdown("### 📊 Model Evaluation")
st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
st.text(classification_report(y_test, y_pred))
st.write(confusion_matrix(y_test, y_pred))

# SHAP Analysis - Correct implementation
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)  # New API returns single object

# Get values and base value
shap_values_array = shap_values.values[:, :, 1]  # For binary classification
base_value = shap_values.base_values[0, 1]  # Base value for positive class

# Verify dimensions
st.write(f"SHAP values shape: {shap_values_array.shape}")
st.write(f"X_test shape: {X_test.shape}")

# Sample selection
st.markdown("### 🔎 Select a Sample for Explanation")
selected_index = st.number_input(
    f"Row Index (0-{len(X_test)-1})",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
)
sample = X_test.iloc[selected_index]
sample_pred = model.predict([sample])[0]
st.markdown(
    f"**Prediction:** {'Suspicious' if sample_pred == 1 else 'Normal'}"
)

# Corrected Force Plot
st.markdown("### 📉 SHAP Force Plot")
try:
    plt.figure()
    shap.plots.force(
        base_value,
        shap_values_array[selected_index],
        sample.values,
        feature_names=features,
        matplotlib=True,
        show=False,
    )
    st.pyplot(plt.gcf(), bbox_inches="tight")
    plt.clf()
except Exception as e:
    st.error(f"Force plot error: {str(e)}")

# Corrected Summary Plot
with st.expander("📈 SHAP Summary Plot"):
    try:
        plt.figure()
        shap.summary_plot(
            shap_values_array,
            X_test.values,
            feature_names=features,
            show=False,
        )
        st.pyplot(plt.gcf(), bbox_inches="tight")
        plt.clf()
    except Exception as e:
        st.error(f"Summary plot error: {str(e)}")

# Corrected Feature Importance
with st.expander("📊 Top Contributing Features"):
    try:
        # Get SHAP values for selected sample
        sample_shap = shap_values_array[selected_index]

        # Create DataFrame with consistent dimensions
        df_importance = pd.DataFrame(
            {
                "Feature": features,
                "SHAP Value": sample_shap,
                "Absolute": np.abs(sample_shap),
            }
        ).sort_values("Absolute", ascending=False)

        fig = px.bar(
            df_importance,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            title="Feature Importance",
            color="SHAP Value",
            color_continuous_scale="RdBu",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Feature importance error: {str(e)}")


# Corrected SHAP Values Table
@st.cache_data
def get_shap_df():
    return pd.DataFrame(
        data=shap_values_array, columns=features, index=X_test.index
    )


try:
    shap_df = get_shap_df()
    st.download_button(
        "📥 Download SHAP Values",
        shap_df.to_csv(),
        "shap_values.csv",
        "text/csv",
    )
    if st.button("🧾 Show SHAP Values"):
        st.write(shap_df)
except Exception as e:
    st.error(f"SHAP table error: {str(e)}")
