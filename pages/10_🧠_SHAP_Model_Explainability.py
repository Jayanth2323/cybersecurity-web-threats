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

st.set_page_config(page_title="SHAP Explainability", layout="wide")
st.title("🧠 SHAP Explainability: Why This Was Flagged")


# Load & clean data
@st.cache_data
def load_data():
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


df = load_data()
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
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

# SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Sample selection
st.markdown("### 🔎 Select a Sample for Explanation")
selected_index = st.number_input(
    f"Row Index (0 - {len(X_test) - 1})",
    min_value=0,
    max_value=len(X_test) - 1,
    step=1,
)
sample = X_test.iloc[[selected_index]]
sample_pred = model.predict(sample)[0]
label = "Suspicious" if sample_pred == 1 else "Normal"
st.markdown(f"**Prediction for selected row:** `{label}`")

# SHAP Explanation (Safe Static Plot)
st.markdown("### 📉 SHAP Force Plot (Static)")
sample_shap = shap_values[selected_index]
base_value = (
    explainer.expected_value[0]
    if isinstance(explainer.expected_value, (list, np.ndarray))
    else explainer.expected_value
)

explanation = shap.Explanation(
    values=sample_shap.values,
    base_values=base_value,
    data=sample.values[0],
    feature_names=sample.columns.tolist(),
)

fig, ax = plt.subplots(figsize=(10, 1))
shap.plots.force(
    base_value=explanation.base_values,  # Corrected order of arguments
    shap_values=explanation.values,
    features=explanation.data,
    feature_names=explanation.feature_names,
    matplotlib=True,
    show=False,
)
plt.tight_layout()
st.pyplot(fig)

# Top Features Chart
with st.expander("📊 Top Contributing Features", expanded=False):
    shap_row = shap_values[selected_index].values
    abs_shap_values = np.abs(shap_row)
    feature_importance = pd.DataFrame(
        {
            "Feature": sample.columns,
            "SHAP Value": shap_row,
            "Absolute SHAP": abs_shap_values,
        }
    ).sort_values(by="Absolute SHAP", ascending=False)

    fig_bar = px.bar(
        feature_importance,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        title="Top Contributing Features",
        color="SHAP Value",
        color_continuous_scale="RdBu",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Summary Plot
with st.expander("📈 SHAP Summary Plot (All Samples)"):
    summary_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
    summary_df["Predicted"] = model.predict(X_test)
    melted = summary_df.melt(
        id_vars="Predicted", var_name="Feature", value_name="SHAP Value"
    )

    fig_summary = px.strip(
        melted,
        x="SHAP Value",
        y="Feature",
        color="Predicted",
        title="SHAP Summary (All Samples)",
        orientation="h",
        stripmode="overlay",
    )
    st.plotly_chart(fig_summary, use_container_width=True)


# SHAP CSV Export
@st.cache_data
def get_shap_values_df():
    return pd.DataFrame(shap_values.values, columns=X_test.columns)


shap_values_df = get_shap_values_df()
csv = shap_values_df.to_csv(index=False)
st.download_button(
    label="📥 Download SHAP Values (CSV)",
    data=csv,
    file_name="shap_values.csv",
    mime="text/csv",
)

# Optional full table
if st.button("🧾 Show All SHAP Values Table"):
    st.write(shap_values_df)
