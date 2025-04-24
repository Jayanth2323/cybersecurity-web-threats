import streamlit as st
import pandas as pd
import shap
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components
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

# Train Random Forest
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
st.write("### Model Evaluation:")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

# SHAP Explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Row selector
st.markdown("### Select a Sample for Explanation")
selected_index = st.number_input(
    f"Select Row Index (0 - {len(X_test) - 1})",
    min_value=0,
    max_value=len(X_test) - 1,
    step=1,
    key="row_index",
)

# Display Prediction
sample = X_test.iloc[[selected_index]]
sample_pred = model.predict(sample)[0]
label = "Suspicious" if sample_pred == 1 else "Normal"
st.markdown(f"**Prediction for this row:** `{label}`")

# SHAP Force Plot (v0.20+ compliant)
st.markdown("### SHAP Force Plot Explanation")
shap.initjs()
base_value = explainer.expected_value
if isinstance(base_value, (list, np.ndarray)):
    base_value = base_value[0]

force_plot = shap.force_plot(
    base_value,
    shap_values[selected_index],
    sample,
    feature_names=sample.columns,
    matplotlib=False,
)

shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
components.html(shap_html, height=300)

# Top contributing features (Bar Chart)
with st.expander(
    "📊 View Top Contributing Features (Bar Chart)", expanded=False
):
    shap_row = shap_values[selected_index].values
    abs_shap_values = np.abs(shap_row)
    feature_importance = pd.DataFrame(
        {
            "Feature": sample.columns,
            "SHAP Value": shap_row,
            "Absolute SHAP": abs_shap_values,
        }
    ).sort_values(by="Absolute SHAP", ascending=False)

    fig = px.bar(
        feature_importance,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        title="Top Contributing Features (Single Sample)",
        color="SHAP Value",
        color_continuous_scale="RdBu",
    )
    st.plotly_chart(fig, use_container_width=True)

# SHAP Summary Plot (Interactive Plotly)
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
        title="SHAP Summary Plot (Interactive)",
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
    label="📥 Download SHAP values as CSV",
    data=csv,
    file_name="shap_values.csv",
    mime="text/csv",
)

# Display SHAP Table
if st.button("🧾 Display All SHAP Values (Table View)"):
    st.write(shap_values_df)
