import streamlit as st
import pandas as pd
import shap
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
st.title("🔍 SHAP Explainability: Why This Was Flagged")


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
st.write("### 📊 Model Evaluation:")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

# SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Row selector
st.markdown("### 🧪 Select a Sample for Explanation")
selected_index = st.number_input(
    f"Select Row Index (0 - {len(X_test) - 1})",
    min_value=0,
    max_value=len(X_test) - 1,
    step=1,
    key="row_index",
)

# Display Prediction
sample_pred = model.predict([X_test.iloc[selected_index]])[0]
label = "Suspicious" if sample_pred == 1 else "Normal"
st.markdown(f"**Prediction for this row:** `{label}`")

# SHAP Force Plot
st.markdown("### 🔬 SHAP Force Plot Explanation")
shap.initjs()
html = shap.plots.force(
    explainer.expected_value[1],
    shap_values[selected_index].values,
    X_test.iloc[selected_index],
    matplotlib=False,
).html()

components.html(html, height=300)
