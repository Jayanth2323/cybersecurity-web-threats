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
        st.error(
            """Error: The file 'data/analyzed_output.csv' was not found.
            Please make sure the file is in the correct location."""
        )
        return None


df = load_data()
if df is None:
    st.stop()

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
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Sample selection
st.markdown("### 🔎 Select a Sample for Explanation")
selected_index = st.number_input(
    f"Row Index (0 - {len(X_test) - 1})",
    min_value=0,
    max_value=len(X_test) - 1,
    step=1,
)
sample = X_test.iloc[selected_index:selected_index + 1]
sample_pred = model.predict(sample)[0]
label = "Suspicious" if sample_pred == 1 else "Normal"
st.markdown(f"**Prediction for selected row:** `{label}`")

# SHAP Explanation (Static Plot)
# SHAP Explanation (Static Plot)
st.markdown("### 📉 SHAP Force Plot (Static)")
plt.figure()
try:
    # Debugging info
    st.write(
        f"""
        SHAP values shape for sample:
        {np.array(shap_values[1][selected_index]).shape}"""
        )
    st.write(f"Sample values shape:{sample.iloc[0].values.shape}")
    st.write(
        f"Expected value shape:{np.array(explainer.expected_value[1]).shape}"
    )

    # Create the force plot for single sample
    shap.force_plot(
        explainer.expected_value[1],  # Base value for class 1
        shap_values[1][selected_index, :],  # SHAP values for this sample
        sample.iloc[0].values,  # Feature values
        feature_names=features,
        matplotlib=True,
        show=False,
    )
    st.pyplot(plt.gcf(), bbox_inches="tight")
    plt.clf()

    # Alternative: Create force plot for all samples (commented out)
    # plt.figure()
    # shap.force_plot(
    #     explainer.expected_value[1],
    #     shap_values[1],
    #     X_test,
    #     feature_names=features,
    #     matplotlib=True,
    #     show=False,
    # )
    # st.pyplot(plt.gcf(), bbox_inches="tight")
    # plt.clf()

except Exception as e:
    st.error(f"Error generating SHAP plot: {str(e)}")
    st.error("Common causes:")
    st.error("- Mismatch between SHAP values and feature dimensions")
    st.error("- Incorrect indexing of SHAP values array")
    st.error("- Try updating SHAP library: pip install --upgrade shap")


shap.force_plot(
    explainer.expected_value[1],
    shap_values[1],
    X_test,
    feature_names=features,
    matplotlib=True,
    show=False,
    link="logit",
)


st.write(f"Sample values shape: {sample.iloc[0].values.shape}")


# Top Features Chart
with st.expander("📊 Top Contributing Features", expanded=False):
    shap_row = shap_values[1][selected_index]
    abs_shap_values = np.abs(shap_row)
    feature_importance = pd.DataFrame(
        {
            "Feature": features,
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
    plt.figure()
    shap.summary_plot(shap_values[1], X_test, show=False)
    st.pyplot(plt.gcf(), bbox_inches="tight")
    plt.clf()


# SHAP CSV Export
@st.cache_data
def get_shap_values_df():
    return pd.DataFrame(shap_values[1], columns=features)


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
