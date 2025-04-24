import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap

# import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Initialize SHAP
shap.initjs()

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("⚖️ Compare Random Forest vs Neural Network")


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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize for NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

# Train Neural Net
nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
nn.fit(X_train_scaled, y_train)
nn_pred = nn.predict(X_test_scaled)
nn_prob = nn.predict_proba(X_test_scaled)[:, 1]


# Metrics comparison
def show_metrics(title, y_true, y_pred, y_prob):
    st.markdown(f"### 🔍 {title}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
    col2.metric("Precision", f"{precision_score(y_true, y_pred):.2f}")
    col3.metric("Recall", f"{recall_score(y_true, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_true, y_pred):.2f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{title} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)


# SHAP explainability function with dimension checks
def explain_model(model, X, X_train, model_type="rf"):
    try:
        if model_type == "rf":
            explainer = shap.TreeExplainer(model)
        else:
            # For NN, use KernelExplainer with background
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X)
        # For binary classification, return positive class values
        return shap_values[1] if isinstance(shap_values, list) else shap_values
    except Exception as e:
        st.error(f"SHAP explanation error: {str(e)}")
        return None


# Main comparison
st.subheader("🧪 Random Forest Metrics")
show_metrics("Random Forest", y_test, rf_pred, rf_prob)

st.subheader("🧠 Neural Network Metrics")
show_metrics("Neural Network", y_test, nn_pred, nn_prob)

# SHAP Analysis Section
# st.header("🔍 Model Explainability with SHAP")

# Sample selection
sample_idx = st.slider("Select sample to explain", 0, len(X_test) - 1, 0)
sample = X_test.iloc[sample_idx: sample_idx + 1]
sample_scaled = scaler.transform(sample)

# Random Forest SHAP
st.subheader("🌳 Random Forest Explanation")
try:
    rf_shap_values = explain_model(rf, X_test, X_train, "rf")

    if rf_shap_values is not None:
        # Verify dimensions
        st.write(f"SHAP values shape: {rf_shap_values.shape}")
        st.write(f"Sample shape: {sample.shape}")

        # Force plot with correct dimensions
        plt.figure()
        shap.force_plot(
            rf.predict_proba(X_train)[:, 1].mean(),
            rf_shap_values[sample_idx, :],  # Correct slicing for single sample
            sample.values[0],  # Feature values as 1D array
            feature_names=features,
            matplotlib=True,
            show=False,
        )
        st.pyplot(plt.gcf(), bbox_inches="tight")
        plt.clf()

        # Summary plot
        st.markdown("#### Feature Importance")
        plt.figure()
        shap.summary_plot(
            rf_shap_values,
            X_test.values,  # Ensure numpy array
            feature_names=features,
            show=False,
        )
        st.pyplot(plt.gcf(), bbox_inches="tight")
        plt.clf()
except Exception as e:
    st.error(f"Error explaining Random Forest: {str(e)}")

# Neural Network SHAP
st.subheader("🧠 Neural Network Explanation")
try:
    nn_shap_values = explain_model(nn, X_test_scaled, X_train_scaled, "nn")

    if nn_shap_values is not None:
        # Verify dimensions
        st.write(f"SHAP values shape: {nn_shap_values.shape}")
        st.write(f"Sample shape: {sample_scaled.shape}")

        # Force plot with correct dimensions
        plt.figure()
        shap.force_plot(
            nn.predict_proba(X_train_scaled)[:, 1].mean(),
            nn_shap_values[sample_idx, :],  # Correct slicing for single sample
            sample_scaled[0],  # Feature values as 1D array
            feature_names=features,
            matplotlib=True,
            show=False,
        )
        st.pyplot(plt.gcf(), bbox_inches="tight")
        plt.clf()

        # Summary plot
        st.markdown("#### Feature Importance")
        plt.figure()
        shap.summary_plot(
            nn_shap_values,
            X_test_scaled,  # Already numpy array
            feature_names=features,
            show=False,
        )
        st.pyplot(plt.gcf(), bbox_inches="tight")
        plt.clf()
except Exception as e:
    st.error(f"Error explaining Neural Network: {str(e)}")

# Performance comparison
st.header("📊 Performance Comparison")
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], "k--")
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_prob)
ax.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC={roc_auc_score(y_test, rf_prob):.2f}",
)
ax.plot(
    fpr_nn,
    tpr_nn,
    label=f"Neural Network (AUC={roc_auc_score(y_test, nn_prob):.2f}",
)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend()
st.pyplot(fig)
