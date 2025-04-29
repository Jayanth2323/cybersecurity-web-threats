import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestClassifier
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

# TensorFlow/Keras for NN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("⚖️ Compare Random Forest vs Neural Network")


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
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return None


df = load_data()
if df is None:
    st.stop()

features = ["bytes_in", "bytes_out", "duration_seconds", "avg_packet_size"]
X = df[features]
y = df["anomaly_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Random Forest (baseline)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]


# Keras Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential(
    [
        Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0,
)

nn_prob = model.predict(X_test_scaled).ravel()
nn_pred = (nn_prob >= 0.5).astype(int)


def show_metrics(title, y_true, y_pred, y_prob):
    st.markdown(f"### 🔍 {title}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
    col2.metric("Precision", f"{precision_score(y_true, y_pred):.2f}")
    col3.metric("Recall", f"{recall_score(y_true, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_true, y_pred):.2f}")

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


def plot_training_history(history):
    hist = history.history
    epochs = range(len(hist["accuracy"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, hist["accuracy"], label="Training Accuracy")
    ax1.plot(epochs, hist["val_accuracy"], label="Validation Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(epochs, hist["loss"], label="Training Loss")
    ax2.plot(epochs, hist["val_loss"], label="Validation Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    st.pyplot(fig)


st.subheader("🌲 Random Forest Metrics")
show_metrics("Random Forest", y_test, rf_pred, rf_prob)

st.subheader("🧠 Neural Network Metrics")
show_metrics("Neural Network (Keras)", y_test, nn_pred, nn_prob)

st.subheader("📊 Neural Network Training History")
plot_training_history(history)
