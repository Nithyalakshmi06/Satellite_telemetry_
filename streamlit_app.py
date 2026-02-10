import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

from src.preprocessing import normalize_data, create_sequences
from src.detect_anomalies import detect_anomalies, calculate_threshold
from src.metrics import reconstruction_metrics
from src.lstm_autoencoder import build_lstm_autoencoder


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Satellite Telemetry Anomaly Detection",
    layout="centered"
)

st.title("🛰️ Satellite Telemetry Anomaly Detection")
st.write(
    "Upload a satellite telemetry `.npy` file to detect anomalies "
    "using an LSTM Autoencoder."
)

TIMESTEPS = 50


# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader(
    "Upload telemetry file (.npy)", type=["npy"]
)

if uploaded_file is not None:

    # -------------------------------
    # LOAD & PREPROCESS DATA
    # -------------------------------
    data = np.load(uploaded_file)
    st.write("📊 Raw data shape:", data.shape)

    data = normalize_data(data)
    X = create_sequences(data, TIMESTEPS)

    st.write("🔢 Sequence shape:", X.shape)

    features = X.shape[2]
    st.write(f"🧠 Detected sensors: {features}")


    # -------------------------------
    # SELECT MODEL BASED ON FEATURES
    # -------------------------------
    model_path = f"models/lstm_autoencoder_{features}.h5"

    if not os.path.exists(model_path):
        st.error(
            f"No trained model found for {features} sensors.\n\n"
            f"Expected file: `{model_path}`"
        )
        st.stop()

    model = build_lstm_autoencoder(
        timesteps=TIMESTEPS,
        features=features
    )

    model.load_weights(model_path)
    st.success(f"Loaded model: {model_path}")


    # -------------------------------
    # ANOMALY DETECTION
    # -------------------------------
    threshold = calculate_threshold(model, X)
    errors, anomalies = detect_anomalies(model, X, threshold)

    if anomalies.any():
        st.error("🚨 Anomalous behavior detected")
    else:
        st.success("✅ Telemetry data appears normal")


    # -------------------------------
    # PLOT
    # -------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(errors, label="Reconstruction Error")
    ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
    ax.set_title("Anomaly Detection Result")
    ax.set_xlabel("Time Window")
    ax.set_ylabel("Reconstruction Error")
    ax.legend()
    st.pyplot(fig)


    # -------------------------------
    # METRICS
    # -------------------------------
    reconstructed = model.predict(X)
    mse, rmse = reconstruction_metrics(X, reconstructed)


    # -------------------------------
    # SUMMARY
    # -------------------------------
    total_windows = len(errors)
    anomaly_windows = int((errors > threshold).sum())
    percentage = (anomaly_windows / total_windows) * 100

    if percentage < 5:
        severity = "LOW"
    elif percentage < 20:
        severity = "MEDIUM"
    else:
        severity = "HIGH"

    st.markdown("## 📊 Analysis Summary")

    st.metric("Reconstruction MSE", f"{mse:.4f}")
    st.metric("Reconstruction RMSE", f"{rmse:.4f}")

    st.write(f"**Total Windows:** {total_windows}")
    st.write(f"**Anomalous Windows:** {anomaly_windows}")
    st.write(f"**Anomaly Percentage:** {percentage:.2f}%")

    if severity == "LOW":
        st.success(f"Severity Level: {severity}")
    elif severity == "MEDIUM":
        st.warning(f"Severity Level: {severity}")
    else:
        st.error(f"Severity Level: {severity}")
