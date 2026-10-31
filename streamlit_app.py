import os
import io

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

TIMESTEPS = 50
MAX_ROWS = 10000          # keeps memory low on free hosting
HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Satellite Telemetry Anomaly Detection", layout="centered")


# ---------------------------------------------------------------
# Core functions (same logic as your src/ files, kept in one file)
# ---------------------------------------------------------------
def normalize_data(data):
    return MinMaxScaler().fit_transform(data)


def create_sequences(data, window_size=TIMESTEPS):
    return np.array([data[i:i + window_size] for i in range(len(data) - window_size)],
                    dtype="float32")


def build_lstm_autoencoder(timesteps, features):
    inputs = Input(shape=(timesteps, features))
    encoded = LSTM(64, activation="relu", return_sequences=False)(inputs)
    encoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(64, activation="relu", return_sequences=True)(encoded)
    decoded = TimeDistributed(Dense(features))(decoded)
    model = Model(inputs, decoded)
    model.compile(optimizer="adam", loss="mse")
    return model


def find_model_file(features):
    for folder in (HERE, os.path.join(HERE, "models")):
        path = os.path.join(folder, f"lstm_autoencoder_{features}.h5")
        if os.path.exists(path):
            return path
    return None


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(features):
    path = find_model_file(features)
    if path is None:
        return None, None
    model = build_lstm_autoencoder(TIMESTEPS, features)
    model.load_weights(path)
    return model, os.path.basename(path)


def make_demo_data(features, seed=7, n=3000):
    """Synthetic SMAP-like telemetry: channel 0 = analog signal, the other
    channels stay quiet (like idle command flags). Three faults are injected
    into channel 0 so the detector has something to find."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    data = np.zeros((n, features), dtype="float32")
    data[:, 0] = np.sin(t / 40) + 0.5 * np.sin(t / 13) + rng.normal(0, 0.05, n)
    data[1200:1260, 0] += 2.5                       # level shift
    data[2100:2130, 0] = rng.normal(0, 1.5, 30)     # noisy burst
    data[2600:2610, 0] -= 3.0                       # short dropout
    return data


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🛰️ Satellite Telemetry Anomaly Detection")
st.write("Detect anomalies in satellite telemetry using an **LSTM Autoencoder** "
         "trained on NASA SMAP data. High reconstruction error = unusual behaviour.")

mode = st.radio("Choose data source",
                ["Use demo data", "Upload my own .npy file"], horizontal=True)

data = None
if mode == "Use demo data":
    sensors = st.selectbox("Number of sensors (matches a trained model)", [25, 55])
    data = make_demo_data(sensors)
    st.info("Demo data is synthetic (SMAP-like) with 3 faults injected around "
            "time steps ~1200, ~2100 and ~2600.")
    buf = io.BytesIO()
    np.save(buf, data)
    st.download_button("⬇️ Download this demo as .npy (to test the upload option)",
                       buf.getvalue(), file_name=f"demo_telemetry_{sensors}.npy")
else:
    up = st.file_uploader("Upload telemetry file (.npy, shape: time × sensors)", type=["npy"])
    if up is not None:
        try:
            data = np.load(up, allow_pickle=False)
        except Exception as e:
            st.error(f"Could not read this file as a .npy array: {e}")
            st.stop()

if data is None:
    st.stop()

# ---- validate ----
if data.ndim == 1:
    data = data.reshape(-1, 1)
if data.ndim != 2:
    st.error(f"Expected a 2-D array (time × sensors), got shape {data.shape}.")
    st.stop()
if not np.isfinite(data).all():
    st.warning("File contains NaN/inf values — replacing them with 0.")
    data = np.nan_to_num(data)
if len(data) > MAX_ROWS:
    st.warning(f"File is long; using the first {MAX_ROWS} rows only.")
    data = data[:MAX_ROWS]
if len(data) <= TIMESTEPS:
    st.error(f"Need more than {TIMESTEPS} time steps, got {len(data)}.")
    st.stop()

st.write("📊 Raw data shape:", data.shape)

features = data.shape[1]
model, model_name = load_model(features)
if model is None:
    st.error(f"No trained model found for {features} sensors. "
             "Available models: 25 sensors (SMAP) and 55 sensors (MSL).")
    st.stop()

X = create_sequences(normalize_data(data))
st.write("🔢 Sequence shape:", X.shape)
st.write(f"🧠 Detected sensors: {features}")
st.success(f"Loaded model: {model_name}")

# ---- predict once, reuse everywhere ----
with st.spinner("Running anomaly detection..."):
    recon = model.predict(X, batch_size=128, verbose=0)

errors = np.mean(np.square(X - recon), axis=(1, 2))
threshold = errors.mean() + 3 * errors.std()
anomaly_mask = errors > threshold
mse = float(np.mean(np.square(X - recon)))
rmse = float(np.sqrt(mse))

if anomaly_mask.any():
    st.error("🚨 Anomalous behavior detected")
else:
    st.success("✅ Telemetry data appears normal")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(errors, label="Reconstruction Error")
ax.scatter(np.where(anomaly_mask)[0], errors[anomaly_mask], color="red", s=12,
           label="Anomaly points", zorder=3)
ax.axhline(threshold, color="black", linestyle="--", label="Threshold")
ax.set_title("Anomaly Detection Result")
ax.set_xlabel("Time Window")
ax.set_ylabel("Reconstruction Error")
ax.legend()
st.pyplot(fig)
plt.close(fig)

total = len(errors)
n_anom = int(anomaly_mask.sum())
pct = n_anom / total * 100
severity = "LOW" if pct < 5 else "MEDIUM" if pct < 20 else "HIGH"

st.markdown("## 📊 Analysis Summary")
c1, c2 = st.columns(2)
c1.metric("Reconstruction MSE", f"{mse:.4f}")
c2.metric("Reconstruction RMSE", f"{rmse:.4f}")
st.write(f"**Total Windows:** {total}")
st.write(f"**Anomalous Windows:** {n_anom}")
st.write(f"**Anomaly Percentage:** {pct:.2f}%")
{"LOW": st.success, "MEDIUM": st.warning, "HIGH": st.error}[severity](f"Severity Level: {severity}")

st.caption("Threshold = mean + 3 × standard deviation of reconstruction error.")
