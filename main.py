from src.data_loader import load_npy
from src.preprocessing import normalize_data, create_sequences
from src.lstm_autoencoder import build_lstm_autoencoder
from src.train import train_model
from src.detect_anomalies import calculate_threshold, detect_anomalies
from src.visualize import plot_anomalies
from src.metrics import reconstruction_metrics
import os


# ===============================
# CONFIGURATION
# ===============================
TRAIN_FILE = "data/train/A-1.npy"   # training file
TEST_FILE  = "data/test/A-2.npy"   # test file
TIMESTEPS  = 50


# ===============================
# LOAD & PREPROCESS TRAIN DATA
# ===============================
train_data = load_npy(TRAIN_FILE)
train_data = normalize_data(train_data)
X_train = create_sequences(train_data, TIMESTEPS)

timesteps = X_train.shape[1]
features  = X_train.shape[2]

print("X_train shape:", X_train.shape)
print("Detected sensors:", features)


# ===============================
# MODEL PATH (AUTO)
# ===============================
MODEL_PATH = f"models/lstm_autoencoder_{features}.h5"


# ===============================
# BUILD & TRAIN MODEL
# ===============================
model = build_lstm_autoencoder(
    timesteps=timesteps,
    features=features
)

train_model(model, X_train)

os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
print("Model saved to:", MODEL_PATH)


# ===============================
# THRESHOLD (TRAIN DATA)
# ===============================
threshold = calculate_threshold(model, X_train)
print("Anomaly threshold:", threshold)


# ===============================
# LOAD & PREPROCESS TEST DATA
# ===============================
test_data = load_npy(TEST_FILE)
test_data = normalize_data(test_data)
X_test = create_sequences(test_data, TIMESTEPS)

print("X_test shape:", X_test.shape)


# ===============================
# SAFETY CHECK
# ===============================
if X_test.shape[2] != features:
    raise ValueError(
        f"Feature mismatch: "
        f"train has {features} sensors, "
        f"test has {X_test.shape[2]} sensors"
    )


# ===============================
# ANOMALY DETECTION
# ===============================
errors, anomalies = detect_anomalies(model, X_test, threshold)


# ===============================
# VISUALIZATION
# ===============================
plot_path = plot_anomalies(errors, threshold)
print("Anomaly plot saved at:", plot_path)


# ===============================
# RECONSTRUCTION METRICS
# ===============================
reconstructed = model.predict(X_test)
mse, rmse = reconstruction_metrics(X_test, reconstructed)

print("Reconstruction MSE :", mse)
print("Reconstruction RMSE:", rmse)


# ===============================
# SUMMARY
# ===============================
total_windows = len(errors)
anomaly_windows = int((errors > threshold).sum())
percentage = (anomaly_windows / total_windows) * 100

print("\n===== SUMMARY =====")
print("Total windows      :", total_windows)
print("Anomalous windows  :", anomaly_windows)
print(f"Anomaly percentage : {percentage:.2f}%")

if percentage < 5:
    print("Severity: LOW")
elif percentage < 20:
    print("Severity: MEDIUM")
else:
    print("Severity: HIGH")

print("\n✅ Anomaly detection pipeline completed successfully")
