import numpy as np

def calculate_threshold(model, X_train):
    recon = model.predict(X_train)
    errors = np.mean(np.square(X_train - recon), axis=(1, 2))
    return errors.mean() + 3 * errors.std()

def detect_anomalies(model, X_test, threshold):
    recon = model.predict(X_test)
    errors = np.mean(np.square(X_test - recon), axis=(1, 2))
    anomalies = errors > threshold
    return errors, anomalies
