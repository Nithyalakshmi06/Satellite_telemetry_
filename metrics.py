import numpy as np

def reconstruction_metrics(X, X_reconstructed):
    mse = np.mean(np.square(X - X_reconstructed))
    rmse = np.sqrt(mse)
    return mse, rmse
