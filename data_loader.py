import numpy as np
import os

def load_npy(file_path):
    """
    Load a single .npy telemetry file
    Shape: (time_steps, features)
    """
    return np.load(file_path)

