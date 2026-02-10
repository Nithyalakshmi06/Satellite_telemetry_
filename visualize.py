import matplotlib.pyplot as plt
import numpy as np
import os


def plot_anomalies(errors, threshold):
    """
    Plots reconstruction error and highlights anomaly points.
    Saves the figure to results/plots and also returns the figure.
    """

    # Create output directory
    os.makedirs("results/plots", exist_ok=True)

    # Identify anomaly points
    anomaly_indices = errors > threshold

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(errors, label="Reconstruction Error")
    ax.scatter(
        np.where(anomaly_indices)[0],
        errors[anomaly_indices],
        color="red",
        s=15,
        label="Anomaly Points"
    )
    ax.axhline(
        threshold,
        color="black",
        linestyle="--",
        label="Threshold"
    )

    ax.set_title("Satellite Telemetry Anomaly Detection")
    ax.set_xlabel("Time Window")
    ax.set_ylabel("Reconstruction Error")
    ax.legend()

    # Save plot
    save_path = "results/plots/anomaly_detection.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # Close figure to avoid memory leaks
    plt.close(fig)

    return save_path
