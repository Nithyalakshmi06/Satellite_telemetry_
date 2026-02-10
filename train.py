from tensorflow.keras.callbacks import EarlyStopping
import os
# no src import needed inside same file
import numpy as np


def train_model(model, X_train):
    early_stop = EarlyStopping(monitor="loss", patience=5)

    model.fit(
        X_train, X_train,
        epochs=50,
        batch_size=32,
        shuffle=True,
        callbacks=[early_stop]
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_autoencoder.h5")
