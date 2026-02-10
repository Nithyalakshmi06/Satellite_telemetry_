from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

def build_lstm_autoencoder(timesteps, features):
    inputs = Input(shape=(timesteps, features))
    
    encoded = LSTM(64, activation="relu", return_sequences=False)(inputs)
    encoded = RepeatVector(timesteps)(encoded)
    
    decoded = LSTM(64, activation="relu", return_sequences=True)(encoded)
    decoded = TimeDistributed(Dense(features))(decoded)
    
    model = Model(inputs, decoded)
    model.compile(optimizer="adam", loss="mse")
    return model

