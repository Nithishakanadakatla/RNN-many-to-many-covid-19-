"""
train.py
--------
Many-to-Many Simple RNN for COVID-19 forecasting.

Input : previous N_STEPS_IN days of a feature (default: Confirmed cases)
Output: next N_STEPS_OUT days of the same feature

Usage:
    python train.py
Produces:
    covid_rnn_model.keras   - trained Keras model
    scaler.pkl              - fitted MinMaxScaler (needed by app.py)
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
CSV_PATH = "day_wise.csv"
FEATURE_COL = "Confirmed"      # change to "Active" or any other column if desired
N_STEPS_IN = 7                 # look-back window
N_STEPS_OUT = 7                # forecast horizon
TEST_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 8
RANDOM_STATE = 42
MODEL_PATH = "covid_rnn_model.keras"
SCALER_PATH = "scaler.pkl"


def load_series(csv_path: str, feature_col: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df = df.sort_values("Date").reset_index(drop=True)
    series = df[feature_col].values.astype(float).reshape(-1, 1)
    return series


def make_sequences(series: np.ndarray, n_in: int, n_out: int):
    """
    Turn a 1D series into overlapping (X, y) windows for a many-to-many model.
    X shape: (samples, n_in, 1)
    y shape: (samples, n_out)
    """
    X, y = [], []
    for i in range(len(series) - n_in - n_out + 1):
        X.append(series[i:i + n_in, 0])
        y.append(series[i + n_in:i + n_in + n_out, 0])
    return np.array(X), np.array(y)


def build_model(n_steps_in: int, n_steps_out: int) -> keras.Model:
    """Simple RNN (NOT LSTM/GRU) many-to-many regressor."""
    model = keras.Sequential([
        layers.Input(shape=(n_steps_in, 1)),
        layers.SimpleRNN(64, activation="tanh", return_sequences=True),
        layers.SimpleRNN(32, activation="tanh"),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_steps_out)   # many outputs at once (7 future values)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    print(f"Loading '{CSV_PATH}' ...")
    series = load_series(CSV_PATH, FEATURE_COL)
    print(f"Series length: {len(series)} days | Feature: {FEATURE_COL}")

    # Scale to [0, 1] - important for RNN training stability
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series)

    # Build supervised many-to-many windows
    X, y = make_sequences(series_scaled, N_STEPS_IN, N_STEPS_OUT)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    print(f"Total sequences: {X.shape[0]} | X shape: {X.shape} | y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    model = build_model(N_STEPS_IN, N_STEPS_OUT)
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2
    )

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest MSE (scaled): {test_loss:.6f} | Test MAE (scaled): {test_mae:.6f}")

    # Save model (.keras format, as requested)
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    # Save scaler + config so app.py can reproduce the same preprocessing
    with open(SCALER_PATH, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "feature_col": FEATURE_COL,
            "n_steps_in": N_STEPS_IN,
            "n_steps_out": N_STEPS_OUT
        }, f)
    print(f"Scaler + config saved to: {SCALER_PATH}")


if __name__ == "__main__":
    main()