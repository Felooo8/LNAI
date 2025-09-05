"""CNN, LSTM and GRU baselines for option price forecasting.

This module trains a handful of lightweight deep learning models on the
cleaned option dataset.  It demonstrates how to obtain the standardized
feature set via :func:`features.get_feature_list` and how to normalize the
train/validation/test splits using :func:`preprocessing.scale_splits`.
The resulting metrics are written to a CSV or JSON file compatible with the
aggregation utilities in this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import callbacks, layers, models
import random
import tensorflow as tf

from features import get_feature_list
from preprocessing import scale_splits

TARGET = "price"
SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 5
SEEDS = [0, 1, 2, 3, 4]


def set_seed(seed: int) -> None:
    """Seed RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def _time_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Slice the dataframe into train/val/test ranges."""

    train = df[df["QUOTE_DATE"].between("2016-01-01", "2019-12-31")].copy()
    val = df[df["QUOTE_DATE"].between("2020-01-01", "2020-12-31")].copy()
    test = df[df["QUOTE_DATE"] >= "2021-01-01"].copy()
    return train, val, test


def _build_sequences(df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return rolling sequences and targets for sequential models."""

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    for _, grp in df.groupby("option_id", sort=False):
        if len(grp) <= SEQ_LEN:
            continue
        X = grp[features].values
        y = grp["target"].values
        for j in range(SEQ_LEN - 1, len(grp)):
            X_list.append(X[j - SEQ_LEN + 1 : j + 1])
            y_list.append(y[j])
    return np.stack(X_list), np.array(y_list)


def load_dataset(path: str, horizon: int):
    """Load dataset and produce scaled train/val/test splits with sequences."""

    df = pd.read_parquet(path)
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])
    df.sort_values(["option_id", "QUOTE_DATE"], inplace=True)

    # Shift future price by the desired horizon
    df["target"] = df.groupby("option_id")[TARGET].shift(-horizon)
    df.dropna(subset=["target"], inplace=True)

    train_df, val_df, test_df = _time_splits(df)

    features = get_feature_list(forecasting=True)
    _, targ_scaler = scale_splits(train_df, val_df, test_df, features, "target")

    X_train, y_train = _build_sequences(train_df, features)
    X_val, y_val = _build_sequences(val_df, features)
    X_test, y_test = _build_sequences(test_df, features)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), targ_scaler


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_cnn(input_shape: Tuple[int, int]) -> models.Model:
    model = models.Sequential(
        [
            layers.Conv1D(16, 3, activation="relu", input_shape=input_shape),
            layers.MaxPooling1D(),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def build_lstm(input_shape: Tuple[int, int]) -> models.Model:
    model = models.Sequential(
        [layers.LSTM(32, input_shape=input_shape), layers.Dense(1)]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def build_gru(input_shape: Tuple[int, int]) -> models.Model:
    model = models.Sequential(
        [layers.GRU(32, input_shape=input_shape), layers.Dense(1)]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_and_eval(
    builder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
) -> Dict[str, float]:
    """Train ``builder`` model and return MAE/RMSE on the test split."""

    model = builder(input_shape=X_train.shape[1:])
    cb = callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[cb],
        verbose=0,
    )
    pred = model.predict(X_test, verbose=0).squeeze()
    y_pred = scaler.inverse_transform(pred.reshape(-1, 1)).squeeze()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).squeeze()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"mae": mae, "rmse": rmse}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    global EPOCHS, BATCH_SIZE

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-path",
        default="data/cleaned/aapl-options.parquet",
        help="Path to cleaned option dataset",
    )
    p.add_argument(
        "--output",
        default="baselines.json",
        help="Where to store aggregated metrics (extension decides JSON/CSV)",
    )
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = p.parse_args()
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    results: List[Dict[str, float]] = []
    for horizon in (7, 30):
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = load_dataset(
            args.data_path, horizon
        )
        builders = {
            "cnn": build_cnn,
            "lstm": build_lstm,
            "gru": build_gru,
        }
        for name, builder in builders.items():
            seed_metrics = []
            for seed in SEEDS:
                set_seed(seed)
                metrics = train_and_eval(
                    builder,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    scaler,
                )
                seed_metrics.append(metrics)
            avg_mae = float(np.mean([m["mae"] for m in seed_metrics]))
            avg_rmse = float(np.mean([m["rmse"] for m in seed_metrics]))
            results.append(
                {
                    "model": name,
                    "horizon": horizon,
                    "seeds": len(SEEDS),
                    "mae": avg_mae,
                    "rmse": avg_rmse,
                }
            )

    out_path = Path(args.output)
    df = pd.DataFrame(results)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_json(out_path, orient="records", indent=2)
    print(df)


if __name__ == "__main__":
    main()
