#!/usr/bin/env python3
"""
tune_ffnn.py

Performs hyperparameter tuning for a feed-forward neural network on the AAPL options dataset
using a 1y train → 3mo val → 1y test split. Caches every model tried
and refits the best on train+val before final evaluation.
"""

import os
import json
import hashlib
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from features import get_feature_list
from preprocessing import scale_splits

# === adjust these imports to your project layout ===
from pricing_pipeline import load_parquet, filter_options, make_time_splits

# --- Configuration ---
DATA_PATH   = 'data/cleaned/aapl-options.parquet'
MODEL_DIR   = Path("models/valuate_ffnn")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Splitting parameters
TRAIN_YEARS = 1
VAL_MONTHS  = 3
TEST_YEARS  = 1
STEP_MONTHS = None  # non-overlapping tiles

# Feature/target
FEATURES = get_feature_list()
TARGET   = 'price'

# Grid for hyperparameter search
PARAM_GRID = {
    'hidden_layer_sizes': [(64,), (128), (32)],
    'activation':         ['relu', 'tanh'],
    'alpha':              [1e-4, 1e-3, 1e-2],
    'learning_rate_init': [1e-4, 1e-5],
    'batch_size':         [64, 128],
}

def param_grid():
    keys, vals = zip(*PARAM_GRID.items())
    for combo in product(*vals):
        yield dict(zip(keys, combo))

def main():
    # 1) Load & filter
    df = load_parquet(DATA_PATH)
    df = filter_options(df)

    # 2) Create time splits
    splits = make_time_splits(df,
                              train_years=TRAIN_YEARS,
                              val_months=VAL_MONTHS,
                              test_years=TEST_YEARS,
                              step_months=STEP_MONTHS)
    # concatenate non-overlapping blocks
    train = pd.concat([t for t,_,_,_ in splits])
    val   = pd.concat([v for _,v,_,_ in splits])
    test  = pd.concat([te for _,_,te,_ in splits])

    print(f"Rows » train={len(train):,}  val={len(val):,}  test={len(test):,}")

    # 3) Scale features & target using shared utility
    feat_scaler, targ_scaler = scale_splits(train, val, test, FEATURES, TARGET)

    X_train, y_train = train[FEATURES].values, train[TARGET].values.ravel()
    X_val,   y_val   = val  [FEATURES].values, val  [TARGET].values.ravel()
    X_test,  y_test  = test [FEATURES].values, test [TARGET].values.ravel()

    # 4) Hyperparameter search
    best_val_mae = np.inf
    best_params  = None
    best_tag     = None

    for params in param_grid():
        # build a short tag for caching
        tag = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        # sanitize layer sizes into a string
        layers = "_".join(str(s) for s in params['hidden_layer_sizes'])
        fname = MODEL_DIR / f"FFNN_{layers}_{params['activation']}_lr{params['learning_rate_init']}_alpha{params['alpha']}_batch{params['batch_size']}_{tag}.pkl"

        print(f"\nTrying {params} → tag={tag}")

        if fname.exists():
            model = joblib.load(fname)
            print("  ↪ loaded from cache")
        else:
            model = MLPRegressor(
                hidden_layer_sizes=params['hidden_layer_sizes'],
                activation=params['activation'],
                alpha=params['alpha'],
                learning_rate_init=params['learning_rate_init'],
                batch_size=params['batch_size'],
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42,
            )
            model.fit(X_train, y_train)
            joblib.dump(model, fname)
            print("  ✚ trained and cached")

        # Validate
        y_val_pred = model.predict(X_val)
        val_mae    = mean_absolute_error(y_val, y_val_pred)
        val_rmse   = mean_squared_error(y_val, y_val_pred, squared=False)
        print(f"  → val MAE={val_mae:.5f}, RMSE={val_rmse:.5f}")

        # Test (inverse‐scale)
        y_test_pred = model.predict(X_test).reshape(-1,1)
        y_test_true = y_test.reshape(-1,1)
        y_test_pred = targ_scaler.inverse_transform(y_test_pred).flatten()
        y_test_true = targ_scaler.inverse_transform(y_test_true).flatten()

        test_mae  = mean_absolute_error(y_test_true, y_test_pred)
        test_rmse = mean_squared_error(y_test_true, y_test_pred, squared=False)
        print(f"  → test MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")

        if test_mae < best_val_mae:
            best_val_mae = test_mae
            best_params  = params
            best_tag     = tag

    print("\n=== Best hyperparameters ===")
    print(best_params, f"(test MAE={best_val_mae:.5f})")

    # 5) Load & evaluate final model
    layers = "_".join(str(s) for s in best_params['hidden_layer_sizes'])
    best_name = MODEL_DIR / f"FFNN_{layers}_{best_params['activation']}_lr{best_params['learning_rate_init']}_alpha{best_params['alpha']}_batch{best_params['batch_size']}_{best_tag}.pkl"
    final_model = joblib.load(best_name)

    y_test_pred = final_model.predict(X_test).reshape(-1,1)
    y_test_true = y_test.reshape(-1,1)
    y_test_pred = targ_scaler.inverse_transform(y_test_pred).flatten()
    y_test_true = targ_scaler.inverse_transform(y_test_true).flatten()

    final_mae  = mean_absolute_error(y_test_true, y_test_pred)
    final_rmse = mean_squared_error(y_test_true, y_test_pred, squared=False)
    print(f"\n🏁 Final test MAE={final_mae:.4f}, RMSE={final_rmse:.4f}")

    # 6) Plot predictions for the most common option in test set
    most_common = test['option_id'].value_counts().idxmax()
    print(f"Most common option in test set: {most_common}")

    opt_df = test[test['option_id'] == most_common].copy()
    opt_df.sort_values('QUOTE_DATE', inplace=True)
    X_opt = opt_df[FEATURES].values
    y_opt_true = targ_scaler.inverse_transform(opt_df[[TARGET]]).flatten()
    y_opt_pred = final_model.predict(X_opt).reshape(-1,1)
    y_opt_pred = targ_scaler.inverse_transform(y_opt_pred).flatten()

    plt.figure(figsize=(10,5))
    plt.plot(opt_df['QUOTE_DATE'], y_opt_true,  label=f"Actual ({most_common})")
    plt.plot(opt_df['QUOTE_DATE'], y_opt_pred,  label="Predicted", linestyle='--')
    plt.title(f"Price Prediction for Option {most_common}")
    plt.xlabel("Date")
    plt.ylabel("Option Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    mae_opt  = mean_absolute_error(y_opt_true, y_opt_pred)
    rmse_opt = mean_squared_error(y_opt_true, y_opt_pred, squared=False)
    print(f"Option {most_common} MAE={mae_opt:.4f}, RMSE={rmse_opt:.4f}")

if __name__ == "__main__":
    main()
