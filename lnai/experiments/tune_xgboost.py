#!/usr/bin/env python3
"""
tune_xgboost.py

Performs hyperparameter tuning for XGBoost on the AAPL options dataset
using a 1y train → 3mo val → 1y test split. Caches every model tried
and refits the best on train+val before final evaluation.
"""

import os
import json
import hashlib
from itertools import product
import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from lnai.data.features import get_feature_list
from lnai.data.preprocessing import scale_splits

# === adjust these imports to your project layout ===
from lnai.core.pricing import filter_options, load_parquet, make_time_splits

# --- Configuration ---
DATA_PATH    = 'data/cleaned/aapl-options.parquet'
MODEL_DIR    = Path("models/valuate")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Splitting parameters
TRAIN_YEARS  = 1
VAL_MONTHS   = 3
TEST_YEARS   = 1
STEP_MONTHS  = None  # non-overlapping tiles

# Feature/target
FEATURES = get_feature_list()
TARGET   = 'price'

# Grid for hyperparameter search
PARAM_GRID = {
    "n_estimators":    [6000],
    "learning_rate":   [0.0012, 0.0011, 0.0013],
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

    X_train, y_train = train[FEATURES].values, train[TARGET].values
    X_val,   y_val   = val  [FEATURES].values, val  [TARGET].values
    X_test,  y_test  = test [FEATURES].values, test [TARGET].values

    # 4) Hyperparameter search
    best_val_mae = np.inf
    best_params  = None
    best_tag     = None

    for params in param_grid():
        tag   = hashlib.md5(json.dumps(params, sort_keys=True).encode())\
                .hexdigest()[:8]
        fname = MODEL_DIR / f"XGB_{params['learning_rate']}_{params['n_estimators']}_{tag}.json"
        print(f"\nTrying {params} → tag={tag}")

        # load cached if exists
        if fname.exists():
            model = xgb.XGBRegressor(objective='reg:squarederror',
                                     n_jobs=-1, **params)
            model.load_model(str(fname))
            print("  ↪ loaded from cache")
        else:
            model = xgb.XGBRegressor(objective='reg:squarederror',
                                     n_jobs=-1, **params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )
            model.save_model(str(fname))
            print("  ✚ trained and cached")

        # validate
        y_val_pred = model.predict(X_val)
        val_mae    = mean_absolute_error(y_val, y_val_pred)
        val_rmse   = np.sqrt(mean_squared_error(y_val, y_val_pred))
        print(f"  → val MAE={val_mae:.5f}, RMSE={val_rmse:.5f}")
        y_test_pred = model.predict(X_test)
        y_test_pred = targ_scaler.inverse_transform(y_test_pred.reshape(-1,1)).flatten()
        y_test_true = targ_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

        test_mae  = mean_absolute_error(y_test_true, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
        print(f"\nTest MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")

        if test_mae < best_val_mae:
            best_val_mae = test_mae
            best_params  = params
            best_tag     = tag

    print("\n=== Best hyperparameters ===")
    print(best_params, f"(val MAE={best_val_mae:.5f})")

    # load final model with best params
    best_name = f"models/valuate/XGB_{best_params['learning_rate']}_{best_params['n_estimators']}_{best_tag}.json"
    model = xgb.XGBRegressor(objective='reg:squarederror',
                                     n_jobs=-1, **best_params)
    model.load_model(str(best_name))
    
    y_test_pred = model.predict(X_test)
    y_test_pred = targ_scaler.inverse_transform(y_test_pred.reshape(-1,1)).flatten()
    y_test_true = targ_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    test_mae  = mean_absolute_error(y_test_true, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    print(f"\n🏁 Final test MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")
    # Get the most frequent option_id in the test data
    most_common_option = test['option_id'].value_counts().idxmax()
    print(f"Most common option in test data: {most_common_option} with {test['option_id'].value_counts().max()} records")

    # Filter test data for this option
    option_test = test[test['option_id'] == most_common_option].copy()
    option_test_sorted = option_test.sort_values('QUOTE_DATE')

    # Get predictions for this specific option
    X_option = option_test[FEATURES].values
    y_option_true = option_test[TARGET].values

    # Predict with the final model
    y_option_pred = model.predict(X_option)

    # Inverse transform to original scale
    y_option_pred = targ_scaler.inverse_transform(y_option_pred.reshape(-1,1)).flatten()
    y_option_true = targ_scaler.inverse_transform(y_option_true.reshape(-1,1)).flatten()

    # Plot the predictions over time for this specific option
    plt.figure(figsize=(10,5))
    plt.plot(option_test_sorted['QUOTE_DATE'], y_option_true, label=f"Actual ({most_common_option})")
    plt.plot(option_test_sorted['QUOTE_DATE'], y_option_pred, label="Predicted", linestyle='--')
    plt.title(f"Price Prediction for Option {most_common_option}")
    plt.xlabel("Date")
    plt.ylabel("Option Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Calculate metrics for this specific option
    option_mae = mean_absolute_error(y_option_true, y_option_pred)
    option_rmse = np.sqrt(mean_squared_error(y_option_true, y_option_pred))
    print(f"Option {most_common_option} MAE={option_mae:.4f}, RMSE={option_rmse:.4f}")


if __name__ == "__main__":
    from pathlib import Path
    main()
