
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM  # new
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from lnai.core.pricing import binomial_price, bs_price, mc_price
from lnai.core.pricing import filter_options, load_parquet, make_time_splits

# --- Configuration --- #
DATA_PATH = 'data/cleaned/aapl-options.parquet' # Placeho   lder path
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 30
ID = 9
SEEDS = [0, 1, 2, 3, 4]


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)

# --- Data Loading and Preprocessing --- #
def load_and_preprocess_data(data_path):
    df = load_parquet(data_path)
    df_all = filter_options(df)
    splits = make_time_splits(df_all,
                              train_years=1,
                              val_months=3,
                              test_years=1,
                              step_months=27)
    return splits

# --- Model Definitions --- #
def build_ffnn_model(input_dim):
    model = Sequential([
        Dense(8, activation='relu', input_dim=input_dim),
        Dropout(0.08),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm_model(input_dim, timesteps=1):
    model = Sequential([
        LSTM(4, activation='tanh', input_shape=(timesteps, input_dim)),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Main Execution --- #
if __name__ == "__main__":
    splits = load_and_preprocess_data(DATA_PATH)
    train_df = pd.concat([t for t,_,_,_ in splits])
    val_df   = pd.concat([v for _,v,_,_ in splits])
    test_df  = pd.concat([te for _,_,te,_ in splits])# collect all the date‐dicts
    dates_list = [d for _,_,_,d in splits]

    # Define features and target
    features = ['is_call', 'moneyness', 'ttm', 
                'DELTA', 'GAMMA', 'VEGA', 'THETA', 'RHO', 'IV', 'VOLUME']
    features = ['is_call', 'moneyness', 'ttm', 'IV']
    target = 'price'

    # Chronological splits
    # train_df = df[df['QUOTE_DATE'].between('2016-01-01','2019-12-31')].copy()
    # val_df   = df[df['QUOTE_DATE'].between('2020-01-01','2020-12-31')].copy()
    # test_df  = df[df['QUOTE_DATE'].between('2021-01-01','2023-12-31')].copy()
    print(f"Rows \u00bb train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # Scale features and target
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_df[features] = feature_scaler.fit_transform(train_df[features])
    val_df[features]   = feature_scaler.transform(val_df[features])
    test_df[features]  = feature_scaler.transform(test_df[features])

    train_df[[target]] = target_scaler.fit_transform(train_df[[target]])
    val_df[[target]]   = target_scaler.transform(val_df[[target]])
    test_df[[target]]  = target_scaler.transform(test_df[[target]])

    X_train, y_train = train_df[features].values, train_df[target].values
    X_val, y_val = val_df[features].values, val_df[target].values
    X_test, y_test = test_df[features].values, test_df[target].values

    results = {name: {"mae": [], "rmse": []} for name in ["FFNN", "XGBoost", "LSTM"]}

    os.makedirs("models", exist_ok=True)

    n_feats = len(features)
    X_train_l = X_train.reshape(-1, 1, n_feats)
    X_val_l   = X_val.reshape(-1,   1, n_feats)
    X_test_l  = X_test.reshape(-1,  1, n_feats)

    for seed in SEEDS:
        set_seed(seed)
        models = {
            "FFNN": build_ffnn_model(X_train.shape[1]),
            "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=6000,
                                         learning_rate=0.0013, n_jobs=-1, random_state=seed),
            "LSTM": build_lstm_model(X_train.shape[1], timesteps=1),
        }

        for name, model in models.items():
            print(f"\n--- Seed {seed} Training/Loading {name} Model ---")

            model_exists = False
            if name == "FFNN" and os.path.exists(f"models/valuate/{name}_model_{ID}_seed{seed}.keras"):
                try:
                    model = tf.keras.models.load_model(f"models/valuate/{name}_model_{ID}_seed{seed}.keras")
                    model_exists = True
                    print(f"Loaded existing {name} model for seed {seed}")
                except Exception:
                    print(f"Error loading {name} model, will retrain")
            elif name == "XGBoost" and os.path.exists(f"models/valuate/{name}_model_{ID}_seed{seed}.json"):
                try:
                    model = xgb.XGBRegressor()
                    model.load_model(f"models/valuate/{name}_model_{ID}_seed{seed}.json")
                    model_exists = True
                    print(f"Loaded existing {name} model for seed {seed}")
                except Exception:
                    print(f"Error loading {name} model, will retrain")
            elif name == "LSTM" and os.path.exists(f"models/valuate/{name}_model_{ID}_seed{seed}.keras"):
                try:
                    model = tf.keras.models.load_model(f"models/valuate/{name}_model_{ID}_seed{seed}.keras")
                    model_exists = True
                    print(f"Loaded existing {name} model for seed {seed}")
                except Exception:
                    print(f"Error loading {name} model, will retrain")

            if not model_exists:
                print(f"Training new {name} model (seed {seed})")
                if name == "FFNN":
                    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
                    model.fit(X_train, y_train,
                              epochs=EPOCHS,
                              batch_size=BATCH_SIZE,
                              validation_data=(X_val, y_val),
                              callbacks=[early_stopping],
                              verbose=1)
                    model.save(f"models/valuate/{name}_model_{ID}_seed{seed}.keras")
                elif name == "XGBoost":
                    model.fit(X_train, y_train,
                              eval_set=[(X_val, y_val)],
                              verbose=False)
                    model.save_model(f"models/valuate/{name}_model_{ID}_seed{seed}.json")
                elif name == "LSTM":
                    model.fit(X_train_l, y_train,
                              validation_data=(X_val_l, y_val),
                              epochs=EPOCHS//2, batch_size=BATCH_SIZE,
                              verbose=1)
                    model.save(f"models/valuate/{name}_model_{ID}_seed{seed}.keras")

            print(f"\n--- Evaluating {name} Model (seed {seed}) ---")
            if name == "LSTM":
                y_pred_scaled = model.predict(X_test_l).flatten()
            else:
                y_pred_scaled = model.predict(X_test).flatten()

            y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
            y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1))

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            results[name]["mae"].append(mae)
            results[name]["rmse"].append(rmse)

            print(f"{name} MAE: {mae:.4f}")
            print(f"{name} RMSE: {rmse:.4f}")

    for name, stats in results.items():
        mae_arr = np.array(stats["mae"])
        rmse_arr = np.array(stats["rmse"])
        print(f"\n{name} MAE mean={mae_arr.mean():.4f} std={mae_arr.std(ddof=1):.4f}")
        print(f"{name} RMSE mean={rmse_arr.mean():.4f} std={rmse_arr.std(ddof=1):.4f}")

    # reload models from the first seed for further analysis
    seed0 = SEEDS[0]
    ffnn_model = tf.keras.models.load_model(f"models/valuate/FFNN_model_{ID}_seed{seed0}.keras")
    xgb_model  = xgb.XGBRegressor()
    xgb_model.load_model(f"models/valuate/XGBoost_model_{ID}_seed{seed0}.json")
    lstm_model = tf.keras.models.load_model(f"models/valuate/LSTM_model_{ID}_seed{seed0}.keras")

    # 1) Recover the original test set (unscaled) for BS/binomial/mc
    orig_df = load_parquet(DATA_PATH)
    orig_df = filter_options(orig_df)
    splits_orig = make_time_splits(orig_df,
                                train_years=1,
                                val_months=3,
                                test_years=1,
                                step_months=27)
    test_orig = pd.concat([te for _,_,te,_ in splits_orig]).reset_index(drop=True)

    # 2) True prices
    y_true_all = test_orig['price'].values

    # 3) BS / Binomial / Monte Carlo on full test set
    bs_all  = test_orig.apply(lambda r: bs_price(
                    r.UNDERLYING_LAST, r.STRIKE, r.ttm,
                    0.06, sigma=r.IV, is_call=r.is_call), axis=1)
    bin_all = test_orig.apply(lambda r: binomial_price(r, r_flat=0.06, steps=1000), axis=1)
    mc_all  = test_orig.apply(lambda r: mc_price(r, r_flat=0.06, num_paths=100), axis=1)

    # 4) FFNN / XGB on full test set (we already have X_test & target_scaler)
    y_ffnn_s_all = ffnn_model.predict(X_test).flatten()
    y_xgb_s_all  = xgb_model.predict(X_test)
    y_lstm_s_all  = lstm_model.predict(X_test_l).flatten()
    y_ffnn_all   = target_scaler.inverse_transform(y_ffnn_s_all.reshape(-1,1)).flatten()
    y_xgb_all    = target_scaler.inverse_transform(y_xgb_s_all.reshape(-1,1)).flatten()
    y_lstm_all   = target_scaler.inverse_transform(y_lstm_s_all.reshape(-1,1)).flatten()

    # 5) Build summary table
    results = []
    for name, preds in [
        ("Black–Scholes",   bs_all),
        ("Binomial (CRR)",  bin_all),
        ("Monte Carlo",     mc_all),
         ("LSTM", y_lstm_all),   # new
        ("FFNN",            y_ffnn_all),
        ("XGBoost",         y_xgb_all),
    ]:
        mae  = mean_absolute_error(y_true_all, preds)
        rmse = np.sqrt(mean_squared_error(y_true_all, preds))
        results.append({"Model": name, "MAE": mae, "RMSE": rmse})

    summary_df = pd.DataFrame(results)

    print("\nOverall Test-Set Performance:")
    print(summary_df.to_string(index=False))

    for i, (train_i, val_i, test_i, _) in enumerate(splits, start=1):
        # pick the single most-common contract
        top_contract = test_i['option_id'].value_counts().idxmax()
        df_ct = (test_i[test_i['option_id'] == top_contract]
                .sort_values('QUOTE_DATE')
                .reset_index(drop=True))

        # 1) actual dates & prices
        dates_plot = df_ct['QUOTE_DATE']
        y_true     = df_ct['price'].values

        # 2) FFNN + XGB predictions
        X_ct   = feature_scaler.transform(df_ct[features].values)
        y_ffnn = ffnn_model.predict(X_ct).flatten()
        y_xgb  = xgb_model.predict(X_ct)
        y_ffnn = target_scaler.inverse_transform(y_ffnn.reshape(-1,1)).flatten()
        y_xgb  = target_scaler.inverse_transform(y_xgb.reshape(-1,1)).flatten()
        y_lstm = lstm_model.predict(X_ct.reshape(-1,1,n_feats)).flatten()
        y_lstm = target_scaler.inverse_transform(y_lstm.reshape(-1,1)).flatten()

        # 3) Black–Scholes, Binomial & Monte-Carlo
        y_bs  = df_ct.apply(lambda r: bs_price(
                    r.UNDERLYING_LAST,
                    r.STRIKE,
                    r.ttm,
                    0.06,
                    sigma=r.IV,
                    is_call=r.is_call
                ), axis=1)
        y_bin = df_ct.apply(lambda r: binomial_price(
                    r,
                    r_flat=0.06,
                    steps=1000
                ), axis=1)
        y_mc  = df_ct.apply(lambda r: mc_price(
                    r,
                    r_flat=0.06,
                    num_paths=100
                ), axis=1)

        # 4) plot them all against the SAME dates_plot
        plt.figure(figsize=(12,5))
        plt.plot(dates_plot, y_true, marker='o', label='Actual')
        plt.plot(dates_plot, y_ffnn, marker='x', label='FFNN')
        plt.plot(dates_plot, y_xgb,  marker='s', label='XGBoost')
        plt.plot(dates_plot, y_bs,   marker='^', label='Black–Scholes')
        plt.plot(dates_plot, y_bin,  marker='v', label='Binomial (CRR)')
        plt.plot(dates_plot, y_mc,   marker='d', label='Monte Carlo')
        plt.plot(dates_plot, y_lstm, marker='d', label='LSTM')
        plt.title(f"Window {i} — Contract {top_contract}")
        plt.xlabel("Quote Date")
        plt.ylabel("Option Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    print("nScript execution complete. Please replace 'aapl-options.parquet' with your actual data file.")


