import pickle
from pathlib import Path
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Informer2020.models.model import Informer
import matplotlib.pyplot as plt
from features import get_feature_list
from preprocessing import scale_splits

from pricing_pipeline import load_parquet, filter_options, make_time_splits

TARGET = 'price'
SEQ_LEN = 30            # look-back window
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2, 3, 4]


def make_parser() -> argparse.ArgumentParser:
    """Return an argument parser pre-configured for training."""

    p = argparse.ArgumentParser(description="Train the Informer baseline")
    p.add_argument("--horizon", type=int, default=30, help="Forecast horizon")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    p.add_argument("--run-id", type=int, default=44, help="Model cache identifier")
    p.add_argument("--data-id", type=int, default=22, help="Dataset cache identifier")
    p.add_argument(
        "--data-path",
        default="data/cleaned/aapl-options.parquet",
        help="Path to input dataset",
    )
    p.add_argument(
        "--cache-root",
        default="cache_inf_forecast",
        help="Directory for cached datasets and models",
    )
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



# ------------  DATA & SPLITS  -------------------------------
def prepare_splits(path: str, horizon: int):
    """Return train/val/test splits for a given horizon."""

    min_ttm_days = horizon + 30
    df = filter_options(
        load_parquet(path), min_data_points=61, min_ttm_days=min_ttm_days
    )
    df = df.sort_values("QUOTE_DATE").reset_index(drop=True)

    # 2) tile into back-to-back 1y / 3mo / 1y blocks
    splits = make_time_splits(df,
                              train_years=1,
                              val_months=3,
                              test_years=1,
                              step_months=1)

    # 3) concat each block type
    train_df = pd.concat([t for t,_,_,_ in splits]).reset_index(drop=True)
    val_df   = pd.concat([v for _,v,_,_ in splits]).reset_index(drop=True)
    test_df  = pd.concat([te for _,_,te,_ in splits]).reset_index(drop=True)

    return train_df, val_df, test_df, splits

# ------------  DATA & SPLITS  -------------------------------
def prepare_splits_simple(path: str, horizon: int, asset: str = 'aapl'):
    """Simpler deterministic time windows used by trading script."""

    min_ttm_days = horizon + 30
    df = filter_options(
        load_parquet(path), min_data_points=61, asset=asset, min_ttm_days=min_ttm_days
    )
    # 2) ensure datetime
    df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'])
    # 3) slice by explicit windows
    if asset == 'btc':
        train = df[df.QUOTE_DATE.between('2021-06-01','2023-01-31')].copy()
        val   = df[df.QUOTE_DATE.between('2023-02-01','2023-07-31')].copy()
        test  = df[df.QUOTE_DATE.between('2023-08-01','2024-09-30')].copy()
    else:
        train = df[df.QUOTE_DATE.between('2016-01-01', '2019-12-31')].copy()
        val   = df[df.QUOTE_DATE.between('2020-01-01', '2020-12-31')].copy()
        test  = df[df.QUOTE_DATE.between('2021-01-01', '2023-12-31')].copy()
    return train, val, test, []


# ------------------------- DATASET -------------------------------
class InformerForecastDS(Dataset):
    def __init__(self, df: pd.DataFrame, pred_len: int, label_len: int, features):
        self.pred_len = pred_len
        self.label_len = label_len
        self.dates = []  # new list to store date origins

        feats = features  # e.g. ['is_call', 'moneyness', ...]
        enc_list, dec_list, y_list = [], [], []

        for _, g in df.groupby('option_id', sort=False):
            if len(g) < SEQ_LEN + pred_len:
                continue
            X = g[feats].values                    # shape [T, n_feats]
            Y = g[TARGET].values                  # shape [T]

            # indices for deterministic features known at prediction time
            call_idx = feats.index('is_call') if 'is_call' in feats else None
            ttm_idx  = feats.index('ttm') if 'ttm' in feats else None

            # rolling windows -------------------------------------------------
            max_i = len(g) - pred_len
            for end in range(SEQ_LEN, max_i):
                forecast_origin = g.iloc[end - 1]["QUOTE_DATE"]
                self.dates.append(forecast_origin)
                start   = end - SEQ_LEN
                enc_x   = X[start:end]            # [seq_len, n_feats]

                label_slice = X[end - self.label_len:end]

                # build future decoder features using only deterministic info
                future = np.zeros((pred_len, len(feats)))
                if call_idx is not None:
                    future[:, call_idx] = X[end - 1, call_idx]
                if ttm_idx is not None:
                    current_ttm = g.iloc[end - 1]["ttm"]
                    future[:, ttm_idx] = np.maximum(
                        current_ttm - np.arange(1, pred_len + 1), 0
                    )

                dec_x = np.vstack([label_slice, future])

                target = Y[end:end + pred_len]    # [pred_len]

                enc_list.append(enc_x)
                dec_list.append(dec_x)
                y_list.append(target)

        # stack once – much faster --------------------------------------------
        self.enc_x = torch.tensor(np.stack(enc_list), dtype=torch.float32)
        self.dec_x = torch.tensor(np.stack(dec_list), dtype=torch.float32)
        self.y     = torch.tensor(np.stack(y_list),  dtype=torch.float32)
        self.dates = np.array(self.dates)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # return only the tensors we’ll actually use
        return self.enc_x[i], self.dec_x[i], self.y[i]



# ------------------------ MODEL ----------------------------------
def build_model(
    n_feats: int,
    pred_len: int,
    label_len: int,
    *,
    d_model: int = 32,
    n_heads: int = 3,
    e_layers: int = 1,
    d_layers: int = 1,
    d_ff: int = 8,
    dropout: float = 0.06,
):
    """Instantiate an Informer model with configurable hyperparameters.

    Parameters mirror the paper's notation and can now be tuned from
    external utilities (e.g. grid search).
    """
    return Informer(
        enc_in=n_feats,
        dec_in=n_feats,
        c_out=1,
        seq_len=SEQ_LEN,
        label_len=label_len,
        out_len=pred_len,
        factor=3,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=d_layers,
        d_ff=d_ff,
        dropout=dropout,
        attn="full",
        embed="fixed",
        freq="d",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix=False,
        device=DEVICE,
    ).to(DEVICE)


# ----------------------- TRAIN / EVAL ----------------------------

def run_epoch(model, loader, criterion, opt=None):
    is_train = opt is not None
    model.train() if is_train else model.eval()
    losses, total_batches = [], len(loader)

    for idx, (enc_x, dec_x, y) in enumerate(loader):
        enc_x, dec_x = enc_x.to(DEVICE), dec_x.to(DEVICE)
        y = y.to(DEVICE).unsqueeze(-1)            # [B, pred_len, 1]

        out  = model(enc_x, dec_x)                # Informer call
        loss = criterion(out, y)

        if is_train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        losses.append(loss.item())
        print(f"Batch {idx+1}/{total_batches}  ({(idx+1)/total_batches:5.1%})", end="\r")

    return float(np.mean(losses))

# ------------  MAIN  ----------------------------------------
def train(cfg=None):
    """Train the Informer model and return aggregate test metrics."""

    cfg = cfg or {}
    horizon = cfg.get("horizon", 30)
    batch = cfg.get("batch", cfg.get("batch_size", 64))
    epochs = cfg.get("epochs", 10)
    lr = cfg.get("lr", 0.05)
    run_id = cfg.get("run_id", 44)
    data_id = cfg.get("data_id", 22)
    data_path = cfg.get("data_path", "data/cleaned/aapl-options.parquet")
    cache_root = Path(cfg.get("cache_root", "cache_inf_forecast"))
    seeds = cfg.get("seeds", SEEDS)

    label_len = 4 if horizon == 7 else 15
    features = get_feature_list(forecasting=horizon > 0)

    train_df, val_df, test_df, _ = prepare_splits_simple(data_path, horizon)
    print(
        f"Rows » train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}"
    )

    feats = [f for f in features if f != TARGET]
    scaler_X, scaler_y = scale_splits(train_df, val_df, test_df, feats, TARGET)

    pred_len = horizon
    cache_dir = cache_root / f"h{pred_len}"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"ds_s_{data_id}.pkl"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            train_ds, val_ds, test_ds = pickle.load(f)
        print("🔄 Loaded cached datasets")
    else:
        train_ds = InformerForecastDS(train_df, pred_len, label_len, features)
        val_ds = InformerForecastDS(val_df, pred_len, label_len, features)
        test_ds = InformerForecastDS(test_df, pred_len, label_len, features)
        with open(cache_file, "wb") as f:
            pickle.dump((train_ds, val_ds, test_ds), f)
        print("💾 Cached new datasets")

    print(
        f"Rows » train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}"
    )

    metrics = {"mae": [], "rmse": []}

    for seed in seeds:
        set_seed(seed)
        model = build_model(len(features), pred_len, label_len)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        mse = torch.nn.MSELoss()
        mpath = cache_dir / f"model_s_{run_id}_seed{seed}.pth"

        trn = DataLoader(
            train_ds,
            batch_size=batch,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed),
        )
        vld = DataLoader(val_ds, batch_size=batch, shuffle=False, drop_last=True)
        tst = DataLoader(test_ds, batch_size=batch, shuffle=False, drop_last=False)

        if mpath.exists():
            print(f"🔄 Loading model from {mpath}")
            model.load_state_dict(torch.load(mpath, map_location=DEVICE))
        else:
            best_val = float("inf")
            patience = 30
            counter = 0
            for ep in range(1, epochs + 1):
                tl = run_epoch(model, trn, mse, opt)
                vl = run_epoch(model, vld, mse)
                print(
                    f"[Seed {seed} Epoch {ep:2d}/{epochs}] train={tl:.5f}  val={vl:.5f}"
                )
                if vl < best_val:
                    best_val = vl
                    counter = 0
                    torch.save(model.state_dict(), mpath)
                    print(f"✓ Model improved, saved to {mpath}")
                elif not mpath.exists():
                    torch.save(model.state_dict(), mpath)
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"⚠ Early stopping at epoch {ep}")
                        break

        model.load_state_dict(torch.load(mpath, map_location=DEVICE))
        preds, trues = [], []
        with torch.no_grad():
            for enc_x, dec_x, y in tst:
                out = model(enc_x.to(DEVICE), dec_x.to(DEVICE))
                preds.append(out.cpu().numpy())
                trues.append(y.unsqueeze(-1).cpu().numpy())

        preds = np.concatenate(preds, axis=0).reshape(-1, 1)
        trues = np.concatenate(trues, axis=0).reshape(-1, 1)
        preds = scaler_y.inverse_transform(preds).flatten()
        trues = scaler_y.inverse_transform(trues).flatten()

        mae = mean_absolute_error(trues, preds)
        rmse = np.sqrt(mean_squared_error(trues, preds))
        metrics["mae"].append(mae)
        metrics["rmse"].append(rmse)
        print(f"✅ Seed {seed} Test ({pred_len}-day) → MAE={mae:.4f}  RMSE={rmse:.4f}")

    mae_arr = np.array(metrics["mae"])
    rmse_arr = np.array(metrics["rmse"])
    print(
        f"\nAggregate MAE mean={mae_arr.mean():.4f} std={mae_arr.std(ddof=1):.4f}"
    )
    print(
        f"Aggregate RMSE mean={rmse_arr.mean():.4f} std={rmse_arr.std(ddof=1):.4f}"
    )

    return {
        "mae_mean": float(mae_arr.mean()),
        "mae_std": float(mae_arr.std(ddof=1)),
        "rmse_mean": float(rmse_arr.mean()),
        "rmse_std": float(rmse_arr.std(ddof=1)),
    }


def main():
    parser = make_parser()
    args = parser.parse_args()
    train(vars(args))


if __name__ == "__main__":
    main()
