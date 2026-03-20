from __future__ import annotations
import argparse
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from types import SimpleNamespace

from lnai.data.preprocessing import scale_splits

from lnai.experiments.informer_forecasting import (
    SEQ_LEN, TARGET, DEVICE,
    prepare_splits_simple, build_model, InformerForecastDS,
)
from lnai.data.features import get_feature_list

# Runtime configuration – values populated via CLI in ``__main__``
HORIZON = 30
BATCH = 64
CACHE_ROOT = Path("cache_inf_forecast")
LABEL_LEN = 4 if HORIZON == 7 else 15
FEATURES = get_feature_list(forecasting=HORIZON > 0)

##############################################################################
# ---------------- model map + cost constants -------------------------------
##############################################################################
model_map = {
     9: "CNN",
    13: "LSTM",
    12: "GRU",
     2: "Pyraformer",
     3: "Autoformer",
     4: "Fedformer",
     6: "Informer"
}

ASSETS = ["aapl", "nvda", "spx", "btc"]
FIXED_COST     = 0.70
PROP_COST_RATE = 0.005
SPREAD_RATE    = 0.003
OPT_MULT       = 1


##############################################################################
# -------------- helpers reused for every checkpoint -------------------------
##############################################################################

# ---------------------------------------------------------------------------
# Model loader utilities
# ---------------------------------------------------------------------------
def _wrap_transformer(model: nn.Module) -> nn.Module:
    """Wrap transformer models expecting time features.

    Many third-party implementations (Autoformer, FEDformer) expect additional
    time feature tensors beyond ``enc_x`` and ``dec_x``.  The trading pipeline
    provides only the core feature matrices, so this wrapper injects zero
    placeholders for the missing arguments and returns the model output.
    """

    class Wrapped(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.m = m

        def forward(self, enc_x, dec_x):
            mark_enc = torch.zeros(enc_x.shape[0], enc_x.shape[1], 4, device=enc_x.device)
            mark_dec = torch.zeros(dec_x.shape[0], dec_x.shape[1], 4, device=dec_x.device)
            out = self.m(enc_x, mark_enc, dec_x, mark_dec)
            if isinstance(out, tuple):
                out = out[0]
            return out

    return Wrapped(model)


def _informer_loader(path: Path) -> nn.Module:
    model = build_model(len(FEATURES), pred_len=HORIZON, label_len=LABEL_LEN)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def _autoformer_loader(path: Path) -> nn.Module:
    from Autoformer.models.Autoformer import Model as Autoformer

    cfg = SimpleNamespace(
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=HORIZON,
        output_attention=False,
        moving_avg=25,
        enc_in=len(FEATURES),
        dec_in=len(FEATURES),
        c_out=1,
        d_model=32,
        n_heads=3,
        e_layers=1,
        d_layers=1,
        d_ff=64,
        dropout=0.05,
        embed="fixed",
        freq="d",
        factor=3,
        activation="gelu",
    )

    model = Autoformer(cfg)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return _wrap_transformer(model)


def _fedformer_loader(path: Path) -> nn.Module:
    from FEDformer.models.FEDformer import Model as FEDformer

    cfg = SimpleNamespace(
        version="Wavelets",
        mode_select="random",
        modes=32,
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=HORIZON,
        output_attention=False,
        moving_avg=25,
        enc_in=len(FEATURES),
        dec_in=len(FEATURES),
        c_out=1,
        d_model=32,
        n_heads=3,
        e_layers=1,
        d_layers=1,
        d_ff=64,
        dropout=0.05,
        embed="fixed",
        freq="d",
        factor=3,
        activation="gelu",
        L=1,
        base="legendre",
        cross_activation="tanh",
    )

    model = FEDformer(cfg)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return _wrap_transformer(model)


def _pyraformer_loader(path: Path) -> nn.Module:
    model = torch.load(path, map_location=DEVICE)
    if hasattr(model, "eval"):
        model.eval()

    class Wrapped(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.m = m

        def forward(self, enc_x, dec_x):
            out = self.m(enc_x)
            if isinstance(out, tuple):
                out = out[0]
            return out

    return Wrapped(model)


class CNNModel(nn.Module):
    def __init__(self, n_feats: int, pred_len: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_feats, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, pred_len)

    def forward(self, enc_x, dec_x):
        x = enc_x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        out = self.fc(x)
        return out.unsqueeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, n_feats: int, pred_len: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_feats, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, pred_len)

    def forward(self, enc_x, dec_x):
        out, _ = self.lstm(enc_x)
        out = out[:, -1]
        out = self.fc(out)
        return out.unsqueeze(-1)


class GRUModel(nn.Module):
    def __init__(self, n_feats: int, pred_len: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(n_feats, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, pred_len)

    def forward(self, enc_x, dec_x):
        out, _ = self.gru(enc_x)
        out = out[:, -1]
        out = self.fc(out)
        return out.unsqueeze(-1)


def _cnn_loader(path: Path) -> nn.Module:
    model = CNNModel(len(FEATURES), HORIZON)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def _lstm_loader(path: Path) -> nn.Module:
    model = LSTMModel(len(FEATURES), HORIZON)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def _gru_loader(path: Path) -> nn.Module:
    model = GRUModel(len(FEATURES), HORIZON)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


MODEL_LOADERS = {
    "Informer": _informer_loader,
    "Autoformer": _autoformer_loader,
    "Fedformer": _fedformer_loader,
    "Pyraformer": _pyraformer_loader,
    "CNN": _cnn_loader,
    "LSTM": _lstm_loader,
    "GRU": _gru_loader,
}
def rebuild_objects(data_fp: Path, asset: str, cache_dir: Path):
    """Return scalers, test-loader, and aligned enc_x, y_true, opt_id for the full dataset.
    Tries to load from cache first, rebuilds and caches if not available."""
    cache_file = cache_dir / f"{asset}_trading_objects.pkl"
    
    # Check if cached data exists
    if cache_file.exists():
        print(f"Loading cached trading objects for {asset} from {cache_file}")
        cached_data = joblib.load(cache_file)
        return (
            cached_data['sx'], 
            cached_data['sy'], 
            cached_data['loader'], 
            cached_data['enc_x'], 
            cached_data['y_true'], 
            cached_data['opt_id'], 
            cached_data['test_df']
        )
    
    print(f"Building new trading objects for {asset}...")
    train_df, val_df, test_df, _ = prepare_splits_simple(data_fp, HORIZON, asset)
    feats = [f for f in FEATURES if f != TARGET]
    sx, sy = scale_splits(train_df, val_df, test_df, feats, TARGET)

    ds_test = InformerForecastDS(test_df, HORIZON, LABEL_LEN, FEATURES)
    loader  = DataLoader(ds_test, batch_size=BATCH, shuffle=False, drop_last=False)

    enc_x  = ds_test.enc_x.numpy()
    y_true = ds_test.y.numpy()

    opt_id = []
    for oid, g in test_df.groupby("option_id", sort=False):
        if len(g) < SEQ_LEN + HORIZON:
            continue
        max_i = len(g) - HORIZON
        for _ in range(SEQ_LEN, max_i):
            opt_id.append(oid)
    opt_id = pd.Series(opt_id).reset_index(drop=True)

    # Save to cache
    joblib.dump({
        'sx': sx,
        'sy': sy,
        'loader': loader,
        'enc_x': enc_x,
        'y_true': y_true,
        'opt_id': opt_id,
        'test_df': test_df
    }, cache_file)

    print(f"Saved trading objects to cache: {cache_file}")
    return sx, sy, loader, enc_x, y_true, opt_id, test_df


def prepare_single_option_data(test_df: pd.DataFrame, sx: MinMaxScaler, sy: MinMaxScaler, target_option_id: str):
    """
    Prepares data for a single option_id.
    Takes scaled test_df, scalers, and target_option_id.
    Returns loader, enc_x, y_true, and opt_id for that specific option.
    """
    # Filter the test_df for the specific option_id
    single_option_df = test_df[test_df["option_id"] == target_option_id].copy()

    if len(single_option_df) < SEQ_LEN + HORIZON:
        print(f"Warning: Option {target_option_id} has insufficient data for windowing. Skipping.")
        return None, None, None, None

    # Rebuild dataset and loader for this specific option
    ds_single_option = InformerForecastDS(
        single_option_df, HORIZON, LABEL_LEN, FEATURES
    )
    loader_single_option = DataLoader(ds_single_option, batch_size=BATCH, shuffle=False, drop_last=False)

    enc_x_single_option = ds_single_option.enc_x.numpy()
    y_true_single_option = ds_single_option.y.numpy()

    # Rebuild opt_id for this single option (it will be all the same option_id)
    opt_id_single_option = []
    # This logic must mirror how InformerForecastDS creates its samples (every single sliding window).
    if len(single_option_df) >= SEQ_LEN + HORIZON:
        # CORRECTED: Iterate step by 1 to match the number of samples InformerForecastDS generates
        for _ in range(len(single_option_df) - SEQ_LEN - HORIZON + 1):
            opt_id_single_option.append(target_option_id)
    opt_id_single_option = pd.Series(opt_id_single_option).reset_index(drop=True)

    return loader_single_option, enc_x_single_option, y_true_single_option, opt_id_single_option


def load_checkpoint(mid: int) -> torch.nn.Module:
    """Load model ``mid`` using the appropriate architecture-specific loader."""

    ckpt = CACHE_ROOT / f"h{HORIZON}" / f"model_s_{mid}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    name = model_map.get(mid)
    if name is None:
        raise ValueError(f"Unknown model id: {mid}")

    loader = MODEL_LOADERS.get(name)
    if loader is None:
        raise ValueError(f"No loader registered for model {name}")

    return loader(ckpt)


def cost_one_side(price0: float, fixed_cost, prop_cost_rate) -> float:
    """Open **or** close leg cost."""
    return fixed_cost + prop_cost_rate * price0 * OPT_MULT


##############################################################################
# --------------------------- trading engine ---------------------------------
##############################################################################
def trade_one(model_id: int,
              sx, sy, loader,
              enc_x, y_true, opt_id,
              initial_capital: float = 10000.0) -> dict:

    model = load_checkpoint(model_id)

    preds = []
    with torch.no_grad():
        for enc, dec, _ in loader:
            out = model(enc.to(DEVICE), dec.to(DEVICE)).cpu().numpy()
            preds.append(out)

    if not preds:
        return {
            "trades": 0,
            "total_pl": 0.0,
            "hit_rate": float('nan'),
            "sharpe": float('nan'),
            "equity": np.array([0.0]),
            "total_return_percent": 0.0,
            "avg_profit_per_trade": float('nan'),
            "frac_profitable_trades": float('nan'),
            "total_pl_no_cost": 0.0,
            "equity_no_cost": np.array([0.0]),
            "total_return_percent_no_cost": 0.0,
            "avg_profit_per_trade_no_cost": float('nan'),
            "frac_profitable_trades_no_cost": float('nan'),
            "all_trade_profits_with_costs": [],
            "all_trade_profits_no_costs": [],
        }

    preds = np.concatenate(preds, axis=0)

    original_preds_shape = preds.shape
    preds = sy.inverse_transform(preds.reshape(-1, 1)).reshape(original_preds_shape)
    y_true = sy.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)

    pnl, trades = 0.0, 0
    equity_path = [0.0]
    trade_profits = [] # This stores profits WITH costs

    pnl_no_cost, trades_no_cost = 0.0, 0
    equity_path_no_cost = [0.0]
    trade_profits_no_costs = [] # This stores profits WITHOUT costs. Corrected name to match the key below.

    STEP = HORIZON

    for i in range(0, len(preds), STEP):
        if i >= len(opt_id) or i >= len(enc_x) or i >= len(y_true):
            print(f"Warning: Index {i} out of bounds for data arrays. "
                  f"preds len: {len(preds)}, opt_id len: {len(opt_id)}, "
                  f"enc_x len: {len(enc_x)}, y_true len: {len(y_true)}. Breaking loop.")
            break

        current_option_id = opt_id.iloc[i]

        last_px = sy.inverse_transform(
            enc_x[i, -1:, FEATURES.index(TARGET)].reshape(-1,1)
        )[0,0]
        fut_pred = preds[i]
        fut_real = y_true[i]

        direction = 1 if fut_pred[-1] > last_px else -1
        gross = direction * (fut_real[-1] - last_px) * OPT_MULT

        open_cost    = cost_one_side(last_px, FIXED_COST, PROP_COST_RATE)
        close_cost   = cost_one_side(fut_real[-1], FIXED_COST, PROP_COST_RATE)
        spread_pen   = SPREAD_RATE * last_px * OPT_MULT
        net          = gross - (open_cost + close_cost + spread_pen)

        pnl += net
        trades += 1
        equity_path.append(pnl)
        trade_profits.append({"option_id": current_option_id, "profit": net, "type": "with_costs"})

        net_no_cost = gross
        pnl_no_cost += net_no_cost
        trades_no_cost += 1
        equity_path_no_cost.append(pnl_no_cost)
        trade_profits_no_costs.append({"option_id": current_option_id, "profit": net_no_cost, "type": "no_costs"})

    ret = np.array(equity_path)
    ret_no_cost = np.array(equity_path_no_cost)

    sharpe_ratio = float("nan")
    if trades > 1:
        profits_for_sharpe = np.array([t["profit"] for t in trade_profits if t["type"] == "with_costs"])
        if profits_for_sharpe.std(ddof=1) != 0:
            sharpe_ratio = (profits_for_sharpe.mean() / profits_for_sharpe.std(ddof=1)) * math.sqrt(252 / STEP)


    total_return_percent = (pnl / initial_capital) * 100 if initial_capital != 0 else float('nan')
    total_return_percent_no_cost = (pnl_no_cost / initial_capital) * 100 if initial_capital != 0 else float('nan')


    if trades > 0 :
        avg_profit_per_trade = np.mean([t["profit"] for t in trade_profits if t["type"] == "with_costs"])
        frac_profitable_trades = np.mean([t["profit"] > 0 for t in trade_profits if t["type"] == "with_costs"])
    else:
        avg_profit_per_trade = float('nan')
        frac_profitable_trades = float('nan')

    if trades_no_cost > 0:
        avg_profit_per_trade_no_cost = np.mean([t["profit"] for t in trade_profits_no_costs if t["type"] == "no_costs"]) # Corrected: use trade_profits_no_costs
        frac_profitable_trades_no_cost = np.mean([t["profit"] > 0 for t in trade_profits_no_costs if t["type"] == "no_costs"]) # Corrected: use trade_profits_no_costs
    else:
        avg_profit_per_trade_no_cost = float('nan')
        frac_profitable_trades_no_cost = float('nan')

    return dict(
        trades          = trades,
        total_pl        = pnl,
        hit_rate        = frac_profitable_trades,
        sharpe          = sharpe_ratio,
        equity          = ret,
        total_return_percent    = total_return_percent,
        avg_profit_per_trade = avg_profit_per_trade,
        frac_profitable_trades = frac_profitable_trades,
        total_pl_no_cost = pnl_no_cost,
        equity_no_cost  = ret_no_cost,
        total_return_percent_no_cost = total_return_percent_no_cost,
        avg_profit_per_trade_no_cost = avg_profit_per_trade_no_cost,
        frac_profitable_trades_no_cost = frac_profitable_trades_no_cost,
        all_trade_profits_with_costs = trade_profits,
        all_trade_profits_no_costs = trade_profits_no_costs, # THIS WAS THE ERROR: Changed from trade_profits_no_costs to all_trade_profits_no_costs
    )

##############################################################################
# --------------------------------- main ------------------------------------
##############################################################################
def main(
    asset: str,
    data_dir: Path,
    run_whole_dataset: bool = True,
    run_longest_options: bool = True,
):
    data_fp = data_dir / f"{asset}-options.parquet"
    cache_dir = Path("cache_trading") / asset
    cache_dir.mkdir(parents=True, exist_ok=True)
    sx, sy, full_loader, full_enc_x, full_y_true, full_opt_id, full_test_df = rebuild_objects(data_fp, asset, cache_dir)

    PLOT_INITIAL_CAPITAL = 10000.0

    if run_whole_dataset:
        print("\n===== Running trading simulation on the WHOLE DATASET =====")
        summary = []
        curves  = {}
        curves_no_cost = {}

        for mid, name in model_map.items():
            stats = trade_one(mid, sx, sy, full_loader, full_enc_x, full_y_true, full_opt_id, initial_capital=PLOT_INITIAL_CAPITAL)
            summary.append({
                "Model":   name,
                "Trades":  stats["trades"],
                "Total $": stats["total_pl"],
                "Avg Prof $": stats["avg_profit_per_trade"],
                "Frac Prof %": stats["frac_profitable_trades"] * 100,
                "Total Return %": stats["total_return_percent"],
                "Total $ (No Cost)": stats["total_pl_no_cost"],
                "Avg Prof $ (No Cost)": stats["avg_profit_per_trade_no_cost"],
                "Frac Prof % (No Cost)": stats["frac_profitable_trades_no_cost"] * 100,
                "Total Return % (No Cost)": stats["total_return_percent_no_cost"],
            })
            curves[name] = stats["equity"]
            curves_no_cost[name] = stats["equity_no_cost"]
            print(f"✓ finished {name} for whole dataset")

        print(asset)
        df = (pd.DataFrame(summary)
                .sort_values("Total $", ascending=False)
                .reset_index(drop=True))
        
        cols_with_costs = ["Model", "Trades", "Total $", "Avg Prof $", "Frac Prof %", "Total Return %"]
        print("\n========== COST-ADJUSTED TEST RESULTS (WHOLE DATASET) ==========")
        print(df[cols_with_costs].to_string(index=False, formatters={
            "Total $"       : "{:12.2f}".format,
            "Avg Prof $"    : "{:10.2f}".format,
            "Frac Prof %"   : "{:10.2f}".format,
            "Total Return %": "{:14.2f}".format,
        }))

        cols_no_costs = ["Model", "Trades", "Total $ (No Cost)", "Avg Prof $ (No Cost)", "Frac Prof % (No Cost)", "Total Return % (No Cost)"]
        print("\n========== WITHOUT TRANSACTION COSTS TEST RESULTS (WHOLE DATASET) ==========")
        print(df[cols_no_costs].to_string(index=False, formatters={
            "Total $ (No Cost)"       : "{:12.2f}".format,
            "Avg Prof $ (No Cost)"    : "{:10.2f}".format,
            "Frac Prof % (No Cost)"   : "{:10.2f}".format,
            "Total Return % (No Cost)": "{:14.2f}".format,
        }))

        plt.figure(figsize=(12, 6))
        for name, equity_curve in curves.items():
            if len(equity_curve) > 1:
                percentage_return = ((PLOT_INITIAL_CAPITAL + equity_curve) / PLOT_INITIAL_CAPITAL - 1) * 100
                plt.plot(percentage_return, label=f"{name} (With Costs)")
            else:
                print(f"  No sufficient trades for {name} on whole dataset (with costs) to plot.")

        if plt.gca().lines:
            plt.title(f"Cumulative P/L % Return (With Costs) for {asset.upper()} - Whole Dataset")
            plt.xlabel("Trade Number")
            plt.ylabel("Cumulative P/L Percentage (%)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(cache_dir / f"{asset}_whole_dataset_pl_with_costs.png")
        else:
            print(f"  No data to plot for whole dataset (with costs).")
        plt.close()

        plt.figure(figsize=(12, 6))
        for name, equity_curve_no_cost in curves_no_cost.items():
            if len(equity_curve_no_cost) > 1:
                percentage_return_no_cost = ((PLOT_INITIAL_CAPITAL + equity_curve_no_cost) / PLOT_INITIAL_CAPITAL - 1) * 100
                plt.plot(percentage_return_no_cost, label=f"{name} (No Costs)")
            else:
                print(f"  No sufficient trades for {name} on whole dataset (no costs) to plot.")

        if plt.gca().lines:
            plt.title(f"Cumulative P/L % Return (No Costs) for {asset.upper()} - Whole Dataset")
            plt.xlabel("Trade Number")
            plt.ylabel("Cumulative P/L Percentage (%)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(cache_dir / f"{asset}_whole_dataset_pl_no_costs.png")
        else:
            print(f"  No data to plot for whole dataset (no costs).")
        plt.close()

    if run_longest_options:
        print("\n===== Running trading simulation on 5 LONGEST OPTION IDs =====")

        calls_df = full_test_df[full_test_df["is_call"] == 1]
        puts_df  = full_test_df[full_test_df["is_call"] == 0]

        def get_longest(df, n=5):
            counts = {}
            for oid, g in df.groupby("option_id", sort=False):
                if len(g) >= SEQ_LEN + HORIZON:
                    counts[oid] = len(g) - (SEQ_LEN + HORIZON) + 1
            return pd.Series(counts).sort_values(ascending=False).head(n).index.tolist()

        longest_calls = get_longest(calls_df, n=5)
        longest_puts  = get_longest(puts_df,  n=5)
        longest_option_ids = longest_calls + longest_puts

        for opt_id_val in longest_option_ids:
            print(f"\nProcessing for specific option_id: {opt_id_val}")
            single_opt_loader, single_opt_enc_x, single_opt_y_true, single_opt_opt_id = \
                prepare_single_option_data(full_test_df, sx, sy, opt_id_val)

            if single_opt_loader is None:
                continue

            opt_summary = []
            opt_curves = {}
            opt_curves_no_cost = {}

            for mid, name in model_map.items():
                stats = trade_one(mid, sx, sy, single_opt_loader, single_opt_enc_x, single_opt_y_true, single_opt_opt_id, initial_capital=PLOT_INITIAL_CAPITAL)
                opt_summary.append({
                    "Model":   name,
                    "Trades":  stats["trades"],
                    "Total $": stats["total_pl"],
                    "Avg Prof $": stats["avg_profit_per_trade"],
                    "Frac Prof %": stats["frac_profitable_trades"] * 100,
                    "Total Return %": stats["total_return_percent"],
                    "Total $ (No Cost)": stats["total_pl_no_cost"],
                    "Avg Prof $ (No Cost)": stats["avg_profit_per_trade_no_cost"],
                    "Frac Prof % (No Cost)": stats["frac_profitable_trades_no_cost"] * 100,
                    "Total Return % (No Cost)": stats["total_return_percent_no_cost"],
                })
                opt_curves[name] = stats["equity"]
                opt_curves_no_cost[name] = stats["equity_no_cost"]

            df_opt = (pd.DataFrame(opt_summary)
                        .sort_values("Total $", ascending=False)
                        .reset_index(drop=True))
            print(f"\n========== COST-ADJUSTED TEST RESULTS (Option: {opt_id_val}) ==========")


            plt.figure(figsize=(12, 6))
            for name, equity_curve in opt_curves.items():
                if len(equity_curve) > 1:
                    percentage_return = ((PLOT_INITIAL_CAPITAL + equity_curve) / PLOT_INITIAL_CAPITAL - 1) * 100
                    plt.plot(percentage_return, label=f"{name}")
                else:
                    print(f"  No sufficient trades for {name} on option_id {opt_id_val} (with costs) to plot.")

            if plt.gca().lines:
                plt.title(f"Cumulative P/L % Return (With Costs) for {asset.upper()} - Option: {opt_id_val}")
                plt.xlabel("Trade Number")
                plt.ylabel("Cumulative P/L Percentage (%)")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(cache_dir / f"{asset}_option_{opt_id_val}_pl_with_costs.png")
            else:
                print(f"  No data to plot for option_id {opt_id_val} (with costs).")
            plt.close()

            plt.figure(figsize=(12, 6))
            for name, equity_curve_no_cost in opt_curves_no_cost.items():
                if len(equity_curve_no_cost) > 1:
                    percentage_return_no_cost = ((PLOT_INITIAL_CAPITAL + equity_curve_no_cost) / PLOT_INITIAL_CAPITAL - 1) * 100
                    plt.plot(percentage_return_no_cost, label=f"{name}")
                else:
                    print(f"  No sufficient trades for {name} on option_id {opt_id_val} (no costs) to plot.")

            if plt.gca().lines:
                plt.title(f"Cumulative P/L % Return (No Costs) for {asset.upper()} - Option: {opt_id_val}")
                plt.xlabel("Trade Number")
                plt.ylabel("Cumulative P/L Percentage (%)")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(cache_dir / f"{asset}_option_{opt_id_val}_pl_no_costs.png")
            else:
                print(f"  No data to plot for option_id {opt_id_val} (no costs).")
            plt.close()

    print("\nAll requested trading simulations and plots completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset", choices=ASSETS + ["all"], default="all",
        help="Asset to trade or 'all' for every asset",
    )
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--cache-root", default="cache_inf_forecast",
        help="Directory containing trained Informer checkpoints",
    )
    parser.add_argument(
        "--data-dir", default="data/cleaned",
        help="Directory containing <asset>-options.parquet files",
    )
    parser.add_argument("--no_whole", action="store_true",
                        help="Skip running on whole dataset")
    parser.add_argument("--no_longest", action="store_true",
                        help="Skip running on longest option IDs")
    args = parser.parse_args()

    HORIZON = args.horizon
    BATCH = args.batch_size
    CACHE_ROOT = Path(args.cache_root)
    LABEL_LEN = 4 if HORIZON == 7 else 15
    FEATURES = get_feature_list(forecasting=HORIZON > 0)

    assets = ASSETS if args.asset == "all" else [args.asset]
    data_dir = Path(args.data_dir)
    for asset in assets:
        main(
            asset,
            data_dir,
            run_whole_dataset=not args.no_whole,
            run_longest_options=not args.no_longest,
        )
