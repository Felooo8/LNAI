"""Utility functions and classical option pricing models.

This module gathers shared helpers for loading and filtering the option
pricing dataset as well as a few textbook pricing models used as
baselines throughout the project.
"""
from __future__ import annotations

from math import erf, exp, log, sqrt
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

__all__ = [
    "load_parquet",
    "filter_options",
    "make_time_splits",
    "bs_price",
    "binomial_price",
    "mc_price",
]

# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a parquet file into a ``DataFrame``.

    Parameters
    ----------
    path : str or Path
        Location of the parquet dataset.
    **kwargs : dict
        Additional keyword arguments passed to :func:`pandas.read_parquet`.
    """
    df = pd.read_parquet(path, **kwargs)
    # ensure the date column is a ``datetime64`` type for later filtering
    if "QUOTE_DATE" in df.columns:
        df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])
    return df


def filter_options(
    df: pd.DataFrame,
    *,
    asset: Optional[str] = None,
    min_data_points: int = 30,
    min_ttm_days: int = 0,
) -> pd.DataFrame:
    """Filter the raw options ``DataFrame``.

    The function provides a light‑weight preprocessing step used by the
    experiments.  Filtering keeps options with a sufficient history and,
    optionally, restricts the underlying asset and minimum time to
    maturity.
    """
    df = df.copy()

    if asset is not None:
        # Different datasets use different column names for the underlying
        for col in ("underlying", "underlying_symbol", "root"):
            if col in df.columns:
                df = df[df[col].str.lower() == asset.lower()]
                break

    if "ttm" in df.columns:
        df = df[df["ttm"] >= min_ttm_days]

    if "option_id" in df.columns and min_data_points:
        counts = df.groupby("option_id").size()
        valid_ids = counts[counts >= min_data_points].index
        df = df[df["option_id"].isin(valid_ids)]

    if "QUOTE_DATE" in df.columns:
        df = df.sort_values("QUOTE_DATE")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Time series splits
# ---------------------------------------------------------------------------

def make_time_splits(
    df: pd.DataFrame,
    *,
    train_years: int = 1,
    val_months: int = 3,
    test_years: int = 1,
    step_months: Optional[int] = None,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.Timestamp]]]:
    """Create rolling train/validation/test splits.

    Each split is determined purely by ``QUOTE_DATE`` and the provided
    horizons.  The function returns a list of tuples of the form
    ``(train_df, val_df, test_df, date_dict)`` where ``date_dict`` stores
    the boundaries used for that split.
    """
    if "QUOTE_DATE" not in df.columns:
        raise KeyError("DataFrame must contain a 'QUOTE_DATE' column")

    df = df.sort_values("QUOTE_DATE").reset_index(drop=True)
    start_date = df["QUOTE_DATE"].min()
    end_date = df["QUOTE_DATE"].max()

    total_months = train_years * 12 + val_months + test_years * 12
    if step_months is None:
        step_months = total_months

    splits = []
    cur_start = start_date
    while True:
        train_end = cur_start + pd.DateOffset(years=train_years)
        val_end = train_end + pd.DateOffset(months=val_months)
        test_end = val_end + pd.DateOffset(years=test_years)

        if test_end > end_date + pd.Timedelta(days=1):
            break

        train_mask = (df["QUOTE_DATE"] >= cur_start) & (df["QUOTE_DATE"] < train_end)
        val_mask = (df["QUOTE_DATE"] >= train_end) & (df["QUOTE_DATE"] < val_end)
        test_mask = (df["QUOTE_DATE"] >= val_end) & (df["QUOTE_DATE"] < test_end)

        train_df = df.loc[train_mask].reset_index(drop=True)
        val_df = df.loc[val_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)

        date_info = {
            "train_start": cur_start,
            "train_end": train_end - pd.Timedelta(days=1),
            "val_start": train_end,
            "val_end": val_end - pd.Timedelta(days=1),
            "test_start": val_end,
            "test_end": test_end - pd.Timedelta(days=1),
        }
        splits.append((train_df, val_df, test_df, date_info))

        cur_start += pd.DateOffset(months=step_months)
        if cur_start >= end_date:
            break

    return splits


# ---------------------------------------------------------------------------
# Pricing models
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_price(
    spot: float,
    strike: float,
    ttm: float,
    r: float,
    *,
    sigma: float,
    is_call: bool = True,
) -> float:
    """Black–Scholes price of a European option.

    Parameters
    ----------
    spot, strike : float
        Underlying price and option strike.
    ttm : float
        Time to maturity **in days**.
    r : float
        Risk‑free interest rate (annualised, in continuous compounding).
    sigma : float
        Implied volatility of the option (annualised).
    is_call : bool, default ``True``
        ``True`` for a call option, ``False`` for a put.
    """
    T = max(ttm, 0) / 365.0
    if T == 0 or sigma == 0:
        intrinsic = max(spot - strike, 0) if is_call else max(strike - spot, 0)
        return float(intrinsic)

    d1 = (log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if is_call:
        price = spot * _norm_cdf(d1) - strike * exp(-r * T) * _norm_cdf(d2)
    else:
        price = strike * exp(-r * T) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    return float(price)


def _extract_row(row_or_vals):
    """Internal helper to support both row-based and explicit arguments."""
    if isinstance(row_or_vals, pd.Series):
        S0 = row_or_vals.get("UNDERLYING_LAST", row_or_vals.get("spot"))
        K = row_or_vals.get("STRIKE", row_or_vals.get("strike"))
        T = row_or_vals.get("ttm", row_or_vals.get("T")) / 365.0
        sigma = row_or_vals.get("IV", row_or_vals.get("sigma"))
        is_call = bool(row_or_vals.get("is_call", True))
    else:  # assume tuple-like (S, K, ttm_days, sigma, is_call)
        S0, K, ttm_days, sigma, is_call = row_or_vals
        T = ttm_days / 365.0
    return float(S0), float(K), float(T), float(sigma), bool(is_call)


def binomial_price(
    row_or_vals,
    *,
    r_flat: float = 0.0,
    steps: int = 100,
) -> float:
    """Cox–Ross–Rubinstein binomial tree price.

    ``row_or_vals`` can either be a :class:`pandas.Series` with the
    required fields or a tuple ``(spot, strike, ttm_days, sigma, is_call)``.
    """
    S0, K, T, sigma, is_call = _extract_row(row_or_vals)
    dt = T / steps
    if dt <= 0:
        intrinsic = max(S0 - K, 0) if is_call else max(K - S0, 0)
        return float(intrinsic)

    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp(r_flat * dt) - d) / (u - d)
    disc = exp(-r_flat * dt)

    # terminal prices
    prices = S0 * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1)
    values = np.maximum(prices - K, 0) if is_call else np.maximum(K - prices, 0)

    for _ in range(steps):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
    return float(values[0])


def mc_price(
    row_or_vals,
    *,
    r_flat: float = 0.0,
    num_paths: int = 1000,
) -> float:
    """Monte Carlo estimate of a European option price under GBM."""
    S0, K, T, sigma, is_call = _extract_row(row_or_vals)
    if T <= 0:
        intrinsic = max(S0 - K, 0) if is_call else max(K - S0, 0)
        return float(intrinsic)

    rand = np.random.standard_normal(num_paths)
    ST = S0 * np.exp((r_flat - 0.5 * sigma ** 2) * T + sigma * sqrt(T) * rand)
    if is_call:
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    return float(exp(-r_flat * T) * payoff.mean())

