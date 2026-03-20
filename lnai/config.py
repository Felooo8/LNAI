"""Shared repository-level configuration helpers.

These helpers keep the top-level experiment scripts aligned on default
paths while still allowing environment overrides for local workflows.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = Path(os.getenv("LNAI_DATA_PATH", "data/cleaned/aapl-options.parquet"))
DEFAULT_CACHE_ROOT = Path(os.getenv("LNAI_CACHE_ROOT", "cache_inf_forecast"))
DEFAULT_RAW_DATA_DIR = Path(os.getenv("LNAI_RAW_DATA_DIR", "data"))
DEFAULT_CLEAN_DATA_DIR = DEFAULT_RAW_DATA_DIR / "cleaned"
DEFAULT_MACRO_PATH = DEFAULT_RAW_DATA_DIR / "macro.csv"
DEFAULT_HYPERPARAM_PATH = Path(os.getenv("LNAI_HYPERPARAM_PATH", "config/hyperparams.yaml"))
