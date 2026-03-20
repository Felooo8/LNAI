"""Lightweight grid search utilities for transformer forecasters.

This module exposes a minimal interface to evaluate different hyper-parameter
configurations for Informer, Autoformer, FEDformer and Pyraformer. Results are
cached to avoid redundant work.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

from lnai.config import DEFAULT_HYPERPARAM_PATH

try:  # Optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML may be missing
    yaml = None

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TransformerConfig:
    d_model: int
    n_heads: int
    e_layers: int
    d_layers: int
    dropout: float


CONFIG_FIELDS = {f.name for f in fields(TransformerConfig)}


def load_param_grid(model: str, path: Path = DEFAULT_HYPERPARAM_PATH) -> Dict[str, Iterable]:
    """Load hyper-parameter ranges for ``model`` from ``path``.

    The configuration file is expected to be YAML but falls back to JSON if the
    :mod:`yaml` package is unavailable.  Extra keys not present in
    :class:`TransformerConfig` are ignored so the YAML can contain values used by
    specific trainers.
    """

    text = path.read_text()
    data = yaml.safe_load(text) if yaml else json.loads(text)
    params = data.get(model.lower())
    if params is None:
        raise KeyError(f"{model} not found in {path}")
    return {k: v for k, v in params.items() if k in CONFIG_FIELDS}


# ---------------------------------------------------------------------------
# Training hooks
# ---------------------------------------------------------------------------
# The grid search expects callables with the following signature.  Each
# trainer should carry out a full training/validation loop and return a single
# scalar metric where lower is better.
Trainer = Callable[[str, int, TransformerConfig], float]


def informer_trainer(asset: str, horizon: int, cfg: TransformerConfig) -> float:
    """Train Informer with the provided hyper-parameters.

    Parameters
    ----------
    asset, horizon:
        Identify which dataset to load and the forecasting horizon in days.
    cfg:
        Hyper-parameters for the transformer backbone.

    Returns
    -------
    float
        Validation loss returned by :func:`lnai.experiments.informer_forecasting.train`.
    """

    from lnai.experiments.informer_forecasting import train as informer_train

    params = {
        "data_path": f"data/cleaned/{asset}-options.parquet",
        "horizon": horizon,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "e_layers": cfg.e_layers,
        "d_layers": cfg.d_layers,
        "dropout": cfg.dropout,
    }

    metrics = informer_train(params)
    # Prefer MAE but fall back to any available metric
    return float(
        metrics.get("mae_mean")
        if isinstance(metrics, dict)
        else metrics  # type: ignore[arg-type]
    )


def autoformer_trainer(asset: str, horizon: int, cfg: TransformerConfig) -> float:
    """Train Autoformer with ``cfg`` and return a validation metric."""

    import argparse

    from Autoformer.run import run as autoformer_run

    args = argparse.Namespace(
        is_training=1,
        model_id="grid-search",
        model="Autoformer",
        data="custom",
        root_path=".",
        data_path=f"data/cleaned/{asset}-options.parquet",
        features="M",
        target="price",
        seq_len=30,
        label_len=15,
        pred_len=horizon,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        e_layers=cfg.e_layers,
        d_layers=cfg.d_layers,
        dropout=cfg.dropout,
        train_epochs=1,
        batch_size=32,
        patience=1,
        learning_rate=1e-4,
    )

    metrics = autoformer_run(args)
    return float(metrics.get("mae", metrics.get("mae_mean", 0.0)))


def fedformer_trainer(asset: str, horizon: int, cfg: TransformerConfig) -> float:
    """Train FEDformer with ``cfg`` and return a validation metric."""

    import argparse

    from FEDformer.run import run as fedformer_run

    args = argparse.Namespace(
        is_training=1,
        model_id="grid-search",
        model="FEDformer",
        data="custom",
        root_path=".",
        data_path=f"data/cleaned/{asset}-options.parquet",
        features="M",
        target="price",
        seq_len=30,
        label_len=15,
        pred_len=horizon,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        e_layers=cfg.e_layers,
        d_layers=cfg.d_layers,
        dropout=cfg.dropout,
        train_epochs=1,
        batch_size=32,
        patience=1,
        learning_rate=1e-4,
    )

    metrics = fedformer_run(args)
    return float(metrics.get("mae", metrics.get("mae_mean", 0.0)))


def pyraformer_trainer(asset: str, horizon: int, cfg: TransformerConfig) -> float:
    """Train Pyraformer with ``cfg`` and return a validation metric."""

    import argparse

    from Pyraformer.run import run as pyraformer_run

    args = argparse.Namespace(
        is_training=1,
        model_id="grid-search",
        model="Pyraformer",
        data="custom",
        root_path=".",
        data_path=f"data/cleaned/{asset}-options.parquet",
        features="M",
        target="price",
        seq_len=30,
        label_len=15,
        pred_len=horizon,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        e_layers=cfg.e_layers,
        d_layers=cfg.d_layers,
        dropout=cfg.dropout,
        train_epochs=1,
        batch_size=32,
        patience=1,
        learning_rate=1e-4,
    )

    metrics = pyraformer_run(args)
    return float(metrics.get("mae", metrics.get("mae_mean", 0.0)))


# Mapping of model identifiers to their respective trainers
TRAINERS: Dict[str, Trainer] = {
    "informer": informer_trainer,
    "autoformer": autoformer_trainer,
    "fedformer": fedformer_trainer,
    "pyraformer": pyraformer_trainer,
}


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(
    model: str,
    asset: str,
    horizon: int,
    param_grid: Optional[Dict[str, Iterable]] = None,
    cache_dir: Path = Path("grid_cache"),
) -> Tuple[TransformerConfig, float]:
    """Evaluate a grid of hyper-parameters and return the best configuration.

    If ``param_grid`` is ``None`` the ranges are loaded from
    :func:`load_param_grid`.  Metrics for each configuration are cached under
    ``cache_dir`` so that repeated runs skip already evaluated settings.
    """
    trainer = TRAINERS[model.lower()]
    if param_grid is None:
        param_grid = load_param_grid(model)
    cache_dir = cache_dir / model / asset / f"h{horizon}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    best_cfg, best_metric = None, float("inf")

    keys = sorted(param_grid)
    for values in itertools.product(*[param_grid[k] for k in keys]):
        cfg = TransformerConfig(**dict(zip(keys, values, strict=True)))
        cache_file = cache_dir / f"{cfg}.json"
        if cache_file.exists():
            metric = json.loads(cache_file.read_text())["metric"]
        else:
            metric = trainer(asset, horizon, cfg)
            cache_file.write_text(json.dumps({"metric": metric, **asdict(cfg)}))

        if metric < best_metric:
            best_metric, best_cfg = metric, cfg

    assert best_cfg is not None  # grid must not be empty
    return best_cfg, best_metric


__all__ = ["TransformerConfig", "grid_search", "TRAINERS", "load_param_grid"]
