"""Train and evaluate multiple forecasting models with shared configuration.

This module provides a command line interface and a :func:`train_and_eval`
utility that dispatches to the individual model runners present in the
repository. Results from each model are collected and written to a JSON or
CSV file for easy comparison.

Example
-------
Run Autoformer and Informer on the default dataset and save metrics:

.. code-block:: bash

    python train_all_models.py --models informer autoformer --output results.json

Use ``--dry-run`` to see which commands would execute without running the
training loops.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def train_and_eval(model_name: str, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Dispatch training/evaluation to the appropriate runner.

    Parameters
    ----------
    model_name:
        Identifier of the model to train (e.g. ``"informer"``).
    cfg:
        Configuration dictionary passed through to the runner. The exact
        fields depend on the model.
    """

    name = model_name.lower()
    task = cfg.get("task", "forecast").lower()

    if task == "valuate" or cfg.get("horizon", 0) == 0:
        if name != "informer":
            raise ValueError("Valuation task currently supports only the Informer model")
        from transformer_valuate import train as valuate_train

        return valuate_train(cfg)

    if name == "informer":
        from informer_runner import train as informer_train

        return informer_train(cfg)

    import argparse as _ap  # local import to avoid polluting namespace

    if name == "autoformer":
        from Autoformer.run import run as autoformer_run

        return autoformer_run(_ap.Namespace(**cfg))
    if name == "fedformer":
        from FEDformer.run import run as fedformer_run

        return fedformer_run(_ap.Namespace(**cfg))
    if name == "pyraformer":
        from Pyraformer.run import run as pyraformer_run

        return pyraformer_run(_ap.Namespace(**cfg))

    raise ValueError(f"Unknown model: {model_name}")


def _save_table(results: List[Dict[str, Any]], path: Path) -> None:
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    else:
        with path.open("w") as f:
            json.dump(results, f, indent=2)


def _standardize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize metric keys so outputs share a common schema."""
    key_map = {
        "mae": "mae",
        "mae_mean": "mae",
        "rmse": "rmse",
        "rmse_mean": "rmse",
        "hit_rate": "hit_rate",
        "hit_rate_mean": "hit_rate",
    }
    out: Dict[str, Any] = {}
    for k, v in (metrics or {}).items():
        lk = k.lower()
        out[key_map.get(lk, lk)] = v
    return out


def _load_grid_best(
    model: str, asset: str, horizon: int, cache_root: Path = Path("grid_cache")
) -> Optional[Dict[str, Any]]:
    """Return best cached hyper‑parameters for ``model`` if available."""

    cache_dir = cache_root / model / asset / f"h{horizon}"
    if not cache_dir.exists():
        return None

    best_cfg: Optional[Dict[str, Any]] = None
    best_metric = float("inf")
    for f in cache_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        metric = data.get("metric")
        if metric is None:
            continue
        if metric < best_metric:
            best_metric = metric
            best_cfg = {k: v for k, v in data.items() if k != "metric"}

    return best_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["forecast", "valuate"],
        default="forecast",
        help="Choose forecasting or valuation task",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["informer", "autoformer", "fedformer", "pyraformer"],
        help="List of models to train",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Where to store aggregated metrics (extension decides JSON/CSV)",
    )
    parser.add_argument(
        "--data-path",
        default="data/cleaned/aapl-options.parquet",
        help="Dataset path used by Informer baseline",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Forecast horizon used for grid search lookup",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--run-id", type=int, default=44, help="Model cache identifier")
    parser.add_argument("--data-id", type=int, default=22, help="Dataset cache identifier")
    parser.add_argument(
        "--cache-root",
        default="cache_inf_forecast",
        help="Directory for Informer caches",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing training",
    )
    parser.add_argument(
        "--use-grid-best",
        action="store_true",
        help="Load best hyper-parameters from grid search cache if available",
    )

    args = parser.parse_args()

    if args.task == "valuate":
        args.horizon = 0

    cfg = {
        "data_path": args.data_path,
        "horizon": args.horizon,
        "batch": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "run_id": args.run_id,
        "data_id": args.data_id,
        "cache_root": args.cache_root,
        "task": args.task,
    }
    asset = Path(args.data_path).stem.split("-")[0]
    results: List[Dict[str, Any]] = []

    for name in args.models:
        model_cfg = dict(cfg)
        if args.use_grid_best:
            grid_cfg = _load_grid_best(name, asset, args.horizon)
            if grid_cfg:
                model_cfg.update(grid_cfg)
            else:
                print(f"No grid cache found for {name}, using defaults")

        if args.dry_run:
            print(f"[DRY-RUN] Would train {name} with cfg={model_cfg}")
            continue

        print(f"===== Training {name} =====")
        metrics = _standardize_metrics(train_and_eval(name, model_cfg))
        results.append({"model": name, **metrics})

    if results:
        _save_table(results, Path(args.output))


if __name__ == "__main__":
    main()

