"""Aggregate baseline, grid-search and transformer metrics into one table."""

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    """Load a CSV or JSON file into a DataFrame with lowercase columns."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        with path.open() as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
    return df.rename(columns=str.lower)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate baseline, grid-search and transformer metrics into one table."
    )
    parser.add_argument("--baselines", required=True, help="Baseline metrics file")
    parser.add_argument("--grid-search", required=True, help="Grid search best-score file")
    parser.add_argument("--transformers", required=True, help="Transformer evaluation file")
    parser.add_argument("--output", default="aggregate_results.csv", help="Output CSV/JSON path")
    parser.add_argument(
        "--baseline-seeds",
        type=int,
        default=1,
        help="Number of seeds used to average baseline metrics",
    )
    args = parser.parse_args()

    tables: List[pd.DataFrame] = [
        _read_table(Path(args.baselines)).assign(seeds=args.baseline_seeds),
        _read_table(Path(args.grid_search)).assign(seeds=1),
        _read_table(Path(args.transformers)).assign(seeds=1),
    ]
    table = pd.concat(tables, ignore_index=True)
    out_path = Path(args.output)
    if out_path.suffix.lower() == ".csv":
        table.to_csv(out_path, index=False)
    else:
        table.to_json(out_path, orient="records", indent=2)
    print(table)


if __name__ == "__main__":
    main()
