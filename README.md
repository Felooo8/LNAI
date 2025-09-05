# LNAI Forecasting Models

This repository hosts a collection of research models for option price
forecasting, including Informer, Autoformer, FEDformer and Pyraformer.

## Prepare data

Final datasets used by the models are produced with `clean_data.py`. The script
scans raw `spx*.csv` files under `data/`, applies basic quality filters and
exports cleaned CSV and Parquet files to `data/cleaned/`.

```bash
python clean_data.py
```

Run this step before training to ensure the models consume the cleaned
datasets.

## Deep learning baselines

`deep_baselines.py` trains simple CNN, LSTM and GRU models for 7‑day and
30‑day forecasting horizons.  Metrics are saved to a CSV or JSON file and can
be combined with transformer results via `aggregate_results.py`.

```bash
python deep_baselines.py --data-path data/cleaned/aapl-options.parquet \
       --output baselines.csv
```

## Train all models

The script `train_all_models.py` orchestrates training and evaluation across
multiple architectures with a shared configuration. Metrics from each model
are stored in a CSV or JSON file for easy comparison.

```bash
python train_all_models.py --models informer autoformer --output results.json
```

Use `--dry-run` to preview the actions without executing the training loops.

## Hyper-parameter tuning

`transformer_grid_search.py` evaluates grids of transformer hyper-parameters and
caches every configuration under `grid_cache/<model>/<asset>/h<horizon>/`. A
minimal example:

```python
from transformer_grid_search import grid_search, TransformerConfig

param_grid = {
    "d_model": [32, 64],
    "n_heads": [2, 4],
    "e_layers": [1],
    "d_layers": [1],
    "dropout": [0.1, 0.2],
}

grid_search("informer", "aapl", 30, param_grid)
```

After running the search, train the models using the best cached
hyper‑parameters:

```bash
python train_all_models.py --models informer autoformer \
       --use-grid-best --data-path data/cleaned/aapl-options.parquet \
       --horizon 30
```

## Aggregate results

Once baselines, grid‑search scores, and transformer evaluations are
available, combine them into the comparison tables used in the paper with
`aggregate_results.py`:

```bash
python aggregate_results.py --baselines baselines.csv \
                            --grid-search grid_best.json \
                            --transformers results.json \
                            --output table.csv
```

The script accepts CSV or JSON inputs and writes the merged table to the path
specified by `--output`.

## Hyper-parameter grids

`transformer_grid_search.py` pulls its search ranges from `hyperparams.yaml`
in the repository root.  When experiment ranges change, update this YAML file
so that all contributors share a consistent configuration.

