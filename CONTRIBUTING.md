# Contributing

This repository accompanies research on transformer architectures for option valuation, forecasting, and trading.

## Principles

- preserve the paper workflow and research intent,
- prefer small, well-scoped changes,
- keep experiment entry points and shared utilities clearly separated,
- avoid unnecessary edits in bundled upstream model directories.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

Install additional heavy dependencies such as PyTorch, TensorFlow, or XGBoost only when you need to run the corresponding experiments.

## Checks

```bash
ruff check lnai/config.py lnai/data lnai/core/pricing.py lnai/experiments/aggregate_results.py lnai/experiments/deep_baselines.py lnai/experiments/grid_search.py lnai/experiments/informer_forecasting.py lnai/experiments/informer_valuation.py lnai/experiments/train_all_models.py tests
pytest
```

## Structure

- `lnai/data/`: dataset preparation, feature utilities, preprocessing
- `lnai/core/`: pricing utilities and shared experiment helpers
- `lnai/experiments/`: valuation, forecasting, tuning, trading entry points
- `lnai/analysis/`: exploratory scripts not used as main orchestration entry points

## Pull requests

Please describe:

1. the exact workflow or module changed,
2. why the change improves correctness, clarity, or reproducibility,
3. whether the change touches bundled upstream model code.
