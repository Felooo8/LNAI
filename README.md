# Transformer Architectures for Option Pricing: Valuation, Prediction, and Trading

Repository accompanying the LNAI (extended ICAART) paper on transformer-based option modeling.

## Paper

**Transformer Architectures for Option Pricing: Valuation, Prediction, and Trading**  
Jarosław A. Chudziak and Feliks Bańka  
Faculty of Electronics and Information Technology, Warsaw University of Technology, Poland

This repository contains the experimental code used to compare **Informer, Autoformer, FEDformer, and Pyraformer** on three related tasks:

- **option valuation**,
- **short- and medium-term option price forecasting**,
- **rule-based trading driven by model forecasts**.

The study benchmarks transformer models against classical analytical pricing methods, shallow machine-learning baselines, and deep sequence baselines across **equities, indices, and cryptocurrency options**.

## Abstract summary

Accurate option pricing and forecasting remain difficult because of volatility clustering, regime shifts, and the gap between analytical assumptions and observed market behavior. This codebase evaluates whether long-sequence transformer architectures can improve pricing accuracy and forecast robustness across assets and horizons, and whether those gains remain useful in a simple trading setting.

## What is in this repository

The repository is organized around the actual paper workflow:

1. **data cleaning and dataset preparation**,
2. **shared option-pricing and preprocessing utilities**,
3. **valuation, forecasting, and trading experiments**,
4. **result aggregation and comparison**,
5. **bundled upstream transformer implementations** used by the experiments.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── config/
│   └── hyperparams.yaml
├── lnai/
│   ├── analysis/              # exploratory and analysis scripts
│   ├── core/                  # pricing models and shared split logic
│   ├── data/                  # cleaning, features, preprocessing
│   └── experiments/           # valuation, forecasting, tuning, trading
├── tests/                     # regression tests for shared utilities
├── Autoformer/
├── FEDformer/
├── Informer2020/
└── Pyraformer/
```

## Main entry points

The root has been intentionally trimmed; the main project code now lives under `lnai/`.

Common commands:

```bash
python -m lnai.data.cleaning
python -m lnai.experiments.deep_baselines --data-path data/cleaned/aapl-options.parquet
python -m lnai.experiments.train_all_models --models informer autoformer fedformer pyraformer
python -m lnai.experiments.aggregate_results --baselines baselines.json --grid-search grid_best.json --transformers results.json
```

If installed in editable mode, the same flows are available as:

```bash
lnai-clean-data
lnai-deep-baselines --data-path data/cleaned/aapl-options.parquet
lnai-train-all --models informer autoformer fedformer pyraformer
lnai-aggregate-results --baselines baselines.json --grid-search grid_best.json --transformers results.json
```

## Tasks covered

### 1) Option valuation

Informer-based valuation experiments are implemented in `lnai/experiments/informer_valuation.py` and use the shared preprocessing and split utilities under `lnai/data/` and `lnai/core/`.

### 2) Forecasting

Forecasting experiments cover both transformer models and deep sequence baselines. The unified orchestration entry point is `lnai/experiments/train_all_models.py`.

### 3) Trading

`lnai/experiments/informer_trading.py` evaluates a simple rule-based trading setup driven by model forecasts.

## Setup

### Lightweight development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

### Additional experiment dependencies

The root development install covers repository tooling and lightweight shared dependencies. Full experiment runs additionally require the relevant modeling stacks used by each script, such as:

- PyTorch,
- TensorFlow / Keras,
- XGBoost.

## Configuration

- Default dataset and cache paths can be set through `.env.example` variables.
- Transformer grid-search ranges live in `config/hyperparams.yaml`.
- Shared defaults are defined in `lnai/config.py`.

## Testing

```bash
ruff check lnai/config.py lnai/data lnai/core/pricing.py lnai/experiments/aggregate_results.py lnai/experiments/deep_baselines.py lnai/experiments/grid_search.py lnai/experiments/informer_forecasting.py lnai/experiments/informer_valuation.py lnai/experiments/train_all_models.py tests
pytest
```

## Notes on bundled model code

The `Autoformer/`, `FEDformer/`, `Informer2020/`, and `Pyraformer/` directories are bundled upstream or adapted research implementations that the LNAI experiments call into. They are kept largely intact to preserve reproducibility with the original model code.

## Citation

If you use this repository, please cite the corresponding paper once the final bibliographic record is available.
