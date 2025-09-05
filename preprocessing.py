from typing import Iterable, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure categorical columns are numeric (e.g., is_call as int)."""
    if 'is_call' in df.columns:
        df['is_call'] = df['is_call'].astype(int)
    # ensure macro features are numeric if present
    for col in ('vix', 'treasury_yield', 'inflation'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def scale_splits(train: pd.DataFrame,
                 val: pd.DataFrame,
                 test: pd.DataFrame,
                 features: Iterable[str],
                 target: str) -> Tuple[MinMaxScaler, MinMaxScaler]:
    """Fit MinMax scalers on the train split and transform all splits.

    Returns
    -------
    feat_scaler, target_scaler : fitted scalers for features and target.
    """
    # ensure consistent categorical encoding
    for df in (train, val, test):
        encode_categoricals(df)

    feat_scaler = MinMaxScaler().fit(train[list(features)])
    targ_scaler = MinMaxScaler().fit(train[[target]])

    for df in (train, val, test):
        df[list(features)] = feat_scaler.transform(df[list(features)])
        df[[target]] = targ_scaler.transform(df[[target]])

    return feat_scaler, targ_scaler
