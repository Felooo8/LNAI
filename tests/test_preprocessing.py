import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import pandas as pd

from lnai.data.preprocessing import encode_categoricals, scale_splits


def _frame(offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "is_call": [1.0 + offset, 0.0 + offset],
            "moneyness": [0.9 + offset, 1.1 + offset],
            "ttm": [30 + offset, 45 + offset],
            "price": [4.0 + offset, 5.0 + offset],
            "vix": [15.0 + offset, 16.0 + offset],
        }
    )


def test_encode_categoricals_casts_expected_columns() -> None:
    df = pd.DataFrame({"is_call": [0.0, 1.0], "vix": ["14.1", "15.2"]})

    encoded = encode_categoricals(df)

    assert list(encoded["is_call"]) == [0, 1]
    assert encoded["vix"].dtype.kind in {"f", "i"}


def test_scale_splits_uses_train_fit_for_all_splits() -> None:
    train = _frame(0)
    val = _frame(1)
    test = _frame(2)

    feat_scaler, target_scaler = scale_splits(
        train=train,
        val=val,
        test=test,
        features=["is_call", "moneyness", "ttm", "vix"],
        target="price",
    )

    assert feat_scaler.n_features_in_ == 4
    assert target_scaler.n_features_in_ == 1
    assert train["price"].between(0, 1).all()
    assert val["price"].max() > 1
    assert test["price"].max() > val["price"].max()
