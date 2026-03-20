import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("numpy")

from lnai.core.pricing import binomial_price, bs_price, filter_options, make_time_splits, mc_price


def test_filter_options_applies_asset_history_and_ttm_filters() -> None:
    rows = []
    for i in range(35):
        rows.append(
            {
                "underlying": "AAPL",
                "option_id": "keep",
                "ttm": 45,
                "QUOTE_DATE": f"2020-01-{(i % 28) + 1:02d}",
            }
        )
    for i in range(10):
        rows.append(
            {
                "underlying": "MSFT",
                "option_id": "drop_asset",
                "ttm": 45,
                "QUOTE_DATE": f"2020-02-{(i % 28) + 1:02d}",
            }
        )
    rows.append(
        {
            "underlying": "AAPL",
            "option_id": "drop_ttm",
            "ttm": 10,
            "QUOTE_DATE": "2020-03-01",
        }
    )
    df = pd.DataFrame(rows)

    filtered = filter_options(df, asset="aapl", min_data_points=30, min_ttm_days=30)

    assert filtered["option_id"].unique().tolist() == ["keep"]


def test_make_time_splits_builds_expected_windows() -> None:
    dates = pd.date_range("2020-01-01", periods=900, freq="D")
    df = pd.DataFrame({"QUOTE_DATE": dates, "value": range(len(dates))})

    splits = make_time_splits(df, train_years=1, val_months=3, test_years=1)

    assert len(splits) == 1
    train_df, val_df, test_df, info = splits[0]
    assert not train_df.empty and not val_df.empty and not test_df.empty
    assert info["train_start"] == pd.Timestamp("2020-01-01")
    assert info["val_start"] > info["train_start"]
    assert info["test_start"] > info["val_start"]


def test_make_time_splits_requires_quote_date_column() -> None:
    with pytest.raises(KeyError):
        make_time_splits(pd.DataFrame({"x": [1, 2, 3]}))


def test_pricing_models_return_non_negative_values() -> None:
    inputs = (100.0, 95.0, 30.0, 0.2, True)

    assert bs_price(100.0, 95.0, 30.0, 0.05, sigma=0.2, is_call=True) >= 0
    assert binomial_price(inputs, r_flat=0.05, steps=50) >= 0
    assert mc_price(inputs, r_flat=0.05, num_paths=500) >= 0
