from lnai.data.features import BASE_FEATURES, get_feature_list


def test_get_feature_list_valuation_mode_returns_base_features() -> None:
    assert get_feature_list(forecasting=False) == BASE_FEATURES


def test_get_feature_list_forecasting_inserts_price_and_prev_mid() -> None:
    features = get_feature_list(forecasting=True)

    assert features[:4] == ["is_call", "moneyness", "ttm", "price"]
    assert features[4] == "IV"
    assert features[-4:] == ["prev_mid", "vix", "treasury_yield", "inflation"]
