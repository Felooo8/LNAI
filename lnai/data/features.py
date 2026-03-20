BASE_FEATURES = [
    'is_call',
    'moneyness',
    'ttm',
    'IV',
    'underlying_close',
    'underlying_log_ret',
    # macro features appended at the end
    'vix',
    'treasury_yield',
    'inflation',
]

def get_feature_list(forecasting: bool = False):
    """Return the ordered list of features.

    Parameters
    ----------
    forecasting : bool, default False
        If True, include past price information used for forecasting
        (current price and previous day's mid price) at the appropriate
        positions. When False, only the base valuation features are
        returned.
    """
    feats = BASE_FEATURES.copy()
    if forecasting:
        # price as feature right after IV for forecasting tasks
        feats.insert(3, 'price')
        # previous day's mid price before macro features
        if 'vix' in feats:
            feats.insert(feats.index('vix'), 'prev_mid')
        else:
            # fallback: append if macro features missing
            feats.append('prev_mid')
    return feats
