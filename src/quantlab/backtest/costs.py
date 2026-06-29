def apply_transaction_costs(gross_returns, signal, cost_per_trade):
    """
    gross_returns : pd.Series
        Strategy returns before transaction costs.

    signal : pd.Series
        Position held through time. Typical values are {-1, 0, 1},
        but any numerical position size is allowed.

    cost_per_trade : float
        Transaction cost paid per unit of turnover.
    """
    turnover = abs(signal - signal.shift(1))
    return gross_returns - cost_per_trade * turnover