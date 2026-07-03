def portfolio_returns(strategies_returns, weights):
    """
    strategy_returns: pd.DataFrame
        Columns are strategy returns.
    weights: dict or pd.Series
        Portfolio weights for each strategy.
    """
    return (strategies_returns * weights).sum(1)