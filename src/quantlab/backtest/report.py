import pandas as pd
from .metrics import annualized_expected_return, annualized_volatility, equity_curve, max_drawdown, sharpe, historical_var, expected_shortfall

def performance_summary(df):
    idx = [
        'total_return',
        'annualized_return',
        'annualized_volatility',
        'sharpe',
        'var_95',
        'exp_shortfall_95',
        'max_drawdown',
    ]

    values = [
        equity_curve(df).iloc[-1] - 1,
        annualized_expected_return(df),
        annualized_volatility(df),
        sharpe(df),
        historical_var(df, 0.95),
        expected_shortfall(df, 0.95),
        max_drawdown(equity_curve(df)),
    ]

    return pd.Series(index=idx, data=values)