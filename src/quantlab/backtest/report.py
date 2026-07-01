import pandas as pd
from .metrics import annualized_expected_return, annualized_volatility, equity_curve, max_drawdown, sharpe

def performance_summary(df):
    idx = [
        'total_return',
        'annualized_return',
        'annualized_volatility',
        'sharpe',
        'max_drawdown',
    ]

    values = [
        equity_curve(df).iloc[-1] - 1,
        annualized_expected_return(df),
        annualized_volatility(df),
        sharpe(df),
        max_drawdown(equity_curve(df)),
    ]

    return pd.Series(index=idx, data=values)