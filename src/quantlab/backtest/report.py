"""Tables that compare strategies."""

import numpy as np
import pandas as pd

from .metrics import (DAYS_PER_YEAR, total_return, cagr, annualized_volatility, sharpe, sharpe_se,
                      historical_var, expected_shortfall, max_drawdown)


def performance_table(strategies, days=DAYS_PER_YEAR):
    """
    One row per strategy.

    strategies : dict
        Maps the name of a strategy to its pd.Series of daily returns.
    """
    rows = {}
    for name, returns in strategies.items():
        rows[name] = {
            'total_return': total_return(returns),
            'cagr': cagr(returns, days),
            'ann_volatility': annualized_volatility(returns, days),
            'sharpe': sharpe(returns, days),
            'sharpe_se': sharpe_se(returns, days),
            'var_95': historical_var(returns, 0.95),
            'exp_shortfall_95': expected_shortfall(returns, 0.95),
            'max_drawdown': max_drawdown(returns),
        }
    return pd.DataFrame(rows).T


def format_table(table):
    """
    Readable copy of a performance table: percentages, and two decimals for the Sharpe columns.
    """
    out = pd.DataFrame(index=table.index)
    for col in table.columns:
        if col == 'sharpe':
            fmt = '{:+.2f}'
        elif col == 'sharpe_se':
            fmt = '{:.2f}'
        else:
            fmt = '{:.1%}'
        out[col] = table[col].map(fmt.format)
    return out


def sharpe_verdict(sharpe_value, se):
    """
    Plain-language reading of a Sharpe ratio against its standard error.
    """
    if np.isnan(sharpe_value) or np.isnan(se):
        return 'not defined'
    if abs(sharpe_value) < 2 * se:
        return 'indistinguishable from zero'
    return 'different from zero at two standard errors'
