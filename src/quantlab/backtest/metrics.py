"""Performance and risk statistics of a series of daily returns.

Every function takes a pd.Series of daily returns (NaN are ignored).
"""

import numpy as np

# BTC/USDT trades every day of the year, so daily figures are annualized with 365.
DAYS_PER_YEAR = 365


def equity_curve(returns):
    """
    Given an initial capital V_0 = 1 and daily returns (r_1, r_2, r_3), the portfolio value is
    V_1 = V_0 * (1 + r_1)
    V_2 = V_1 * (1 + r_2) = V_0 * (1 + r_1) * (1 + r_2)
    V_3 = V_2 * (1 + r_3) = V_0 * (1 + r_1) * (1 + r_2) * (1 + r_3)
    """
    return (1 + returns.dropna()).cumprod()


def running_drawdown(returns):
    """
    Percentage distance of the equity curve from its previous peak (zero at a new high).
    """
    equity = equity_curve(returns)
    return equity / equity.cummax() - 1


def max_drawdown(returns):
    """
    Largest peak-to-valley fall in percentage.
    """
    return running_drawdown(returns).min()


def total_return(returns):
    return equity_curve(returns).iloc[-1] - 1


def cagr(returns, days=DAYS_PER_YEAR):
    """
    Compound annual growth rate: the constant yearly return that gives the same total return.
    """
    r = returns.dropna()
    return (1 + r).prod() ** (days / len(r)) - 1


def annualized_volatility(returns, days=DAYS_PER_YEAR):
    return np.sqrt(days) * returns.dropna().std()


def sharpe(returns, days=DAYS_PER_YEAR, rf_annual=0.0):
    """
    Annualized Sharpe ratio. Convention used in the notebooks: risk-free rate = 0.
    """
    r = returns.dropna() - rf_annual / days
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return np.sqrt(days) * r.mean() / r.std()


def sharpe_se(returns, days=DAYS_PER_YEAR):
    """
    Standard error of the annualized Sharpe ratio (Lo, 2002, i.i.d. returns).

    For the per-period Sharpe ratio s estimated on n observations,
    se(s) = sqrt((1 + s^2 / 2) / n). Annualizing multiplies by sqrt(days),
    which gives approximately 1 / sqrt(number of years).
    """
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    s = r.mean() / r.std()
    return np.sqrt(days * (1 + 0.5 * s ** 2) / len(r))


def historical_var(returns, confidence_level=0.95):
    """
    Historical Value at Risk, reported as a positive loss.
    """
    q = returns.dropna().quantile(1 - confidence_level)
    return -min(0.0, q)


def expected_shortfall(returns, confidence_level=0.95):
    """
    Average loss on the days beyond the VaR threshold, reported as a positive number.
    """
    r = returns.dropna()
    tail_returns = r[r <= r.quantile(1 - confidence_level)]
    if tail_returns.empty:
        return 0.0
    return max(-tail_returns.mean(), 0.0)
