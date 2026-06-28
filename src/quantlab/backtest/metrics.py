import numpy as np

def equity_curve(df):
    """
    Given an initial capital V_0 and first days returns (r_1, r_2, r_3), then the portfolio value is
    V_1 = V_0 * (1 + r_1)
    V_2 = V_1 * (1 + r_2) = V_0 * (1 + r_1) * (1 + r_2) 
    V_3 = V_2 * (1 + r_3) = V_0 * (1 + r_1) * (1 + r_2) * (1 + r_3)
    """
    returns = _returns(df)
    return (returns + 1).cumprod()

def annualized_expected_return(df):
    """
    Approximate annualized expected return.
    """
    N = 252
    returns = _returns(df)
    return N * returns.mean()

def annualized_volatility(df):
    N = 252
    returns = _returns(df)
    return np.sqrt(N) * returns.std()

def sharpe(df):
    vol = annualized_volatility(df)
    if vol == 0:
        return np.nan
    return annualized_expected_return(df) / vol

def max_drawdown(equity):
    """
    Largest peak-to-valley fall in percentage
    """    
    return (equity/equity.cummax() - 1).min()

def _returns(df):
    if 'returns' in df.columns:
        return df['returns'].dropna()
    return (df['close']/df['close'].shift(1) - 1).dropna()