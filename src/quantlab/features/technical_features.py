import numpy as np

def add_returns(df):
    """
    Standard returns: r_t = ( p_t - p_{t-1} ) / p_{t-1}
    """
    df['returns'] = df['close'] / df['close'].shift(1) - 1
    return df

def add_log_returns(df):
    """
    Log returns: log(1 + r_t)
    """
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    return df

def add_rolling_vol(df, window):
    """
    Rolling volatility taken on log_returns
    """
    df[f'rolling_vol_{window}'] = (np.log(df['close'] / df['close'].shift(1))).rolling(window).std()
    return df

def add_sma(df, window):
    df[f'sma_{window}'] = df['close'].rolling(window).mean()
    return df

def add_momentum(df, window):
    """
    Past return over a rolling window
    """
    df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) -  1
    return df