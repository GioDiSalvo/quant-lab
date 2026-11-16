"""From a feature to strategy returns: signal, position, transaction costs."""

import pandas as pd


def sign_signal(x, threshold=0.0):
    """
    +1 where x > threshold, -1 where x < -threshold, 0 otherwise.

    Where x is NaN (for example during the warm-up of a rolling feature)
    the signal is 0: no position is taken on information that does not exist yet.
    """
    signal = pd.Series(0.0, index=x.index)
    signal[x > threshold] = 1.0
    signal[x < -threshold] = -1.0
    return signal


def positions(signal):
    """
    Position held on day t = signal known at the close of day t-1 (no look-ahead).
    """
    return signal.shift(1).fillna(0.0)


def gross_returns(signal, returns):
    """
    Strategy returns before transaction costs.
    """
    return positions(signal) * returns


def net_returns(signal, returns, cost=0.001):
    """
    Strategy returns after proportional transaction costs.

    signal : pd.Series
        Typical values are {-1, 0, 1}, but any numerical position size is allowed.

    cost : float
        Transaction cost paid per unit of turnover. Going from +1 to -1
        is a trade of size 2 and pays twice.
    """
    pos = positions(signal)
    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])
    return pos * returns - cost * turnover
