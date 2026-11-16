"""Tests of dependence in a time series."""

import numpy as np
import pandas as pd
from scipy.stats import chi2


def noise_band(n):
    """
    Approximate 95% band for a sample correlation of n pairs when the true value is 0.
    """
    return 2 / np.sqrt(n)


def autocorrelations(x, lags=20):
    """
    Sample autocorrelations at lags 1..lags (standard estimator).
    """
    e = pd.Series(x).dropna().to_numpy(dtype=float)
    e = e - e.mean()
    denom = np.sum(e ** 2)
    return pd.Series([np.sum(e[k:] * e[:-k]) / denom for k in range(1, lags + 1)],
                     index=range(1, lags + 1))


def ljung_box(x, lags=20, robust=False):
    """
    Ljung-Box test that the first `lags` autocorrelations are all zero.

    Returns (Q, p-value). A small p-value is evidence of autocorrelation.

    The standard test assumes constant variance. When volatility clusters, the
    sample autocorrelation at lag k has variance (1 + gamma_k / sigma^4) / n, where
    gamma_k is the autocovariance of the squared (demeaned) series at lag k, instead
    of 1 / n. With robust=True each squared autocorrelation is divided by that factor
    (Diebold, 1986; Lobato, Nankervis and Savin, 2001), so the test keeps its size
    under conditional heteroskedasticity.
    """
    e = pd.Series(x).dropna().to_numpy(dtype=float)
    e = e - e.mean()
    n = len(e)
    var = np.mean(e ** 2)
    rho = autocorrelations(e, lags).to_numpy()
    q = 0.0
    for k in range(1, lags + 1):
        factor = 1.0
        if robust:
            factor = np.mean(e[k:] ** 2 * e[:-k] ** 2) / var ** 2
        q += rho[k - 1] ** 2 / ((n - k) * factor)
    q *= n * (n + 2)
    return q, chi2.sf(q, df=lags)
