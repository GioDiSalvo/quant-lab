import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import metrics
from quantlab.backtest.report import performance_table, format_table, sharpe_verdict
from quantlab.backtest.strategy import sign_signal, positions, gross_returns, net_returns
from quantlab.data.binance import KLINE_COLUMNS, clean_data, validate_data
from quantlab.features.technical_features import add_returns, add_momentum, add_sma
from quantlab.utils.stats import autocorrelations, ljung_box, noise_band

DAYS = pd.date_range('2024-01-01', periods=8)


# ---------------------------------------------------------------- signals

def test_signal_is_zero_where_the_feature_is_undefined():
    x = pd.Series([np.nan, np.nan, 0.5, -0.5, 0.01, -0.01, 0.0, 0.3], index=DAYS)
    assert list(sign_signal(x)) == [0, 0, 1, -1, 1, -1, 0, 1]


def test_signal_threshold_is_symmetric_around_zero():
    x = pd.Series([np.nan, 0.5, -0.5, 0.01, -0.01, 0.02, -0.02, 0.03], index=DAYS)
    assert list(sign_signal(x, threshold=0.02)) == [0, 1, -1, 0, 0, 0, 0, 1]


def test_position_is_the_signal_of_the_previous_day():
    signal = pd.Series([1, -1, 0, 1, 1, -1, 0, 0], index=DAYS, dtype=float)
    assert list(positions(signal)) == [0, 1, -1, 0, 1, 1, -1, 0]


def test_no_look_ahead_in_strategy_returns():
    # the signal turns long on day 2: the return of day 2 must not be earned
    signal = pd.Series([0, 0, 1, 1, 0, 0, 0, 0], index=DAYS, dtype=float)
    returns = pd.Series([0.0, 0.0, 0.10, 0.02, 0.03, 0.0, 0.0, 0.0], index=DAYS)
    gross = gross_returns(signal, returns)
    assert gross.iloc[2] == 0.0
    assert gross.iloc[3] == pytest.approx(0.02)
    assert gross.iloc[4] == pytest.approx(0.03)


# ---------------------------------------------------------------- costs

def test_costs_are_proportional_to_the_size_of_the_trade():
    # enter long (size 1), flip to short (size 2), go flat (size 1)
    signal = pd.Series([1, 1, -1, -1, 0, 0, 0, 0], index=DAYS, dtype=float)
    returns = pd.Series([np.nan, 0.01, 0.02, -0.03, 0.01, 0.0, 0.0, 0.0], index=DAYS)
    paid = (gross_returns(signal, returns) - net_returns(signal, returns, cost=0.001)).fillna(0)
    assert np.allclose(paid, [0, 0.001, 0, 0.002, 0, 0.001, 0, 0])


def test_no_cost_without_trading():
    signal = pd.Series(1.0, index=DAYS)
    returns = pd.Series(0.01, index=DAYS)
    paid = gross_returns(signal, returns) - net_returns(signal, returns, cost=0.001)
    assert paid.iloc[2:].abs().max() == pytest.approx(0.0)   # only the entry on day 1 is charged
    assert paid.iloc[1] == pytest.approx(0.001)


# ---------------------------------------------------------------- metrics

def test_total_return_and_drawdown():
    returns = pd.Series([0.10, -0.50, 0.10])
    assert metrics.total_return(returns) == pytest.approx(1.1 * 0.5 * 1.1 - 1)
    assert metrics.max_drawdown(returns) == pytest.approx(-0.5)
    assert list(metrics.running_drawdown(returns).round(6)) == [0.0, -0.5, -0.45]


def test_cagr_of_a_constant_daily_return():
    daily = 1.10 ** (1 / 365) - 1
    assert metrics.cagr(pd.Series([daily] * 730)) == pytest.approx(0.10)


def test_nan_are_ignored():
    returns = pd.Series([np.nan, 0.01, -0.02, 0.03])
    assert metrics.total_return(returns) == pytest.approx(1.01 * 0.98 * 1.03 - 1)


def test_var_and_expected_shortfall_are_positive_losses():
    returns = pd.Series(np.linspace(-0.10, 0.10, 201))
    var = metrics.historical_var(returns, 0.95)
    es = metrics.expected_shortfall(returns, 0.95)
    assert var == pytest.approx(0.09)
    assert es > var > 0


def test_sharpe_standard_error_is_about_one_over_sqrt_years():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.025, 999))
    assert metrics.sharpe_se(returns) == pytest.approx(1 / np.sqrt(999 / 365), abs=0.01)


def test_sharpe_standard_error_matches_simulation():
    rng = np.random.default_rng(1)
    estimates = [metrics.sharpe(pd.Series(rng.normal(0.0005, 0.025, 999))) for _ in range(1000)]
    assert np.std(estimates) == pytest.approx(1 / np.sqrt(999 / 365), rel=0.1)


def test_a_strategy_that_never_trades_has_no_sharpe():
    flat = pd.Series(np.zeros(100))
    assert np.isnan(metrics.sharpe(flat))
    assert np.isnan(metrics.sharpe_se(flat))
    assert sharpe_verdict(np.nan, np.nan) == 'not defined'


def test_performance_table_has_one_row_per_strategy():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.0005, 0.025, 500))
    table = performance_table({'a': returns, 'b': -returns})
    assert list(table.index) == ['a', 'b']
    assert table.loc['a', 'sharpe'] == pytest.approx(-table.loc['b', 'sharpe'])
    assert format_table(table).shape == table.shape


# ---------------------------------------------------------------- dependence tests

def garch_series(rng, n=1000):
    """No autocorrelation in the level, strong autocorrelation in the variance."""
    r, var = np.empty(n), 0.025 ** 2
    for i in range(n):
        r[i] = np.sqrt(var) * rng.standard_normal()
        var = 0.00003 + 0.10 * r[i] ** 2 + 0.85 * var
    return r


def test_autocorrelation_of_an_ar1_process():
    rng = np.random.default_rng(3)
    x = np.zeros(5000)
    for i in range(1, len(x)):
        x[i] = 0.5 * x[i - 1] + rng.standard_normal()
    rho = autocorrelations(x, lags=2)
    assert rho[1] == pytest.approx(0.5, abs=0.05)
    assert rho[2] == pytest.approx(0.25, abs=0.05)
    assert abs(rho[1]) > noise_band(len(x))
    assert ljung_box(x, lags=20)[1] < 1e-6
    assert ljung_box(x, lags=20, robust=True)[1] < 1e-6


def test_ljung_box_rejects_about_five_percent_of_iid_series():
    rng = np.random.default_rng(4)
    rejections = [ljung_box(rng.standard_normal(500), lags=10)[1] < 0.05 for _ in range(400)]
    assert 0.02 < np.mean(rejections) < 0.09


def test_robust_ljung_box_keeps_its_size_under_volatility_clustering():
    rng = np.random.default_rng(5)
    standard, robust = [], []
    for _ in range(200):
        r = garch_series(rng)
        standard.append(ljung_box(r, lags=20)[1] < 0.05)
        robust.append(ljung_box(r, lags=20, robust=True)[1] < 0.05)
    assert np.mean(standard) > 0.10     # the standard test rejects far too often
    assert np.mean(robust) < 0.10       # the robust test stays close to 5%


# ---------------------------------------------------------------- features

def test_features_are_undefined_during_the_warm_up():
    df = pd.DataFrame({'close': np.linspace(100, 130, 31)})
    df = add_sma(add_momentum(add_returns(df), 20), 20)
    assert df['returns'].isna().sum() == 1
    assert df['momentum_20'].isna().sum() == 20
    assert df['sma_20'].isna().sum() == 19
    assert df['momentum_20'].iloc[20] == pytest.approx(120 / 100 - 1)


# ---------------------------------------------------------------- data checks

def raw_klines(n=5, start='2024-01-01'):
    """Raw candles in the format returned by Binance."""
    day = 86_400_000
    t0 = int(pd.Timestamp(start, tz='UTC').timestamp() * 1000)
    rows = [[t0 + i * day, '100.0', '110.0', '90.0', '105.0', '10.0', t0 + (i + 1) * day - 1,
             '1000.0', 42, '5.0', '500.0', '0'] for i in range(n)]
    return pd.DataFrame(rows, columns=KLINE_COLUMNS)


def test_clean_data_returns_a_numeric_frame_indexed_by_time():
    df = clean_data(raw_klines())
    assert len(df) == 5
    assert df.index[0] == pd.Timestamp('2024-01-01')
    assert df['close'].dtype == float
    assert 'close_time' not in df.columns
    validate_data(df)


def test_clean_data_refuses_a_candle_that_is_still_open():
    raw = raw_klines()
    raw.loc[raw.index[-1], 'close_time'] = int((pd.Timestamp.now(tz='UTC') + pd.Timedelta(hours=1)).timestamp() * 1000)
    with pytest.raises(AssertionError, match='still open'):
        clean_data(raw)


def test_clean_data_refuses_missing_days():
    raw = raw_klines().drop(index=2)
    with pytest.raises(AssertionError, match='missing'):
        clean_data(raw)


def test_validate_data_refuses_inconsistent_bars():
    df = clean_data(raw_klines())
    df.loc[df.index[1], 'low'] = 120.0   # low above high
    with pytest.raises(AssertionError, match='inconsistent bars'):
        validate_data(df)
