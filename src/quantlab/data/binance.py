import requests
import pandas as pd

KLINES_URL = 'https://api.binance.com/api/v3/klines'

KLINE_COLUMNS = [
    'open_time',
    'open',
    'high',
    'low',
    'close',
    'volume',
    'close_time',
    'quote_asset_volume',
    'number_of_trades',
    'taker_buy_base_volume',
    'taker_buy_quote_volume',
    'ignore',
]

FLOAT_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'quote_asset_volume',
    'taker_buy_base_volume',
    'taker_buy_quote_volume',
]


def download_binance_klines(symbol, interval, start, end, limit=1000):
    """
    Download OHLCV candles from the Binance Spot API for a fixed window.

    A fixed window makes the download reproducible: the same call always
    returns the same candles.

    Parameters
    ----------
    symbol : str, e.g., 'BTCUSDT'
    interval: str, e.g. '1d'
    start, end : str or pd.Timestamp
        First and last opening time of the window, in UTC.
    limit: int
        Maximum number of candles (Binance returns at most 1000 per call).
    """
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': _to_milliseconds(start),
        'endTime': _to_milliseconds(end),
        'limit': limit,
    }

    response = requests.get(KLINES_URL, params=params, timeout=30)
    response.raise_for_status() # checks the download went fine
    data = response.json()

    df = pd.DataFrame(data, columns=KLINE_COLUMNS)
    df = clean_data(df)
    validate_data(df)

    return df


def clean_data(df):
    df = df.copy()

    # Binance sends prices and volumes as strings
    df[FLOAT_COLUMNS] = df[FLOAT_COLUMNS].astype(float)

    # Binance also returns the candle of the current period, which is still open:
    # every candle must be complete
    now_ms = pd.Timestamp.now(tz='UTC').timestamp() * 1000
    assert (df['close_time'] < now_ms).all(), 'the last candle is still open: move `end` back'

    # timestamps are milliseconds since the epoch, in UTC
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

    # sanity checks for the open_time before setting it as index
    assert not df['open_time'].duplicated().any(), 'duplicated timestamps'
    assert df['open_time'].is_monotonic_increasing, 'timestamps are not sorted'
    assert df['open_time'].diff().dropna().nunique() <= 1, 'candles are not equally spaced: some are missing'

    df = df.set_index('open_time')

    # remove the columns that are not needed
    df = df.drop(columns=['close_time', 'ignore'])

    # be sure there is no NaN
    assert not df.isna().any().any(), 'missing values'
    return df


def validate_data(df):
    # every bar must satisfy low <= open, close <= high
    checks = {
        'low <= high': df['low'] <= df['high'],
        'low <= open': df['low'] <= df['open'],
        'low <= close': df['low'] <= df['close'],
        'open <= high': df['open'] <= df['high'],
        'close <= high': df['close'] <= df['high'],
    }

    for name, check in checks.items():
        assert check.all(), 'inconsistent bars ({}), first ones: {}'.format(name, list(df.index[~check][:5]))


def save_parquet(df, path):
    """
    Save a DataFrame to a parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str
        Destination path.
    """
    df.to_parquet(path)


def load_parquet(path):
    """
    Load a DataFrame from a parquet file.
    """
    return pd.read_parquet(path)


def _to_milliseconds(timestamp):
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    return int(ts.timestamp() * 1000)
