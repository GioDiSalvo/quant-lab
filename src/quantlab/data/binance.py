import requests
import pandas as pd

def download_binance_klines(symbol, interval, limit = 1000):
    """
    Download OHLCV data from Binance Spot API.

    Parameters
    ----------
    symbol : str, e.g., 'BTCUSDT'
    interval: str, e.g. '1d'
    limit: int
    """
    url = 'https://api.binance.com/api/v3/klines'

    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
    }

    response = requests.get(url, params=params)
    response.raise_for_status() # checks the download went fine
    data = response.json()

    columns = [
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

    df = pd.DataFrame(data, columns=columns)
    df = clean_data(df)
    validate_data(df)

    return df

def clean_data(df):
    # put the correct dtype for the features
    float_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'quote_asset_volume',
        'taker_buy_base_volume',
        'taker_buy_quote_volume',
    ]
    df[float_cols] = df[float_cols].astype(float)

    # convert from ms (standard for Binance) to dd-mm-yyyy format
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms').dt.round('s')

    # sanity check for the open_time before setting it as index
    assert not df['open_time'].duplicated().any()
    assert df['open_time'].is_monotonic_increasing

    df = df.set_index('open_time')

    # remove useless column
    df = df.drop(columns=['close_time', 'ignore'])

    # be sure there is no NAN
    assert not df.isna().any().any()
    return df

def validate_data(df):
    # basic sanity checks
    checks = [
        df['low'] <= df['high'],
        df['low'] <= df['open'],
        df['low'] <= df['close'],
        df['open'] <= df['high'],
        df['close'] <= df['high'],
    ]

    for check in checks:
        if not check.all():
            print('error at ', check)
        assert check.all()
