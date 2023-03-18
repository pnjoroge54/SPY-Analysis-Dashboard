import os
import pandas as pd
import pickle
from datetime import timedelta, datetime as dt


def get_rf_data():
    '''Returns dataframe of 90-day T-Bill Rates'''

    return pd.read_csv('data/T-Bill Rates.csv', index_col=0, parse_dates=True)


def get_SPY_info():
    '''Returns dataframe of info about S&P 500 companies'''
    
    df = pd.read_csv('data/market_data/spy_data/SPY_Info.csv', index_col=0)
    cols = {'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Sub-Industry'}
    df.rename(columns=cols, inplace=True)

    return df


def get_SPY_data():
    '''Returns DataFrame of S&P 500 market data'''

    df = pd.read_csv('data/market_data/spy_data/1d/SPY.csv')
    df.index = pd.to_datetime(df['Date'].apply(lambda x: x.split(' ')[0]))
    df.drop(columns='Date', inplace=True)

    return df


def get_ticker_data(ticker):
    '''Load ticker's market data'''
    
    fname = os.path.join('data/market_data/1d', f'{ticker}.csv')
    df = pd.read_csv(fname, index_col=0, parse_dates=True)
    df.rename(columns={'adjclose': 'adj close'}, inplace=True)
    df.drop(columns='ticker', inplace=True)
    df.index.name = 'Date'
    df.columns = df.columns.str.title()

    return df


def resample_data(ticker, timeframe):
    '''Load ticker's market data'''

    path = 'data/market_data'  
        
    if timeframe.endswith('m'):
        x = int(timeframe.split('m')[0])
        folder = '5m' if x >= 5 else '1m'
        freq = f'{x}T'
        
        if ticker == '^GSPC':
            fpath = os.path.join(path, 'spy_data', folder)
            fname = os.path.join(fpath, os.listdir(fpath)[-1])
        else:
            fname = os.path.join(path, folder, f'{ticker}.csv')
        
        df = pd.read_csv(fname)
        col = df.columns[0]
        fmt = ':00-0'
        df.index = pd.to_datetime(df[col].apply(lambda x: x.split(fmt)[0]))
        df.drop(columns=col, index=df.index[-1], inplace=True)
        df.index.name = 'Date'

    else:
        freq = 'W-FRI' if timeframe == 'W1' else 'BM'
        if ticker == '^GSPC':
            df = get_SPY_data()
        else:
            df = get_ticker_data(ticker)

    if timeframe not in ('1m', '5m'):
        resampled_df = pd.DataFrame()
        offset = timedelta(minutes=30)
        resampled_df['Open'] = df['Open'].resample(freq, offset=offset).first()
        resampled_df['High'] = df['High'].resample(freq, offset=offset).max()
        resampled_df['Low'] = df['Low'].resample(freq, offset=offset).min()
        resampled_df['Close'] = df['Close'].resample(freq, offset=offset).last()
        resampled_df['Adj Close'] = df['Adj Close'].resample(freq, offset=offset).last()
        resampled_df['Volume'] = df['Volume'].resample(freq, offset=offset).sum()
        df = resampled_df.dropna()

    return df


def get_financial_statements():
    '''Load dict of available financial statements of S&P 500 stocks'''

    file = r'data\financial_statements\financial_statements.pickle'
    
    with open(file, 'rb') as f:
        d = pickle.load(f)
    
    return d


def get_ticker_info():
    fname = 'data/market_data/spy_data/spy_tickers_info.pickle'
    
    with open(fname, 'rb') as f:
        info = pickle.load(f)

    return info