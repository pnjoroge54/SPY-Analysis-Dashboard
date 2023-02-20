import os
import pandas as pd
import pickle


def get_rf_data():
    '''Returns dataframe of 90-day T-Bill Rates'''

    return pd.read_csv(r'data\T-Bill Rates.csv', index_col=0, parse_dates=True)


def get_SPY_info():
    '''Returns dataframe of info about S&P 500 companies'''
    
    df = pd.read_csv(r'data\spy_data\SPY_Info.csv', index_col=0)
    cols = {'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Sub-Industry'}
    df.rename(columns=cols, inplace=True)

    return df


def get_SPY_data():
    '''Returns DataFrame of S&P 500 market data'''

    df = pd.read_csv('data/spy_data/SPY.csv')
    df.index = pd.to_datetime(df['Date'].apply(lambda x: x.split(' ')[0]))
    df.drop(columns='Date', inplace=True)

    return df


def get_ticker_data(ticker):
    '''Load ticker's market data'''
    
    file = os.path.join('data/market_data/daily', f'{ticker}.csv')
    df = pd.read_csv(file) # , index_col=0, parse_dates=True
    df.index = pd.to_datetime(df['Date'].apply(lambda x: x.split(' ')[0]))
    df.drop(columns='Date', inplace=True)
    
    return df


def get_interval_market_data(ticker, interval):
    '''Load ticker's market data'''
    
    if interval.endswith('Min'):
        folder = interval.split(' Min')[0] + 'm'
        col = 'Datetime'
        fmt = ':00-0'
    elif interval == 'Weekly':
        folder = '1wk'
        col = 'Date'
        fmt = ' '
    elif interval == 'Monthly':
        folder = '1mo'

    file = os.path.join(f'data/market_data/{folder}', f'{ticker}.csv')
    df = pd.read_csv(file)
    df.index = pd.to_datetime(df[col].apply(lambda x: x.split(fmt)[0]))
    df.drop(columns=col, inplace=True)
    
    return df


def get_financial_statements():
    '''Load dict of available financial statements of S&P 500 stocks'''

    file = r'data\financial_statements\financial_statements.pickle'

    with open(file, 'rb') as f:
        d = pickle.load(f)
    
    return d


def get_ticker_info():
    fname = 'data/spy_data/spy_tickers_info.pickle'
    with open(fname, 'rb') as f:
        info = pickle.load(f)

    return info
