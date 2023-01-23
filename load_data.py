import os
import pandas as pd
import pickle


def get_rf_data():
    '''Returns dataframe of 90-day T-Bill Rates'''

    return pd.read_csv(r'data\T-Bill Rates.csv', index_col='Date', parse_dates=True)


def get_SPY_info():
    '''Returns dataframe of info about S&P 500 companies'''
    
    df = pd.read_csv(r'data\spy_data\SPY_Info.csv', index_col=0)
    cols = {'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Sub-Industry'}
    df.rename(columns=cols, inplace=True)

    return df


def get_SPY_data():
    '''Returns DataFrame of S&P 500 market data'''

    df = pd.read_csv('data/spy_data/SPY.csv', index_col='Date', parse_dates=['Date'])
    df.index = pd.to_datetime(df.index, utc=True)

    return df


def get_ticker_data(ticker):
    file = os.path.join(r'data\market_data', f'{ticker.upper()}.csv')
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    
    return df


def get_financial_statements():
    '''Load dict of available financial statements of S&P 500 stocks'''

    file = r'data\financial_statements\financial_statements.pickle'

    with open(file, 'rb') as f:
        d = pickle.load(f)
    
    return d