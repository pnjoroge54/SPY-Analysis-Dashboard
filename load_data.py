import numpy as np
import pandas as pd
from datetime import timedelta


def get_rf_data():
    '''Make DataFrame of 90-day T-Bill Rates'''

    return pd.read_csv(r'data\T-Bill Rates.csv', index_col='Date', parse_dates=True)


def get_SPY_info():
    '''Make DataFrame of info about S&P 500 companies'''
    
    return pd.read_csv(r'data\SPY-Info.csv')


def get_SPY_data():
    '''Make DataFrame of S&P 500 market data'''

    df = pd.read_csv(r'data\SPY.csv')
    df.index = pd.to_datetime(df['Date'].apply(lambda x: x.split(' ')[0]))
    df.drop(columns='Date', inplace=True)
    df['Return'] = np.log1p(df['Close'].pct_change())

    return df
