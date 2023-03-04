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
    
    file = os.path.join('data/market_data/1d', f'{ticker}.csv')
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    df.rename(columns={'adjclose': 'adj close'}, inplace=True)
    df.drop(columns='ticker', inplace=True)
    df.columns = df.columns.str.title()

    return df


def resample_data(ticker, interval):
    '''Load ticker's market data'''
        
    if interval.endswith('m'):
        fmt = ':00-0'
        x = int(interval.split('m')[0])
        folder = '5m' if x >= 5 else '1m'
        freq = f'{x}T'
    else:
        fmt = ' '
        folder = '1d'
        freq = 'W-FRI' if interval == 'Weekly' else 'BM'

    file = os.path.join(f'data/market_data/{folder}', f'{ticker}.csv')
    df = pd.read_csv(file)
    col = df.columns[0]
    df.index = pd.to_datetime(df[col].apply(lambda x: x.split(fmt)[0]))
    df.drop(columns=col, inplace=True)
    df.index.name = 'Date'

    if not interval.endswith('m'):
        df.rename(columns={'adjclose': 'adj close'}, inplace=True)
        df.drop(columns='ticker', inplace=True)
        df.columns = df.columns.str.title()

    if interval not in ('1m', '5m'):
        resampled_df = pd.DataFrame()
        resampled_df['Open'] = df['Open'].resample(freq).first()
        resampled_df['High'] = df['High'].resample(freq).max()
        resampled_df['Low'] = df['Low'].resample(freq).min()
        resampled_df['Close'] = df['Close'].resample(freq).last()
        resampled_df['Adj Close'] = df['Adj Close'].resample(freq).last()
        resampled_df['Volume'] = df['Volume'].resample(freq).sum()
        resampled_df.dropna(inplace=True)
        df = resampled_df

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