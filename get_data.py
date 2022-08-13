import os
import sys
import pickle
import shutil
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
from pytz import timezone

import pandas_datareader as pdr
import yfinance as yf
import yahoo_fin.stock_info as si
import FundamentalAnalysis as fa
from urllib.request import Request, urlopen
from html_table_parser.parser import HTMLTableParser 
import streamlit as st


def get_SPY_companies():
    '''Get a list of the companies comprising the SPY'''

    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    c_df = pd.read_csv('data/SPY-Info.csv')
    
    # Change '.' to '-' in ticker before df is written
    for row in range(len(df)):
        df.loc[row, 'Symbol'] = df.loc[row, 'Symbol'].replace('.', '-')
    
    if c_df.equals(df):
        print('\nSPY info is up to date!\n')
    else: 
        df.to_csv('data/SPY-Info.csv', index=False)
        print('\nSPY info updated!\n')


def get_tickers():
    df = pd.read_csv('data/SPY-Info.csv')
    tickers = df['Symbol'].to_list()

    return tickers


def url_get_contents(url):
    # Opens a website and read its binary contents (HTTP Response Body)
    # making request to the website
    req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
    f = urlopen(req)
    
    return f.read() # reading contents of the website


def get_SPY_weights():
    '''Gets market cap weights for stocks, sectors and sub-industries'''

    url = 'https://www.slickcharts.com/sp500'
    xhtml = url_get_contents(url).decode('utf-8')
    p = HTMLTableParser() # Defining the HTMLTableParser object
    p.feed(xhtml) # feeding the html contents in the HTMLTableParser object  
    df = pd.DataFrame(p.tables[0])
    new_header = df.iloc[0] # grab the first row for the header
    df = df[1:] # take the data less the header row
    df.columns = new_header # set the header row as the df header
    df.drop(['#', 'Price', 'Chg', '% Chg'], axis=1, inplace=True)
    df['Weight'] = pd.to_numeric(df['Weight'])
    
    for i in df.index:
        df.loc[i, 'Symbol'] = df.loc[i, 'Symbol'].replace('.', '-')
    
    df.set_index('Symbol', inplace=True)
    df.to_csv('data/SPY Weights.csv')
    print('SPY weights have been updated!\n')


def get_market_data():  
    '''
    Get historical price & volume data for each stock in the SPY,
    as well as SPY data
    '''

    tickers = get_tickers()
    n = len(tickers)
    i = 0

    # download stock data
    for ticker in tickers:
        i += 1
        data = si.get_data(ticker)
        data.to_csv(f'data/market_data/{ticker}.csv')
        sys.stdout.write("\r")
        sys.stdout.write(f"{i}/{n} ({i/n * 100:.2f}%) of SPY market data downloaded")
        sys.stdout.flush()

    # download index data
    SPY = yf.Ticker('^GSPC').history(period='max')
    SPY.to_csv('data/SPY.csv')
    print('\n')


def move_market_data():
    '''Move tickers that have been removed from the SPY to their own folder'''

    tickers = get_tickers()
    files = [x.replace('.csv', '') for x in os.listdir('data/market_data')]

    if set(files) == set(tickers):
        print('No new companies in SPY\n')
    else:
        missing = set(files) - set(tickers)
        for ticker in missing:
            file = ticker + '.csv'
            shutil.move(f'data/market_data/{file}', f'data/removed_from_index/{file}')
            print(f'{ticker} is no longer in SPY.')
    
        
def ratios_to_update():
    tickers = get_tickers()
    not_current = []
    no_data = []
    not_downloaded = []
    f = 'data/financial_ratios/Annual'

    for ticker in tickers:
        file = ticker + '.csv'
        
        if file in os.listdir(f):
            df = pd.read_csv(os.path.join(f, file))
            if not df.empty:
                if str(dt.now().year - 1) != df.columns[1]:
                    not_current.append(ticker)
            else:
                no_data.append(ticker)
        else:
            not_downloaded.append(ticker)

    print(f'{len(not_current) if not_current is not None else 0} in not_current: {not_current}')
    print(f'{len(no_data) if no_data is not None else 0} in no_data: {no_data}')
    
    to_update = not_current + no_data + not_downloaded
    
    return to_update


def get_financial_ratios(i, n):
    '''
    Downloads annual financial ratios

    Parameters
    ----------

    i : int
        A counter of the tickers whose ratios have been downloaded
    n : int
        Calls the API key according to the number, e.g., key{n}

    Returns
    -------
    ratios.to_csv : csv
        CSVs of all downloaded ratios for SPY stocks
    '''

    f = 'data/financial_ratios/Annual'
    to_update = ratios_to_update()
    
    # Use recursion to continue looping through tickers when API calls
    # for a key reach the limit (250 requests/day)
    if i < len(to_update):
        try:
            for ticker in to_update[i: i + 250]:
                ratios = fa.financial_ratios(ticker, st.secrets[f'FUNDAMENTAL_ANALYSIS_API_KEY{n}'],
                                             period='annual')
                ratios.to_csv(os.path.join(f, f'{ticker}.csv'))
                i += 1
                sys.stdout.write("\r")
                sys.stdout.write(f"{i} / {len(to_update)} outdated financial ratios downloaded")
                sys.stdout.flush()
        except Exception as e:
            if e == '<urlopen error [Errno 11001] getaddrinfo failed>':
                print(e)
                return -1
            else:
                if n <= 4:
                    print(f'\nAPI Key {n} has maxed out its requests\n')
                    n += 1

        return get_financial_ratios(i, n)

    else:
        print('\n\nAnnual financial ratios are up to date!\n')       


def get_TTM_financial_ratios(i, n, d):
    '''
    Downloads trailing twelve month (TTM) financial ratios

    Parameters
    ----------

    i : int
        A counter of the tickers whose ratios have been downloaded
    n : int
        Calls the API key according to the number, e.g., key{n}
    d : dictionary
        Dictionary of the ratios downloaded for each ticker

    Returns
    -------
    d : dict
        Dictionary of all TTM ratios for SPY stocks
    '''

    tickers = get_tickers()

    # Use recursion to continue building the dict of TTM ratios when API calls
    # for a key reach the limit (250 requests/day)
    if i < len(tickers):
        try:
            for ticker in tickers[i: i + 250]:
                ratios = fa.financial_ratios(ticker, st.secrets[f'FUNDAMENTAL_ANALYSIS_API_KEY{n}'],
                                             period='annual', TTM=True)
                d[ticker] = ratios.to_dict()
                i += 1
                sys.stdout.write("\r")
                sys.stdout.write(f"{i}/{len(tickers)} current financial ratios downloaded")
                sys.stdout.write("\033[F") # back to previous line 
                sys.stdout.write("\033[K") # clear line 
        except Exception as e:
            if e == '<urlopen error [Errno 11001] getaddrinfo failed>':
                print(e)
                return -1
            else:
                if n <= 4:
                    print(f'\nAPI Key {n} has maxed out its requests\n')
                    n += 1

        return get_TTM_financial_ratios(i, n, d) 

    else:
        print('\nCurrent ratios are up to date!\n')
        return d

            
def save_TTM_financial_ratios():
    '''Save ratios as pickle file'''

    # Set datetime object to EST timezone
    tz = timezone('EST')
    cdate = dt.now(tz)
    hour = cdate.hour

    # Sets the file name to today's date only after the US stock market
    # has closed, otherwise uses the previous day's date. Also sets
    # weekends to Friday's date.
    if cdate.weekday() != 5 and cdate.weekday() != 6 and cdate.weekday() != 0:
        if hour < 16:
            cdate -= timedelta(days=1)
    else:
        days = 0

        if cdate.weekday() == 5:
            days = 1
        elif cdate.weekday() == 6:
            days = 2
        elif cdate.weekday() == 0:
            if hour < 16:
                days = 3
        
        cdate -= timedelta(days=days)
    
    file = cdate.strftime('%d-%m-%Y') + '.pickle'
    f = 'data/financial_ratios/Current'
    d = get_TTM_financial_ratios(0, 1, {})
    tickers = get_tickers()

    if len(d) == len(tickers):
        with open(os.path.join(f, file), 'wb') as f1:
            pickle.dump(d, f1)

        print(file, 'saved\n')

    else:
        print(f'{len(tickers) - len(d)} / {len(tickers)} ratios not downloaded\n')


def get_risk_free_rates():
    rf_rates = pdr.fred.FredReader('DTB3', dt(1954, 1, 4), dt.now()).read()
    rf_rates.to_csv(r'data\T-Bill Rates.csv')
    print('\nT-Bill Rates saved\n')


if __name__ == "__main__":           
    get_SPY_companies()
    get_SPY_weights()
    get_market_data()
    move_market_data()
    get_risk_free_rates()
    save_TTM_financial_ratios()
    get_financial_ratios(0, 1)
    

