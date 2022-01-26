
import sys
from pytz import timezone
import os
from os.path import join
import pickle
import shutil
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import yfinance as yf
import yahoo_fin.stock_info as si
import FundamentalAnalysis as fa
from urllib.request import Request, urlopen
from html_table_parser.parser import HTMLTableParser 
import streamlit as st


'''Get a list of the companies comprising the S&P500'''
def get_SPY_companies():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    c_df = pd.read_csv('data/S&P500-Info.csv', index_col='Unnamed: 0')

    # Change '.' to '-' in ticker before df is written
    for row in df.index:
        df.loc[row, 'Symbol'] = df.loc[row, 'Symbol'].replace('.', '-')

    if c_df.equals(df):
        print('S&P 500 info is up to date')
    else: 
        df.to_csv('data/S&P500-Info.csv')
        print('S&P 500 info updated')

def get_tickers():
    df = pd.read_csv('data/S&P500-Info.csv')
    tickers = df['Symbol'].to_list()

    return tickers

'''Get historical price & volume data for each stock in the S&P500,
   as well as S&P500 data'''
def get_market_data():  
    i = 0
    n = 505
    tickers = get_tickers()

    for ticker in tickers:
        i += 1
        # tickerData = yf.Ticker(symbol)
        # data = tickerData.history(period='max')
        data = si.get_data(ticker)
        data.to_csv(f'data/market_data/{ticker}.csv')
        sys.stdout.write("\r")
        sys.stdout.write(f"{i}/{n} ({i/n * 100:.2f}%) of S&P 500 market data downloaded")
        sys.stdout.flush()

    SPY = yf.Ticker('^GSPC').history(period='max')
    SPY.to_csv('data/SPY.csv')

'''Move tickers that have been removed from the S&P500 to their own folder'''
def move_market_data():
    tickers = get_tickers()

    for file in os.listdir('data/market_data'):
        ticker = file.replace('.csv', '')

        if ticker not in tickers:
            shutil.move(f'data/market_data/{file}', f'data/removed_from_index/{file}')
            print(f'{ticker} is no longer in the S&P 500.')

'''Download annual financial ratios'''
def get_financial_ratios():
    tickers = get_tickers()
    not_current = []
    no_data = []
    not_downloaded = []
    f = 'data/financial_ratios/Annual'
    i = 0

    for ticker in tickers:
        file = ticker + '.csv'
        df = pd.read_csv(join(f, file))

        if not df.empty:
            if str(dt.now().year - 1) != df.columns[1]:
                not_current.append(ticker)
        else:
            no_data.append(ticker)

    print(len(not_current, 'not_current'))
    print('no_data:', no_data)
    # for i, ticker in enumerate(tickers[250:500], start=251):
    #     file = ticker + '.csv'
        
    #     if file in os.listdir(f):
    #         pass
    #     else:
    #         try:
    #             fr = fa.financial_ratios(ticker, st.secrets['FUNDAMENTAL_ANALYSIS_API_KEY2'], period='annual')
    #             fr.to_csv(f'{f}/{ticker}.csv')
    #             print(i, ticker, 'downloaded')
    #         except ValueError as e:
    #             print(e)
    #             not_downloaded.append(ticker)

    # for i, ticker in enumerate(tickers[500:], start=501):
    #     file = ticker + '.csv'

    #     if file in os.listdir(f):
    #         pass
    #     else:
    #         try:
    #             fr = fa.financial_ratios(ticker, st.secrets['FUNDAMENTAL_ANALYSIS_API_KEY3'], period='annual')
    #             fr.to_csv(f'{f}/{ticker}.csv')
    #             print(i, ticker, 'downloaded')
    #         except ValueError as e:
    #             print(e)
    #             not_downloaded.append(ticker)

    # print(len(not_downloaded), 'annual ratios not downloaded')
    # print(not_downloaded)

'''Gets market cap weights for stocks, sectors and sub-industries
   from https://www.slickcharts.com/sp500'''
def get_SPY_weights():
    def url_get_contents(url):
        # Opens a website and read its binary contents (HTTP Response Body)
        # making request to the website
        req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
        f = urlopen(req)
        return f.read() # reading contents of the website
 
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
    df.to_csv('data\S&P500 Weights.csv')
    print('S&P 500 weights are up to date!')

def get_TTM_financial_ratios(i, n, d):
    tickers = get_tickers()

    if i < 505:
        try:
            for ticker in tickers[i: i + 250]:
                ratios = fa.financial_ratios(
                                ticker, st.secrets[f'FUNDAMENTAL_ANALYSIS_API_KEY{n}'],
                                period='annual', TTM=True
                                )
                d[ticker] = ratios.to_dict()
                i += 1
                sys.stdout.write("\r")
                sys.stdout.write(f"{i}/{len(tickers)} current financial ratios downloaded")
                sys.stdout.flush()
        except:
            sys.stdout.write("\033[F") # back to previous line 
            sys.stdout.write("\033[K") # clear line 
            
            # Use recursion to continue building the dict of current ratios when API calls
            # for a key reach the limit (250 requests/day)
            if n < 4:
                print(f'API Key {n} has maxed out its requests \
                        \n{i} financial ratios downloaded\n')
                n += 1
                get_TTM_financial_ratios(i, n, d)
            else:
                print(f'API Key {n} has maxed out its requests \
                        \n{i} financial ratios downloaded\n')
        
        return d
    else:
        print('Current ratios are up to date!')
        return d        

def save_TTM_financial_ratios():
    # Set datetime object to EST timezone
    tz = timezone('EST')
    cdate = dt.now(tz)
    hour = cdate.hour

    # Sets the file name to today's date only after the US stock market
    # has closed, otherwise uses the previous day's date. Also sets
    # weekends to Friday's date.
    if cdate.weekday() != 5 & cdate.weekday() != 6 & cdate.weekday() != 0:
        if hour >= 16:
            file = cdate.strftime('%d-%m-%Y') + '.pickle'
        else:
            cdate = cdate - timedelta(days=1)
            file = cdate.strftime('%d-%m-%Y') + '.pickle'
    else:
        if cdate.weekday() == 5:
            cdate = cdate - timedelta(days=1)
        if cdate.weekday() == 6:
            cdate = cdate - timedelta(days=2)
        if cdate.weekday() == 0:
            if hour < 16:
                cdate = cdate - timedelta(days=3)

        file = cdate.strftime('%d-%m-%Y') + '.pickle'

    f = r'data/financial_ratios/Current'
    d = get_TTM_financial_ratios(0, 1, {})

    if len(d) == 505:
        with open(join(f, file), 'wb') as f1:
            pickle.dump(d, f1)
    else:
        print(f'{505 - len(d)}/505 ratios not downloaded\n')

           
get_SPY_companies()
# get_SPY_weights()
# get_market_data()
# move_market_data()
save_TTM_financial_ratios()