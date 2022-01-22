
import sys
import time
import os
import shutil
import requests
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import yfinance as yf
import yahoo_fin.stock_info as si
import FundamentalAnalysis as fa
import streamlit as st

'''Get a list of the companies comprising the S&P500'''
def get_SPY_companies():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    # Change '.' to '-' in ticker before df is written
    for row in df.index:
        df.loc[row, 'Symbol'] = df.loc[row, 'Symbol'].replace('.', '-')
    df.to_csv('data/S&P500-Info.csv')

'''Get historical price & volume data for each stock in the S&P500'''
def get_market_data():  
    i = 0
    n = 505

    with open('data/S&P500-Info.csv') as f:
        lines = f.read().splitlines()
        for company in lines[1:]:
            i += 1
            ticker = company.split(',')[1]
            # tickerData = yf.Ticker(symbol)
            # data = tickerData.history(period='max')
            data = si.get_data(ticker)
            data.to_csv(f'data/market_data/{ticker}.csv')
            sys.stdout.write("\r")
            sys.stdout.write(f"{((i/n) * 100):.2f}% complete")
            sys.stdout.flush()

    index_data = yf.Ticker('^GSPC').history(period='max')
    index_data.to_csv('data/SPY.csv')

'''Move tickers that have been removed from the S&P500 to their own folder'''
def move_market_data():
    df = pd.read_csv('data/S&P500-Info.csv')
    symbols = df['Symbol'].to_list()
    for file in os.listdir('data/market_data'):
        ticker = file.replace('.csv', '')
        if ticker not in symbols:
            shutil.move(f'data/market_data/{file}', f'data/removed_from_index/{file}')
            print(f'{ticker} is no longer in the S&P 500.')

'''Download annual financial ratios'''
def get_financial_ratios():
    df = pd.read_csv('data/S&P500-Info.csv')
    tickers = df['Symbol'].to_list()
    not_downloaded = []
    f = 'data/financial_ratios/Annual'

    for i, ticker in enumerate(tickers[:250], start=1):
        file = ticker + '.csv'

        if file in os.listdir(f):
            pass
        else:
            try:
                fr = fa.financial_ratios(ticker, st.secrets['FUNDAMENTAL_ANALYSIS_API_KEY1'],
                                         period='annual')
                fr.to_csv(f'{f}/{ticker}.csv')
                print(i, ticker, 'downloaded')
            except ValueError as e:
                print(e)
                not_downloaded.append(ticker)

    for i, ticker in enumerate(tickers[250:500], start=251):
        file = ticker + '.csv'
        
        if file in os.listdir(f):
            pass
        else:
            try:
                fr = fa.financial_ratios(ticker, st.secrets['FUNDAMENTAL_ANALYSIS_API_KEY2'], period='annual')
                fr.to_csv(f'{f}/{ticker}.csv')
                print(i, ticker, 'downloaded')
            except ValueError as e:
                print(e)
                not_downloaded.append(ticker)

    for i, ticker in enumerate(tickers[500:], start=501):
        file = ticker + '.csv'

        if file in os.listdir(f):
            pass
        else:
            try:
                fr = fa.financial_ratios(ticker, st.secrets['FUNDAMENTAL_ANALYSIS_API_KEY3'], period='annual')
                fr.to_csv(f'{f}/{ticker}.csv')
                print(i, ticker, 'downloaded')
            except ValueError as e:
                print(e)
                not_downloaded.append(ticker)

    print(len(not_downloaded), 'annual ratios not downloaded')
    print(not_downloaded)

'''Gets news from the last year for each stock. 
   Ensure that your dates are in yyyy-mm-dd format.'''
def get_historical_news(start_date, end_date):
    # Calculates number of days between start and end dates
    ndays = (dt.strptime(end_date, "%Y-%m-%d") - dt.strptime(start_date, "%Y-%m-%d")).days

    if ndays > 365:
        return 'We can only obtain news from the last 1 year. Re-enter your start and end dates.'
    elif ndays < 0:
        return 'Your start date is older than your end date. Re-enter your start and end dates.'
    else:
        df = pd.read_csv('data/S&P500-Info.csv')
        tickers = df['Symbol'].to_list()
        max_calls = 60
        nrequests = 0
        date = start_date
        date_obj = dt.strptime(start_date, "%Y-%m-%d")

        for i, ticker in enumerate(tickers, start=1):
            file = ticker + '.xlsx'

            if file in os.listdir('data/news'):
                print(i, ticker, 'already downloaded')
            else:
                print('\n')
                data = []

                for item in range(ndays + 1):
                    try:
                        nrequests += 1
                        print(nrequests, ticker, date, 'downloading...')
                        r = requests.get('https://finnhub.io/api/v1/company-news?symbol=' \
                                         + ticker + '&from=' + date + '&to=' + date \
                                         + '&token=' + st.secrets['FINNHUB_API_KEY'])
                        data += r.json()
                        date_obj = date_obj + timedelta(days=1)
                        date  = date_obj.strftime("%Y-%m-%d")

                        # Stops API calls for a minute after the max_calls per minute are reached
                        if nrequests == max_calls:
                            print('\nSleeping...')
                            # Prints out the countdown while it sleeps
                            for i in range(60, 0, -1):
                                sys.stdout.write("\r")
                                sys.stdout.write(f"{i:2d} seconds remaining")
                                sys.stdout.flush()
                                time.sleep(1)

                            sys.stdout.write("\rSleeping done                  \n")
                            print('\n')
                    except Exception as e:
                        print(ticker, date, e)
                        pass

                df = pd.DataFrame.from_dict(data)
                df.to_excel(f'data/news/{ticker}.xlsx')
                print('\n', i, ticker, 'done')
                date = start_date
                date_obj = dt.strptime(start_date, "%Y-%m-%d")
                nrequests = 0

end = dt.today().strftime("%Y-%m-%d")
start = (dt.today() - timedelta(365)).strftime("%Y-%m-%d")
                
# get_SPY_companies()
get_market_data()  
# move_market_data()
# get_financial_ratios()
# get_historical_news(start, end)
