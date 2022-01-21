
import os
import shutil
import pandas as pd
import yfinance as yf
import yahoo_fin.stock_info as si
import FundamentalAnalysis as fa
import streamlit as st
import requests
import pickle
import time
import sys
import dateutil.parser
from datetime import datetime as dt
from datetime import timedelta
from yahoo_earnings_calendar import YahooEarningsCalendar

def get_SPY_companies():
    '''Get a list of the companies comprising the S&P500'''
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    # Change '.' to '-' in ticker before df is written
    for row in df.index:
        df.loc[row, 'Symbol'] = df.loc[row, 'Symbol'].replace('.', '-')
    df.to_csv('data/S&P500-Info.csv')

def get_market_data():
    '''Get historical price & volume data for each stock in the S&P500'''    
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

def move_market_data():
    '''Move tickers that have been removed from the S&P500 to their own folder'''
    df = pd.read_csv('data/S&P500-Info.csv')
    symbols = df['Symbol'].to_list()
    for file in os.listdir('data/market_data'):
        ticker = file.replace('.csv', '')
        if ticker not in symbols:
            shutil.move(f'data/market_data/{file}', f'data/removed_from_index/{file}')
            print(f'{ticker} is no longer in the S&P 500.')

def get_financials():
    '''Gets financial statements for the last 4 years for each company'''
    df = pd.read_csv('data/S&P500-Info.csv')
    tickers = df['Symbol'].to_list()
    a_not_downloaded = []
    q_not_downloaded = []
    for i, ticker in enumerate(tickers, start=1):
        file = ticker + '.csv'
        if file in os.listdir('data/financial_statements/Annual/Statements of Financial Position'):
            pass
        else:
            try:
                sfp = si.get_balance_sheet(ticker)
                soci = si.get_income_statement(ticker)
                cfs = si.get_cash_flow(ticker)
                sfp.to_csv(f'data/financial_statements/Annual/Statements of Financial Position/{ticker}.csv')
                soci.to_csv(f'data/financial_statements/Annual/Statements of Comprehensive Income/{ticker}.csv')
                cfs.to_csv(f'data/financial_statements/Annual/Statements of Cash Flows/{ticker}.csv')
                print(i, ticker, 'Annual')
            except:
                a_not_downloaded.append(ticker)
                print(i, ticker, 'Annual not downloaded')

        if file in os.listdir('data/financial_statements/Quarterly/Statements of Financial Position'):
            pass
        else:
            try:
                qsfp = si.get_balance_sheet(ticker, yearly=False)
                qsoci = si.get_income_statement(ticker, yearly=False)
                qcfs = si.get_cash_flow(ticker, yearly=False)
                qsfp.to_csv(f'data/financial_statements/Quarterly/Statements of Financial Position/{ticker}.csv')
                qsoci.to_csv(f'data/financial_statements/Quarterly/Statements of Comprehensive Income/{ticker}.csv')
                qcfs.to_csv(f'data/financial_statements/Quarterly/Statements of Cash Flows/{ticker}.csv')
                print(i, ticker, 'Quarter')
            except:
                a_not_downloaded.append(ticker)
                print(i, ticker, 'Quarterly not downloaded')
        
    print('Missing yearly statements: ', a_not_downloaded)
    print('Missing quarterly statements: ', q_not_downloaded)

def get_financial_ratios():
    '''Download annual financial ratios'''

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
                fr = fa.financial_ratios(ticker, st.secrets['FUNDAMENTAL_ANALYSIS_API_KEY1'], period='annual')
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
                fr = fa.financial_ratios(ticker, st.secrets['FUNDAMENTALS_ANALYSIS_API_KEY3'], period='annual')
                fr.to_csv(f'{f}/{ticker}.csv')
                print(i, ticker, 'downloaded')
            except ValueError as e:
                print(e)
                not_downloaded.append(ticker)
    
    print(not_downloaded)

def get_historical_news(start_date, end_date):
    '''Gets news from the last year for each stock. 
       Ensure that your dates are in yyyy-mm-dd format.'''

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
                        r = requests.get('https://finnhub.io/api/v1/company-news?symbol=' + ticker + '&from=' +
                                        date + '&to=' + date + '&token=' + st.secrets['FINNHUB_API_KEY'])
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
                            nrequests = 0

                    except Exception as e:
                        print(ticker, date, e)
                        pass

                df = pd.DataFrame.from_dict(data)
                df.to_excel(f'data/news/{ticker}.xlsx')
                print('\n', i, ticker, 'done')
                date = start_date
                date_obj = dt.strptime(start_date, "%Y-%m-%d")

def get_reporting_dates():
    '''Extract annual reporting dates from financial statements'''
    f= r'data\financial_statements\Annual\Statements of Cash Flows'
    ticker_reporting_dates = []
    for file in os.listdir(f):
        df = pd.read_csv(f + '/' + file, index_col='Breakdown')
        reporting_date = dt.strptime(df.columns[0], '%Y-%m-%d')
        ticker = file.replace('.csv', '')
        ticker_reporting_dates.append((ticker, reporting_date))
    
    with open(r'data\financial_statements\Reporting Dates\reporting_dates.pickle', 'wb') as f1:
        pickle.dump(ticker_reporting_dates, f1)

def get_filing_dates():
    '''Download earnings calendars of companies.
       By default, requests are delayed by 1.8 sec to avoid exceeding the 2000/hour rate limit. 
       You can override the default delay by passing an argument to the YahooEarningsCalendar constructor.
    '''
    seconds_delay = 0.5
    yec = YahooEarningsCalendar(seconds_delay)  
    df = pd.read_csv('data/S&P500-Info.csv')
    tickers = df['Symbol'].to_list()
    f = r'data\financial_statements\Filing Dates'
    not_downloaded = []
    i = 0
    n = 505 - len(os.listdir(f))

    for ticker in tickers:
        file = ticker + '.csv'
        if file in os.listdir(f):
            pass
        else:
            try:
                i += 1
                earnings_list = yec.get_earnings_of(ticker)
                earnings_df = pd.DataFrame(earnings_list)
                earnings_df['filing_date'] = earnings_df['startdatetime'].apply(lambda x: dateutil.parser.isoparse(x).date())
                earnings_df = earnings_df[['ticker', 'filing_date', 'startdatetimetype']] 
                earnings_df.to_csv(f'{f}/{file}')
                sys.stdout.write("\r")
                sys.stdout.write(f"{(n - i):2d}/{n} files remaining")
                sys.stdout.flush()
            except Exception as e:
                sys.stdout.write(f"\r{ticker} Error: {e}            \n")
                not_downloaded.append(ticker)

    print(f'\nNot downloaded: {len(not_downloaded)} \n{not_downloaded}')

get_SPY_companies()
get_market_data()  
move_market_data()
# get_financials()
# get_reporting_dates()
# get_financial_ratios()
# get_historical_news('2020-07-03', '2021-07-02')
# get_filing_dates()
