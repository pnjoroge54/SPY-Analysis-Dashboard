import os
import sys
import pickle
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
from pytz import timezone

import pandas_datareader.data as web
import yfinance as yf
import yahoo_fin.stock_info as si
import fundamentalanalysis as fa
from urllib.request import Request, urlopen
from html_table_parser.parser import HTMLTableParser 
import streamlit as st


def get_SPY_companies():
    '''Get a list of the companies comprising the SPY'''

    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    current = table[0]
    hist = table[1]

    # Create historical df I can use for re-creating index components at different dates
    added = hist['Date'].join(hist['Added'])
    added['inSPY'] = True
    added['Date'] = pd.to_datetime(added['Date'])
    
    removed = hist['Date'].join(hist['Removed'])
    removed['inSPY'] = False
    removed['Date'] = pd.to_datetime(removed['Date'])

    hist = pd.concat([added, removed]).sort_values('Date', ascending=False)
    hist.dropna(inplace=True)

    # Change '.' to '-' in ticker before df is written
    for row in range(len(current)):
        try:
            current.loc[row, 'Symbol'] = current.loc[row, 'Symbol'].replace('.', '-')
        except:
            pass

    for row in range(len(hist)):
        try:
            hist.loc[row, 'Ticker'] = hist.loc[row, 'Ticker'].replace('.', '-')
        except:
            pass
    
    o_current = pd.read_csv('data/SPY-Info.csv')
    o_hist = pd.read_csv('data/SPY-Historical.csv')

    if o_current.equals(current):
        print('SPY-Info is up to date!\n')
    else: 
        current.to_csv('data/SPY-Info.csv', index=False)
        print('SPY-Info updated!\n')

    if o_hist.equals(hist):
        print('SPY-Historical is up to date!\n')
    else: 
        hist.to_csv('data/SPY-Historical.csv', index=False)
        print('SPY-Historical updated!\n')


def get_tickers():
    df = pd.read_csv('data/SPY-Info.csv')
    df1 = pd.read_csv('data/SPY-Historical.csv', index_col='Date')
    df1[:'2015-01-01']
    
    tickers = df['Symbol'].to_list()
    hist_tickers = list(set(tickers + df1['Ticker'].to_list()))

    return tickers, hist_tickers


def url_get_contents(url):
    '''
    Opens a website and reads its binary contents 
    (HTTP Response Body) making request to the website
    '''

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
    '''Get historical data for S&P 500 & for each stock in S&P 500'''

    SPY = yf.Ticker('^GSPC').history(period='max') # download index data
    SPY.to_csv('data/SPY.csv')

    tickers = get_tickers()[0]
    n = len(tickers)
    i = 0
    not_downloaded = []

    # download stock data
    for ticker in tickers:
        path = f'data/market_data/{ticker}.csv'
        # if os.path.exists(path):
        #     ts = int(os.stat(path).st_mtime)
        #     last_changed = dt.now() - dt.utcfromtimestamp(ts)
        #     last_changed = last_changed.days
        # else:
        #     last_changed = -1

        # if last_changed > 0:
        try:
            data = si.get_data(ticker)
            data.to_csv(path)
            i += 1
            sys.stdout.write("\r")
            sys.stdout.write(f"{i}/{n} ({i / n * 100:.2f}%) of SPY market data downloaded")
            sys.stdout.flush()
        except Exception as e:
            print(e)
            not_downloaded.append(ticker)            
    if not_downloaded:
        print('\n', not_downloaded, '\n')


def remove_replaced_tickers():
    '''Move tickers that have been removed from the SPY to their own folder'''

    tickers = get_tickers()[0]
    mkt_path = r'data\market_data'
    ratios_path = r'data\financial_ratios\Annual'
    mkt = [x.replace('.csv', '') for x in os.listdir(mkt_path)]
    ratios = [x.replace('.csv', '') for x in os.listdir(ratios_path)]

    if set(mkt) == set(tickers):
        print('No new companies in SPY\n')
    else:
        mkt = set(mkt) - set(tickers)
        ratios = set(ratios) - set(tickers)
        removed = mkt | ratios
        for ticker in removed:
            file = ticker + '.csv'
            try:
                os.remove(f'{mkt_path}\{file}')
            except:
                pass
            try:
                os.remove(f'{ratios_path}\{file}')
            except:
                continue
            # print(f'{ticker} is no longer in SPY')
    
        
def ratios_to_update():
    tickers = get_tickers()[0]
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
    
    to_update = not_current + no_data + not_downloaded
    
    return to_update


def get_financial_ratios(i=0, n=1):
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
                ratios = fa.financial_ratios(ticker, 
                                             st.secrets[f'FUNDAMENTAL_ANALYSIS_API_KEY{n}'],
                                             period='annual')
                ratios.to_csv(os.path.join(f, f'{ticker}.csv'))
                i += 1
                sys.stdout.write("\r")
                sys.stdout.write(f"{i}/{len(to_update)} outdated financial ratios downloaded")
                sys.stdout.flush()
        except Exception as e:
            if e == '<urlopen error [Errno 11001] getaddrinfo failed>':
                print(e)
            else:
                if n < 5:
                    print(f'\nAPI Key {n} has maxed out its requests\n')
                    n += 1

        return get_financial_ratios(i, n)

    else:
        print('\nAnnual financial ratios are up to date!\n')       


def get_TTM_financial_ratios(i=0, n=1, d={}):
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
    
    tickers = get_tickers()[0]

    # Use recursion to continue building the dict of TTM ratios when API calls
    # for a key reach the limit (250 requests/day)
    if i < len(tickers):
        try:
            for ticker in tickers[i: i + 250]:
                ratios = fa.financial_ratios(ticker, 
                                             st.secrets[f'FUNDAMENTAL_ANALYSIS_API_KEY{n}'],
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
            else:
                if n < 5:
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
    d = get_TTM_financial_ratios()
    tickers = get_tickers()[0]

    if len(d) == len(tickers):
        with open(os.path.join(f, file), 'wb') as f1:
            pickle.dump(d, f1)
        print(file, 'saved\n')
    else:
        print(f'{len(tickers) - len(d)}/{len(tickers)} ratios not downloaded\n')


def get_risk_free_rates():
    # rf_rates = pdr.fred.FredReader('DTB3', dt(1954, 1, 4), dt.now()).read()
    rf_rates = yf.download('^IRX', progress=False)
    rf_rates.to_csv(r'data\T-Bill Rates.csv')
    print('T-Bill Rates saved\n')


def get_multi_factor_model_data():
    start_date = '1954-01-01' 
    df_three_factor = web.DataReader('F-F_Research_Data_Factors', 'famafrench', 
                                    start=start_date)[0]
    df_three_factor.index = df_three_factor.index.format()

    df_mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start=start_date)[0]
    df_mom.index = df_mom.index.format()
    df_four_factor = df_three_factor.join(df_mom)
    df_five_factor = web.DataReader('F-F_Research_Data_5_Factors_2x3', 
                                    'famafrench', start=start_date)[0]
    df_five_factor.index = df_five_factor.index.format()

    df_three_factor.to_csv('data/multi-factor_models/F-F_Research_Data_Factors.csv')
    df_four_factor.to_csv('data/multi-factor_models/Carhart_4_Factors.csv')
    df_five_factor.to_csv('data/multi-factor_models/F-F_Research_Data_5_Factors_2x3.csv')

    print('Multi-factor model data downloaded\n')


def get_financial_statements():
    statements = {}
    tickers = get_tickers()[0]
    i = 0
    n = len(tickers)

    for ticker in tickers:
        base_url = f"https://stockrow.com/api/companies/{ticker}/financials.xlsx?dimension=A&section="
        sofp = f"{base_url}Balance%20Sheet&sort=desc"
        soci = f"{base_url}Income%20Statement&sort=desc"
        socf = f"{base_url}Cash%20Flow&sort=desc"
        try:        
            df1 = pd.read_excel(sofp) # balance sheet data           
            df2 = pd.read_excel(soci) # income statement data
            df3 = pd.read_excel(socf) # cashflow statement data  
            df = pd.concat([df1, df2, df3]) # combining all extracted information
            columns = df.columns.values
            
            for i in range(len(columns)):
                if columns[i] == "Unnamed: 0":
                    columns[i] = "Item"
                else:
                    columns[i] = columns[i].strftime("%Y-%m-%d")
            
            df.columns = columns
            df.set_index("Item", inplace=True)
            statements[ticker] = df
            
            i += 1
            sys.stdout.write("\r")
            sys.stdout.write(f"{i}/{n} ({i / n * 100:.2f}%) statements downloaded")
            sys.stdout.flush()

        except Exception as e:
            print(f'i: {ticker} - {e}')

    with open('data/financial_statements.pickle', 'wb') as f:
        pickle.dump(statements, f)


if __name__ == "__main__":           
    get_SPY_companies()
    get_SPY_weights()
    get_risk_free_rates()
    get_multi_factor_model_data()
    get_market_data()
    remove_replaced_tickers()
    save_TTM_financial_ratios()
    get_financial_ratios()
    get_financial_statements()
