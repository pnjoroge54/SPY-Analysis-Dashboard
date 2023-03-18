import os
import pickle
import pandas as pd
import time
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
import winsound


f_start = time.time()

def get_SPY_companies():
    '''Get a list of the companies comprising the S&P 500'''

    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    current = table[0]
    hist = table[1]

    # Create historical df for re-creating index components at different dates
    added = hist['Date'].join(hist['Added'])
    added['inSPY'] = True
    added['Date'] = pd.to_datetime(added['Date'])
    
    removed = hist['Date'].join(hist['Removed'])
    removed['inSPY'] = False
    removed['Date'] = pd.to_datetime(removed['Date'])

    hist = pd.concat([added, removed]).sort_values('Date', ascending=False).dropna()

    current['Symbol'] = current['Symbol'].str.replace('.', '-', regex=False)
    hist['Ticker'] = hist['Ticker'].str.replace('.', '-', regex=False)
    
    current_fname = r'data\market_data\spy_data\SPY_Info.csv'
    hist_fname = r'data\market_data\spy_data\SPY_Historical.csv'
    o_current = pd.read_csv(current_fname)
    o_hist = pd.read_csv(hist_fname)

    if o_current.equals(current):
        print('\nSPY_Info is up to date\n')
    else: 
        current.to_csv(current_fname, index=False)
        print('\nSPY_Info updated\n')

    if o_hist.equals(hist):
        print('SPY_Historical is up to date\n')
    else: 
        hist.to_csv(hist_fname, index=False)
        print('SPY_Historical updated\n')


def get_tickers():
    df = pd.read_csv(r'data\market_data\spy_data\SPY_Info.csv')
    tickers = df['Symbol'].to_list()

    return tickers


def url_get_contents(url):
    '''Opens website and reads the binary contents (HTTP Response Body)'''

    req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
    f = urlopen(req)
    
    return f.read() # reading contents of the website


def get_SPY_weights():
    '''Download market cap weights for stocks as a CSV file'''

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
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    df.set_index('Symbol', inplace=True)
    df.to_csv(r'data\market_data\spy_data\SPY_Weights.csv')
    print('S&P 500 weights updated \n')

    
def get_interval_market_data(intervals=['1d', '1m', '5m'], path=r'data\market_data'):
    t_start = time.time()
    tickers = get_tickers()
    end = dt.now()
    weekday = end.weekday()
    days = 0

    if weekday >= 5:
        days += weekday - 5
    elif weekday == 0 and end.hour < 24:
        days += 2
    
    end -= timedelta(days)
    spy_path = os.path.join(path, 'spy_data')
    
    for interval in intervals:
        t_path = os.path.join(path, interval)
        os.makedirs(spy_path, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        n = len(tickers)
        j = 0
        start = end - timedelta(7 - days) if interval == '1m' else end - timedelta(60 - days)
        f_end = end - timedelta(1)
        if interval == '1d':
            df = yf.Ticker('^GSPC').history('max', end=f_end) # download index data
            fname = os.path.join(spy_path, interval, f'SPY.csv')
        else:
            df = yf.download('^GSPC', start=start, end=end, interval=interval, progress=False)
            f_date = dt.strftime(f_end, "%d.%m.%y")
            fname = os.path.join(spy_path, interval, f'SPY_{f_date}.csv')
        if not df.empty:
            df.to_csv(fname)
        for i, ticker in enumerate(tickers, 1):
            try:
                fname = os.path.join(t_path, f'{ticker}.csv')
                if interval == '1d':
                    df = si.get_data(ticker, end_date=f_end)
                else:
                    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
                if not df.empty:
                    df.to_csv(fname)
                t_end = time.time()
                mm, ss = divmod(t_end - t_start, 60)
                print(f"\r{mm:.0f}m:{ss:.0f}s {i}/{n} ({i / n:.2%})" \
                      f" of {interval} SPY market data downloaded".ljust(70, ' '),
                      end='', flush=True)
            except Exception as e:
                j += 1
                t_end = time.time()
                mm, ss = divmod(t_end - t_start, 60)
                print(f'\r{mm:.0f}m:{ss:.0f}s {j}/{n}: {ticker} - {e}',
                      end='', flush=True) 
        
        t_end = time.time()
        mm, ss = divmod(t_end - t_start, 60)
        print(f'\n{n - j}/{n} of S&P 500 stock {interval} data downloaded in {mm:.0f}m:{ss:.0f}s\n')


def get_tickers_info():
    t_start = time.time()
    tickers = get_tickers()
    n = len(tickers)
    fname = r'data\market_data\spy_data\spy_tickers_info.pickle'
    
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            info = pickle.load(f)
    else:
        info = {}
    
    for i, ticker in enumerate(tickers, 1):
        if ticker not in info:
            info[ticker] = {}
            try:
                ticker_info = yf.Ticker(ticker).info
                info[ticker]['Security'] = ticker_info['longName']
                info[ticker]['Sector'] = ticker_info['sector']
                info[ticker]['Industry'] = ticker_info['industry']
                info[ticker]['Business Summary'] = ticker_info['longBusinessSummary']
                info[ticker]['Website'] = ticker_info['website']
            except Exception as e:
                info[ticker]['Security'] = 'N/A'
                info[ticker]['Sector'] = 'N/A'
                info[ticker]['Industry'] = 'N/A'
                info[ticker]['Business Summary'] = 'N/A'
                info[ticker]['Website'] = 'N/A'
                print(f'\r{i}/{n}: {ticker} - {e}'.ljust(70, ' '))
        
        t_end = time.time()
        mm, ss = divmod(t_end - t_start, 60)
        print(f'\r{mm:.0f}m:{ss:.0f}s: {i}/{n} ({i / n:.2%}) business summaries downloaded',
                end='', flush=True)

    with open(fname, 'wb') as f:
        pickle.dump(info, f)
    
    t_end = time.time()
    mm, ss = divmod(t_end - t_start, 60)
    print(f'\nBusiness summaries saved in {mm:.0f}m:{ss:.0f}s\n')
    
        
def ratios_to_update():
    tickers = get_tickers()[0]
    not_current = []
    no_data = []
    not_downloaded = []
    f = 'data/financial_ratios/Annual'

    for ticker in tickers:
        fname = ticker + '.csv'   
        if fname in os.listdir(f):
            df = pd.read_csv(os.path.join(f, fname))
            if str(dt.now().year - 1) != df.columns[1]:
                not_current.append(ticker)
            if df.empty:
                no_data.append(ticker)
        else:
            not_downloaded.append(ticker)
    
    to_update = not_current + no_data + not_downloaded
    
    return to_update


def get_financial_ratios(i=0, n=1):
    '''
    Downloads annual financial ratios for each S&P 500 stock as a CSV file

    Parameters
    ----------

    i : int
        A counter of the tickers whose ratios have been downloaded
    n : int
        Calls the API key according to the number, e.g., key{n}
    '''

    t_start = time.time()
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
                t_end = time.time()
                mm, ss = divmod(t_end - t_start, 60)
                print(f"\r{i}/{len(to_update)} outdated financial ratios downloaded",
                      end='', flush=True)
        except Exception as e:
            if e == '<urlopen error [Errno 11001] getaddrinfo failed>':
                print('\r', e.ljust(100, ' '), end='', flush=True)
            elif n < 5:
                print(f'\nAPI Key {n} has maxed out its requests')
                n += 1

        return get_financial_ratios(i, n)

    else:
        t_end = time.time()
        mm, ss = divmod(t_end - t_start, 60)
        print(f'\nAnnual financial ratios downloaded in {mm:.0f}m:{ss:.0f}s\n')       


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
    '''
    
    t_start = time.time()
    tickers = get_tickers()[0]
    ntickers = len(tickers)

    # Use recursion to continue building the dict of TTM ratios when API calls
    # for a key reach the limit (250 requests/day)
    if i < ntickers:
        try:
            for ticker in tickers[i: i + 250]:
                ratios = fa.financial_ratios(ticker, 
                                             st.secrets[f'FUNDAMENTAL_ANALYSIS_API_KEY{n}'],
                                             period='annual',
                                             TTM=True)
                d[ticker] = ratios.to_dict()
                i += 1
                t_end = time.time()
                mm, ss = divmod(t_end - t_start, 60)
                print(f"\r{mm:.0f}m:{ss:.0f}s {i}/{ntickers} ({i / ntickers:.2%}) TTM financial ratios downloaded",
                      end='', flush=True)
        except Exception as e:
            if e == '<urlopen error [Errno 11001] getaddrinfo failed>':
                print('\n', e)
            elif n < 5:
                print(f'\nAPI Key {n} has maxed out its requests\n')
                n += 1

        return get_TTM_financial_ratios(i, n, d) 

    else:
        print('\nCurrent ratios are up to date!\n')
        return d

            
def save_TTM_financial_ratios():
    '''Save ratios as pickle file'''

    t_start = time.time()
    date = dt.now(tz=timezone('EST'))  # Set datetime object to EST timezone

    # Sets file name to today's date only after US stock market
    # closes, otherwise uses previous day's date. Also sets
    # weekends to Friday's date.
    if date.weekday() > 4:
        date -= timedelta(date.weekday() - 4)
    elif date.hour < 16:
        date -= timedelta(1)
        
    fname = date.strftime('%d-%m-%Y') + '.pickle'
    path = r'data\financial_ratios\Current'
    d = get_TTM_financial_ratios()
    nd = len(d)
    tickers = get_tickers()[0]
    ntickers = len(tickers)

    if nd == ntickers:
        with open(os.path.join(path, fname), 'wb') as f:
            pickle.dump(d, f)
        print(fname, 'saved\n')
    else:
        print(f'{ntickers - nd}/{ntickers} ratios not downloaded\n')

    t_end = time.time()
    mm, ss = divmod(t_end - t_start, 60)
    print(f'TTM financial ratios downloaded in {mm:.0f}m:{ss:.0f}s')


def get_risk_free_rates():
    df = si.get_data('^IRX')
    df.columns = df.columns.str.title()
    df.to_csv(r'data\T-Bill Rates.csv')
    print('T-Bill Rates saved\n')


def get_factor_model_data():
    start_date = '1954-01-01' 
    df_three_factor = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=start_date)[0]
    df_three_factor.index = df_three_factor.index.format()
    df_mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start=start_date)[0]
    df_mom.index = df_mom.index.format()
    df_four_factor = df_three_factor.join(df_mom)
    df_five_factor = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=start_date)[0]
    df_five_factor.index = df_five_factor.index.format()
    path = r'data\factor_models'
    df_three_factor.to_csv(os.path.join(path, 'F-F_Research_Data_Factors.csv'))
    df_four_factor.to_csv(os.path.join(path, 'Carhart_4_Factors.csv'))
    df_five_factor.to_csv(os.path.join(path, 'F-F_Research_Data_5_Factors_2x3.csv'))

    print('Factor model data downloaded\n')


def get_financial_statements():
    t_start = time.time()
    tickers = get_tickers()
    n = len(tickers)
    path = r'data\financial_statements'
    d_tickers = {'META': 'FB', 'BALL': 'BLL'}
    dict_file = os.path.join(path, 'financial_statements.pickle')

    if os.path.isfile(dict_file):
        with open(dict_file, 'rb') as f:
            statements = pickle.load(f)
    else:
        statements = {}

    for i, ticker in enumerate(tickers, 1):
        if ticker in d_tickers:
            d_ticker = d_tickers[ticker]
        else:
            d_ticker = ticker.replace('-', '.')
        base_url = f"https://stockrow.com/api/companies/{d_ticker}/" \
                    "financials.xlsx?dimension=Q&section="
        sofp = f"{base_url}Balance%20Sheet&sort=desc"
        soci = f"{base_url}Income%20Statement&sort=desc"
        socf = f"{base_url}Cash%20Flow&sort=desc"
        download = False

        if ticker not in statements:
            download = True
        elif dt.strptime(statements[ticker].columns[0], "%Y-%m-%d").year < dt.now().year - 2:
            download = True            

        if download:
            try:        
                df1 = pd.read_excel(sofp) # balance sheet data           
                df2 = pd.read_excel(soci) # income statement data
                df3 = pd.read_excel(socf) # cashflow statement data
                dfs = [df1, df2, df3]
                fname = f'{ticker}.xlsx'
                folders = ['sofp', 'soci', 'socf']
                for df, f in zip(dfs, folders):
                    df.set_index(df.columns[0], inplace=True)
                    df.index.name = 'Item'
                    columns = [x.strftime("%Y-%m-%d") for x in df.columns]
                    df.columns = columns
                    fpath = os.path.join(path, f, fname)
                    df.to_excel(fpath)
                df = pd.concat(dfs) # combining all extracted information
                statements[ticker] = df
                t_end = time.time()
                mm, ss = divmod(t_end - t_start, 60)
            except Exception as e:
                print(f'\r{i}/{n}: {ticker} - {e}'.ljust(70, ' '))
        
        t_end = time.time()
        mm, ss = divmod(t_end - t_start, 60)
        print(f"\r{mm:.0f}m:{ss:.0f}s {i}/{n} ({i / n:.2%}) statements downloaded", end='', flush=True)

    with open(dict_file, 'wb') as f:
        pickle.dump(statements, f)

    t_end = time.time()
    mm, ss = divmod(t_end - t_start, 60)
    print(f'\nFinancial statements downloaded in {mm:.0f}m:{ss:.0f}s\n')


def redownload_market_data(path=r'data\market_data'):
    t_start = time.time()
    end = dt.now()
    weekday = end.weekday() 
    tickers = get_tickers()
    days = 0

    if weekday >= 5:
        days += weekday - 5
    elif weekday == 0 and end.hour < 24:
        days += 2
    
    for f in os.listdir(path):
        if f != 'spy_data':
            fpath = os.path.join(path, f)
            files = os.listdir(fpath)
            missing_files = set([f'{x}.csv' for x in tickers]) - set(files)
            to_update = list(missing_files)
            start = end - timedelta(7 - days) if f == '1m' else end - timedelta(60 - days)
            f_end = end - timedelta(1) if f == '1d' else end
            for fname in files:
                fname = os.path.join(fpath, fname)
                ti_m = os.path.getmtime(fname)
                date = dt.fromtimestamp(ti_m)
                delta = dt.now() - date
                if delta.days > 0:
                    to_update.append(fname)
                    t_end = time.time()
                    mm, ss = divmod(t_end - t_start, 60)
                    print(f"\r{f} scanned in {mm:.0f}m:{ss:.3f}s".ljust(70, ' '),
                          end='', flush=True)

            n = len(to_update)
            j = 0

            for i, filepath in enumerate(to_update, 1):
                fname = os.path.split(filepath)[1]
                ticker = os.path.splitext(fname)[0]
                try:
                    if f == '1d':
                        df = si.get_data(ticker, end_date=f_end)
                    else:
                        df = yf.download(ticker, start=start, end=f_end, interval=f, progress=False)

                    if not df.empty:
                        df.to_csv(filepath)

                    t_end = time.time()
                    mm, ss = divmod(t_end - t_start, 60)
                    print(f"\r{mm:.0f}m:{ss:.0f}s {i}/{n} ({i / n:.2%})" \
                          f" of {f} SPY market data downloaded".ljust(70, ' '),
                          end='', flush=True)

                except Exception as e:
                    j += 1
                    t_end = time.time()
                    mm, ss = divmod(t_end - t_start, 60)
                    print(f'\r{mm:.0f}m:{ss:.0f}s {j}/{n}: {ticker} - {e}',
                          end='', flush=True)

        else:
            ticker = '^GSPC'
            for dirpath, dirnames, filenames in os.walk(fpath):
                if not dirnames:
                    fname = os.path.join(dirpath, filenames[-1])
                    ti_m = os.path.getmtime(fname)
                    date = dt.fromtimestamp(ti_m)
                    delta = dt.now() - date
                    f = os.path.split(dirpath)[-1]
                    start = end - timedelta(7 - days) if f == '1m' else end - timedelta(60 - days)
                    f_end = end - timedelta(1) if f == '1d' else end
                    if delta.days > 0:
                        if f == '1d':
                            df = si.get_data(ticker, end_date=f_end)
                            fname = os.path.join(dirpath, f'SPY.csv')
                        else:
                            df = yf.download(ticker, start=start, end=f_end, interval=f, progress=False)
                            f_date = dt.strftime(f_end, "%d.%m.%y")
                            fname = os.path.join(dirpath, f'SPY_{f_date}.csv')
                        
                        if not df.empty:
                            df.to_csv(fname)
                        
                        t_end = time.time()
                        mm, ss = divmod(t_end - t_start, 60)
                        print(f"\r{mm:.0f}m:{ss:.0f}s {fname} downloaded".ljust(70, ' '),
                              end='', flush=True)
        
    t_end = time.time()
    mm, ss = divmod(t_end - t_start, 60)
    print(f'\nRedownloaded in {mm:.0f}m:{ss:.0f}s') 


if __name__ == "__main__":           
    # get_SPY_companies()
    # get_SPY_weights()
    # get_risk_free_rates()
    # get_factor_model_data()
    get_interval_market_data()
    # save_TTM_financial_ratios()
    # get_financial_ratios()
    # get_financial_statements()
    # get_tickers_info()
    redownload_market_data()
    
    f_end = time.time()
    mm, ss = divmod(f_end - f_start, 60)
    winsound.Beep(440, 500)
    print(f'\nDone in {mm:.0f}m:{ss:.0f}s!!!\n')