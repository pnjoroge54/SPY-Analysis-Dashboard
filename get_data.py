import os
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
    
    current_fname = r'data\spy_data\SPY_Info.csv'
    hist_fname = r'data\spy_data\SPY_Historical.csv'
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
    df1 = pd.read_csv(r'data\spy_data\SPY_Info.csv')
    df2 = pd.read_csv(r'data\spy_data\SPY_Historical.csv', index_col='Date')
    df2[:'2015-01-01']
    
    tickers = df1['Symbol'].to_list()
    hist_tickers = list(set(tickers + df2['Ticker'].to_list()))

    return tickers, hist_tickers


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
    df.to_csv(r'data\spy_data\SPY_Weights.csv')
    print('S&P 500 weights updated \n')


def get_market_data():  
    '''Get historical data for S&P 500 index & for each constituent stock'''

    SPY = yf.Ticker('^GSPC').history(period='max') # download index data
    SPY.to_csv(r'data\spy_data\SPY.csv')

    tickers = get_tickers()[0]
    n = len(tickers)
    not_downloaded = []
    path = r'data\market_data'

    for i, ticker in enumerate(tickers, 1):
        fname = os.path.join(path, f'{ticker}.csv')
        try:
            data = si.get_data(ticker) # download stock data
            data.to_csv(fname)
            print(f"\r{i}/{n} ({i / n:.2%}) of SPY market data downloaded", end='', flush=True)
        except Exception as e:
            print(f'\r{i}/{n}: {ticker} - {e}')
            not_downloaded.append(ticker)  

    print('\nS&P 500 stock data downloaded \n')

    if not_downloaded:
        print(f'{len(not_downloaded)} stocks not downloaded \n')


def remove_replaced_tickers():
    '''Move tickers that have been removed from the S&P 500 to their own folder'''

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
    Downloads annual financial ratios for each S&P 500 stock as a CSV file

    Parameters
    ----------

    i : int
        A counter of the tickers whose ratios have been downloaded
    n : int
        Calls the API key according to the number, e.g., key{n}
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
                print(f"\r{i}/{len(to_update)} outdated financial ratios downloaded", end='', flush=True)
        except Exception as e:
            if e == '<urlopen error [Errno 11001] getaddrinfo failed>':
                print('\r', e.ljust(100, ' '), end='', flush=True)
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
    '''
    
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
                print(f"\r{i}/{ntickers} ({i / ntickers:.2%}) TTM financial ratios downloaded",
                      end='', flush=True)
        except Exception as e:
            if e == '<urlopen error [Errno 11001] getaddrinfo failed>':
                print('\n', e)
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
    weekday = cdate.weekday()

    # Sets the file name to today's date only after the US stock market
    # has closed, otherwise uses the previous day's date. Also sets
    # weekends to Friday's date.
    if weekday != 5 and weekday != 6 and weekday != 0 and hour < 16:
        days = 1
    elif weekday == 5:
        days = 1
    elif weekday == 6:
        days = 2
    elif weekday == 0 and hour < 16:
        days = 3
    else:
        days = 0
        
    cdate -= timedelta(days=days)   
    file = cdate.strftime('%d-%m-%Y') + '.pickle'
    path = r'data\financial_ratios\Current'
    d = get_TTM_financial_ratios()
    nd = len(d)
    tickers = get_tickers()[0]
    ntickers = len(tickers)

    if nd == ntickers:
        with open(os.path.join(path, file), 'wb') as f:
            pickle.dump(d, f)
        print(file, 'saved\n')
    else:
        print(f'{ntickers - nd}/{ntickers} ratios not downloaded\n')


def get_risk_free_rates():
    # rf_rates = pdr.fred.FredReader('DTB3', dt(1954, 1, 4), dt.now()).read()
    rf_rates = yf.download('^IRX', progress=False)
    rf_rates.to_csv(r'data\T-Bill Rates.csv')
    print('T-Bill Rates saved\n')


def get_multi_factor_model_data():
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
    tickers = get_tickers()[0]
    n = len(tickers)
    path = r'data\financial_statements'
    base_url = 'https://stockrow.com/api/companies/'
    d_tickers = {'META': 'FB',
                 'BALL': 'BLL'
                 }
    dict_file = os.path.join(path, 'financial_statements.pickle')

    try:
        with open(dict_file, 'rb') as f:
            statements = pickle.load(f)
    except:
        statements = {}

    for i, ticker in enumerate(tickers, 1):
        if ticker in d_tickers:
            d_ticker = d_tickers[ticker]
        else:
            d_ticker = ticker.replace('-', '.')

        base_url += f"{d_ticker}/financials.xlsx?dimension=Q&section="
        sofp = f"{base_url}Balance%20Sheet&sort=desc"
        soci = f"{base_url}Income%20Statement&sort=desc"
        socf = f"{base_url}Cash%20Flow&sort=desc"
        download = False

        if d_ticker not in statements:
            download = True
        elif dt.strptime(statements[d_ticker].columns[0], "%Y-%m-%d").year < dt.now().year - 2:
            download = True            

        if download:
            try:        
                df1 = pd.read_excel(sofp) # balance sheet data           
                df2 = pd.read_excel(soci) # income statement data
                df3 = pd.read_excel(socf) # cashflow statement data
                df = pd.concat([df1, df2, df3]) # combining all extracted information
                df.set_index("Unnamed: 0", inplace=True)
                df.index.name = 'Item'
                columns = [x.strftime("%Y-%m-%d") for x in df.columns]
                df.columns = columns
                statements[ticker] = df
                fname = f'{ticker}.xlsx'
                df1.to_excel(os.path.join(path, 'sofp', fname), index=False)
                df2.to_excel(os.path.join(path, 'soci', fname), index=False)
                df3.to_excel(os.path.join(path, 'socf', fname), index=False)
            except Exception as e:
                print(f'\r{i}/{n}: {ticker} - {e}'.ljust(100, ' '))
        
        print(f"\r{i}/{n} ({i / n:.2%}) statements downloaded", end='', flush=True)

    with open(dict_file, 'wb') as f:
        pickle.dump(statements, f)

    print(f'\nFinancial statements saved\n')
        

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