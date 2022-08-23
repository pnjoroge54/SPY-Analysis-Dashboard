import os
import math
import numpy as np
import pandas as pd
import requests
import pickle
from datetime import datetime as dt
from datetime import timedelta
from pytz import timezone

import cufflinks as cf
import plotly.graph_objects as go
import streamlit as st


f = 'data/market_data/'


def getIndexOfTuple(lst, index, value):
    for pos, t in enumerate(lst):
        if t[index] == value:
            return pos + 1
            
    # Matches behavior of list.index
    raise ValueError("list.index(x): x not in list")


@st.cache
def get_SPY_info():
    '''Make dataframe of info about SPY companies'''
    
    return pd.read_csv('data/SPY-Info.csv')


@st.cache
def get_SPY_data():
    '''Make dataframe of SPY market data'''

    return pd.read_csv('data/SPY.csv', index_col='Date', parse_dates=True)


@st.cache
def get_first_dates():
    '''Get the earliest date for which market data is available for each company'''

    first_dates = []

    for ticker in ticker_list:
        df = pd.read_csv(os.path.join(f, f'{ticker}.csv'), index_col='Unnamed: 0', parse_dates=True)
        first_date = df.iloc[0].name
        first_dates.append((ticker, first_date))
        first_dates = sorted(first_dates, key=lambda x: x[1])

    return first_dates


@st.cache
def make_combined_returns_df():
    '''Make dataframe of returns for all SPY stocks, as well as SPY index'''

    combined_returns = SPY_df.copy()
    combined_returns['Return'] = combined_returns['Close'].pct_change() * 100
    combined_returns.drop([x for x in combined_returns.columns if x != 'Return'],
                          axis=1, inplace=True)
    combined_returns.rename(columns={'Return': 'SPY'}, inplace=True)
    
    for ticker in ticker_list:
        df = pd.read_csv(os.path.join(f, f'{ticker}.csv'), index_col='Unnamed: 0', parse_dates=True)
        df['Return'] = df['adjclose'].pct_change() * 100
        combined_returns = combined_returns.join(df['Return'], how='left')
        combined_returns.rename(columns={'Return': ticker}, inplace=True)

    return combined_returns


def get_all_current_ratios():
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
    f = r'data\financial_ratios\Current'
    
    if file in os.listdir(f):
        s = "The data reported is today's."
    else:
        dates = []

        for file in os.listdir(f):
            date = file.replace('.pickle', '')
            date = dt.strptime(date, '%d-%m-%Y')
            dates.append(date)

        dates = sorted(dates)
        date = dt.strftime(dates[-1], '%B %d, %Y')
        file = dt.strftime(dates[-1], '%d-%m-%Y') + '.pickle'
        s = f'The data reported is from {date}.'

    with open(os.path.join(f, file), 'rb') as f1:
        d = pickle.load(f1)
    
    return d, s


@st.cache
def get_current_ratios(ratios, ratio):
    r = ratios[ratio]
    sector_ratios = {}
    subIndustry_ratios = {}
    ticker_ratios = {}
    subIndustry_list = SPY_info_df['GICS Sub-Industry'].unique()
    
    for subIndustry in subIndustry_list:
        subIndustry_ratios[subIndustry] = []

    for sector in sector_list:
        sector_tickers = SPY_info_df[SPY_info_df['GICS Sector'] == sector] \
                         ['Symbol'].to_list()
        sector_ratio = []
        for ticker in sector_tickers:
            # Get sub-industry of ticker
            t_subIndustry = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                            ['GICS Sub-Industry'].item()
            # Get weight to use in weighted average calculation
            weight = ticker_weights[ticker]
            ticker_sector_weight = weight / sector_weights[sector]
            ticker_subIndustry_weight = weight / subIndustry_weights[t_subIndustry]
            # Ratio result
            try:
                res = current_ratios[ticker][r]
            except:
                res = 0

            # Append ratio to its sector list
            sector_ratio.append(res * ticker_sector_weight)
            # Append ratio to its sub-industry list
            subIndustry_ratios[t_subIndustry].append(res * ticker_subIndustry_weight)
            # Get ratio, name, sector, sub-industry of each ticker
            ticker_ratios[ticker] = {}
            ticker_ratios[ticker]['Company'] = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                                                ['Security'].item()
            ticker_ratios[ticker]['Sector'] = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                                                ['GICS Sector'].item()
            ticker_ratios[ticker]['Sub-Industry'] = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                                                    ['GICS Sub-Industry'].item()
            ticker_ratios[ticker][ratio] = res                    

        # Calculate sector ratios
        sector_res = sum(sector_ratio)
        sector_ratios[sector] = sector_res

    # Get sub-industry ratios
    for subIndustry in subIndustry_list:
        subIndustry_ratios[subIndustry] = sum(subIndustry_ratios[subIndustry])

    df = pd.DataFrame.from_dict(sector_ratios, orient='index', columns=[ratio])

    return df, subIndustry_ratios, ticker_ratios 


@st.cache
def calculate_beta(df, ticker, start_date, end_date):
    df = df[start_date: end_date]
    SPY_std = df['SPY'].std()
    ticker_std = df[ticker].std()
    correlation_df = df.corr()
    SPY_ticker_corr = correlation_df.loc[ticker, 'SPY']
    beta = SPY_ticker_corr * (ticker_std / SPY_std)
    
    return beta


@st.cache
def get_weights():
    # Assign weights by sectors & sub-industries
    
    weights_df = pd.read_csv('data/SPY Weights.csv', index_col='Symbol')
    sector_weights = {}
    subIndustry_weights = {}
    ticker_weights = {}

    for sector in sector_list:
        subIndustry_list = SPY_info_df[SPY_info_df['GICS Sector'] == sector] \
                           ['GICS Sub-Industry'].to_list()
        sector_weight = 0
        for subIndustry in subIndustry_list:
            tickers = SPY_info_df[SPY_info_df['GICS Sub-Industry'] == subIndustry] \
                      ['Symbol'].to_list()
            subIndustry_weight = 0
            for ticker in tickers:
                weight = weights_df.loc[ticker, 'Weight']
                ticker_weights[ticker] = weight
                subIndustry_weight += weight

            subIndustry_weights[subIndustry] = subIndustry_weight
            sector_weight += subIndustry_weight
            
        sector_weights[sector] = sector_weight
       
    return sector_weights, subIndustry_weights, ticker_weights


@st.cache
def find_stocks_missing_data(start_date, end_date):
    # Identify stocks that lack market data before the start date

    s, s1, s2 = '', '', ''
    
    if start_date < first_dates[0][1]:
        s += f"**The first date for which there is data is \
                {first_dates[0][1].strftime('%B %d, %Y')}**"
    if end_date > last_date:
        s1 += f"**The last date for which there is data is \
                {last_date.strftime('%B %d, %Y')}**"
    
    missing = list(filter(lambda x: x[1] > start_date, first_dates))
    missing = [x[0] for x in missing]

    # Make dict of stocks by sector & sub-industry
    d = {sector: {} for sector in sector_list}
    
    for ticker in missing:
        sector = SPY_info_df[SPY_info_df['Symbol'] == ticker]['GICS Sector'].item()
        subIndustry = SPY_info_df[SPY_info_df['Symbol'] == ticker]['GICS Sub-Industry'].item()
        d[sector][subIndustry] = []
    for ticker in missing:
        sector = SPY_info_df[SPY_info_df['Symbol'] == ticker]['GICS Sector'].item()
        subIndustry = SPY_info_df[SPY_info_df['Symbol'] == ticker]['GICS Sub-Industry'].item()
        d[sector][subIndustry] += [ticker]

    # Delete empty dict keys
    for sector in sector_list:
        if len(d[sector]) == 0:
            del d[sector]

    if len(missing) > 0:
        s2 += f"{len(missing)}/{len(ticker_list)} stocks have data that begins after \
                {start_date.strftime('%B %d, %Y')}. \
                \nMissing data affects the accuracy of results displayed below."

    return d, s, s1, s2


@st.cache
def get_returns_and_volatility(start_date, end_date):
    sector_returns = {}
    subIndustry_returns = {}
    sector_vols = {}
    subIndustry_vols = {}
    sector_sharpes = {}
    subIndustry_sharpes = {}
    ticker_cols = {}
    rf_rates = pd.read_csv(r'data\T-Bill Rates.csv', index_col='Date', parse_dates=True)
    rf_rates.rename(columns={'Close': 'DTB3'}, inplace=True)
    subIndustry_list = SPY_info_df['GICS Sub-Industry'].unique()
    
    for subIndustry in subIndustry_list:
        subIndustry_returns[subIndustry] = []
        subIndustry_vols[subIndustry] = []
        subIndustry_sharpes[subIndustry] = []

    for sector in sector_list:
        sector_tickers = SPY_info_df[SPY_info_df['GICS Sector'] == sector] \
                         ['Symbol'].to_list()
        sector_return = []
        sector_vol = []
        sector_sharpe = []
        for ticker in sector_tickers:
            # Get sub-industry of ticker
            t_subIndustry = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                            ['GICS Sub-Industry'].item()
            df = pd.read_csv(os.path.join(f, ticker + '.csv'), index_col='Unnamed: 0', parse_dates=True)
            df = df[start_date: end_date]
            df = pd.concat([df, rf_rates.DTB3], axis=1, join='inner')
            df.ffill(inplace=True)
            df['Daily T-Bill Rate'] = (1 + df['DTB3'] / 100 * (90 / 360))**(1 / 90) - 1
            df['Daily Return'] = df['adjclose'].pct_change()
            df['Daily Excess Return'] = df['Daily Return'] - df['Daily T-Bill Rate']
            df['Cumulative Return'] = (1 + df['Daily Return']).cumprod() - 1 
            df['Cumulative Excess Return'] = (1 + df['Daily Excess Return']).cumprod() - 1 
            df_return = df['Cumulative Return'][-1] * 100
            df_ereturn = df['Cumulative Excess Return'][-1] * 100
            df_std = df['Daily Return'].std() * 100
            df_estd = df['Daily Excess Return'].std() * 100
            df_sharpe = df_ereturn / df_estd

            # Get weight to use in weighted average calculation
            weight = ticker_weights[ticker]
            ticker_sector_weight = weight / sector_weights[sector]
            ticker_subIndustry_weight = weight / subIndustry_weights[t_subIndustry]
            # Append result to its sector list
            sector_return.append(df_return * ticker_sector_weight)
            sector_vol.append(df_std * ticker_sector_weight)
            sector_sharpe.append(df_sharpe * ticker_sector_weight)
            # Append result to its sub-industry list
            subIndustry_returns[t_subIndustry].append(df_return * ticker_subIndustry_weight)
            subIndustry_vols[t_subIndustry].append(df_std * ticker_subIndustry_weight)
            subIndustry_sharpes[t_subIndustry].append(df_sharpe * ticker_subIndustry_weight)
            # Get results for each ticker
            ticker_cols[ticker] = {}
            ticker_cols[ticker]['Company'] = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                                             ['Security'].item()
            ticker_cols[ticker]['Sector'] = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                                            ['GICS Sector'].item()
            ticker_cols[ticker]['Sub-Industry'] = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                                                  ['GICS Sub-Industry'].item()
            ticker_cols[ticker]['Return (%)'] = df_return
            ticker_cols[ticker]['Volatility (%)'] = df_std
            ticker_cols[ticker]['Sharpe Ratio'] = df_sharpe

        sector_returns[sector] =  sum(sector_return)
        sector_vols[sector] = sum(sector_vol)
        sector_sharpes[sector] = sum(sector_sharpe)

    # Get sub-industry returns
    for subIndustry in subIndustry_list:
        subIndustry_returns[subIndustry] = sum(subIndustry_returns[subIndustry])
        subIndustry_vols[subIndustry] = sum(subIndustry_vols[subIndustry])
        subIndustry_sharpes[subIndustry] = sum(subIndustry_sharpes[subIndustry])
    
    sector_returns_df = pd.DataFrame.from_dict(sector_returns, orient='index', columns=['Return (%)'])
    sector_vols_df = pd.DataFrame.from_dict(sector_vols, orient='index', columns=['Volatility (%)'])
    sector_sharpes_df = pd.DataFrame.from_dict(sector_sharpes, orient='index', columns=['Sharpe Ratio'])
    df = SPY_df[start_date: end_date]
    df = pd.concat([df, rf_rates.DTB3], axis=1, join='inner')
    df.ffill(inplace=True)
    df['Daily T-Bill Rate'] = (1 + df['DTB3'] / 100 * (90 / 360))**(1 / 90) - 1      
    df['Daily Return'] = df['Close'].pct_change()
    df['Daily Excess Return'] = df['Daily Return'] - df['Daily T-Bill Rate']
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod() - 1 
    df['Cumulative Excess Return'] = (1 + df['Daily Excess Return']).cumprod() - 1 
    SPY_return = df['Cumulative Return'][-1] * 100
    SPY_ereturn = df['Cumulative Excess Return'][-1] * 100
    SPY_std = df['Daily Return'].std() * 100
    SPY_estd = df['Daily Excess Return'].std() * 100
    SPY_sharpe = SPY_ereturn / SPY_estd
    
    return sector_returns_df, subIndustry_returns, ticker_cols, sector_vols_df, subIndustry_vols, \
           sector_sharpes_df, subIndustry_sharpes, SPY_return, SPY_std, SPY_sharpe

    
@st.cache
def get_betas(start_date, end_date):
    sector_betas = {}
    subIndustry_betas = {}
    ticker_betas = {}
    subIndustry_list = SPY_info_df['GICS Sub-Industry'].unique()
    
    for subIndustry in subIndustry_list:
        subIndustry_betas[subIndustry] = []

    for sector in sector_list:
        sector_tickers = SPY_info_df[SPY_info_df['GICS Sector'] == sector] \
                        ['Symbol'].to_list()
        sector_beta = []
        for ticker in sector_tickers:
            # Get sub-industry of ticker
            t_subIndustry = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                            ['GICS Sub-Industry'].item()
            # Get beta for each ticker
            beta = calculate_beta(combined_returns_df, ticker, start_date, end_date)
            # Get weight to use in weighted average calculation
            weight = ticker_weights[ticker]
            ticker_sector_weight = weight / sector_weights[sector]
            ticker_subIndustry_weight = weight / subIndustry_weights[t_subIndustry]
            # Append beta to its sector list
            sector_beta.append(beta * ticker_sector_weight)  
            # Append beta to its sub-industry list
            subIndustry_betas[t_subIndustry].append(beta * ticker_subIndustry_weight)
            # Get beta, name, sector, sub-industry of each ticker
            ticker_betas[ticker] = {}
            ticker_betas[ticker]['Company'] = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                                              ['Security'].item()
            ticker_betas[ticker]['Sector'] = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                                             ['GICS Sector'].item()
            ticker_betas[ticker]['Sub-Industry'] = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                                                   ['GICS Sub-Industry'].item()
            ticker_betas[ticker]['Beta'] = beta       

        # Calculate sector betas
        sector_betas[sector] = sum(sector_beta)

    # Get sub-industry betas
    for subIndustry in subIndustry_list:
        subIndustry_betas[subIndustry] = sum(subIndustry_betas[subIndustry])

    df = pd.DataFrame.from_dict(sector_betas, orient='index', columns=['Beta'])

    return df, subIndustry_betas, ticker_betas


@st.cache
def TTM_Squeeze():

    def in_squeeze(df):
        return df['lower_band'] > df['lower_keltner'] and df['upper_band'] < df['upper_keltner']

    coming_out = []

    for file in os.listdir(f):
        ticker = file.split('.')[0]
        start_date = yr_ago
        df = pd.read_csv(os.path.join(f, file), index_col='Unnamed: 0', parse_dates=True)
        df = df[start_date: last_date]
        df['20sma'] = df['close'].rolling(window=20).mean()
        df['std deviation'] = df['close'].rolling(window=20).std()
        df['lower_band'] = df['20sma'] - (2 * df['std deviation'])
        df['upper_band'] = df['20sma'] + (2 * df['std deviation'])        
        df['TR'] = abs(df['high'] - df['low'])
        df['ATR'] = df['TR'].rolling(window=20).mean()
        df['lower_keltner'] = df['20sma'] - (df['ATR'] * 1.5)
        df['upper_keltner'] = df['20sma'] + (df['ATR'] * 1.5)

        df['squeeze_on'] = df.apply(in_squeeze, axis=1)

        if df.iloc[-2]['squeeze_on'] and not df.iloc[-1]['squeeze_on']:
            coming_out.append((ticker, df.iloc[-1].name))
        elif df.iloc[-3]['squeeze_on'] and not df.iloc[-2]['squeeze_on']:
            coming_out.append((ticker, df.iloc[-2].name))
        elif df.iloc[-4]['squeeze_on'] and not df.iloc[-3]['squeeze_on']:
            coming_out.append((ticker, df.iloc[-3].name))
        elif df.iloc[-5]['squeeze_on'] and not df.iloc[-4]['squeeze_on']:
            coming_out.append((ticker, df.iloc[-4].name))
        elif df.iloc[-6]['squeeze_on'] and not df.iloc[-5]['squeeze_on']:
            coming_out.append((ticker, df.iloc[-5].name))

    return coming_out
    

def make_TTM_squeeze_charts(lst):
    for item in lst:
        ticker = item[0]
        date = item[1].strftime('%b %d')
        df = pd.read_csv(os.path.join(f, ticker + '.csv'), index_col='Unnamed: 0', parse_dates=True)
        start_date = last_date - timedelta(days=180)
        df = df[start_date: last_date]
        df['20sma'] = df['close'].rolling(window=20).mean()
        df['std deviation'] = df['close'].rolling(window=20).std()
        df['lower_band'] = df['20sma'] - (2 * df['std deviation'])
        df['upper_band'] = df['20sma'] + (2 * df['std deviation'])               
        df['TR'] = abs(df['high'] - df['low'])
        df['ATR'] = df['TR'].rolling(window=20).mean()
        df['lower_keltner'] = df['20sma'] - (df['ATR'] * 1.5)
        df['upper_keltner'] = df['20sma'] + (df['ATR'] * 1.5)
        name = SPY_info_df[SPY_info_df['Symbol'] == ticker]['Security'].item()
        title = f'{name} ({ticker})'
        candlestick = go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                        low=df['low'], close=df['close'], name=ticker)
        upper_band = go.Scatter(x=df.index, y=df['upper_band'],
                                name='Upper Bollinger Band',
                                line={'color': 'blue', 'width': 0.75})
        lower_band = go.Scatter(x=df.index, y=df['lower_band'],
                                name='Lower Bollinger Band',
                                line={'color': 'blue', 'width': 0.75})
        upper_keltner = go.Scatter(x=df.index, y=df['upper_keltner'],
                                    name='Upper Keltner Channel', 
                                    line={'color': 'red', 'width': 0.75})
        lower_keltner = go.Scatter(x=df.index, y=df['lower_keltner'],
                                    name='Lower Keltner Channel', 
                                    line={'color': 'red', 'width': 0.75})
        layout = go.Layout(plot_bgcolor='#ECECEC', paper_bgcolor='#ECECEC')
        fig = go.Figure(data=[candlestick, upper_band, lower_band,
                              upper_keltner, lower_keltner],
                        layout=layout)
                        
        # Set candlestick line and fill colors
        cs = fig.data[0]
        cs.increasing.fillcolor = '#B7E9F7'
        cs.increasing.line.color = '#45B6FE'
        cs.decreasing.fillcolor = '#BEBEBE'
        cs.decreasing.line.color = '#808080'
        fig.update_layout(title=title, yaxis_title='Price')
        fig.add_annotation(x=item[1], y=df.loc[item[1], 'upper_keltner'],
                            text=f'Breaks out on {date}', showarrow=True, arrowhead=1)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#BEBEBE')              
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#BEBEBE')
        fig.layout.xaxis.rangeslider.visible = False
        st.plotly_chart(fig)


def plot_fibonacci_levels(ticker, start_date, end_date):
    df = pd.read_csv(os.path.join(f, ticker + '.csv'), index_col='Unnamed: 0', parse_dates=True)
    df = df[start_date: end_date]
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    highest_swing = -1
    lowest_swing = -1

    for i in range(1, df.shape[0] - 1):
        if df['high'][i] > df['high'][i - 1] and df['high'][i] > df['high'][i + 1] \
            and (highest_swing == -1 or df['high'][i] > df['high'][highest_swing]):
            highest_swing = i
        if df['low'][i] < df['low'][i - 1] and df['low'][i] < df['low'][i + 1] \
            and (lowest_swing == -1 or df['low'][i] < df['low'][lowest_swing]):
            lowest_swing = i

    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    colors = ["black", "red", "green", "blue", "cyan", "magenta", "gold"]
    levels = []
    max_level = df['high'][highest_swing]
    min_level = df['low'][lowest_swing]

    for ratio in ratios:
        # Uptrend
        if highest_swing > lowest_swing:
            levels.append(max_level - (max_level - min_level) * ratio)
        # Downtrend
        else:
            levels.append(min_level + (max_level - min_level) * ratio)

    sy = []

    for x in levels:
        a = []
        for i in df.index:
            a.append(x)
        sy.append(a)
    
    name = SPY_info_df[SPY_info_df['Symbol'] == ticker]['Security'].item()
    title = f'{name} ({ticker})'
    candlesticks = go.Candlestick(x=df['date'], open=df['open'], high=df['high'],
                                  low=df['low'], close=df['close'], name=ticker)
    frl = go.Scatter(x=df.date, y=sy[0], name='0%', line={'color': colors[0], 'width': 0.75})
    frl1 = go.Scatter(x=df.date, y=sy[1], name='23.6%', line={'color': colors[1], 'width': 0.75})
    frl2 = go.Scatter(x=df.date, y=sy[2], name='38.2%', line={'color': colors[2], 'width': 0.75})
    frl3 = go.Scatter(x=df.date, y=sy[3], name='50.0%', line={'color': colors[3], 'width': 0.75})
    frl4 = go.Scatter(x=df.date, y=sy[4], name='61.8%', line={'color': colors[4], 'width': 0.75})
    frl5 = go.Scatter(x=df.date, y=sy[5], name='78.6%', line={'color': colors[5], 'width': 0.75})
    frl6 = go.Scatter(x=df.date, y=sy[6], name='100%', line={'color': colors[6], 'width': 0.75})
    layout = go.Layout(plot_bgcolor='#ECECEC', paper_bgcolor='#ECECEC')
    fig = go.Figure(data=[candlesticks, frl, frl1, frl2, frl3, frl4, frl5, frl6], layout=layout)
    cs = fig.data[0]
    cs.increasing.fillcolor = '#B7E9F7'
    cs.increasing.line.color = '#45B6FE'
    cs.decreasing.fillcolor = '#BEBEBE'
    cs.decreasing.line.color = '#808080'
    fig.update_layout(title=title, yaxis_title='Price')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.layout.xaxis.rangeslider.visible = False
    st.plotly_chart(fig)


# Check if stock had a SMA crossover in the last 5 trading days
@st.cache
def find_SMA_crossovers(crossover):
    golden, death = [], []
    s1 = crossover.split('/')
    sma1 = int(s1[0])
    csma1 = str(sma1) + 'sma'
    s2 = s1[1].split(' ')
    sma2 = int(s2[0])
    csma2 = str(sma2) + 'sma'
    
    for ticker in ticker_list:
        df = pd.read_csv(os.path.join(f, ticker + '.csv'), index_col='Unnamed: 0', parse_dates=True)
        
        if len(df) < sma2:
            continue
        else:
            df[csma1] = df['close'].rolling(window=sma1).mean()
            df[csma2] = df['close'].rolling(window=sma2).mean()
            df['signal'] = 0
            df['signal'][sma2:] = np.where(df[csma1][sma2:] > df[csma2][sma2:], 1, 0)
            df['crossover'] = df['signal'].diff()
        
            if df['crossover'].iloc[-5:].any():
                cross = df['crossover'].iloc[-5:].to_list()
                try:
                    val = cross.index(1)
                    golden.append(ticker)
                except ValueError:
                    death.append(ticker)

    return golden, death


def make_crossover_charts(crossover, lst, n):        
    s1 = crossover.split('/')
    sma1 = int(s1[0])
    s2 = s1[1].split(' ')
    sma2 = int(s2[0])
    
    for ticker in lst[n: n + 10]:
        df = pd.read_csv(os.path.join(f, ticker + '.csv'), index_col='Unnamed: 0', parse_dates=True)
        df = df.iloc[-sma2 * 3:]
        name = SPY_info_df[SPY_info_df['Symbol'] == ticker]['Security'].item()
        title = f'{name} ({ticker})'
        qf = cf.QuantFig(df, name=ticker, title=title)
        qf.add_sma(periods=sma1, column='close', colors='green', width=1)
        qf.add_sma(periods=sma2, column='close', colors='purple', width=1)
        fig = qf.iplot(asFigure=True, yTitle='Price')
        st.plotly_chart(fig)


def get_news(ticker, date):
    r = requests.get(f'''https://finnhub.io/api/v1/company-news?symbol={ticker}&from=
                         {date}&to={date}&token={st.secrets["FINNHUB_API_KEY"]}''')
    data = r.json()
    
    return data


SPY_df = get_SPY_data()
SPY_info_df = get_SPY_info()
ticker_list = SPY_info_df['Symbol'].to_list()
sector_list = SPY_info_df['GICS Sector'].unique()
last_date = SPY_df.iloc[-1].name.date()
yr_ago = last_date - timedelta(days=365)
first_dates = get_first_dates()
combined_returns_df = make_combined_returns_df()
current_ratios, ratios_data_report = get_all_current_ratios()
sector_weights, subIndustry_weights, ticker_weights = get_weights()
coming_out = TTM_Squeeze()