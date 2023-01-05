import os
import numpy as np
import pandas as pd
import requests
import pickle
from datetime import datetime as dt
from datetime import timedelta
from pytz import timezone

import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from load_data import *


rf_rates = get_rf_data()
SPY_df = get_SPY_data()
SPY_info_df = get_SPY_info()
ticker_list = SPY_info_df['Symbol'].to_list()
sector_list = SPY_info_df['GICS Sector'].unique()
first_date = SPY_df.iloc[0].name
last_date = SPY_df.iloc[-1].name
yr_ago = last_date - timedelta(days=365)


def getIndexOfTuple(lst, index, value):
    for pos, t in enumerate(lst):
        if t[index] == value:
            return pos + 1
            
    # Matches behavior of list.index
    raise ValueError("list.index(x): x not in list")


@st.cache
def get_ticker_data(ticker):
    file = os.path.join(r'data\market_data', f'{ticker}.csv')
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    df['Return'] = np.log1p(df['adjclose'].pct_change())
    
    return df


@st.cache
def get_first_dates():
    '''Get the first date for which market data is available for each company'''

    first_dates = []

    for ticker in ticker_list:
        df = get_ticker_data(ticker)
        first_date = df.iloc[0].name
        first_dates.append((ticker, first_date))
        first_dates = sorted(first_dates, key=lambda x: x[1])

    return first_dates


def get_all_current_ratios():
    tz = timezone('EST')
    cdate = dt.now(tz)  # Set datetime object to EST timezone
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
    sector_weights, subIndustry_weights, ticker_weights = get_weights()

    for sector in sector_list:
        sector_tickers = SPY_info_df[SPY_info_df['GICS Sector'] == sector] \
                         ['Symbol'].to_list()
        sector_ratios[sector] = 0
        for ticker in sector_tickers:
            # Get sub-industry of ticker
            t_si = SPY_info_df[SPY_info_df['Symbol'] == ticker]['GICS Sub-Industry'].item()
            res = current_ratios[ticker][r] # Ratio result

            # Get weight to use in weighted average calculation
            weight = ticker_weights[ticker]
            ticker_sector_weight = weight / sector_weights[sector]
            ticker_subIndustry_weight = weight / subIndustry_weights[t_si]
            
            sector_ratios[sector] += (res * ticker_sector_weight)
            
            subIndustry_ratios.setdefault(t_si, 0)
            subIndustry_ratios += (res * ticker_subIndustry_weight)
    
            ticker_ratios[ticker] = {}
            ticker_ratios[ticker]['Company'] = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                                                ['Security'].item()
            ticker_ratios[ticker]['Sector'] = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                                                ['GICS Sector'].item()
            ticker_ratios[ticker]['Sub-Industry'] = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                                                    ['GICS Sub-Industry'].item()
            ticker_ratios[ticker][ratio] = res                    

    sector_ratios_df = pd.DataFrame.from_dict(sector_ratios, orient='index', columns=[ratio])

    return sector_ratios_df, subIndustry_ratios, ticker_ratios 


@st.cache
def calculate_beta(ticker, start_date, end_date):
    df1 = SPY_df[start_date : end_date]
    df1.rename(columns={'Return': 'SPY'}, inplace=True)
    df2 = get_ticker_data(ticker)[start_date : end_date]
    df2.rename(columns={'Return': ticker}, inplace=True)
    df = pd.concat([df1['SPY'], df2[ticker]], axis=1, join='inner')
    SPY_std = df['SPY'].std()
    ticker_std = df[ticker].std()
    corr_df = df.corr()
    SPY_ticker_corr = corr_df.loc[ticker, 'SPY']
    beta = SPY_ticker_corr * (ticker_std / SPY_std)
    
    return beta


@st.cache
def get_weights():
    # Assign weights by sectors & sub-industries
    
    weights_df = pd.read_csv(r'data\SPY Weights.csv', index_col='Symbol')
    sectors, subIndustries, tickers = {}, {}, {}

    for sector in sector_list:
        sector_tickers = SPY_info_df[SPY_info_df['GICS Sector'] == sector] \
                         ['Symbol'].to_list()
        sectors[sector] = 0
        for ticker in sector_tickers:
            t_si = SPY_info_df[SPY_info_df['Symbol'] == ticker]['GICS Sub-Industry'].item()
            weight = weights_df.loc[ticker, 'Weight']
            tickers[ticker] = weight
            subIndustries.setdefault(t_si, 0) 
            subIndustries[t_si] += weight
            sectors[sector] += weight
       
    return sectors, subIndustries, tickers


@st.cache
def find_stocks_missing_data(start_date, end_date):
    '''Identify stocks that lack market data before the start date'''

    first_dates = get_first_dates()
    s1, s2, s3 = '', '', ''
    
    if start_date < first_dates[0][1]:
        s1 += f"**The first date for which there is data is \
                {first_dates[0][1].strftime('%B %d, %Y')}**"
    if end_date > last_date:
        s2 += f"**The last date for which there is data is \
                {last_date.strftime('%B %d, %Y')}**"
    
    missing = filter(lambda x: x[1] > start_date, first_dates)
    missing = [x[0] for x in missing]

    d = {sector: {} for sector in sector_list} # Dict of stocks by sector & sub-industry
    
    for ticker in missing:
        sector = SPY_info_df[SPY_info_df['Symbol'] == ticker]['GICS Sector'].item()
        subIndustry = SPY_info_df[SPY_info_df['Symbol'] == ticker]['GICS Sub-Industry'].item()
        d[sector].setdefault(subIndustry, []).append(ticker)

    # Delete empty dict keys
    for sector in sector_list:
        if len(d[sector]) == 0:
            del d[sector]

    if len(missing) > 0:
        s3 += f'''{len(missing)}/{len(ticker_list)} stocks have data that begins after 
                  {start_date.strftime('%B %d, %Y')}, which affects the accuracy of 
                  results displayed below.'''

    return d, s1, s2, s3


@st.cache
def calculate_metrics(start_date, end_date):
    rf = rf_rates.loc[start_date : end_date, 'Close'].mean() / 100
    df = SPY_df[start_date : end_date]
    t = len(df) / 252   
    cagr = ((df['Close'][-1] / df['Open'][0])**(1 / t) - 1)
    std = df['Return'].std() * np.sqrt(252) # Annualised std
    sr = (cagr - rf) / std # Sharpe Ratio
    SPY = pd.Series({'Return': cagr, 'Volatility': std, 'Sharpe Ratio': sr, 'Beta': 1, 'RF': rf})

    sector_weights, subIndustry_weights, ticker_weights = get_weights()
    sectors, subIndustries, tickers = {}, {}, {}
    
    for sector in sector_list:
        sector_tickers = SPY_info_df[SPY_info_df['GICS Sector'] == sector] \
                         ['Symbol'].to_list()
        sectors[sector] = {'Return': 0, 'Volatility': 0, 'Sharpe Ratio': 0, 'Beta': 0}
        for ticker in sector_tickers:
            # Get sub-industry of ticker
            t_si = SPY_info_df[SPY_info_df['Symbol'] == ticker]['GICS Sub-Industry'].item()
            df = get_ticker_data(ticker)[start_date : end_date]
            t = len(df) / 252
            cagr = ((df['adjclose'][-1] / df['open'][0])**(1 / t) - 1)
            std = df['Return'].std() * np.sqrt(252) # Annualised std
            sr = (cagr - rf) / std # Sharpe Ratio
            beta = calculate_beta(ticker, start_date, end_date)

            # Get weights to use in weighted average calculation
            weight = ticker_weights[ticker]
            t_sector_weight = weight / sector_weights[sector]
            t_si_weight = weight / subIndustry_weights[t_si]

            sectors[sector]['Return'] += (cagr * t_sector_weight)
            sectors[sector]['Volatility'] += (std * t_sector_weight)
            sectors[sector]['Sharpe Ratio'] += (sr * t_sector_weight)
            sectors[sector]['Beta'] += (beta * t_sector_weight)

            subIndustries.setdefault(t_si,
                {'Sector': sector, 'Return': 0, 'Volatility': 0, 'Sharpe Ratio': 0, 'Beta': 0})
            subIndustries[t_si]['Return'] += (cagr * t_si_weight)
            subIndustries[t_si]['Volatility'] += (std * t_si_weight)
            subIndustries[t_si]['Sharpe Ratio'] += (sr * t_si_weight)
            subIndustries[t_si]['Beta'] += (beta * t_si_weight)

            tickers[ticker] = {}
            tickers[ticker]['Company'] = SPY_info_df[SPY_info_df['Symbol'] == ticker] \
                                             ['Security'].item()
            tickers[ticker]['Sector'] = sector
            tickers[ticker]['Sub-Industry'] = t_si
            tickers[ticker]['Return'] = cagr
            tickers[ticker]['Volatility'] = std
            tickers[ticker]['Sharpe Ratio'] = sr
            tickers[ticker]['Beta'] = beta

    sectors = pd.DataFrame.from_dict(sectors, orient='index')
    sectors.index.name = 'Sector'
    subIndustries = pd.DataFrame.from_dict(subIndustries, orient='index')
    subIndustries.index.name = 'Sub-Industry'
    tickers = pd.DataFrame.from_dict(tickers, orient='index')
    tickers.index.name = 'Ticker'

    return sectors, subIndustries, tickers, SPY, rf


def set_form_dates():
    with st.form(key='form1'):
        c1, c2 = st.columns(2)
        start = c1.date_input('Start Date', yr_ago, min_value=first_date)
        end = c2.date_input('End Date', last_date, max_value=last_date)
        submit_btn = c1.form_submit_button(label='Submit')

    return start, end


@st.cache
def make_sector_chart(sector_df, SPY_df, metric):
    sector_df.sort_values(by=metric, ascending=False, inplace=True)    
    fig = px.bar(sector_df, x=sector_df.index, y=metric, opacity=0.65)

    if metric == 'Return' or metric == 'Volatility':
        fig.layout.yaxis.tickformat = ',.0%'
        text1 = f'S&P 500 ({SPY_df[metric] * 100:,.2f}%)'
    else:
        fig.layout.yaxis.tickformat = ',.2'
        text1 = f'S&P 500 ({SPY_df[metric]:,.2f})'

    fig.add_hline(y=SPY_df[metric],
                  line_color='red',
                  line_width=1,
                  annotation_text=text1, 
                  annotation_bgcolor='#FF7F7F',
                  annotation_bordercolor='red')
    
    if metric != 'Volatility':
        title = f'Sector {metric}s'
    else:
        title = 'Sector Volatilities'
    
    if metric == 'Sharpe Ratio':
        xtitle = f'Risk-free rate = {SPY_df.RF * 100:,.2f}%'
    else:
        xtitle = ''

    fig.update_layout(title=title, xaxis_title=xtitle)

    return fig


@st.cache
def make_subIndustry_chart(sector, sector_df, subIndustries_df, SPY_df, metric):
    subIndustries_df = subIndustries_df[subIndustries_df.Sector == sector] \
                        .sort_values(by=metric, ascending=False)
    sector_metric = sector_df.loc[sector, metric]
    SPY_metric = SPY_df[metric]

    # Set positions of annotation texts
    if SPY_metric > sector_metric:
        pos1 = 'top right'
        pos2 = 'bottom right'
    else:
        pos1 = 'bottom right'
        pos2 = 'top right'
    
    fig = px.bar(subIndustries_df, x=subIndustries_df.index, y=metric, opacity=0.65)

    if metric == 'Return' or metric == 'Volatility':
        fig.layout.yaxis.tickformat = ',.0%'
        text1 = f'S&P 500 ({SPY_metric * 100:,.2f}%)'
        text2 = f'{sector} ({sector_metric * 100:,.2f}%)'
    else:
        fig.layout.yaxis.tickformat = ',.2'
        text1 = f'S&P 500 ({SPY_metric:,.2f})'
        text2 = f'{sector} ({sector_metric:,.2f})'

    fig.add_hline(y=SPY_metric,
                  line_color='red',
                  line_width=1,
                  annotation_text=text1, 
                  annotation_position=pos1,
                  annotation_bgcolor='#FF7F7F',
                  annotation_bordercolor='red')
    fig.add_hline(y=sector_metric,
                  line_color='green',
                  line_width=1,
                  annotation_text=text2,
                  annotation_position=pos2, 
                  annotation_bgcolor='green',
                  annotation_bordercolor='green')
    
    if metric != 'Volatility':
        title = f'{sector} Sub-Industry {metric}s'
    else:
        title = f'{sector} Sub-Industry Volatilities'
    
    if metric == 'Sharpe Ratio':
        xtitle = f'Risk-free rate = {SPY_df.RF * 100:,.2f}%'
    else:
        xtitle = ''
        
    fig.update_layout(title=title, xaxis_title=xtitle)

    return fig


@st.cache
def make_subIndustry_tickers_chart(sector, subIndustry, sector_df, subIndustries_df, 
                                   tickers_df, SPY_df, metric):
    tickers_df = tickers_df[tickers_df['Sub-Industry'] == subIndustry] \
                    .sort_values(by=metric, ascending=False)
    sector_metric = sector_df.loc[sector, metric]
    subIndustry_metric = subIndustries_df.loc[subIndustry, metric]
    SPY_metric = SPY_df[metric]
    metrics = [sector_metric, subIndustry_metric, SPY_metric]

    # Set positions of annotation texts
    if SPY_metric == min(metrics):
        pos1 = 'bottom left'
    else:
        pos1 = 'top left'

    if subIndustry_metric < sector_metric:
        pos2 = 'top right'
        pos3 = 'bottom right'
    else:
        pos2 = 'bottom right'
        pos3 = 'top right'
    
    fig = px.bar(tickers_df, x=tickers_df.index, y=metric,
                 opacity=0.65, hover_data={'Company':True})

    if metric == 'Return' or metric == 'Volatility':
        fig.layout.yaxis.tickformat = ',.0%'
        text1 = f'S&P 500 ({SPY_metric * 100:,.2f}%)'
        text2 = f'{sector} ({sector_metric * 100:,.2f}%)'
        text3 = f'{subIndustry} ({subIndustry_metric * 100:,.2f}%)'
    else:
        fig.layout.yaxis.tickformat = ',.2'
        text1 = f'S&P 500 ({SPY_metric:,.2f})'
        text2 = f'{sector} ({sector_metric:,.2f})'
        text3 = f'{subIndustry} ({subIndustry_metric:,.2f})'

    fig.add_hline(y=SPY_metric,
                  line_color='red',
                  line_width=1,
                  annotation_text=text1, 
                  annotation_position=pos1,
                  annotation_bgcolor='#FF7F7F',
                  annotation_bordercolor='red')
    fig.add_hline(y=sector_metric,
                  line_color='green',
                  line_width=1,
                  annotation_text=text2,
                  annotation_position=pos2, 
                  annotation_bgcolor='green',
                  annotation_bordercolor='green')
    fig.add_hline(y=subIndustry_metric,
                  line_color='purple',
                  line_width=1,
                  annotation_text=text3,
                  annotation_position=pos3, 
                  annotation_bgcolor='purple',
                  annotation_bordercolor='purple')
    
    if metric != 'Volatility':
        title = f'{subIndustry} Company {metric}s'
    else:
        title = f'{subIndustry} Company Volatilities'
    
    if metric == 'Sharpe Ratio':
        xtitle = f'Risk-free rate = {SPY_df.RF * 100:,.2f}%'
    else:
        xtitle = ''
        
    fig.update_layout(title=title, xaxis_title=xtitle)

    return fig


def TTM_Squeeze():

    def in_squeeze(df):
        return df['lower_band'] > df['lower_keltner'] and df['upper_band'] < df['upper_keltner']

    coming_out = []

    for ticker in ticker_list:
        start_date = yr_ago
        df = get_ticker_data(ticker)
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
    

def make_TTM_squeeze_charts(ticker, date):
    fdate = date.strftime('%b %d')
    df = get_ticker_data(ticker)
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
    layout = go.Layout(plot_bgcolor='#ECECEC', paper_bgcolor='#ECECEC', 
                        font_color='black')
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
    fig.add_annotation(x=date, y=df.loc[date, 'upper_keltner'],
                        text=f'Breaks out on {fdate}', showarrow=True, arrowhead=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#BEBEBE')              
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#BEBEBE')
    fig.layout.xaxis.rangeslider.visible = False
    st.plotly_chart(fig)


def plot_fibonacci_levels(ticker, start_date, end_date):
    df = get_ticker_data(ticker)
    df = df[start_date : end_date]
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    highest_swing = -1
    lowest_swing = -1

    for i in range(1, df.shape[0] - 1):
        if (df['high'][i] > df['high'][i - 1] and df['high'][i] > df['high'][i + 1]) \
            and (highest_swing == -1 or df['high'][i] > df['high'][highest_swing]):
            highest_swing = i
        if (df['low'][i] < df['low'][i - 1] and df['low'][i] < df['low'][i + 1]) \
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

    sy = [[x] * len(df) for x in levels]

    name = SPY_info_df[SPY_info_df['Symbol'] == ticker]['Security'].item()
    title = f'{name} ({ticker})'
    candlesticks = go.Candlestick(x=df['date'], open=df['open'], high=df['high'],
                                  low=df['low'], close=df['close'], name=ticker)
    frl0 = go.Scatter(x=df.date, y=sy[0], name='0%',    line={'color': colors[0], 'width': 0.75})
    frl1 = go.Scatter(x=df.date, y=sy[1], name='23.6%', line={'color': colors[1], 'width': 0.75})
    frl2 = go.Scatter(x=df.date, y=sy[2], name='38.2%', line={'color': colors[2], 'width': 0.75})
    frl3 = go.Scatter(x=df.date, y=sy[3], name='50.0%', line={'color': colors[3], 'width': 0.75})
    frl4 = go.Scatter(x=df.date, y=sy[4], name='61.8%', line={'color': colors[4], 'width': 0.75})
    frl5 = go.Scatter(x=df.date, y=sy[5], name='78.6%', line={'color': colors[5], 'width': 0.75})
    frl6 = go.Scatter(x=df.date, y=sy[6], name='100%',  line={'color': colors[6], 'width': 0.75})
    layout = go.Layout(plot_bgcolor='#ECECEC', paper_bgcolor='#ECECEC', font_color='black')
    fig = go.Figure(data=[candlesticks, frl0, frl1, frl2, frl3, frl4, frl5, frl6], layout=layout)
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


@st.cache
def find_SMA_crossovers(crossover):
    # Check if stock had a SMA crossover in the last 5 trading days

    golden, death = [], []
    s1 = crossover.split('/')
    sma1 = int(s1[0])
    csma1 = str(sma1) + 'sma'
    s2 = s1[1].split(' ')
    sma2 = int(s2[0])
    csma2 = str(sma2) + 'sma'

    for ticker in ticker_list:
        df = get_ticker_data(ticker)
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


def make_crossover_charts(crossover, ticker):        
    s1 = crossover.split('/')
    sma1 = int(s1[0])
    s2 = s1[1].split(' ')
    sma2 = int(s2[0])
    df = get_ticker_data(ticker)
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


def get_financial_statements():
    with open(r'data\financial_statements.pickle', 'rb') as f:
        d = pickle.load(f)
    
    return d


current_ratios, ratios_data_report = get_all_current_ratios()
coming_out = TTM_Squeeze()