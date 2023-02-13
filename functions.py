import os
import numpy as np
import pandas as pd
import requests
import pickle
from datetime import datetime as dt
from datetime import timedelta
from pytz import timezone
from operator import itemgetter
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def get_rf_data():
    '''Returns DataFrame of 90-day T-Bill Rates'''

    return pd.read_csv('data/T-Bill Rates.csv', index_col=0, parse_dates=True)


def get_SPY_info():
    '''Returns DataFrame of info about S&P 500 companies'''
    
    df = pd.read_csv('data/spy_data/SPY_Info.csv', index_col=0)
    cols = {'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Sub-Industry'}
    df.rename(columns=cols, inplace=True)

    return df


def get_SPY_data():
    '''Returns DataFrame of S&P 500 market data'''

    df = pd.read_csv('data/spy_data/SPY.csv') # , index_col=0, parse_dates=True
    df.index = pd.to_datetime(df['Date'].apply(lambda x: x.split(' ')[0]))
    df.drop(columns='Date', inplace=True)

    return df


@st.cache
def get_ticker_data(ticker):
    '''Load ticker's market data'''
    
    file = os.path.join('data/market_data/daily', f'{ticker}.csv')
    df = pd.read_csv(file) # , index_col=0, parse_dates=True
    df.index = pd.to_datetime(df['Date'].apply(lambda x: x.split(' ')[0]))
    df.drop(columns='Date', inplace=True)
    
    return df


@st.cache
def get_interval_market_data(ticker, interval):
    '''Load ticker's market data'''
    
    if interval.endswith('Min'):
        folder = interval.split(' Min')[0] + 'm'
        col = 'Datetime'
        fmt = '-0:'
    elif interval == 'Weekly':
        folder = '1wk'
        col = 'Date'
        fmt = ' '
    elif interval == 'Monthly':
        folder = '1mo'

    file = os.path.join(f'data/market_data/{folder}', f'{ticker}.csv')
    df = pd.read_csv(file) # , index_col=0, parse_dates=True 
    df.index = pd.to_datetime(df[col].apply(lambda x: x.split(fmt)[0]))
    df.drop(columns=col, inplace=True)
    # df.drop(columns=['Adj Close'], inplace=True)
    
    return df


@st.cache
def get_ticker_info():
    fname = 'data/spy_data/spy_tickers_info.pickle'
    with open(fname, 'rb') as f:
        info = pickle.load(f)

    return info


@st.cache
def get_first_dates():
    '''Get the first date for which market data is available for each company'''

    first_dates = []

    for ticker in ticker_list:
        df = get_ticker_data(ticker)
        first_date = df.iloc[0].name
        first_dates.append((ticker, first_date))
        first_dates = sorted(first_dates, key=itemgetter(1))

    return first_dates


@st.cache
def find_stocks_missing_data(start_date, end_date):
    '''Identify stocks that lack market data before the start date'''

    first_dates = get_first_dates()
    s1, s2, s3 = '', '', ''
    first_date = first_dates[0][1]
    
    if start_date < first_date:
        s1 += f"**The first date for which there is data is {first_date.strftime('%B %d, %Y')}**"
    if end_date > last_date:
        s2 += f"**The last date for which there is data is {last_date.strftime('%B %d, %Y')}**"
    
    missing = filter(lambda x: x[1] > start_date, first_dates)
    missing = [x[0] for x in missing]

    d = {} # Dict of stocks by sector & sub-industry
        
    for ticker in missing:
        sector = SPY_info_df.loc[ticker, 'Sector']
        subIndustry = SPY_info_df.loc[ticker, 'Sub-Industry']
        d.setdefault(sector, {})
        d[sector].setdefault(subIndustry, []).append(ticker)

    if len(missing) > 0:
        s3 += f'''{len(missing)}/{len(ticker_list)} stocks have data that begins after 
                  {start_date.strftime('%B %d, %Y')}, which affects the accuracy of 
                  results displayed below.'''

    return d, s1, s2, s3


def load_TTM_ratios():
    '''Loads a dict of the latest available Trailing Twelve-Month (TTM) ratios from file'''

    tz = timezone('EST')
    cdate = dt.now(tz)  # Set datetime object to EST timezone
    file = cdate.strftime('%d-%m-%Y') + '.pickle'
    path = 'data/financial_ratios/Current'
    
    if file in os.listdir(path):
        s = "The data reported is today's."
    else:
        dates = sorted([dt.strptime(os.path.splitext(x)[0], '%d-%m-%Y')
                        for x in os.listdir(path)])
        date = dt.strftime(dates[-1], '%B %d, %Y')
        file = dt.strftime(dates[-1], '%d-%m-%Y') + '.pickle'
        s = f'Data as of {date}'

    with open(os.path.join(path, file), 'rb') as f:
        d = pickle.load(f)
    
    return d, s


@st.cache
def get_weights():
    '''Assign market cap weights by sector, sub-industry & ticker'''
    
    weights_df = pd.read_csv('data/spy_data/SPY_Weights.csv', index_col='Symbol')
    sectors, subIndustries, tickers = {}, {}, {}

    for sector in sector_list:
        sector_tickers = SPY_info_df[SPY_info_df['Sector'] == sector].index.to_list()
        sectors[sector] = 0
        for ticker in sector_tickers:
            t_si = SPY_info_df.loc[ticker, 'Sub-Industry']
            weight = weights_df.loc[ticker, 'Weight']
            tickers[ticker] = weight
            subIndustries.setdefault(t_si, 0) 
            subIndustries[t_si] += weight
            sectors[sector] += weight
       
    return sectors, subIndustries, tickers


def set_form_dates():
    '''Returns a streamlit form for selecting date inputs'''

    with st.form(key='form1'):
        c1, c2 = st.columns(2)
        start = c1.date_input('Start Date', yr_ago, min_value=first_date)
        end = c2.date_input('End Date', last_date, max_value=last_date)
        st.form_submit_button(label='Submit')

    return pd.to_datetime(start), pd.to_datetime(end)

    
@st.cache
def plot_returns_histogram(df):
    '''Histogram of daily returns'''

    gt0 = (len(df[df['Return'] >= 0]) / (len(df) - 1))
    lt0 = (len(df[df['Return'] < 0]) / (len(df) - 1))
    
    xtitle = f'Negative Daily Returns: {lt0:,.0%} | Positive Daily Returns: {gt0:,.0%}'
    title = 'Daily Returns Distribution'
    
    if df['Return'].mean() < df['Return'].median():
        pos1 = 'top left'
        pos2 = 'top right'
    else:
        pos1 = 'top right'
        pos2 = 'top left'
        
    fig = px.histogram(df, x='Return', title=title, opacity=0.5)
    fig.add_vline(x=df['Return'].mean(),
                  line_color='red',
                  line_width=0.65, 
                  annotation_text=f"Mean ({df['Return'].mean():.2%})",
                  annotation_position=pos1, 
                  annotation_bgcolor='indianred',
                  annotation_bordercolor='red')
    fig.add_vline(x=df['Return'].median(),
                  line_color='limegreen',
                  line_width=0.65, 
                  annotation_text=f"Median ({df['Return'].median():.2%})",
                  annotation_position=pos2, 
                  annotation_bgcolor='limegreen',
                  annotation_bordercolor='green')
    fig.update_annotations(font=dict(color='white'))
    fig.update_layout(xaxis_title=xtitle)
    fig.layout.xaxis.tickformat = ',.2%'

    return fig


@st.cache
def calculate_beta(ticker, start_date, end_date):
    '''Stock's beta relative to S&P 500'''
    
    df1 = get_SPY_data()[start_date : end_date]
    df1['Return'] = np.log1p(df1['Close'].pct_change())
    df1.rename(columns={'Return': 'SPY'}, inplace=True)
    df2 = get_ticker_data(ticker)[start_date : end_date]
    df2['Return'] = np.log1p(df2['Adj Close'].pct_change())
    df2.rename(columns={'Return': ticker}, inplace=True)
    df = pd.concat([df1['SPY'], df2[ticker]], axis=1, join='inner')
    SPY_std = df['SPY'].std()
    ticker_std = df[ticker].std()
    corr_df = df.corr()
    SPY_ticker_corr = corr_df.loc[ticker, 'SPY']
    beta = SPY_ticker_corr * (ticker_std / SPY_std)
    
    return beta
    

@st.cache
def calculate_metrics(start_date, end_date):
    '''
    Calculate return, volatility, sharpe ratio, and beta
    for index, stocks, sectors and sub-industries
    '''
    
    rf = rf_rates.loc[start_date : end_date, 'Close'].mean() / 100
    df = SPY_df[start_date : end_date]
    df['Return'] = np.log1p(df['Close'].pct_change())
    t = len(df) / 252   
    cagr = ((df['Close'][-1] / df['Open'][0])**(1 / t) - 1)
    std = df['Return'].std() * np.sqrt(252) # Annualised std
    sr = (cagr - rf) / std # Sharpe Ratio
    SPY = pd.Series({'Return': cagr, 'Volatility': std, 'Sharpe Ratio': sr, 'Beta': 1, 'RF': rf})

    sector_weights, subIndustry_weights, ticker_weights = get_weights()
    sectors, subIndustries, tickers = {}, {}, {}
    
    for sector in sector_list:
        sector_tickers = SPY_info_df[SPY_info_df['Sector'] == sector].index.to_list()
        sectors[sector] = {'Return': 0, 'Volatility': 0, 'Sharpe Ratio': 0, 'Beta': 0}
        for ticker in sector_tickers:
            # Get sub-industry of ticker
            t_si = SPY_info_df.loc[ticker, 'Sub-Industry']
            df = get_ticker_data(ticker)[start_date : end_date]
            df['Return'] = np.log1p(df['Adj Close'].pct_change())
            t = len(df) / 252
            cagr = ((df['Adj Close'][-1] / df['Open'][0])**(1 / t) - 1)
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
            
            d = {'Sector': sector, 'Return': 0, 'Volatility': 0, 'Sharpe Ratio': 0, 'Beta': 0}
            subIndustries.setdefault(t_si, d)
            subIndustries[t_si]['Return'] += (cagr * t_si_weight)
            subIndustries[t_si]['Volatility'] += (std * t_si_weight)
            subIndustries[t_si]['Sharpe Ratio'] += (sr * t_si_weight)
            subIndustries[t_si]['Beta'] += (beta * t_si_weight)

            tickers[ticker] = {}
            tickers[ticker]['Company'] = SPY_info_df.loc[ticker, 'Security']
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


@st.cache
def get_TTM_ratios(ratios, ratio):
    '''
    Returns dicts of Trailing Twelve-Month (TTM) financial ratios 
    by sector, sub-industry, and ticker
    '''

    r = ratios[ratio]
    sectors, subIndustries, tickers = {}, {}, {}
    sector_weights, subIndustry_weights, ticker_weights = get_weights()

    for sector in sector_list:
        sector_tickers = SPY_info_df[SPY_info_df['Sector'] == sector].index.to_list()
        sectors[sector] = 0
        for ticker in sector_tickers:
            t_si = SPY_info_df.loc[ticker, 'Sub-Industry'] # Get sub-industry of ticker
            res = TTM_ratios[ticker][r] # Ratio result

            # Get weights to use in weighted average calculations
            weight = ticker_weights[ticker]
            ticker_sector_weight = weight / sector_weights[sector]
            ticker_subIndustry_weight = weight / subIndustry_weights[t_si]
            
            sectors[sector] += (res * ticker_sector_weight)
            
            subIndustries.setdefault(t_si, {'Sector': sector, ratio: 0})
            subIndustries[t_si][ratio] += (res * ticker_subIndustry_weight)
    
            tickers[ticker] = {}
            tickers[ticker]['Company'] = SPY_info_df.loc[ticker, 'Security']
            tickers[ticker]['Sector'] = SPY_info_df.loc[ticker, 'Sector']
            tickers[ticker]['Sub-Industry'] = SPY_info_df.loc[ticker, 'Sub-Industry']
            tickers[ticker][ratio] = res                    

    sectors = pd.DataFrame.from_dict(sectors, orient='index', columns=[ratio])
    sectors.index.name = 'Sector'
    subIndustries = pd.DataFrame.from_dict(subIndustries, orient='index')
    subIndustries.index.name = 'Sub-Industry'
    tickers = pd.DataFrame.from_dict(tickers, orient='index')
    tickers.index.name = 'Ticker'
    
    return sectors, subIndustries, tickers 


@st.cache(allow_output_mutation=True)
def plot_sma_returns(ticker, start_date, end_date, window):
    '''Returns line charts of simple moving average of returns for ticker and S&P 500'''

    ticker_df = get_ticker_data(ticker)[start_date : end_date]
    ticker_df['Return'] = np.log1p(ticker_df['Adj Close'].pct_change())
    SPY_df = get_SPY_data()[start_date : end_date]
    SPY_df['Return'] = np.log1p(SPY_df['Close'].pct_change())
    beta = calculate_beta(ticker, start_date, end_date)
    yr_days = 365
    years, days = divmod((end_date - start_date).days, yr_days)
    years += days / yr_days

    fig = go.Figure([
            go.Scatter(
                x=SPY_df.index,
                y=SPY_df['Return'].rolling(window=window).mean(),
                name='S&P 500',
                mode='lines',
                line_width=1.25,
                line_color='red',
                showlegend=True),
            go.Scatter(
                x=ticker_df.index,
                y=ticker_df['Return'].rolling(window=window).mean(),
                name=ticker,
                mode='lines',
                line_width=1.25,
                line_color='blue',
                showlegend=True)     
            ])
    fig.layout.yaxis.tickformat = ',.2%'
    fig.update_layout(title=f'{window}-Day Moving Average of Returns',
                      xaxis=dict(title=f'{years:.0f}y Beta = {beta:,.2f}',
                      showgrid=False), 
                      yaxis_title='Return')

    return fig
    

@st.cache(allow_output_mutation=True)
def plot_sector_metric(df1, df2, metric):
    '''
    Returns bar chart of S&P 500 sectors by metric
    
    Parameters
    ----------
    df1: DataFrame of sector metrics
    df2: pd.Series of S&P 500 metrics
    metric: selected from [Return, Volatility, Sharpe Ratio, Beta, Financial Ratio]  
    '''

    df1.sort_values(by=metric, ascending=False, inplace=True)    
    fig = px.bar(df1, x=df1.index, y=metric, opacity=0.65)

    if metric == 'Return' or metric == 'Volatility':
        fig.layout.yaxis.tickformat = ',.2%'
        text1 = f'S&P 500 ({df2[metric]:.2%})'
        fig.add_hline(y=df2[metric],
                  line_color='red',
                  line_width=1,
                  annotation_text=text1, 
                  annotation_position='top right',
                  annotation_bgcolor='indianred',
                  annotation_bordercolor='red')
    else:
        fig.layout.yaxis.tickformat = ',.2f'
        text1 = f'S&P 500 ({df2[metric]:,.2f})'
    
    if metric != 'Volatility':
        title = f'Sector {metric}s'
    else:
        title = 'Sector Volatilities'
    
    if metric == 'Sharpe Ratio':
        xtitle = f'Risk-Free Rate = {df2.RF:.2%}'
    else:
        xtitle = ''

    fig.update_layout(title=title, xaxis_title=xtitle)
    fig.update_annotations(font=dict(color='white'))

    return fig


@st.cache(allow_output_mutation=True)
def plot_subIndustry_metric(df1, df2, df3, sector, metric):
    '''
    Returns bar chart of sub-industries in sector by metric
    
    Parameters
    ----------  
    df1: DataFrame of sector metrics
    df2: DataFrame of sub-industries metrics
    df3: pd.Series of S&P 500 metrics
    sector: user-selected input
    metric: selected from [Return, Volatility, Sharpe Ratio, Beta, Financial Ratio] 
    '''

    df2 = df2[df2.Sector == sector].sort_values(by=metric, ascending=False)
    sector_metric = df1.loc[sector, metric]
    SPY_metric = df3[metric]

    if metric != 'Volatility':
        title = f'{sector} Sub-Industry {metric}s'
    else:
        title = f'{sector} Sub-Industry Volatilities'
    
    if metric == 'Sharpe Ratio':
        xtitle = f'Risk-Free rate = {df3.RF:.2%}'
    else:
        xtitle = ''

    pos1 = 'top right'
    pos2 = 'bottom left'
    
    fig = px.bar(df2, x=df2.index, y=metric, opacity=0.65)

    if metric == 'Return' or metric == 'Volatility':
        fig.layout.yaxis.tickformat = ',.2%'
        text1 = f'S&P 500 ({SPY_metric:.2%})'
        text2 = f'{sector} ({sector_metric:.2%})'
    else:
        fig.layout.yaxis.tickformat = ',.2f'
        text1 = f'S&P 500 ({SPY_metric:,.2f})'
        text2 = f'{sector} ({sector_metric:,.2f})'

    fig.add_hline(y=SPY_metric,
                  line_color='red',
                  line_width=1,
                  annotation_text=text1, 
                  annotation_position=pos1,
                  annotation_bgcolor='indianred',
                  annotation_bordercolor='red')
    fig.add_hline(y=sector_metric,
                  line_color='limegreen',
                  line_width=1,
                  annotation_text=text2,
                  annotation_position=pos2, 
                  annotation_bgcolor='limegreen',
                  annotation_bordercolor='green')       
    fig.update_layout(title=title, xaxis_title=xtitle)
    fig.update_annotations(font=dict(color='white'))

    return fig


@st.cache(allow_output_mutation=True)
def plot_si_tickers_metric(df1, df2, df3, df4, sector, subIndustry, metric, ticker=None):
    '''
    Returns bar chart of sub-industries in sector by metric
    
    Parameters
    ----------  
    df1: DataFrame of sector metrics
    df2: DataFrame of sub-industries metrics
    df3: DataFrame of tickers metrics
    df4: pd.Series of S&P 500 metrics
    sector: user-selected input
    metric: selected from [Return, Volatility, Sharpe Ratio, Beta, Financial Ratio]
    ticker: user-selected input
    '''

    df3 = df3[df3['Sub-Industry'] == subIndustry].sort_values(by=metric, ascending=False)
    sector_metric = df1.loc[sector, metric]
    subIndustry_metric = df2.loc[subIndustry, metric]
    SPY_metric = df4[metric]

    if metric != 'Volatility':
        title = f'{subIndustry} Company {metric}s'
    else:
        title = f'{subIndustry} Company Volatilities'
    
    if metric == 'Sharpe Ratio':
        xtitle = f'Risk-Free rate = {df4.RF:.2%}'
    else:
        xtitle = ''

    pos1 = 'top right'
    
    if sector_metric < subIndustry_metric:
        pos2 = 'bottom left'
        pos3 = 'top left'
    else:
        pos2 = 'top left'
        pos3 = 'bottom left'
    
    fig = px.bar(df3, x=df3.index, y=metric, opacity=0.65, hover_data={'Company': True})

    if metric == 'Return' or metric == 'Volatility':
        fig.layout.yaxis.tickformat = ',.2%'
        text1 = f'S&P 500 ({SPY_metric:.2%})'
        text2 = f'{sector} Sector ({sector_metric:.2%})'
        text3 = f'{subIndustry} Sub-Industry ({subIndustry_metric:.2%})'
    else:
        fig.layout.yaxis.tickformat = ',.2f'
        text1 = f'S&P 500 ({SPY_metric:,.2f})'
        text2 = f'{sector} Sector ({sector_metric:,.2f})'
        text3 = f'{subIndustry} Sub-Industry ({subIndustry_metric:,.2f})'
        
    fig.add_hline(y=SPY_metric,
                  line_color='red',
                  line_width=1,
                  annotation_text=text1, 
                  annotation_position=pos1,
                  annotation_bgcolor='indianred',
                  annotation_bordercolor='red')
    fig.add_hline(y=sector_metric,
                  line_color='limegreen',
                  line_width=1,
                  annotation_text=text2,
                  annotation_position=pos2, 
                  annotation_bgcolor='limegreen',
                  annotation_bordercolor='green')
    fig.add_hline(y=subIndustry_metric,
                  line_color='blue',
                  line_width=1,
                  annotation_text=text3,
                  annotation_position=pos3, 
                  annotation_bgcolor='blue',
                  annotation_bordercolor='blue')

    if ticker:
        if metric != 'Volatility':
            title = f'{subIndustry} Sub-Industry {metric}s'
        else:
            title = f'{subIndustry} Sub-Industry Volatilities'

        rank_df = df3.reset_index()
        rank_df.index += 1
        si_rank = rank_df[rank_df['Ticker'] == ticker].index.item()
        fig.add_annotation(x=ticker, 
                           y=df3.loc[ticker, metric],
                           text=f'{ticker} is ranked {si_rank}/{len(rank_df)} in sub-industry', 
                           showarrow=True, 
                           arrowhead=3,
                           arrowwidth=2,
                           arrowcolor='fuchsia',
                           bordercolor='purple',
                           bgcolor='fuchsia'
                        )
                        
    fig.update_layout(title=title, xaxis_title=xtitle)
    fig.update_annotations(font=dict(color='white'))
                        
    return fig


@st.cache
def plot_sector_tickers_metric(df1, df2, df3, df4, sector, subIndustry, metric, ticker):
    '''
    Returns bar chart of tickers in sector by metric
    
    Parameters
    ----------  
    df1: DataFrame of sector metrics
    df2: DataFrame of sub-industries metrics
    df3: DataFrame of tickers metrics
    df4: pd.Series of S&P 500 metrics
    sector: user-selected input
    metric: selected from [Return, Volatility, Sharpe Ratio, Beta, Financial Ratio]
    ticker: user-selected input
    '''

    df3 = df3[df3['Sector'] == sector].sort_values(by=metric, ascending=False)
    sector_metric = df1.loc[sector, metric]
    subIndustry_metric = df2.loc[subIndustry, metric]
    SPY_metric = df4[metric]
    rank_df = df3.reset_index()
    rank_df.index += 1
    sector_rank = rank_df[rank_df['Ticker'] == ticker].index.item()

    if metric != 'Volatility':
        title = f'{sector} Sector {metric}s'
    else:
        title = f'{sector} Sector Volatilities'
    
    if metric == 'Sharpe Ratio':
        xtitle = f'Risk-Free rate = {df4.RF:.2%}'
    else:
        xtitle = ''

    pos1 = 'top right'
    
    if sector_metric < subIndustry_metric:
        pos2 = 'bottom left'
        pos3 = 'top left'
    else:
        pos2 = 'top left'
        pos3 = 'bottom left'
    
    fig = px.bar(df3, x=df3.index, y=metric, opacity=0.65, hover_data={'Company': True})

    if metric == 'Return' or metric == 'Volatility':
        fig.layout.yaxis.tickformat = ',.2%'
        text1 = f'S&P 500 ({SPY_metric:.2%})'
        text2 = f'{sector} Sector ({sector_metric:.2%})'
        text3 = f'{subIndustry} Sub-Industry ({subIndustry_metric:.2%})'
    else:
        fig.layout.yaxis.tickformat = ',.2f'
        text1 = f'S&P 500 ({SPY_metric:,.2f})'
        text2 = f'{sector} Sector ({sector_metric:,.2f})'
        text3 = f'{subIndustry} Sub-Industry ({subIndustry_metric:,.2f})'

    fig.add_hline(y=SPY_metric,
                  line_color='red',
                  line_width=1,
                  annotation_text=text1, 
                  annotation_position=pos1,
                  annotation_bgcolor='indianred',
                  annotation_bordercolor='red')
    fig.add_hline(y=sector_metric,
                  line_color='limegreen',
                  line_width=1,
                  annotation_text=text2,
                  annotation_position=pos2, 
                  annotation_bgcolor='limegreen',
                  annotation_bordercolor='green')
    fig.add_hline(y=subIndustry_metric,
                  line_color='blue',
                  line_width=1,
                  annotation_text=text3,
                  annotation_position=pos3, 
                  annotation_bgcolor='blue',
                  annotation_bordercolor='blue')
    fig.add_annotation(x=ticker, 
                       y=df3.loc[ticker, metric],
                       text=f'{ticker} is ranked {sector_rank}/{len(rank_df)} in sector', 
                       arrowhead=3,
                       arrowwidth=2,
                       arrowcolor='fuchsia',
                       bordercolor='purple',
                       bgcolor='fuchsia',
                       )
    fig.update_layout(title=title, xaxis_title=xtitle)
    fig.update_annotations(font=dict(color='white'))

    return fig


@st.cache(allow_output_mutation=True)
def plot_sector_financial_ratios(df, ratio):
    df.sort_values(by=ratio, ascending=False, inplace=True) 
    fig = px.bar(df, x=df.index, y=ratio, opacity=0.65)
    fig.update_layout(title=f'Sector {ratio}', xaxis_title='')
    
    return fig
                          

# @st.cache(allow_output_mutation=True)
# def plot_si_financial_ratios(df1, df2, ratio, sector):
#     sector_ratio = df1.loc[sector].item()
#     df2 = df2[df2['Sector'] == sector].sort_values(by=ratio, ascending=False)
    
#     # Chart of sub-industry ratios
#     fig = px.bar(df2, x=df2.index, y=ratio, opacity=0.65)
#     fig.layout.yaxis.tickformat = ',.2f'
#     fig.add_hline(y=sector_ratio, 
#                   line_color='red', 
#                   line_width=1,
#                   annotation_text=f'{sector} {ratio} ({sector_ratio:,.2f})', 
#                   annotation_bgcolor='indianred', 
#                   annotation_bordercolor='red')
#     fig.update_layout(title=f'{sector} Sub-Industry {ratio}s', xaxis_title='')
    
#     return fig


def get_news(ticker, date):
    '''Get news about stock on date using Finnhub API'''

    token = st.secrets["FINNHUB_API_KEY"]
    url = f'https://finnhub.io/api/v1/company-news?symbol={ticker}&from={date}&to={date}&token={token}'
    r = requests.get(url)
    data = r.json()
    
    return data


def get_financial_statements():
    '''Load dict of available financial statements of S&P 500 stocks'''

    file = 'data/financial_statements/financial_statements.pickle'

    with open(file, 'rb') as f:
        d = pickle.load(f)
    
    return d


rf_rates = get_rf_data()
SPY_df = get_SPY_data()
SPY_info_df = get_SPY_info()
tickers_info = get_ticker_info()
ticker_list = SPY_info_df.index.to_list()
sector_list = sorted(SPY_info_df['Sector'].unique().tolist())
first_date = SPY_df.iloc[0].name
last_date = SPY_df.iloc[-1].name
yr_ago = last_date - timedelta(days=365)
TTM_ratios, ratios_data_report = load_TTM_ratios()