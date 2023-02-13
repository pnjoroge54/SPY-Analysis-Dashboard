import numpy as np
from datetime import timedelta
from itertools import zip_longest

import holidays
from scipy.signal import find_peaks
import cufflinks as cf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from functions import *


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
        df['TR'] = abs(df['High'] - df['Low'])
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
    df['TR'] = abs(df['High'] - df['Low'])
    df['ATR'] = df['TR'].rolling(window=20).mean()
    df['lower_keltner'] = df['20sma'] - (df['ATR'] * 1.5)
    df['upper_keltner'] = df['20sma'] + (df['ATR'] * 1.5)
    name = SPY_info_df.loc[ticker, 'Security']
    title = f'{name} ({ticker})'
    candlestick = go.Candlestick(x=df.index, open=df['open'], High=df['High'],
                                 Low=df['Low'], close=df['close'], name=ticker)
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
    fig.add_annotation(x=date, 
                       y=df.loc[date, 'upper_keltner'],
                       text=f'Breaks out on {fdate}', 
                       showarrow=True, 
                       arrowhead=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#BEBEBE')              
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#BEBEBE')
    fig.layout.xaxis.rangeslider.visible = False
    st.plotly_chart(fig)


def plot_fibonacci_levels(ticker, start_date, end_date):
    df = get_ticker_data(ticker)[start_date : end_date]
    # df.reset_index(inplace=True)
    # df.rename(columns={'index': 'date'}, inplace=True)
    highest_swing = -1
    lowest_swing = -1

    for i in range(1, df.shape[0] - 1):
        if (df['High'][i] > df['High'][i - 1] and df['High'][i] > df['High'][i + 1]) \
            and (highest_swing == -1 or df['High'][i] > df['High'][highest_swing]):
            highest_swing = i
        if (df['Low'][i] < df['Low'][i - 1] and df['Low'][i] < df['Low'][i + 1]) \
            and (lowest_swing == -1 or df['Low'][i] < df['Low'][lowest_swing]):
            lowest_swing = i

    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    colors = ["black", "red", "green", "blue", "cyan", "magenta", "gold"]
    levels = []
    max_level = df['High'][highest_swing]
    min_level = df['Low'][lowest_swing]

    for ratio in ratios:
        # Uptrend
        if highest_swing > lowest_swing:
            levels.append(max_level - (max_level - min_level) * ratio)
        # Downtrend
        else:
            levels.append(min_level + (max_level - min_level) * ratio)

    y = [[x] * len(df) for x in levels]

    name = SPY_info_df.loc[ticker, 'Security']
    title = f'{name} ({ticker})'
    candlesticks = go.Candlestick(x=df['date'], open=df['open'], High=df['High'],
                                  Low=df['Low'], close=df['close'], name=ticker)
    frl0 = go.Scatter(x=df.date, y=y[0], name='0%',    line={'color': colors[0], 'width': 0.75})
    frl1 = go.Scatter(x=df.date, y=y[1], name='23.6%', line={'color': colors[1], 'width': 0.75})
    frl2 = go.Scatter(x=df.date, y=y[2], name='38.2%', line={'color': colors[2], 'width': 0.75})
    frl3 = go.Scatter(x=df.date, y=y[3], name='50.0%', line={'color': colors[3], 'width': 0.75})
    frl4 = go.Scatter(x=df.date, y=y[4], name='61.8%', line={'color': colors[4], 'width': 0.75})
    frl5 = go.Scatter(x=df.date, y=y[5], name='78.6%', line={'color': colors[5], 'width': 0.75})
    frl6 = go.Scatter(x=df.date, y=y[6], name='100%',  line={'color': colors[6], 'width': 0.75})
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


def find_SMA_crossovers(crossover):
    '''Find stocks that had a SMA crossover in the last 5 trading days'''

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
    '''Make chart of SMA crossover for stock'''

    s1 = crossover.split('/')
    sma1 = int(s1[0])
    s2 = s1[1].split(' ')
    sma2 = int(s2[0])
    df = get_ticker_data(ticker)
    df = df.iloc[-sma2 * 3:]
    name = SPY_info_df.loc[ticker,'Security']
    title = f'{name} ({ticker})'
    qf = cf.QuantFig(df, name=ticker, title=title)
    qf.add_sma(periods=sma1, column='close', colors='green', width=1)
    qf.add_sma(periods=sma2, column='close', colors='blue', width=1)
    fig = qf.iplot(asFigure=True, yTitle='Price')
    st.plotly_chart(fig)


def isSupport(df, i):
    '''Returns True if value is a price support level'''

    X = df['Low']
    support = X[i] < X[i - 1] \
                and X[i] < X[i + 1] \
                and X[i + 1] < X[i + 2] \
                and X[i - 1] < X[i - 2]

    return support


def isResistance(df, i):
    '''Returns True if value is a price resistance level'''

    X = df['High']
    resistance = X[i] > X[i - 1] \
                    and X[i] > X[i + 1] \
                    and X[i + 1] > X[i + 2] \
                    and X[i - 1] > X[i - 2] 

    return resistance


def sr_levels(df, start, end):
    '''Returns key support/resistance levels for a security'''

    df = df[start : end]
    support = []
    resistance = []
    s = (df['High'] - df['Low']).mean()

    def isFarFromLevel(l):
        '''
        Given a price value, returns False 
        if it is near some previously discovered key level
        '''
        
        levels = support + resistance
        return np.sum([abs(l - x) < s for x in levels]) == 0

    for i in range(2, df.shape[0] - 2):
        if isSupport(df, i):
            l = df['Low'][i]
            if isFarFromLevel(l):
                support.append((i, l))
        elif isResistance(df, i):
            l = df['High'][i]
            if isFarFromLevel(l):
                resistance.append((i, l))

    return support, resistance


@st.cache
def calculate_signals(ticker, start, end, period, short_ma, inter_ma, long_ma):
    if period != 'Daily':
        df = get_interval_market_data(ticker, period)[start : end]
    else:
        df = get_ticker_data(ticker)[start : end]
    
    df.drop(columns=['Adj Close'], inplace=True)
    MAs = [short_ma, inter_ma, long_ma]

    for ma in MAs:
        df[f'MA{ma}'] = df['Close'].rolling(ma).mean()
        # df[f'MA{ma} Peak-and-Trough Reversal'] = 0
        # peaks, _ = find_peaks(df['Close'], height=0)
        # troughs, _ = find_peaks(df['Close'] * -1, height=0)
    
    return df


@st.cache(allow_output_mutation=True)
def add_fibonacci_levels(df, fig):
    highest_swing = -1
    lowest_swing = -1

    for i in range(1, df.shape[0] - 1):
        if (df['High'][i] > df['High'][i - 1] and df['High'][i] > df['High'][i + 1]) \
            and (highest_swing == -1 or df['High'][i] > df['High'][highest_swing]):
            highest_swing = i
        if (df['Low'][i] < df['Low'][i - 1] and df['Low'][i] < df['Low'][i + 1]) \
            and (lowest_swing == -1 or df['Low'][i] < df['Low'][lowest_swing]):
            lowest_swing = i

    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    colors = ["darkgray", "indianred", "green", "blue", "cyan", "magenta", "gold"]
    levels = []
    max_level = df['High'][highest_swing]
    min_level = df['Low'][lowest_swing]

    for ratio in ratios:
        # Uptrend
        if highest_swing > lowest_swing:
            level = max_level - (max_level - min_level) * ratio
        # Downtrend
        else:
            level = min_level + (max_level - min_level) * ratio
        levels.append(level)

    y = [[x] * df.shape[0] for x in levels]

    for i in range(len(ratios)):
        fig.add_scatter(x=df.index,
                        y=y[i],
                        name=f'FRL {ratios[i]:,.2%}',
                        line={'color': colors[i], 'width': 0.75, 'dash': 'dot'},
                        connectgaps=True)
    
    return fig


@st.cache(allow_output_mutation=True)
def plot_trends(ticker, start, end, period, short_ma, inter_ma, long_ma, show_sr, show_prices, show_fib):
    '''
    Returns candlestick chart with support/resistance levels and market cycle trend lines

    Parameters
    ----------
    df: DataFrame of a security's market data
    start: Start Date
    end: End Date
    period: user-input
    short_ma: moving average window
    inter_ma: moving average window
    long_ma: moving average window
    show_prices: display candlestick data on crowded charts
    '''

    df = calculate_signals(ticker, start, end, period, short_ma, inter_ma, long_ma)
    cname = SPY_info_df.loc[ticker, 'Security']
    title1 = f'{cname} - {period} Chart'
    title2 = ''

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True, 
                        vertical_spacing=0.05,
                        subplot_titles=(title1, title2), 
                        row_width=[0.2, 0.8])
    
    cs = go.Candlestick(x=df.index, 
                        open=df['Open'], 
                        high=df['High'],
                        low=df['Low'], 
                        close=df['Close'],
                        name=ticker)
    
    cs.increasing.fillcolor = 'green'
    cs.increasing.line.color = 'darkgreen'
    cs.decreasing.fillcolor = 'red'
    cs.decreasing.line.color = 'indianred'

    data = [cs]
    MAs = [short_ma, inter_ma, long_ma]
    colors = ['red', 'blue', 'limegreen']

    for ma, color in zip(MAs, colors):
        ma = f'MA{ma}'
        sma = go.Scatter(x=df.index,
                         y=df[ma],
                         name=ma,
                         line_width=1,
                         line_color=color,
                         connectgaps=True)
        data.append(sma)
    
    pos = [1] * len(data) # position to add rows, cols in subplot 
    fig.add_traces(data=data, rows=pos, cols=pos)  
    
    support, resistance = sr_levels(df, start, end)
    levels = support + resistance

    # Add support & resistance lines
    if show_sr:
        for i, l in levels:
            n = df.shape[0] - i
            fig.add_scatter(x=df.index[i:],
                            y=[l] * n,
                            name='S/R Level',
                            line_width=0.5,
                            line_color='orange',
                            mode='lines',
                            showlegend=False,
                            connectgaps=True)
            

    if show_fib:
        fig = add_fibonacci_levels(df, fig)
            
    # Volume subplot
    fig.add_trace(go.Bar(x=df.index,
                         y=df['Volume'],
                         name='Volume',
                         marker={'color': 'steelblue'}),
                  row=2, col=1)
    
    us_holidays = list(holidays.US(range(start.year, end.year + 1)).keys())
    fig.update_xaxes(showgrid=True)
    
    if period != 'Weekly':
        rangebreaks = [dict(bounds=["sat", "mon"]), dict(values=us_holidays)]
        if period.endswith('Min'):
            rangebreaks.extend([dict(bounds=[16, 9.5], pattern="hour")])
        fig.update_xaxes(showgrid=True, rangebreaks=rangebreaks)
        
    if show_prices:
        fig.update_layout(hovermode="x unified")
              
    fig.update_yaxes(showgrid=False)
    # fig.layout.annotations[0].update(x=0.1)
    fig.layout.xaxis.rangeslider.visible = False

    return fig


@st.cache
def get_trending_stocks(start, end, period, MAs):
    up = []
    down = []
    short_ma, inter_ma, long_ma, *_ = MAs
    for ticker in ticker_list:
        try:
            df = calculate_signals(ticker, start, end, period, short_ma, inter_ma, long_ma)
            if df[f'MA{long_ma}'][-1] > df[f'MA{inter_ma}'][-1] > df[f'MA{short_ma}'][-1]:
                down.append(ticker)
            if df[f'MA{long_ma}'][-1] < df[f'MA{inter_ma}'][-1] < df[f'MA{short_ma}'][-1]:
                up.append(ticker)
        except:
            continue

    return up, down


@st.cache
def get_trend_aligned_stocks(periods_data, periods, date):
    end = date

    for i, period in enumerate(periods):
        period_d = periods_data[period]
        days = period_d['days']
        days = days + 3 if end.weekday() == 0 else days
        start = end - timedelta(days)
        MAs = period_d['MA']
        up, down = get_trending_stocks(start, end, period, MAs)
        if i == 0:
            up_aligned = set(up)
            down_aligned = set(down)
        else:
            up_aligned.intersection_update(up) 
            down_aligned.intersection_update(down)

    return list(up_aligned), list(down_aligned)


# def get_trending_stocks_data(up, down, period):
#     tickers = up + down
#     d = dict(Up={}, Down={})

#     for ticker in tickers:
#         if ticker in up:
#             d['Up'][ticker] = date

