import bisect
import numpy as np
from datetime import timedelta
from itertools import zip_longest

import holidays
from scipy.signal import find_peaks
import cufflinks as cf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from talib import BBANDS, MACD, RSI
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


def sr_levels(df):
    '''Returns key support/resistance levels for a security'''

    levels = []
    s = (df['High'] - df['Low']).mean()

    def isFarFromLevel(l):
        '''
        Given a price value, returns False 
        if it is near some previously discovered key level
        '''
        
        return np.sum([abs(l - x) < s for x in levels]) == 0

    for i in range(2, df.shape[0] - 2):
        if isSupport(df, i):
            l = df['Low'][i]
            if isFarFromLevel(l):
                levels.append((i, l))
        elif isResistance(df, i):
            l = df['High'][i]
            if isFarFromLevel(l):
                levels.append((i, l))

    levels.sort(key=itemgetter(1))

    return levels


@st.cache(allow_output_mutation=True)
def calculate_signals(ticker, start, end, period, minor_ma, secondary_ma, primary_ma):
    if period != 'Daily':
        if period.endswith('Min'):
            end += timedelta(1)
        df = get_interval_market_data(ticker, period)[start:end]
    else:
        df = get_ticker_data(ticker)[start:end]
        
    df.drop(columns=['Adj Close'], inplace=True)
    MAs = [minor_ma, secondary_ma, primary_ma]

    for ma in MAs:
        df[f'MA{ma}'] = df['Close'].rolling(ma).mean()
        df[f'Advanced MA{ma}'] = df['Close'].rolling(ma, center=True).mean().shift(int(ma**(1/2)))
        # df[f'MA{ma} Peak-and-Trough Reversal'] = 0
        # peaks, _ = find_peaks(df['Close'], height=0)
        # troughs, _ = find_peaks(df['Close'] * -1, height=0)
    
    return df


@st.cache
def calculate_fibonacci_levels(df):
    highest_swing = -1
    lowest_swing = -1
    high = df['High']
    low = df['Low']

    for i in range(1, df.shape[0] - 1):
        if high[i] > high[i - 1] and high[i] > high[i + 1] \
            and (highest_swing == -1 or high[i] > high[highest_swing]):
            highest_swing = i
        if low[i] < low[i - 1] and low[i] < low[i + 1] \
            and (lowest_swing == -1 or low[i] < low[lowest_swing]):
            lowest_swing = i

    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    levels = []
    max_level = high[highest_swing]
    min_level = low[lowest_swing]

    for ratio in ratios:
        # Uptrend
        if highest_swing > lowest_swing:
            level = max_level - (max_level - min_level) * ratio
        # Downtrend
        else:
            level = min_level + (max_level - min_level) * ratio
        levels.append(level)

    return ratios, levels


@st.cache(allow_output_mutation=True)
def plot_trends(graph, ticker, start, end, period, minor_ma, secondary_ma, primary_ma,
                show_vol, show_rsi, show_macd, show_sr, show_fib, show_bb, show_MAs, show_adv_MAs):
    '''
    Returns plot figure

    Parameters
    ----------
    df: DataFrame of a security's market data
    start: Start Date
    end: End Date
    period: user-input
    minor_ma: moving average window
    secondary_ma : moving average window
    primary_ma: moving average window
    show_prices: display candlestick data on crowded charts
    '''

    df = calculate_signals(ticker, start, end, period, minor_ma, secondary_ma, primary_ma)
    cname = SPY_info_df.loc[ticker, 'Security']
    nrows = 1 + show_vol + show_rsi + show_macd
    titles = [f'{cname} - {period} Chart'] + [''] * nrows
    r1 = 1 - 0.1 * nrows
    r2 = (1 - r1) / (nrows - 1)
    row_heights = [r1] + [r2] * (nrows - 1)
    fig_row = 2
    data = []

    fig = make_subplots(rows=nrows, cols=1,
                        shared_xaxes=True, 
                        vertical_spacing=0.05,
                        subplot_titles=titles, 
                        row_heights=row_heights)
    fig.update_xaxes(showgrid=True)          
    fig.update_yaxes(showgrid=False, type='log')

    if graph == 'Candlesticks':
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
        data.append(cs)
    else:
        xy = go.Scatter(x=df.index,
                y=df['Close'],
                name='Close',
                line_width=1.5,
                connectgaps=True)
        data.append(xy)

    if show_MAs or show_adv_MAs:
        MAs = [minor_ma, secondary_ma, primary_ma]
        colors = ['red', 'cyan', 'gold']
        if show_MAs and show_adv_MAs:
            dash = 'dot'
        else:
            dash = 'solid'
        for ma, color in zip(MAs, colors): 
            if show_MAs:
                y = df[f'MA{ma}'].shift(-int(ma / 2)) # centre MA           
                sma = go.Scatter(x=df.index,
                                 y=y,
                                 name=f'MA{ma}',
                                 line_width=1,
                                 line_color=color,
                                 connectgaps=True)
                data.append(sma)
            if show_adv_MAs:
                advanced_sma = go.Scatter(x=df.index,
                                          y=df[f'Advanced MA{ma}'],
                                          name=f'Advanced MA{ma}',
                                          line_width=1.25,
                                          line_color=color,
                                          line_dash=dash,
                                          connectgaps=True)
                data.append(advanced_sma)
    
    pos = [1] * len(data) # position to add rows, cols in subplot 
    fig.add_traces(data=data, rows=pos, cols=pos)  

    # Support & resistance lines
    if show_sr:
        levels = sr_levels(df)
        for i, l in levels:
            n = df.shape[0] - i
            fig.add_scatter(x=df.index[i:],
                            y=[l] * n,
                            name='S/R',
                            line_width=0.5,
                            line_color='orange',
                            mode='lines',
                            showlegend=False,
                            connectgaps=True)   

    # Fibonacci retracements
    if show_fib:
        colors = ["darkgray", "indianred", "green", "blue", "cyan", "magenta", "gold"]
        ratios, levels = calculate_fibonacci_levels(df)
        for i in range(len(ratios)):
            fig.add_scatter(x=df.index,
                            y=[levels[i]] * df.shape[0],
                            name=f'FR {ratios[i]:,.2%}',
                            line_color=colors[i],
                            line_width=0.75,
                            line_dash='dot',
                            connectgaps=True)
    

    # Bollinger bands
    if show_bb:
        up, mid, down = BBANDS(df['Close'], timeperiod=21, nbdevup=2, nbdevdn=2, matype=0)
        bands = {'BB Up': up, 'BB Mid': mid, 'BB Down': down}
        for k, v in bands.items():
            dash = 'dot' if k == 'BB Mid' else 'solid'
            fig.add_scatter(x=v.index,
                            y=v.values,
                            name=k,
                            line_width=1,
                            line_dash=dash,
                            mode='lines',
                            connectgaps=True)   

    # Volume subplot
    if show_vol:   
        fig.add_bar(x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker={'color': 'steelblue'},
                    row=fig_row, col=1)
        fig_row += 1

    # RSI subplot
    if show_rsi:
        rsi = RSI(df['Close'], timeperiod=14)
        fig.add_scatter(x=rsi.index,
                        y=rsi.values,
                        name='RSI',
                        line_width=1,
                        mode='lines',
                        connectgaps=True,
                        row=fig_row, col=1)
        fig.add_hline(70, line_width=0.5, line_dash='dot', line_color='red', 
                      row=fig_row, col=1)
        fig.add_hline(30, line_width=0.5, line_dash='dot', line_color='red', 
                      row=fig_row, col=1)
        fig.update_layout({f'yaxis{fig_row}': {'type': 'linear'}})
        fig_row += 1
    
    # MACD subplot
    if show_macd:
        macd, *_ = MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        fig.add_scatter(x=macd.index,
                        y=macd.values,
                        name='MACD',
                        line_width=1,
                        mode='lines',
                        connectgaps=True,
                        row=fig_row, col=1)
        fig.update_layout({f'yaxis{fig_row}': {'type': 'linear'}})
    
    if period != 'Weekly':
        us_holidays = list(holidays.US(range(start.year, end.year + 1)).keys())
        rangebreaks = [dict(bounds=["sat", "mon"]), dict(values=us_holidays)]
        if period.endswith('Min'):
            rangebreaks.extend([dict(bounds=[16, 9.5], pattern="hour")])
        fig.update_xaxes(rangebreaks=rangebreaks)

    fig.layout.xaxis.rangeslider.visible = False

    return fig


@st.cache
def get_trending_stocks(start, end, period, MAs):
    up = []
    down = []
    minor_ma, secondary_ma, primary_ma, *_ = MAs

    for ticker in ticker_list:
        try:
            df = calculate_signals(ticker, start, end, period, minor_ma, secondary_ma, primary_ma)
            if df[f'MA{primary_ma}'][-1] > df[f'MA{secondary_ma}'][-1] > df[f'MA{minor_ma}'][-1]:
                down.append(ticker)
            if df[f'MA{primary_ma}'][-1] < df[f'MA{secondary_ma}'][-1] < df[f'MA{minor_ma}'][-1]:
                up.append(ticker)
        except:
            continue

    return up, down


@st.cache
def get_trend_aligned_stocks(periods_data, periods, end_date):
    for i, period in enumerate(periods):
        period_d = periods_data[period]
        days = period_d['days']
        start = end_date - timedelta(days)
        MAs = period_d['MA']
        up, down = get_trending_stocks(start, end_date, period, MAs)
        if i == 0:
            up_aligned = set(up)
            down_aligned = set(down)
        else:
            up_aligned.intersection_update(up) 
            down_aligned.intersection_update(down)

    return list(up_aligned), list(down_aligned)