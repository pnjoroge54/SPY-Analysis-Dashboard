import bisect
import numpy as np
import time
from datetime import datetime as dt
from datetime import timedelta
from pprint import pprint
from itertools import zip_longest
import math

import holidays
from scipy.signal import argrelextrema
from scipy.stats import linregress
from sklearn import preprocessing
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


def convert_to_timestamp(x):
    """Convert date objects to integers"""
    
    return time.mktime(x.timetuple())


@st.cache
def sr_levels(df):
    '''Returns key support/resistance levels for a security'''

    df = df.copy()
    df['SR Signal'] = 0
    unit = 'minutes' if df.index[0].minute != 0 else 'days'  
    spt = 0
    rst = 0
    levels = []
    s_levels = []
    prev_date = df.index[0]
    sr_data = {}
    many_tests = {} # dict of bars that test more than 1 level
    s = (df['High'] - df['Low']).mean()
    nr, nc = df.shape

    def isFarFromLevel(l):
        '''Returns True if price is not near a previously discovered support or resistance'''
        
        return np.sum([abs(l - x[1]) < s for x in levels]) == 0

    for i in range(2, nr):
        date = df.index[i]
        s_date = date.strftime('%d-%m-%y')
        high = df['High'][i]
        low = df['Low'][i]
        close = df['Close'][i]
        new_spt = False
        new_rst = False
        sr_switch = False

        if i < nr - 2:
            if isSupport(df, i):
                if isFarFromLevel(low):
                    new_spt = True
                    spt = low
                    df.loc[date, 'Support'] = spt
                    levels.append((i, spt))
                    s_levels = sorted([x[1] for x in levels])
                    # print('NS'.ljust(5), f'- {date.date()} - S: {spt:.2f}, R: {rst:.2f}, hi: {high:.2f}, lo: {low:.2f}')
            
            if isResistance(df, i):
                if isFarFromLevel(high):
                    new_rst = True
                    rst = high
                    df.loc[date, 'Resistance'] = rst
                    levels.append((i, rst))
                    s_levels = sorted([x[1] for x in levels])
                    # print('NR'.ljust(5), f'- {date.date()} - R: {rst:.2f}, S: {spt:.2f}, hi: {high:.2f}, lo: {low:.2f},')    

        # Switch support to resistance & vice versa
        if len(levels) > 1:
            if new_spt:
                ix = bisect.bisect(s_levels, spt)
                rst = s_levels[ix] if ix < len(s_levels) else s_levels[ix - 1]
            if new_rst:
                ix = bisect.bisect_left(s_levels, rst)
                spt = s_levels[ix - 1] if ix > 0 else s_levels[ix]
            if low > rst: # When resistance broken 
                sr_switch = True
                spt = rst
                ix = bisect.bisect(s_levels, low)
                rst = s_levels[ix] if ix < len(s_levels) else s_levels[ix - 1]
                # print('R-S'.ljust(5), f'- {date.date()} - S: {spt:.2f}, R: {rst:.2f}, hi: {high:.2f}, lo: {low:.2f}')
            if high < spt: # When support broken 
                sr_switch = True
                rst = spt
                ix = bisect.bisect_left(s_levels, high)
                spt = s_levels[ix - 1] if ix > 0 else s_levels[ix]
                # print('S-R'.ljust(5), f'- {date.date()} - R: {rst:.2f}, S: {spt:.2f}, hi: {high:.2f}, lo: {low:.2f}')
        
        if new_rst or new_spt or sr_switch:
            cum_vol = df.loc[prev_date:date, 'Volume'].sum()
            delta = date - prev_date # time it takes level to form
            delta = delta.days if unit == 'days' else delta.total_seconds() / 60
            prev_date = date
            d = {'Date': [], 'Timedelta': [], 'Volume': [], 'SR': [], 'Tested': 0, 'Tested Date': []}
            sr_data.setdefault(spt, d)
            sr_data[spt]['Date'].append(s_date)
            sr_data[spt]['Timedelta'].append(delta)
            sr_data[spt]['Volume'].append(cum_vol)
            sr_data[spt]['SR'].append('S')
            # Prevents double-counting when lowest/highest level is both support & resistance
            if spt != rst:
                sr_data.setdefault(rst, d)
                sr_data[rst]['Date'].append(s_date)
                sr_data[rst]['Timedelta'].append(delta)
                sr_data[rst]['Volume'].append(cum_vol)
                sr_data[rst]['SR'].append('R')
                                  
        if spt:
            if close < spt:
                df.loc[date, 'SR Signal'] = 1 # Generate signal
            # Check if S/R levels are tested       
            if high > spt and low < spt:
                sr_data[spt]['Tested'] += 1
                sr_data[spt]['Tested Date'].append(s_date)
                # print('ST'.ljust(5), f'- {date.date()} - S: {spt:.2f}, R: {rst:.2f}, hi: {high:.2f}, lo: {low:.2f}')
                ix = bisect.bisect_left(s_levels, spt)
                n_spt = s_levels[ix - 1] if ix > 0 else s_levels[ix]    
                while low < n_spt and spt != rst and spt != n_spt:
                    # print(f'SH-SL - {date.date()} - NS: {n_spt:.2f}, S: {spt:.2f}, R: {rst:.2f}, hi: {high:.2f}, lo: {low:.2f}')
                    rst = spt
                    spt = n_spt
                    many_tests.setdefault(i, set()).union([spt, rst])
                    if ix > 0:
                        ix -= 1
                        n_spt = s_levels[ix]
                        sr_data[n_spt]['Date'].append(s_date)
                        sr_data[n_spt]['Timedelta'].append(delta)
                        sr_data[n_spt]['Volume'].append(cum_vol)
                        sr_data[n_spt]['SR'].append('S')            
   
        if rst:
            if close > rst:
                df.loc[date, 'SR Signal'] = 1 # Generate signal
            # Check if S/R levels are tested       
            if high > rst and low < rst:
                if spt != rst: # Prevents double-counting
                    sr_data[rst]['Tested'] += 1
                    sr_data[rst]['Tested Date'].append(s_date)
                    # print('RT'.ljust(5), f'- {date.date()} - R: {rst:.2f}, S: {spt:.2f}, hi: {high:.2f}, lo: {low:.2f}')
                    ix = bisect.bisect(s_levels, rst)
                    n_rst = s_levels[ix] if ix < len(s_levels) else s_levels[ix - 1]
                    while high > n_rst and spt != rst and rst != n_rst:
                        # print(f'RL-RH - {date.date()} - NR: {n_rst:.2f}, R: {rst:.2f}, S: {spt:.2f}, hi: {high:.2f}, lo: {low:.2f}')
                        spt = rst
                        rst = n_rst
                        many_tests.setdefault(i, set()).union([spt, rst])
                        if ix < len(s_levels) - 1:
                            ix += 1
                            # print(f'ix: {ix}, {s_levels}')
                            n_rst = s_levels[ix]
                            sr_data[n_rst]['Date'].append(s_date)
                            sr_data[n_rst]['Timedelta'].append(delta)
                            sr_data[n_rst]['Volume'].append(cum_vol)
                            sr_data[n_rst]['SR'].append('R')        

        if spt and rst: 
            df.loc[date:, 'Support'] = spt
            df.loc[date:, 'Resistance'] = rst

    # Calculate significance of levels       
    d = {'SR Level': [], 'Volume': [], 'Timedelta': [], 'Tested': [], 'Date': []}

    try: del sr_data[0]    
    except: pass

    for k, v in sr_data.items():
        d['SR Level'].append(k)
        d['Volume'].append(sum(v['Volume']))
        d['Timedelta'].append(sum(v['Timedelta']))
        d['Tested'].append(v['Tested'])
        d['Date'].append(v['Date'][-1])

    ix = 'SR Level'
    sr_df = pd.DataFrame(d, index=d[ix]).drop(columns=ix)
    sr_df['Date'] = pd.to_datetime(sr_df['Date'])
    sr_df['Date'] = sr_df['Date'].apply(convert_to_timestamp)
    scaler = preprocessing.MinMaxScaler(feature_range=(1, 5))
    sd = scaler.fit_transform(sr_df)
    scaled_df = pd.DataFrame(sd, columns=sr_df.columns, index=d[ix])
    scaled_df['Signal'] = scaled_df.mean(axis=1)
    # print(sr_df)
    # print(scaled_df)
    
    # Make 'SR Signal' last column
    cols = list(df.columns)
    cols.append(cols.pop(cols.index('SR Signal')))
    df = df[cols]
    nr, nc = df.shape
    j = nc - 1 # 'SR Signal' column num

    for i in range(nr):
        if df['SR Signal'][i]:
            if i in many_tests:
                signal = 0
                for l in many_tests[i]:
                    signal += scaled_df.loc[l, 'Signal']
                df.iloc[i, j] = signal
            else:
                if df['Close'][i] > df['Resistance'][i]:
                    l = df['Resistance'][i]
                    df.iloc[i, j] = scaled_df.loc[l, 'Signal']        
                if df['Close'][i] < df['Support'][i]:
                    l = df['Support'][i]
                    df.iloc[i, j] = scaled_df.loc[l, 'Signal']        
    
    return levels, df


@st.cache
def fibonacci_retracements(df):
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

    if highest_swing < lowest_swing:
        ratios.reverse()
        levels.reverse()

    return ratios, levels


@st.cache
def peaks_valleys_trendlines(df):
    '''
    df: DataFrame
    trendline: str 'hl' or 'close', 
               'hl' drawing trendlines from High or Low prices,
               and 'close' from Closing prices
    '''

    df = df.copy().drop(columns='Adj Close')
    close = df['Close']
    peaks = argrelextrema(close.to_numpy(), np.greater)[0]
    valleys = argrelextrema(close.to_numpy(), np.less)[0]
    PV = sorted(list(peaks) + list(valleys))
    first = 'Peak' if min(peaks[0], valleys[0]) == peaks[0] else 'Valley'
    second = 'Peak' if first == 'Valley' else 'Valley'
    first_vals = set(peaks) if first == 'Peak' else set(valleys)
    second_vals = set(peaks) if second == 'Peak' else set(valleys)
    ix = PV[0]
    d0 = abs(close[ix] - close[0])
    dist = [d0]   
    valid_PV = [ix]
    df.loc[df.index[peaks], 'isPeak'] = 1
    df.loc[df.index[valleys], 'isValley'] = 1
    df['Peak'] = close[peaks]
    df['Valley'] = close[valleys]
    trend = -1 if first == 'Valley' else 1
    df['PV Trend'] = trend
    trendlines_c = []
    trendlines_hl = []
    i, j = 0, 1
    cnt = 0
    # print(f'nPeaks: {len(peaks)}, nValleys: {len(valleys)}, nPV: {len(PV)}' \
    #       f'\nfirst peak, valley: {peaks[0], valleys[0]}' \
    #       f'\nlast peak, valley: {peaks[-1], valleys[-1]}\n')
    
    while j < len(PV):
        col = first if not cnt % 2 else second
        o_col = first if cnt % 2 else second
        cnt += 1
        ix = df.index[PV[j]]
        d1 = dist[-1]
        d2 = close[PV[j]] - close[PV[i]]
        retracement = abs(d2) / d1
        # print(f'{cnt}. i: {i}, j: {j} \nPV[{i}]: {PV[i]}, PV[{j}]: {PV[j]}')
        # print(f'd2: {close[PV[j]]:.2f} - {close[PV[i]]:.2f} = {d2:.2f}',
        #       f'\nRetraced: {abs(d2):.2f} / {d1:.2f} = {retracement:.2f}')
        
        if retracement >= 1/3:
            # print(f'Add {o_col} {close[PV[j]]:.2f} on {ix.date()}')
            d1 = abs(close[PV[j]] - close[valid_PV[-1]])
            dist.append(d1)
            valid_PV.append(PV[j])    
            if j < len(PV):
                j += 1      
                i = j - 1
        elif j < len(PV):
            # print(f'Skip {o_col} {close[PV[j]]:.2f} on {ix.date()}')
            # df.loc[ix, f'is{o_col}'] = 0
            if len(dist) > 1:
                dist.pop()
            if len(valid_PV) > 1:
                invalid = valid_PV.pop()
                ix = df.index[invalid]
                # df.loc[ix, f'is{col}'] = 0
                # print(f'Remove {col} {close[invalid]:.2f} on {ix.date()}') 
            i = PV.index(valid_PV[-1])
            j += 1

        # print(f'i: {i}, j: {j}, \n{[round(x, 2) for x in dist]}\n{valid_PV}\n')
            
    # Add last highest peak & last lowest valley if not in valid_PV
    print(f'Last {second} is {close[valid_PV[-1]]:.2f} on {df.index[valid_PV[-1]].date()}')
    print(f'Last {first} is {close[valid_PV[-2]]:.2f} on {df.index[valid_PV[-2]].date()}\n')
    df_ix_list = df.index.to_list()
    nr = df.shape[0]

    if valid_PV[-1] < nr - 1:
        ix = df.index[valid_PV[-1] + 1]
        val = df.loc[ix:, first].dropna() 
        if not val.empty:
            f_ix = val.idxmax() if first == 'Peak' else val.idxmin() # last higest/lowest index
            f_ix_pos = df_ix_list.index(f_ix) # index position    
            if f_ix_pos not in valid_PV and f_ix_pos < nr - 1:
                valid_PV.append(f_ix_pos)
                ix = df.index[f_ix_pos + 1]
                val = df.loc[ix:, second].dropna()
                if not val.empty:
                    s_ix = val.idxmax() if second == 'Peak' else val.idxmin()
                    s_ix_pos = df_ix_list.index(s_ix)
                    valid_PV.append(s_ix_pos)

            # print(f'Updated: \nLast {first} is {close[f_ix_pos]:.2f} on {f_ix.date()} iloc[{f_ix_pos}]\n' \
            #     f'Last {second} is {close[s_ix_pos]:.2f} on {s_ix.date()} iloc[{s_ix_pos}]\n')
    
    first_vals.intersection_update(valid_PV[::2])
    second_vals.intersection_update(valid_PV[1::2])
    first_vals = sorted(list(first_vals))
    second_vals = sorted(list(second_vals))
    # print(f'n{first}: {len(first_vals)}, n{second}: {len(second_vals)}\n')
    n = len(min(first_vals, second_vals, key=len))
    
    # Identify trends
    for i in range(1, n):
        a, pa = first_vals[i], first_vals[i - 1]
        b, pb = second_vals[i], second_vals[i - 1]
        start = df.index[min(pa, pb)] if i == 1 else end
        end = df.index[max(a, b)]
        if close[a] - close[pa] < 0 and close[b] - close[pb] < 0:
            trend = -1
        elif close[a] - close[pa] > 0 and close[b] - close[pb] > 0:
            trend = 1
        else:
            trend = 0
        df.loc[start:, 'PV Trend'] = trend
        # print(f'{i}. start: {start.date()}, end: {end.date()}'\
        #       f'\nprev. {first}: {close[pa]:.2f} on {df.index[pa].date()}, {first}: {close[a]:.2f} on {df.index[a].date()}' \
        #       f'\nprev. {second}: {close[pb]:.2f} on {df.index[pb].date()}, {second}: {close[b]:.2f} on {df.index[b].date()}' \
        #       f'\ntrend: {trend}\n')
    
    # Identify potential trendline ranges  
    df['PV Changepoint'] = df['PV Trend'].diff()
    df['Row'] = np.arange(nr)
    mask = (df['PV Changepoint'] != 0) & (df['PV Changepoint'].notna())
    c_points = df[mask].index
    # print(c_points)
    
    for i in range(1, len(c_points)):
        start = df.index[0] if i == 1 else end
        end = c_points[i]
        npeaks = df.loc[start:end, 'isPeak'].sum()
        nvalleys = df.loc[start:end, 'isValley'].sum()
        uptrend = False
        downtrend = False
        if df.loc[start, 'PV Trend'] < 0 and npeaks >= 2:
            downtrend = True
            indices = df[df['isPeak'] == 1][start:end].index
        if df.loc[start, 'PV Trend'] > 0 and nvalleys >= 2:
            uptrend = True
            indices = df[df['isValley'] == 1][start:end].index
        # print(f"{i}.".ljust(3), 
        #       f"{dt.strftime(start, '%d.%m.%y')} - {dt.strftime(end, '%d.%m.%y')}, " \
        #       f"peaks: {npeaks:.0f}, valleys: {nvalleys:.0f}, " \
        #       f"trend: {df.loc[start, 'PV Trend']:.0f}, c: {df.loc[start, 'PV Changepoint']:.0f}")
    
        if uptrend or downtrend:    
            xs = np.array(df['Row'][indices])
            ys = np.array(close[indices])
            m, c, r, *_ = linregress(xs, ys)
            x0, x2 = indices[0], indices[-1]
            xn = df_ix_list.index(x2) + 4
            xn = xn if xn < nr else nr - 1
            y0 = close[x0]
            yn = m * xn + c
            trendlines_c.append(((x0, df.index[xn]), (y0, yn)))
            r2 = r**2
            slope_angle = math.atan(m)  # slope angle in radians
            slope_angle_degrees = math.degrees(slope_angle)  # slope angle in degrees
            # print('Close Trendline')
            # print(linregress(xs, ys))
            # print(f'R2: {r2:.2f} \nangle_radians: {slope_angle:.2f} \nangle_deg: {slope_angle_degrees:.2f}')
            # print(f'y = mx + c \n{yn:.2f} = {m:.2f} x {xn:.0f} + {c:.2f}')
            # print(f'xs: {xs}, \nys: {ys}\n')
            if uptrend:
                ys = np.array(df.Low[indices])
                m, c, r, *_ = linregress(xs, ys)
                y0 = df.Low[x0]
            if downtrend:
                ys = np.array(df.High[indices])
                m, c, r, *_ = linregress(xs, ys)
                y0 = df.High[x0]
                            
            yn = m * xn + c
            trendlines_hl.append(((x0, df.index[xn]), (y0, yn)))
            r2 = r**2
            slope_angle = math.atan(m)  # slope angle in radians
            slope_angle_degrees = math.degrees(slope_angle)  # slope angle in degrees
            # print('HL Trendline')
            # print(linregress(xs, ys))
            # print(f'R2: {r2:.2f} \nangle_radians: {slope_angle:.2f} \nangle_deg: {slope_angle_degrees:.2f}')
            # print(f'y = mx + c \n{yn:.2f} = {m:.2f} x {xn:.0f} + {c:.2f}')
            # print(f'xs: {xs}, \nys: {ys}\n')
        
    # cols = ['isPeak', 'isValley', 'Peak', 'Valley', 'Row', 'PV Changepoint']
    cols = df.columns[:3].tolist() + ['Volume', 'Row']
    df.drop(columns=cols, inplace=True)
    
    return df, peaks, valleys, valid_PV, trendlines_c, trendlines_hl


@st.cache
def make_dataframe(ticker, period):
    if period == 'D1':
        df = get_ticker_data(ticker).copy()
    else:
        df = resample_data(ticker, period)

    return df


@st.cache
def get_trending_stocks(start, end, period, MAs):
    up = []
    down = []
    end += timedelta(1)
    
    for ticker in ticker_list:
        try:
            df = make_dataframe(ticker, period)[start:end]
            all_MAs = [df['Close'].rolling(ma).mean()[-1] for ma in MAs] # moving averages
            minor_ma, secondary_ma, primary_ma, *_ = all_MAs
            if primary_ma > secondary_ma > minor_ma:
                down.append(ticker)
            elif primary_ma < secondary_ma < minor_ma:
                up.append(ticker)
        except Exception as e:
            print(f'{period}: {ticker} - {e}')

    return up, down


@st.cache
def get_trend_aligned_stocks(periods_data, periods, end):
    for i, period in enumerate(periods):
        days = periods_data[period]['days']
        MAs = periods_data[period]['MA']
        start = end - timedelta(days)
        up, down = get_trending_stocks(start, end, period, MAs)
        if i == 0:
            up_aligned = set(up)
            down_aligned = set(down)
        else:
            up_aligned.intersection_update(up) 
            down_aligned.intersection_update(down)

    return list(up_aligned), list(down_aligned)

@st.cache
def order_peaks_valleys(df):
    '''Remove smallest peak if peaks not followed by valleys & vice versa'''
    
    P = argrelextrema(df.Close.to_numpy(), np.greater)[0].tolist() # peaks
    V = argrelextrema(df.Close.to_numpy(), np.less)[0].tolist() # valleys
    first = 'P' if P[0] < V[0] else 'V'
    lst = P if first == 'P' else V
    fn = min if first == 'P' else max
    removed_vals = []
    
    for i, (p, v) in enumerate(zip(P, V)):
        a = p if first == 'P' else v
        b = v if first == 'P' else p
        if i < len(lst) - 1:
            c = P[i + 1] if first == 'P' else V[i + 1]
            if c < b:
                val = fn((a, df.Close[a]), (c, df.Close[c]), key=itemgetter(1))[0]
                lst.remove(val)
                removed_vals.append(val)
    
    return P, V, removed_vals


@st.cache
def valid_peaks_valleys(df):
    '''
    Remove peaks/valleys with retracement < 1/3

    Parameters
    ----------
    df: DataFrame of a security's market data
    '''

    close = df.Close
    P, V, _ = order_peaks_valleys(df)
    PV = sorted(P + V)
    n = len(PV)
    a, b, c = PV[0], PV[1], PV[2]
    d1 = abs(close[b] - close[a])
    d2 = abs(close[c] - close[b])
    first = 'P' if P[0] < V[0] else 'V'
    lst = P if first == 'P' else V
    pv = {'P', 'V'}
    second = list(pv - {first})[0]
    third = first if c in lst else list(pv - {first})[0]
    valid = [(a, first), (b, second), (c, third)]
    removed_vals = []
    retracement = 1 / 3 * 0.95
    i = 2
    
    while i < n - 1:   
        x = first if c in lst else list(pv - {first})[0]
        y = list(pv - {x})[0] 
        r = d2 / d1

        if r < retracement and i > 2:
            removed_vals.append((c, x))
            rm, px = valid[-1]
            reverse = True if y == 'V' else False
            val = sorted([(b, close[b]), (d, close[d])],
                         key=itemgetter(1), reverse=reverse)
            removed, kept = val[0][0], val[1][0]
            removed_vals.append((removed, y))
            if kept == d:
                i = PV.index(kept)
                try:
                    valid.pop()
                except:
                    pass
                if i == n - 1:
                    valid.append((kept, y))
            elif kept == rm and px == y:
                i += 2
        else:
            val, px = valid[-1]
            if val != c and px != x:
                valid.append((c, x))
            i += 1    
        try:
            a, b = valid[-2][0], valid[-1][0]
            c, d = PV[i], PV[i + 1]
            d1 = abs(close[b] - close[a])
            d2 = abs(close[c] - close[b])
        except:
            try:
                a, b, c = valid[-2][0], valid[-1][0], PV[i]
                d1 = abs(close[b] - close[a])
                d2 = abs(close[c] - close[b])
                r = d2 / d1
                if r >= retracement:
                    valid.append((c,y))
            except:
                pass
        
    P = []
    V = []

    for val, x in sorted(set(valid)):
        lst = P if x == 'P' else V
        lst.append(val)

    return P, V


@st.cache
def trend_changepoints(df):
    close = df.Close
    chg = {k: {'start': [], 'end': []} for k in ['up', 'down', 'ranging']} # trend changepoints
    t = None # trend
    P, V = valid_peaks_valleys(df)
    x, _ = min((P[-1], len(P)), (V[-1], len(V)), key=itemgetter(1))
    pv = 'P' if P[0] < V[0] else 'V'
    lst = P if pv == 'P' else V
    up, down, ranging = [], [], []
    
    # Identify intermediate trend changepoints comprising 5 moves,
    # i.e., 3 trending, 2 retracements 
    for p, v in zip(P[2:], V[2:]):
        ix = p if pv == 'P' else v
        i = lst.index(ix)
        g, h = i - 2, i - 1
        a, b, c = P[g], P[h], p
        d, e, f = V[g], V[h], v
        p0, p1, p2, v0, v1, v2 = close[[a,b,c,d,e,f]]
        if p2 > p1 > p0 and v1 > v0: # Uptrend
            if t and t != 'up':
                chg[t]['end'].append(d)
                chg['up']['start'].append(d)
                up.append(d)
            t = 'up'
        elif v2 < v1 < v0 and p1 < p0: # Downtrend
            if t and t != 'down':
                chg[t]['end'].append(a)
                chg['down']['start'].append(a)
                down.append(a)
            t = 'down'
        else:
            if t in ('up', 'down'): # Ranging
                ix = [a, b] if t == 'up' else [e, d]
                ix = ix[0] if pv == 'P' else ix[1]
                chg[t]['end'].append(ix)
                chg['ranging']['start'].append(ix)
                ranging.append(ix)
            t = 'ranging'
    
    d = chg['ranging']
    if d['end'] and d['start'][0] > d['end'][0]:
        d['end'].pop(0)
    
    try:
        x = sorted(up + down + ranging)[-1]
        for k, v in chg.items():
            for k1, v1 in v.items():
                if k1 == 'start' and x in v1:
                    chg[k]['end'].append(df.shape[0] - 1)
    except: pass

    return chg, P, V


@st.cache(allow_output_mutation=True)
def plot_signals(graph, ticker, start, end, period, plot_data,
                 show_vol, show_rsi, show_macd, show_sr, show_fr, 
                 show_bb, show_MAs, show_adv_MAs, show_trend_analysis, 
                 show_trendlines_c, show_trendlines_hl):
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

    ticker = ticker.split(' - ')[0]
    end = end + timedelta(1) if period.endswith('m') else end
    print(start, end)
    df = make_dataframe(ticker, period)[start:end]
    MAs = plot_data['MAs']
    nrows = 1 + show_vol + show_rsi + show_macd
    r1 = 1 - 0.1 * nrows
    r2 = (1 - r1) / (nrows - 1)
    row_heights = [r1] + [r2] * (nrows - 1)
    fig_row = 2
    data = []
    name = ticker

    fig = make_subplots(rows=nrows, cols=1,
                        shared_xaxes=True, 
                        vertical_spacing=0.05,
                        subplot_titles=[''] * nrows, 
                        row_heights=row_heights)
    fig.update_xaxes(showgrid=True)          
    fig.update_yaxes(showgrid=False, type='log')

    if graph == 'Candlesticks' and not show_trend_analysis and not show_trendlines_c:
        cs = go.Candlestick(x=df.index, 
                            open=df['Open'], 
                            high=df['High'],
                            low=df['Low'], 
                            close=df['Close'],
                            name=name)  
        cs.increasing.fillcolor = 'green'
        cs.increasing.line.color = 'darkgreen'
        cs.decreasing.fillcolor = 'red'
        cs.decreasing.line.color = 'indianred'
        data.append(cs)
    else:
        xy = go.Scatter(x=df.index,
                        y=df['Close'],
                        name=name,
                        line_width=1.5,
                        connectgaps=True)
        data.append(xy)

    # Plot MAs & advanced MAs
    if show_MAs or show_adv_MAs:
        adv_MAs = plot_data['Adv MAs']
        colors = ['red', 'cyan', 'gold']
        dash = 'dot' if show_MAs and show_adv_MAs else 'solid'
        for ma, adv_ma, color in zip(MAs, adv_MAs, colors): 
            if show_MAs:
                y = df['Close'].rolling(ma).mean()
                sma = go.Scatter(x=df.index,
                                 y=y,
                                 name=f'MA{ma}',
                                 line_width=1,
                                 line_color=color,
                                 connectgaps=True)
                data.append(sma)
            if show_adv_MAs:
                y = df['Close'].rolling(ma).mean().shift(adv_ma)
                advanced_sma = go.Scatter(x=df.index,
                                          y=y,
                                          name=f'MA{ma}+{adv_ma}',
                                          line_width=1.25,
                                          line_color=color,
                                          line_dash=dash,
                                          connectgaps=True)
                data.append(advanced_sma)
    
    pos = [1] * len(data) # position to add rows, cols in subplot 
    fig.add_traces(data=data, rows=pos, cols=pos)  

    # Support & resistance lines
    if show_sr:
        levels, _ = sr_levels(df)
        for i, l in levels:
            n = df.shape[0] - i
            fig.add_scatter(x=df.index[i:],
                            y=[l] * n,
                            name='SR',
                            line_width=0.5,
                            line_color='orange',
                            mode='lines',
                            showlegend=False,
                            connectgaps=True)   

    # Fibonacci retracements
    if show_fr:
        colors = ["darkgray", "indianred", "green", "blue", "cyan", "magenta", "gold"]
        ratios, levels = fibonacci_retracements(df)
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

    # Trend Analysis
    if show_trend_analysis:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        P, V, r_PV = order_peaks_valleys(df)
        # keys = ['peaks', 'valleys', 'removed peaks/valleys']
        # colors = ['orange', 'red', 'red']
        # symbols = ['x', 'x', 'circle-x']
        # lws = [0.2, 0.2, 2]
        # size = [5, 5, 10]
        # pv_d = 
        pv_d = {'peaks': {'color': 'orange', 
                          'vals': P, 
                          'sy': 'x', 
                          'lw': 0.2, 
                          'size': 7},
                'valleys': {'color': 'red', 
                            'vals': V, 
                            'sy': 'x', 
                            'lw': 0.2, 
                            'size': 7},
                'removed peaks/valleys': {'color': 'red', 
                                          'vals': r_PV, 
                                          'sy': 'circle-x', 
                                          'lw': 2.5, 
                                          'size': 10}
                }

        # show all peaks/valleys & removed peaks/valleys    
        for k, v in pv_d.items():
            X = v['vals']
            fig.add_scatter(x=df.Close[X].index,
                            y=df.Close[X],
                            name=k,
                            mode='markers',
                            marker=dict(symbol=v['sy'], 
                                        line=dict(color=v['color'],
                                                  width=v['lw']),
                                        color=v['color'],
                                        size=v['size']), 
                            opacity=0.5)
        
        chg, peaks, valleys = trend_changepoints(df)
        X = sorted(peaks + valleys)

        # show valid peaks & valleys
        fig.add_scatter(x=df.Close[X].index,
                        y=df.Close[X],
                        name='valid peaks/valleys',
                        mode='markers',
                        marker=dict(symbol='circle-open', 
                                color='limegreen', 
                                size=12, 
                                line_width=2.5), 
                     
                        opacity=0.5,
                        showlegend=False)
        
        # show trend changepoints
        for k, v in chg.items():
            try:
                x = max(v['start'], v['end'])[-1]
                for x0, x1 in zip_longest(v['start'], v['end'], fillvalue=x):
                    if k == 'up':
                        txt = 'U'
                        color = 'green'
                    elif k == 'down':
                        txt = 'D'
                        color = 'red'
                    else:
                        txt = 'R'
                        color = 'violet'
                    fig.add_vrect(x0=df.index[x0], x1=df.index[x1], 
                                  line_width=0, 
                                  fillcolor=color, 
                                  opacity=0.2,
                                  annotation_text=txt, 
                                  annotation_position="top left")
            except:
                pass
                

    # Trendlines
    if show_trendlines_c or show_trendlines_hl:
        pv_df, peaks, valleys, PV, trendlines_c, trendlines_hl = peaks_valleys_trendlines(df)
        fig.add_scatter(x=df.Close[peaks].index,
                        y=df.Close[peaks],
                        name='Peaks',
                        mode='markers',
                        marker=dict(symbol='x', color='yellow', size=5))
        fig.add_scatter(x=df.Close[valleys].index,
                        y=df.Close[valleys],
                        name='Valleys',
                        mode='markers',
                        marker=dict(symbol='x', color='red', size=5))
        fig.add_scatter(x=df.Close[PV].index,
                        y=df.Close[PV],
                        name='Valid Peaks / Valleys',
                        mode='markers',
                        marker=dict(symbol='circle-open', color='limegreen', size=8))
        if show_trendlines_c:      
            for x, y in trendlines_c:
                fig.add_scatter(x=x,
                                y=y,
                                name='Trendline (Close)',
                                mode='lines',
                                line_color='magenta',
                                opacity=0.5,
                                showlegend=False)
        if show_trendlines_hl:      
            for x, y in trendlines_hl:
                fig.add_scatter(x=x,
                                y=y,
                                name='Trendline (HL)',
                                mode='lines',
                                line_color='cyan',
                                opacity=0.5,
                                showlegend=False)

    # Volume subplot
    if show_vol:
        name = 'Volume'
        fig.add_bar(x=df.index,
                    y=df[name],
                    name=name,
                    marker={'color': 'steelblue'},
                    row=fig_row, col=1)
        # fig.update_layout({f'yaxis{fig_row}': {'title': name}})
        fig_row += 1

    # RSI subplot
    if show_rsi:
        rsi = RSI(df['Close'], timeperiod=14)
        name = 'RSI'
        fig.add_scatter(x=rsi.index,
                        y=rsi.values,
                        name=name,
                        line_width=1,
                        mode='lines',
                        connectgaps=True,
                        showlegend=False,
                        row=fig_row, col=1)
        fig.add_hline(70, line_width=0.5, line_dash='dot', line_color='red', 
                      row=fig_row, col=1)
        fig.add_hline(30, line_width=0.5, line_dash='dot', line_color='red', 
                      row=fig_row, col=1)
        fig.update_layout({f'yaxis{fig_row}': {'type': 'linear', 'title': name}})
        fig_row += 1
    
    # MACD subplot
    if show_macd:
        macd, *_ = MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        name = 'MACD'
        fig.add_scatter(x=macd.index,
                        y=macd.values,
                        name=name,
                        line_width=1,
                        mode='lines',
                        connectgaps=True,
                        showlegend=False,
                        row=fig_row, col=1)
        fig.update_layout({f'yaxis{fig_row}': {'type': 'linear', 'title': name}})
    
    us_holidays = pd.to_datetime(list(holidays.US(range(start.year, end.year + 1)).keys()))
    rangebreaks = []
    rangeselector = []

    if period == 'M1':
        rangeselector = dict(buttons=[
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(count=2, label="2y", step="year", stepmode="backward"),
                                dict(count=3, label="3y", step="year", stepmode="backward"),
                                dict(step="all")
                                ])
    elif period == 'W1':
        rangeselector = dict(buttons=[
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                                ])  
    elif period == 'D1':
        rangebreaks = [dict(bounds=["sat", "mon"])]
    else:
        us_holidays += pd.offsets.Hour(9) + pd.offsets.Minute(30)
        rangebreaks = [dict(bounds=[16, 9.5], pattern="hour"), 
                       dict(bounds=["sat", "mon"])]

    if rangebreaks:
        us_holidays = pd.to_datetime(sorted(list(set(us_holidays) - set(df.index))))
        rangebreaks.append(dict(values=us_holidays))
        fig.update_xaxes(rangebreaks=rangebreaks)

    if rangeselector:
        fig.update_layout(xaxis1=dict(rangeselector=rangeselector))

    cname = SPY_info_df.loc[ticker, 'Security']
    title = f'{cname} ({ticker}) - {period}'

    fig.update_layout(title=dict(text=title, xanchor='center'))
    fig.layout.xaxis.rangeslider.visible = False

    return fig