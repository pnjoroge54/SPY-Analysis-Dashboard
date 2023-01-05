import numpy as np
import pandas as pd
import requests
from datetime import datetime as dt

import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import streamlit as st
import tweepy
from newspaper import Article

from config import TWITTER_USERNAMES
from info import SPY_INFO, FINANCIAL_RATIOS
from functions import *


# nltk.download([
#      "names",
#      "stopwords",
#      "averaged_perceptron_tagger",
#      "vader_lexicon",
#      "punkt"
#       ])
      

# st.title('S&P 500 Dashboard')

options = ('S&P 500 Information', 'Stock Information',
           'Sector Analysis', 'Technical Analysis',
           'News', 'Social Media')

option = st.sidebar.selectbox("Select Dashboard", options)
st.title(option)

if option == 'S&P 500 Information':
    st.info(SPY_INFO)
    st.subheader('Market Data')
    st.write('Chart Display Period')
    start_date, end_date = set_form_dates()

    df = get_SPY_data()[start_date : end_date]
    pos = df[df['Return'] >= 0]
    neg = df[df['Return'] < 0]
    t = len(df) / 252
    cagr = ((df['Close'][-1] / df['Open'][0])**(1 / t) - 1) * 100
    std = df['Return'].std() * np.sqrt(252) * 100
    rf = rf_rates.loc[start_date : end_date, 'Close'].mean() / 100
    sr = (cagr - rf) / std
    s = f'Positive Daily Returns: {(len(pos) / len(df)) * 100:.2f}%'
    s1 = f'Negative Daily Returns: {(len(neg) / len(df)) * 100:.2f}%'
    s2 = f'CAGR: {cagr:,.2f}%'
    s3 = f'Annualised Volatility: {std:,.2f}%'
    s4 = f'Sharpe Ratio (RF = {rf * 100:,.2f}%): {sr:,.2f}'
    s5 = f'{s1} | {s}'

    # Candlestick chart
    qf = cf.QuantFig(df, title='S&P 500', name='SPY')
    qf.add_volume()
    fig = qf.iplot(asFigure=True, yTitle='Value')
    st.plotly_chart(fig)
    st.info(f'{s2}  \n{s3}  \n{s4}  \n{s5}')


if option == 'Stock Information':
    c1, c2 = st.columns(2)
    search = c1.radio('Search by Company', ('Ticker', 'Name'), horizontal=True)
    
    if search == 'Ticker':
        symbol = c2.selectbox('Ticker', ticker_list)
        name = SPY_info_df[SPY_info_df['Symbol'] == symbol]['Security'].item()
    else:
        names_list = SPY_info_df['Security'].to_list()
        name = c2.selectbox('Company', names_list)
        symbol = SPY_info_df[SPY_info_df['Security'] == name]['Symbol'].item()
    
    sector = SPY_info_df[SPY_info_df['Symbol'] == symbol]['GICS Sector'].item()
    subIndustry = SPY_info_df[SPY_info_df['Symbol'] == symbol]['GICS Sub-Industry'].item()
    location = SPY_info_df[SPY_info_df['Symbol'] == symbol]['Headquarters Location'].item()
    founded = SPY_info_df[SPY_info_df['Symbol'] == symbol]['Founded'].item()
    date_added = SPY_info_df[SPY_info_df['Symbol'] == symbol]['Date first added'].item()
    
    try:
        date_added = dt.strptime(date_added, '%Y-%m-%d')
        date_added = date_added.strftime('%B %d, %Y')
    except:
        date_added = 'N/A'

    try:
        tickerData = yf.Ticker(symbol)              
        website = tickerData.info['website']
        summary = tickerData.info['longBusinessSummary']
    except:
        website = 'N/A'
        summary = 'You are currently offline...'

    st.header(f'**{name}**')
    st.info(f'''
            **Sector:** {sector} \n
            **Sub-Industry:** {subIndustry} \n
            **Headquarters:** {location} \n
            **Founded:** {founded} \n
            **First Added to S&P 500:** {date_added} \n
            **Website:** {website}
            ''')
    st.subheader('**Company Bio**')
    st.info(summary)
    st.subheader('Market Data')

    ticker_df = get_ticker_data(symbol)
    first_date = ticker_df.iloc[0].name.date()
    
    with st.form(key='form1'):
        c1, c2 = st.columns(2)
        start_date = c1.date_input('Start Date', yr_ago, min_value=first_date)
        end_date = c2.date_input('End Date', last_date, max_value=last_date)
        submit_btn = c1.form_submit_button(label='Submit')  

    ticker_df = ticker_df[start_date : end_date]
    ticker_df.rename(columns={'volume': 'Volume'}, inplace=True)
  
    if start_date > end_date:
        st.error('*Start Date* must be before *End Date*')
    if start_date < first_date:
        st.error(f"Market data before {first_date.strftime('%B %d, %Y')} is unavailable.")
    if end_date > last_date:
        st.error(f"Market data after {last_date.strftime('%B %d, %Y')} is unavailable.")
    
    # Candlestick charts
    qf = cf.QuantFig(ticker_df, title=name, name=symbol)
    qf.add_bollinger_bands(periods=20,
                           boll_std=2,
                           fill=True,
                           column='close',
                           name='Bollinger Bands')
    qf.add_volume()
    fig = qf.iplot(asFigure=True, yTitle='Price')
    st.plotly_chart(fig)

    # Histogram of daily returns
    ticker_df['Return'] = ticker_df['adjclose'].pct_change() * 100
    gt0 = (len(ticker_df[ticker_df['Return'] >= 0]) / (len(ticker_df) - 1)) * 100
    lt0 = (len(ticker_df[ticker_df['Return'] < 0]) / (len(ticker_df) - 1)) * 100
    
    if ticker_df['Return'].mean() < ticker_df['Return'].median():
        pos = 'top left'
        pos1 = 'top right'
    else:
        pos = 'top right'
        pos1 = 'top left'
    
    s = f'Negative Daily Returns: {lt0:,.2f}% | Positive Daily Returns: {gt0:,.2f}%'
    
    fig = px.histogram(ticker_df,
                       x='Return',
                       title='Daily Returns Distribution',
                       opacity=0.5)
    fig.add_vline(x=ticker_df['Return'].mean(),
                  line_color='red',
                  line_width=0.65, 
                  annotation_text=f"Mean ({ticker_df['Return'].mean():.4f}%)",
                  annotation_position=pos, 
                  annotation_bgcolor='#FF7F7F',
                  annotation_bordercolor='red')
    fig.add_vline(x=ticker_df['Return'].median(),
                  line_color='green',
                  line_width=0.65, 
                  annotation_text=f"Median ({ticker_df['Return'].median():.4f}%)",
                  annotation_position=pos1, 
                  annotation_bgcolor='#90ee90',
                  annotation_bordercolor='green')
    fig.update_layout(xaxis_title=s)
    st.plotly_chart(fig)

    # Comparison to S&P500 daily returns
    SPY_df = get_SPY_data()[start_date : end_date]
    SPY_df['Return'] = SPY_df['Close'].pct_change() * 100
    beta = calculate_beta(symbol, start_date, end_date)

    fig = go.Figure([
            go.Scatter(
                x=SPY_df.index,
                y=SPY_df['Return'].rolling(window=20).mean(),
                name='S&P 500',
                mode='lines',
                line_width=1.25,
                line_color='red',
                showlegend=True),
            go.Scatter(
                x=ticker_df.index,
                y=ticker_df['Return'].rolling(window=20).mean(),
                name=symbol,
                mode='lines',
                line_width=1.25,
                line_color='blue',
                showlegend=True)    
            ])
    fig.update_layout(title='20-Day Moving Average of Returns',
                      yaxis_title='Return (%)',
                      xaxis_title=f"{symbol}'s beta is {beta:,.2f}")
    st.plotly_chart(fig)
    
    # Period returns
    t = len(ticker_df) / 252
    t1 = len(SPY_df) / 252
    ticker_return = ((ticker_df.iloc[-1]['adjclose'] / ticker_df.iloc[0]['open'])**(1 / t) - 1) * 100
    SPY_return = ((SPY_df.iloc[-1]['Close'] / SPY_df.iloc[0]['Open'])**(1 / t1) - 1) * 100    
    
# CREATE A LINK THAT GOES TO STOCK COMPARISON BY SECTOR & THEN DELETE BELOW CODE
    # Sector returns
    # sector_tickers = SPY_info_df[SPY_info_df['GICS Sector'] == sector]['Symbol'].to_list()
    # sector_returns = []
    # sector_ticker_weights = []
    # subIndustry_ticker_weights = []

    # for ticker in sector_tickers:
    #     df = get_ticker_data(ticker)
    #     df = df[start_date : end_date]
    #     df['Daily Return'] = df['adjclose'].pct_change() * 100
    #     df_return = ((1 + df['Daily Return'].mean() / 100)**(len(df) - 1) - 1) * 100
    #     sector_returns.append((ticker, df_return))
    #     sector_ticker_weights.append(ticker_weights[ticker] / sector_weights[sector])

    # sReturns = [x[1] for x in sector_returns]
    # sector_return = sum(np.multiply(sReturns, sector_ticker_weights))
    # sector_returns = sorted(sector_returns, key=lambda x: x[1], reverse=True) 
    # ticker_rank = getIndexOfTuple(sector_returns, 0, symbol)
    # nsector = len(sector_tickers)
    
    # # Sub-Industry returns
    # subIndustry_tickers = SPY_info_df[SPY_info_df['GICS Sub-Industry'] == subIndustry] \
    #                       ['Symbol'].to_list()
    # subIndustry_returns = []

    # for ticker in subIndustry_tickers:
    #     df = get_ticker_data(ticker)
    #     df = df[start_date : end_date]
    #     t = len(df) / 252
    #     df['Daily Return'] = df['adjclose'].pct_change() * 100
    #     df_return = ((df.iloc[-1]['close'] / df.iloc[0]['open'])**(1 / t) - 1) * 100
    #     subIndustry_returns.append((ticker, df_return))
    #     subIndustry_ticker_weights.append(ticker_weights[ticker] / subIndustry_weights[subIndustry])

    # si_returns = [x[1] for x in subIndustry_returns]
    # subIndustry_return = sum(np.multiply(si_returns, subIndustry_ticker_weights))
    # subIndustry_returns = sorted(subIndustry_returns, key=lambda x: x[1], reverse=True) 
    # ticker_rank1 = getIndexOfTuple(subIndustry_returns, 0, symbol) 
    # nsubIndustry = len(subIndustry_tickers)

    # # Graph of returns
    # returns = [ticker_return, sector_return, subIndustry_return, SPY_return]
    
    # if SPY_return == max(returns) or SPY_return > subIndustry_return:
    #     pos = 'top right'
    # else:
    #     pos = 'bottom right'

    # returns_dict = {symbol: ticker_return,
    #                 sector: sector_return,
    #                 subIndustry: subIndustry_return}
    # returns_df = pd.DataFrame.from_dict(returns_dict, orient='index', columns=['Return (%)']).round(2)
    
    # fig = px.bar(returns_df, x=returns_df.index, y='Return (%)', opacity=0.65)
    # fig.add_hline(y=SPY_return, line_color='red', line_width=0.75,
    #               annotation_text=f'S&P 500 Return ({SPY_return:,.2f}%)', 
    #               annotation_bgcolor='#FF7F7F', annotation_bordercolor='red',
    #               annotation_position=pos)
    # fig.update_layout(title='Returns Comparison', xaxis_title='')
    # st.plotly_chart(fig)

    if ticker_return > SPY_return:
        s = f'{symbol} outperforms the S&P 500 by {(ticker_return - SPY_return):,.2f}%'
    elif ticker_return < SPY_return:
        s = f'{symbol} underperforms the S&P 500 by {(SPY_return - ticker_return):,.2f}%'
    else:
        s = f'{symbol} and the S&P 500 have comparable performance'
    
    # if ticker_return > sector_return:
    #     s1 = f'{symbol} outperforms the sector by {(ticker_return - sector_return):,.2f}%'
    # elif ticker_return < sector_return:
    #     s1 = f'{symbol} underperforms the sector by {(sector_return - ticker_return):,.2f}%'
    # else:
    #     s1 = f'{symbol} and the sector have comparable performance.'

    # if ticker_return > subIndustry_return:
    #     s2 = f'{symbol} outperforms the sub-industry by \
    #         {(ticker_return - subIndustry_return):,.2f}%'
    # elif ticker_return < subIndustry_return:
    #     s2 = f'{symbol} underperforms the sub-industry by \
    #         {(subIndustry_return - ticker_return):,.2f}%'
    # else:
    #     s2 = f'{symbol} and the sub-industry have comparable performance'    
    
    # if nsubIndustry == 1:
    #     s3 = f'{symbol} is the only stock in the {subIndustry} sub-industry'
    # else:
    #     s3 = f"- The capitalisation-weighted average CAGR for the {nsubIndustry} \
    #          stocks in the {subIndustry} sub-industry is {subIndustry_return:,.2f}% \
    #          \n- {s2} \
    #          \n- {symbol}'s performance is ranked {ticker_rank1}/{nsubIndustry} \
    #          in the sub-industry"

    # st.write(f"Period: {start_date.strftime('%d/%m/%y')} - \
    #          {end_date.strftime('%d/%m/%y')}")
    # st.info(f"- The {symbol} CAGR is {ticker_return:,.2f}% \
    #         \n- The S&P 500 CAGR is {SPY_return:,.2f}%  \
    #         \n- {s}")
    # st.info(f"- The capitalisation-weighted average CAGR for the {nsector}\
    #         stocks in the {sector} sector is {sector_return:,.2f}% \
    #         \n- {s1} \
    #         \n- {symbol}'s performance is ranked {ticker_rank}/{nsector} \
    #         in the sector")
    # st.info(f"{s3}")


if option == 'Sector Analysis':
    # Metrics to display graphs of
    metrics = ['Return', 'Volatility', 'Sharpe Ratio', 'Beta', 'Financial Ratio']
    metric = st.selectbox('Metric', metrics)
    
    #------------------SECTORS--------------------
    # Date input
    if metric != 'Financial Ratio':
        start, end = set_form_dates()

        sectors_df, subIndustries_df, tickers_df, SPY_metrics, rf = calculate_metrics(start, end)

        # Show date range of ticker data and raise error messages
        if start > end:
            st.error('*Start Date* must be before *End Date*')

        missing_mkt_data, missing_rpt1, missing_rpt2, missing_rpt3 = \
            find_stocks_missing_data(start, end)

        # Provide information on data that is available between chosen dates
        if missing_rpt1 != '':
            st.error(missing_rpt1)
        if missing_rpt2 != '':
            st.error(missing_rpt2)
        if len(missing_mkt_data) > 0:
            with st.expander("Stocks Missing Data"):
                st.warning(missing_rpt3)
                st.write(missing_mkt_data)

        fig = make_sector_chart(sectors_df, SPY_metrics, metric)
        st.plotly_chart(fig)

        #------------------SUB-INDUSTRIES--------------------
        sector = st.selectbox('GICS Sector', sector_list)

        sector_tickers = SPY_info_df[SPY_info_df['GICS Sector'] == sector]['Symbol'].to_list()
        subIndustry_list = subIndustries_df[subIndustries_df.Sector == sector].index.to_list()
        missing = missing_mkt_data.get(sector, {})
        n_missing = len(missing)  

        if n_missing > 0:
            s = f"{n_missing}/{len(sector_tickers)} stocks in the {sector} sector have \
                data that begins after {start_date.strftime('%B %d, %Y')}"
            with st.expander("Stocks Missing Data by Sub-Industry"):
                st.info(s)
                st.write(missing)

        fig = make_subIndustry_chart(sector, sectors_df, subIndustries_df, SPY_metrics, metric)
        st.plotly_chart(fig)

        #------------------COMPANIES--------------------
        subIndustry = st.selectbox('GICS Sub-Industry', subIndustry_list)
        si_tickers = SPY_info_df[SPY_info_df['GICS Sub-Industry'] == subIndustry] \
                        ['Symbol'].to_list()

        missing = missing.get(subIndustry, [])
        n_missing = len(missing)  

        if n_missing > 0:
            s = f"{n_missing}/{len(si_tickers)} stocks in the {subIndustry} sub-industry have \
                  data that begins after {start_date.strftime('%B %d, %Y')}"
            with st.expander("Sub-Industry Stocks Missing Data"):
                st.info(s)
                st.write(missing)

        fig = make_subIndustry_tickers_chart(sector, subIndustry, sectors_df, subIndustries_df,
                                             tickers_df, SPY_metrics, metric)
        st.plotly_chart(fig)

    if metric == 'Financial Ratio':
        c1, c2 = st.columns(2)

        if ratios_data_report != "The data reported is today's.":
            st.info(ratios_data_report)
            
        categories = ['Investment Valuation Ratios',
                      'Profitability Indicator Ratios',
                      'Liquidity Measurement Ratios',
                      'Debt Ratios',
                      'Operating Performance Ratios',
                      'Cash Flow Indicator Ratios']
        
        category = c1.selectbox('Ratio Categories', categories)

        if category == 'Investment Valuation Ratios':
            ratios = {
                'Price to Earnings Ratio': 'priceEarningsRatioTTM', 
                'Price to Book Value Ratio': 'priceToBookRatioTTM', 
                'Price to Sales Ratio': 'priceToSalesRatioTTM', 
                'Price to Earnings to Growth Ratio': 'priceEarningsToGrowthRatioTTM',
                'Price to Free Cash Flows Ratio': 'priceToFreeCashFlowsRatioTTM', 
                'Enterprise Value Multiplier': 'enterpriseValueMultipleTTM', 
                'Dividend Yield': 'dividendYieldTTM'
                }

        if category == 'Profitability Indicator Ratios':
            ratios = {
                'Gross Profit Margin': 'grossProfitMarginTTM',
                'Net Profit Margin': 'netProfitMarginTTM',
                'Operating Profit Margin': 'operatingProfitMarginTTM',
                'Pre-Tax Profit Margin': 'pretaxProfitMarginTTM',
                'Effective Tax Rate': 'effectiveTaxRateTTM',
                'Return On Assets': 'returnOnAssetsTTM',
                'Return On Equity': 'returnOnEquityTTM',
                'Return On Capital Employed': 'returnOnCapitalEmployedTTM'
            }

        if category == 'Liquidity Measurement Ratios':
            ratios = {
                'Current Ratio': 'currentRatioTTM',
                'Quick Ratio': 'quickRatioTTM',
                'Cash Ratio': 'cashRatioTTM',
                'Days Of Sales Outstanding': 'daysOfSalesOutstandingTTM',
                'Days Of Inventory Outstanding': 'daysOfInventoryOutstandingTTM',
                'Operating Cycle': 'operatingCycleTTM',
                'Days Of Payables Outstanding': 'daysOfPayablesOutstandingTTM',
                'Cash Conversion Cycle': 'cashConversionCycleTTM'
            }

        if category == 'Debt Ratios':
            ratios = {
                'Debt Ratio': 'debtRatioTTM',
                'Debt to Equity Ratio': 'debtEquityRatioTTM',
                'Long-Term Debt to Capitalisation': 'longTermDebtToCapitalizationTTM',
                'Total Debt to Capitalisation': 'totalDebtToCapitalizationTTM',
                'Interest Coverage Ratio': 'interestCoverageTTM',
                'Cash Flow to Debt Ratio': 'cashFlowToDebtRatioTTM',
                'Company Equity Multiplier': 'companyEquityMultiplierTTM'
            }

        if category == 'Operating Performance Ratios':
            ratios = {
                'Asset Turnover': 'assetTurnoverTTM',
                'Fixed Asset Turnover': 'fixedAssetTurnoverTTM',
                'Inventory Turnover': 'inventoryTurnoverTTM',
                'Receivables Turnover': 'receivablesTurnoverTTM',
                'Payables Turnover': 'payablesTurnoverTTM'
            }

        if category == 'Cash Flow Indicator Ratios':
            ratios = {
                'Operating Cash Flow per Share': 'operatingCashFlowPerShareTTM',
                'Free Cash Flow per Share': 'freeCashFlowPerShareTTM',
                'Cash per Share': 'cashPerShareTTM',
                'Operating Cash Flow to Sales Ratio': 'operatingCashFlowSalesRatioTTM',
                'Free Cash Flow to Operating Cash Flow Ratio': 'freeCashFlowOperatingCashFlowRatioTTM',
                'Cash Flow Coverage Ratio': 'cashFlowCoverageRatiosTTM',
                'Short-Term Coverage Ratio': 'shortTermCoverageRatiosTTM',
                'Capex Coverage Ratio': 'capitalExpenditureCoverageRatioTTM',
                'Dividend Paid & Capex Coverage Ratio': 'dividendPaidAndCapexCoverageRatioTTM',
                'Dividend Payout Ratio': 'payoutRatioTTM'
            }

        ratio = c2.selectbox('TTM Ratio', list(ratios.keys()))
        formula = FINANCIAL_RATIOS['formulas'][category][ratio]
        definition = FINANCIAL_RATIOS['definitions'][category][ratio]

        with st.expander('Definition'):
            st.write('\n')
            text = st.markdown(f'       {formula}\n{definition}')

        sector_ratios_df, subIndustry_ratios_dict, ticker_ratios_dict = get_current_ratios(ratios, ratio)
            
        # Charts of sector ratios
        fig = px.bar(sector_ratios_df, x=sector_ratios_df.index, y=ratio, opacity=0.65)
        fig.update_layout(title=f'Sector {ratio}', xaxis_title='')
        st.plotly_chart(fig)

        sector = st.selectbox('GICS Sector', sector_list)
        subIndustry_list = SPY_info_df[SPY_info_df['GICS Sector'] == sector]['GICS Sub-Industry'].to_list()

        # Make dataframe of sub-industries
        d = {si: subIndustry_ratios_dict[si] for si in subIndustry_list}
        subIndustry_ratios_df = pd.DataFrame.from_dict(d, orient='index', columns=[ratio])
        subIndustry_ratios_df.index.names = ['Sub-Industry']
        sector_ratio = sector_ratios_df.loc[sector].item()

        fig = px.bar(subIndustry_ratios_df, x=subIndustry_ratios_df.index,
                     y=ratio, opacity=0.65)
        fig.add_hline(y=sector_ratio, line_color='red', line_width=0.75,
                      annotation_text=f'{sector} {ratio} ({sector_ratio:.4f})', 
                      annotation_bgcolor='#FF7F7F', annotation_bordercolor='red')
        fig.update_layout(title=f'{sector} Sub-Industry {ratio}s', xaxis_title='')
        st.plotly_chart(fig)

        subIndustry = st.selectbox('GICS Sub-Industry', subIndustry_list)
        subIndustry_ratio = subIndustry_ratios_df.loc[subIndustry].item()

        # Make dataframe of tickers
        si_tickers = SPY_info_df[SPY_info_df['GICS Sub-Industry'] == subIndustry]['Symbol'].to_list()
        d = {ticker: ticker_ratios_dict[ticker] for ticker in si_tickers}            
        ticker_ratios_df = pd.DataFrame.from_dict(d, orient='index', columns=[ratio, 'Company'])
        ticker_ratios_df.index.names = ['Ticker']

        # Set positions of annotation texts
        if sector_ratio > subIndustry_ratio:
            pos = 'top right'
            pos1 = 'bottom right'
        else:
            pos = 'bottom right'
            pos1 = 'top right'

        y = ratio

        # Chart of ticker ratios
        fig = px.bar(ticker_ratios_df, x=ticker_ratios_df.index, y=y,
                     opacity=0.65, hover_data={y:True, 'Company':True})
        fig.add_hline(y=sector_ratio, line_color='red', line_width=0.75,
                      annotation_text=f'{sector} {ratio} ({sector_ratio:.4f})',
                      annotation_position=pos, annotation_bgcolor='#FF7F7F',
                      annotation_bordercolor='red')
        fig.add_hline(y=subIndustry_ratio, line_color='green', line_width=0.75,
                      annotation_text=f'{subIndustry} {ratio} ({subIndustry_ratio:.4f})',
                      annotation_position=pos1, annotation_bgcolor='#90ee90',
                      annotation_bordercolor='green')
        fig.update_layout(title=f'{subIndustry} Company {ratio}s', xaxis_title='')
        st.plotly_chart(fig)


if option == 'Technical Analysis':
    # 'Bollinger Squeeze'
    indicators = ['Simple Moving Average Crossovers', 'TTM Squeeze', 'Fibonacci Retracement Levels']
    indicator = st.selectbox('Technical Indicator', indicators)
    
    if indicator == 'TTM Squeeze':
        with st.expander('Definition'):         
            st.write('For more details on the TTM Squeeze visit \
                    https://school.stockcharts.com/doku.php?id=technical_indicators:ttm_squeeze')

        if dt.now() == last_date:
            date = 'today'
        else:
            date = last_date.strftime('%B %d, %Y')
    
        # st.info(f'{len(coming_out)} stocks have **broken out of the squeeze** \
        #           in the 5 trading days prior to {date}.')
        
        if len(coming_out) > 0:
            dates = list(set([x[1] for x in coming_out]))
            dates = sorted(dates, reverse=True)
            dates = [date.strftime('%B %d, %Y') for date in dates]

            # with st.expander(f'Stocks that have broken out since {dates[-1]}'):
            #     st.write([x[0] for x in coming_out])
            
            c1, c2 = st.columns(2)
            b_date = c1.selectbox('Breakout Date', dates)
            broke_out = {}

            for item in coming_out:
                if item[1] == dt.strptime(b_date, '%B %d, %Y'):
                    broke_out[item[0]] = item[1]

            ticker = c2.selectbox('Ticker', broke_out.keys())

            # st.info(f'{len(broke_out)} stocks **broke out of the squeeze** on {b_date}.') 

            # with st.expander(f'Stocks that broke out on {b_date}'):
            #     st.write([x[0] for x in broke_out])

            make_TTM_squeeze_charts(ticker, broke_out[ticker])
            
    if indicator == 'Simple Moving Average Crossovers':    
        with st.expander('Definition'):             
            st.write('For more details on Simple Moving Average (SMA) Crossovers \
                    visit https://www.investopedia.com/terms/c/crossover.asp')
        
        crossovers = ['5/15 Day SMA', '10/20 Day SMA', '15/25 Day SMA',
                      '20/50 Day SMA', '50/100 Day SMA', '50/200 Day SMA']
        c1, c2 = st.columns(2)
        crossover = c1.selectbox('Crossovers', crossovers)
        signal = c2.selectbox('Signal', ('Bullish', 'Bearish'))
        golden, death = find_SMA_crossovers(crossover)
        
        if signal == 'Bullish':
            st.info(f'{len(golden)} stocks had a *Golden Cross* in the last 5 days.')

            if len(golden) > 0:
                ticker = st.selectbox('Ticker', golden)
                make_crossover_charts(crossover, ticker)
        
        else:
            st.info(f'{len(death)} stocks had a *Death Cross* in the last 5 days.')
            
            if len(death) > 0:
                ticker = st.selectbox('Ticker', death)
                make_crossover_charts(crossover, ticker)

    if indicator == 'Fibonacci Retracement Levels':
        with st.expander('Definition'):   
            st.write('For more details on Fibonacci Retracement Levels visit \
                    https://www.investopedia.com/terms/f/fibonacciretracement.asp')

        ticker = st.selectbox('Ticker', ticker_list)
        ticker_df = get_ticker_data(ticker)
        first_date = ticker_df.iloc[0].name.date()

        with st.form(key='form2'):
            c1, c2 = st.columns(2)
            start_date = c1.date_input("Start Date", yr_ago, min_value=first_date)
            end_date = c2.date_input("End Date", last_date)
            submit_btn = c1.form_submit_button(label='Submit')
    
        plot_fibonacci_levels(ticker, start_date, end_date)


if option == 'News':
    c1, c2 = st.columns(2)
    ticker = c1.selectbox('Stock Ticker', ticker_list)
    date = c2.date_input('Date', dt.now())
    date = date.strftime('%Y-%m-%d')
    
    try:
        name = yf.Ticker(ticker).info['longName']
    except:
        name = SPY_info_df[SPY_info_df['Symbol'] == ticker]['Security'].item()
    
    news = get_news(ticker, date)
    
    if len(news) == 0:
        st.write(f'There are no stories about {name}.')
    elif len(news) == 1:
        st.write(f'There is {len(news)} story about {name}.')
    else:    
        st.write(f'There are {len(news)} stories about {name}.')

    for story in news:
        headline = story['headline']
        source = story['source']
        url = story['url']
        # Convert timestamp to datetime and get string of hours & min
        published = dt.fromtimestamp(story['datetime']).strftime('%d %b, %Y %I:%M%p')
        
        # Get a summary of the article
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            summary = article.summary
            err = 'Javascript is DisabledYour current browser configurationis not \
                   compatible with this site.'           
            if summary == err:
                summary = story['summary']
        except:
            summary = story['summary']
    
        st.info(f'''
                ### {headline}
                \n\n##### Summary:
                \n\n{summary}
                \n_**Source:** {source}_  
                _**Published:** {published}_ 
                \n_**Full story:** {url}_
                ''')
        

if option == 'Social Media':
    platform = st.selectbox('Platform', ('Twitter', 'StockTwits'))
    
    if platform == 'Twitter':
        # Twitter API settings
        auth = tweepy.OAuthHandler(st.secrets['TWITTER_CONSUMER_KEY'], 
                                   st.secrets['TWITTER_CONSUMER_SECRET'])
        auth.set_access_token(st.secrets['TWITTER_ACCESS_TOKEN'], 
                              st.secrets['TWITTER_ACCESS_TOKEN_SECRET'])
        api = tweepy.API(auth)

        for username in TWITTER_USERNAMES:
            user = api.get_user(screen_name=username)
            tweets = api.user_timeline(screen_name=username)
            st.subheader(username)
            st.image(user.profile_image_url)      
            for tweet in tweets:
                if '$' in tweet.text:
                    words = tweet.text.split(' ')
                    for word in words:
                        if word.startswith('$') and word[1:].isalpha():
                            symbol = word[1:]
                            st.info(tweet.text)
                            st.image(f'https://finviz.com/chart.ashx?t={symbol}')

    if platform == 'StockTwits':
        symbol = st.selectbox('Stock Ticker', ticker_list)
        r = requests.get(f'https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json')
        data = r.json()

        for message in data['messages']:
            st.image(message['user']['avatar_url'])
            st.info(f'''
                    {message['user']['username']} 
                    \n{message['created_at']} 
                    \n{message['body']}
                    ''')
