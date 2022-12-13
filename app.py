from math import sqrt
import numpy as np
import pandas as pd
import requests
from datetime import datetime as dt

import nltk
import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import streamlit as st
import tweepy
from newspaper import Article

from config import TWITTER_USERNAMES
from info import SPY_INFO, FINANCIAL_RATIOS
from functions import (SPY_df, SPY_info_df, ticker_list, sector_list,                         
                       last_date, yr_ago, first_dates, combined_returns_df,
                       ratios_data_report, sector_weights, subIndustry_weights, 
                       ticker_weights, coming_out, getIndexOfTuple, get_ticker_data,
                       calculate_beta, get_betas, get_returns_and_volatility, 
                       get_current_ratios, make_TTM_squeeze_charts, plot_fibonacci_levels, 
                       get_news, find_stocks_missing_data, find_SMA_crossovers, 
                       make_crossover_charts)


nltk.download([
     "names",
     "stopwords",
     "averaged_perceptron_tagger",
     "vader_lexicon",
     "punkt"
      ])
      

st.title('S&P 500 Dashboards')

options = ('S&P 500 Information', 'Stock Information',
           'Stock Comparisons By Sector', 'Technical Analysis',
           'News', 'Social Media')
option = st.sidebar.selectbox("Select A Dashboard", options)

st.header(option)

if option == 'S&P 500 Information':
    st.info(SPY_INFO)
    st.subheader('Market Data')
    st.write('Select Chart Display Period')

    with st.form(key='form1'):
        start_date = st.date_input('Start Date', yr_ago, min_value=SPY_df.iloc[0].name)
        end_date = st.date_input('End Date', last_date)
        submit_button = st.form_submit_button(label='Submit')
    
    SPY_df = SPY_df[start_date: end_date]
    SPY_df['Daily Return'] = SPY_df['Close'].pct_change() * 100
    pos = SPY_df[SPY_df['Daily Return'] >= 0]
    neg = SPY_df[SPY_df['Daily Return'] < 0]
    t = len(SPY_df) / 252
    rtn = ((SPY_df.iloc[-1]['Close'] / SPY_df.iloc[0]['Open'])**(1 / t) - 1) * 100
    std = SPY_df['Daily Return'].std()
    s = f'Positive Daily Returns: {(len(pos) / len(SPY_df)) * 100:.2f}%'
    s1 = f'Negative Daily Returns: {(len(neg) / len(SPY_df)) * 100:.2f}%'
    s2 = f'CAGR is {rtn:.2f}%'
    s3 = f'Annualised Volatility is {std * sqrt(252):.2f}%'
    s4 = f'{s1} | {s}'

    # Candlestick chart
    qf = cf.QuantFig(SPY_df, title='S&P 500 Daily Value', name='SPY')
    fig = qf.iplot(asFigure=True, yTitle='Index Value')
    st.plotly_chart(fig)
    st.info(f'{s2}  \n{s3}  \n{s4}')


if option == 'Stock Information':
    with st.form(key='my_form'):
        start_date = st.date_input('Start Date', yr_ago, min_value=first_dates[0][1])
        end_date = st.date_input('End Date', last_date)
        submit_button = st.form_submit_button(label='Submit')

    tickerSymbol = st.selectbox('Stock Ticker', ticker_list)

    ticker_df = get_ticker_data(tickerSymbol)
    first_date = ticker_df.iloc[0].name.date()
    ticker_df = ticker_df[start_date: end_date]
    ticker_df.rename(columns={'volume': 'Volume'}, inplace=True)
    
    if start_date > end_date:
        st.error('*Start Date* must be before *End Date*')
    if start_date < first_date:
        st.error(f"Market data before {first_date.strftime('%B %d, %Y')} \
                    is unavailable.")
    if end_date > last_date:
        st.error(f"Market data after {last_date.strftime('%B %d, %Y')} \
                    is unavailable.")
    
    st.info(f"**{tickerSymbol}** market data is available from \
        {first_date.strftime('%B %d, %Y')} to {last_date.strftime('%B %d, %Y')}.")

    tickerData = yf.Ticker(tickerSymbol)

    if len(tickerData.info) <= 1:
        name = SPY_info_df[SPY_info_df['Symbol'] == tickerSymbol]['Security'].item()
        sector = SPY_info_df[SPY_info_df['Symbol'] == tickerSymbol]['GICS Sector'].item()
        subIndustry = SPY_info_df[SPY_info_df['Symbol'] == tickerSymbol]['GICS Sub-Industry'].item()
        location = SPY_info_df[SPY_info_df['Symbol'] == tickerSymbol]['Headquarters Location'].item()
        founded = SPY_info_df[SPY_info_df['Symbol'] == tickerSymbol]['Founded'].item()
        
        try:
            date_added = SPY_info_df[SPY_info_df['Symbol'] == tickerSymbol]['Date first added'].item()
            date_added = dt.strptime(date_added, '%Y-%m-%d')
            date_added = date_added.strftime('%B %d, %Y')
        except:
            date_added = 'N/A'
        
        website = 'N/A'
        summary = 'N/A'

    else:
        name = tickerData.info['longName']
        sector = SPY_info_df[SPY_info_df['Symbol'] == tickerSymbol]['GICS Sector'].item()
        subIndustry = SPY_info_df[SPY_info_df['Symbol'] == tickerSymbol]['GICS Sub-Industry'].item()
        city = tickerData.info['city']

        try:
            state = ', ' + tickerData.info['state'] + ', '
        except:
            state = ''  + ', '
            
        country = tickerData.info['country']
        location = city + state + country
        founded = SPY_info_df[SPY_info_df['Symbol'] == tickerSymbol]['Founded'].item()

        try:
            date_added = SPY_info_df[SPY_info_df['Symbol'] == tickerSymbol]['Date first added'].item()
            date_added = dt.strptime(date_added, '%Y-%m-%d')
            date_added = date_added.strftime('%B %d, %Y')
        except:
            date_added = 'N/A'
               
        website = tickerData.info['website']
        summary = tickerData.info['longBusinessSummary']
    
    st.header(f'**{name}**')
    st.info(f'**Sector:** {sector}  \
            \n**Sub-Industry:** {subIndustry}  \
            \n**Headquarters:** {location}  \
            \n**Founded:** {founded}  \
            \n**First Added To S&P 500:** {date_added}  \
            \n**Website:** {website}')
    st.subheader('**Company Bio**')
    st.info(summary)
    st.subheader('Market Data')
    
    # Candlestick charts
    qf = cf.QuantFig(ticker_df, title=name, name=tickerSymbol)
    qf.add_bollinger_bands(periods=20,
                           boll_std=2,
                           fill=True,
                           column='close',
                           name='Bollinger Bands')
    qf.add_volume()
    fig = qf.iplot(asFigure=True, yTitle='Price')
    st.plotly_chart(fig)

    # Histogram of daily returns
    ticker_df['Daily Return'] = ticker_df['adjclose'].pct_change() * 100
    gt0 = (len(ticker_df[ticker_df['Daily Return'] >= 0]) / (len(ticker_df) - 1)) * 100
    lt0 = (len(ticker_df[ticker_df['Daily Return'] < 0]) / (len(ticker_df) - 1)) * 100
    
    if ticker_df['Daily Return'].mean() < ticker_df['Daily Return'].median():
        pos = 'top left'
        pos1 = 'top right'
    else:
        pos = 'top right'
        pos1 = 'top left'
    
    s = f'Negative Daily Returns: {lt0:,.2f}% | Positive Daily Returns: {gt0:,.2f}%'
    
    fig = px.histogram(ticker_df,
                       x='Daily Return',
                       title='Daily Returns Distribution',
                       opacity=0.5)
    fig.add_vline(x=ticker_df['Daily Return'].mean(),
                  line_color='red',
                  line_width=0.65, 
                  annotation_text=f"Mean ({ticker_df['Daily Return'].mean():.4f}%)",
                  annotation_position=pos, 
                  annotation_bgcolor='#FF7F7F',
                  annotation_bordercolor='red')
    fig.add_vline(x=ticker_df['Daily Return'].median(),
                  line_color='green',
                  line_width=0.65, 
                  annotation_text=f"Median ({ticker_df['Daily Return'].median():.4f}%)",
                  annotation_position=pos1, 
                  annotation_bgcolor='#90ee90',
                  annotation_bordercolor='green')
    fig.update_layout(xaxis_title=s)
    st.plotly_chart(fig)

    # Comparison to S&P500 daily returns
    SPY_df = SPY_df[start_date: end_date]
    SPY_df['Daily Return'] = SPY_df['Close'].pct_change() * 100
    beta = calculate_beta(combined_returns_df, tickerSymbol, start_date, end_date)

    fig = go.Figure([
            go.Scatter(
                x=SPY_df.index,
                y=SPY_df['Daily Return'].rolling(window=20).mean(),
                name='S&P 500',
                mode='lines',
                line_width=1.25,
                line_color='red',
                showlegend=True),
            go.Scatter(
                x=ticker_df.index,
                y=ticker_df['Daily Return'].rolling(window=20).mean(),
                name=tickerSymbol,
                mode='lines',
                line_width=1.25,
                line_color='blue',
                showlegend=True)    
            ])
    fig.update_layout(title='20-Day Moving Average of Daily Returns',
                      yaxis_title='Daily Return (%)',
                      xaxis_title=f"{tickerSymbol}'s beta is {beta:,.2f}")
    st.plotly_chart(fig)
    
    # Period returns
    t = len(ticker_df) / 252
    t1 = len(SPY_df) / 252
    ticker_return = ((ticker_df.iloc[-1]['adjclose'] / ticker_df.iloc[0]['open'])**(1 / t) - 1) * 100
    SPY_return = ((SPY_df.iloc[-1]['Close'] / SPY_df.iloc[0]['Open'])**(1 / t1) - 1) * 100    
    
    # Sector returns
    sector_tickers = SPY_info_df[SPY_info_df['GICS Sector'] == sector]['Symbol'].to_list()
    sector_returns = []
    sector_ticker_weights = []
    subIndustry_ticker_weights = []

    for ticker in sector_tickers:
        df = get_ticker_data(ticker)
        df = df[start_date: end_date]
        df['Daily Return'] = df['adjclose'].pct_change() * 100
        df_return = ((1 + df['Daily Return'].mean() / 100)**(len(df) - 1) - 1) * 100
        sector_returns.append((ticker, df_return))
        sector_ticker_weights.append(ticker_weights[ticker] / sector_weights[sector])

    sReturns = [x[1] for x in sector_returns]
    sector_return = sum(np.multiply(sReturns, sector_ticker_weights))
    sector_returns = sorted(sector_returns, key=lambda x: x[1], reverse=True) 
    ticker_rank = getIndexOfTuple(sector_returns, 0, tickerSymbol)
    nsector = len(sector_tickers)
    
    # Sub-Industry returns
    subIndustry_tickers = SPY_info_df[SPY_info_df['GICS Sub-Industry'] == subIndustry] \
                          ['Symbol'].to_list()
    subIndustry_returns = []

    for ticker in subIndustry_tickers:
        df = get_ticker_data(ticker)
        df = df[start_date: end_date]
        t = len(df) / 252
        df['Daily Return'] = df['adjclose'].pct_change() * 100
        df_return = ((df.iloc[-1]['close'] / df.iloc[0]['open'])**(1 / t) - 1) * 100
        subIndustry_returns.append((ticker, df_return))
        subIndustry_ticker_weights.append(ticker_weights[ticker] / subIndustry_weights[subIndustry])

    siReturns = [x[1] for x in subIndustry_returns]
    subIndustry_return = sum(np.multiply(siReturns, subIndustry_ticker_weights))
    subIndustry_returns = sorted(subIndustry_returns, key=lambda x: x[1], reverse=True) 
    ticker_rank1 = getIndexOfTuple(subIndustry_returns, 0, tickerSymbol) 
    nsubIndustry = len(subIndustry_tickers)

    # Graph of returns
    returns = [ticker_return, sector_return, subIndustry_return, SPY_return]
    
    if SPY_return == max(returns) or SPY_return > subIndustry_return:
        pos = 'top right'
    else:
        pos = 'bottom right'

    returns_dict = {tickerSymbol: ticker_return,
                    sector: sector_return,
                    subIndustry: subIndustry_return}
    returns_df = pd.DataFrame.from_dict(returns_dict, orient='index', columns=['Return (%)'])
    
    fig = px.bar(returns_df, x=returns_df.index, y='Return (%)', opacity=0.65)
    fig.add_hline(y=SPY_return, line_color='red', line_width=0.75,
                  annotation_text=f'S&P 500 Return ({SPY_return:,.2f}%)', 
                  annotation_bgcolor='#FF7F7F', annotation_bordercolor='red',
                  annotation_position=pos)
    fig.update_layout(title='Returns Comparison', xaxis_title='')
    st.plotly_chart(fig)

    if ticker_return > SPY_return:
        s = f'{tickerSymbol} outperforms the S&P 500 by {(ticker_return - SPY_return):,.2f}%'
    elif ticker_return < SPY_return:
        s = f'{tickerSymbol} underperforms the S&P 500 by {(SPY_return - ticker_return):,.2f}%'
    else:
        s = f'{tickerSymbol} and the S&P 500 have comparable performance'
    
    if ticker_return > sector_return:
        s1 = f'{tickerSymbol} outperforms the sector by {(ticker_return - sector_return):,.2f}%'
    elif ticker_return < sector_return:
        s1 = f'{tickerSymbol} underperforms the sector by {(sector_return - ticker_return):,.2f}%'
    else:
        s1 = f'{tickerSymbol} and the sector have comparable performance.'

    if ticker_return > subIndustry_return:
        s2 = f'{tickerSymbol} outperforms the sub-industry by \
            {(ticker_return - subIndustry_return):,.2f}%'
    elif ticker_return < subIndustry_return:
        s2 = f'{tickerSymbol} underperforms the sub-industry by \
            {(subIndustry_return - ticker_return):,.2f}%'
    else:
        s2 = f'{tickerSymbol} and the sub-industry have comparable performance'    
    
    if nsubIndustry == 1:
        s3 = f'{tickerSymbol} is the only stock in the {subIndustry} sub-industry'
    else:
        s3 = f"- The capitalisation-weighted average CAGR for the {nsubIndustry} \
             stocks in the {subIndustry} sub-industry is {subIndustry_return:,.2f}% \
             \n- {s2} \
             \n- {tickerSymbol}'s performance is ranked {ticker_rank1}/{nsubIndustry} \
             in the sub-industry"

    st.write(f"Period: {start_date.strftime('%d/%m/%y')} - \
             {end_date.strftime('%d/%m/%y')}")
    st.info(f"- The {tickerSymbol} CAGR is {ticker_return:,.2f}% \
            \n- The S&P 500 CAGR is {SPY_return:,.2f}%  \
            \n- {s}")
    st.info(f"- The capitalisation-weighted average CAGR for the {nsector}\
            stocks in the {sector} sector is {sector_return:,.2f}% \
            \n- {s1} \
            \n- {tickerSymbol}'s performance is ranked {ticker_rank}/{nsector} \
            in the sector")
    st.info(f"{s3}")


if option == 'Stock Comparisons By Sector':
    # Metrics to display graph of
    metrics = ['Returns', 'Returns Volatility', 'Sharpe Ratio', 'Betas', 'Financial Ratios']
    metric = st.selectbox('Display Graphs of', metrics)

    # Additional info about metrics
    if metric == 'Returns Volatility':
        st.info('**Returns Volatility** is calculated as the standard deviation of returns.')
    
    if metric == 'Sharpe Ratio':
        st.info('''
                The **Sharpe Ratio** can be thought of as the *excess return per
                unit of risk*, where risk is the standard deviation of excess
                returns. Sharpe ratios above 1.0 are generally considered “good", 
                as this would suggest that the portfolio is offering excess returns 
                relative to its volatility. Having said that, investors will often 
                compare the Sharpe ratio of a portfolio relative to its peers. 
                Therefore, a portfolio with a Sharpe ratio of 1.0 might be considered 
                inadequate if the competitors in its peer group have an average Sharpe
                ratio above 1.0.
                ''')
    
    # Date input
    if metric != 'Financial Ratios':
        with st.form(key='my_form'):
            start_date = st.date_input("Start Date", yr_ago, min_value=first_dates[0][1])
            end_date = st.date_input("End Date", last_date)
            submit_button = st.form_submit_button(label='Submit')

        # Show date range of ticker data and raise error messages
        if start_date > end_date:
            st.error('*Start Date* must be before *End Date*')

        missing_mkt_data, missing_rpt1, missing_rpt2, missing_rpt3 = \
            find_stocks_missing_data(start_date, end_date)

        # Provide information on data that is available between chosen dates
        if missing_rpt1 != '':
            st.error(missing_rpt1)
        if missing_rpt2 != '':
            st.error(missing_rpt2)
        if missing_rpt3 != '':
            st.warning(missing_rpt3)

        if len(missing_mkt_data) > 0:
            with st.expander("Stocks Missing Data by Sector & Sub-Industry"):
                st.write(missing_mkt_data)

        sector_returns_df, subIndustry_returns_dict, ticker_cols_dict, \
        sector_vols_df, subIndustry_vols_dict, sector_sharpes_df, \
        subIndustry_sharpes_dict, SPY_return, SPY_vol, SPY_sharpe = \
            get_returns_and_volatility(start_date, end_date)
    
    if metric == 'Returns':
        # Charts of sector returns
        fig = px.bar(sector_returns_df,
                     x=sector_returns_df.index,
                     y='Return (%)',
                     opacity=0.65)
        fig.add_hline(y=SPY_return,
                      line_color='red',
                      line_width=0.75,
                      annotation_text=f'S&P 500 Return ({SPY_return:,.2f}%)', 
                      annotation_bgcolor='#FF7F7F',
                      annotation_bordercolor='red')
        fig.update_layout(title='Sector Returns', xaxis_title='')
        st.plotly_chart(fig)

    if metric == 'Returns Volatility':       
        # Charts of sector volatilities
        fig = px.bar(sector_vols_df,
                     x=sector_vols_df.index,
                     y='Volatility (%)',
                     opacity=0.65)
        fig.add_hline(y=SPY_vol, line_color='red',
                      line_width=0.75,
                      annotation_text=f'S&P 500 Returns Volatility ({SPY_vol:,.2f}%)', 
                      annotation_bgcolor='#FF7F7F',
                      annotation_bordercolor='red')
        fig.update_layout(title='Sector Returns Volatility', xaxis_title='')
        st.plotly_chart(fig)

    if metric == 'Sharpe Ratio':        
        # Charts of sector Sharpe Ratio
        fig = px.bar(sector_sharpes_df,
                     x=sector_sharpes_df.index,
                     y='Sharpe Ratio',
                     opacity=0.65)
        fig.add_hline(y=SPY_sharpe,
                      line_color='red',
                      line_width=0.75,
                      annotation_text=f'S&P 500 Sharpe Ratio ({SPY_sharpe:.4f})', 
                      annotation_bgcolor='#FF7F7F',
                      annotation_bordercolor='red')
        fig.update_layout(title='Sector Sharpe Ratio', xaxis_title='')
        st.plotly_chart(fig)

    if metric == 'Betas':
        sector_betas_df, subIndustry_betas_dict, ticker_betas_dict = get_betas(start_date, end_date)
        
        # Charts of sector betas
        fig = px.bar(sector_betas_df,
                     x=sector_betas_df.index,
                     y='Beta',
                     opacity=0.65)
        fig.add_hline(y=1,
                      line_color='red',
                      line_width=0.75,
                      annotation_text='S&P 500 Beta (1.00)', 
                      annotation_position='bottom right',
                      annotation_bgcolor='#FF7F7F',
                      annotation_bordercolor='red')
        fig.update_layout(title='Sector Betas', xaxis_title='')
        st.plotly_chart(fig)

    if metric == 'Financial Ratios':
        if ratios_data_report != "The data reported is today's.":
            st.info(ratios_data_report)
            
        categories = ['Investment Valuation Ratios',
                      'Profitability Indicator Ratios',
                      'Liquidity Measurement Ratios',
                      'Debt Ratios',
                      'Operating Performance Ratios',
                      'Cash Flow Indicator Ratios']
        col1, col2 = st.columns(2)
        category = col1.selectbox('Ratio Categories', categories)

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

        ratio = col2.selectbox('Ratio', list(ratios.keys()))
        formula = FINANCIAL_RATIOS['formulas'][category][ratio]
        definition = FINANCIAL_RATIOS['definitions'][category][ratio]
        st.write('\n')
        text = st.markdown(f'       {formula}\n{definition}')

        sector_ratios_df, subIndustry_ratios_dict, ticker_ratios_dict = get_current_ratios(ratios, ratio)
            
        # Charts of sector ratios
        fig = px.bar(sector_ratios_df, x=sector_ratios_df.index, y=ratio, opacity=0.65)
        fig.update_layout(title=f'Sector {ratio}', xaxis_title='')
        st.plotly_chart(fig)

    # Get GICS sectors
    sector = st.selectbox('GICS Sector', sector_list)

    # Show how many stocks in the sector are missing data
    if metric != 'Financial Ratios':
        sector_tickers = SPY_info_df[SPY_info_df['GICS Sector'] == sector]['Symbol'].to_list()
        
        try:
            missing = missing_mkt_data[sector]
            i = 0
            for stocks in missing.values():
                i += len(stocks)
        except:
            missing = {}
                
        if len(missing) > 0:
            s = f"{i}/{len(sector_tickers)} stocks in the {sector} sector have \
                data that begins after {start_date.strftime('%B %d, %Y')}"
            st.info(s)
            with st.expander("Sector Stocks Missing Data By Sub-Industry"):
                st.write(missing)

    # Get GICS sub-industries for chosen sector
    subIndustry_dict = SPY_info_df[SPY_info_df['GICS Sector'] == sector] \
                        ['GICS Sub-Industry'].value_counts().to_dict()
    subIndustry_list = list(subIndustry_dict.keys())

    if metric == 'Returns':
        # Make dataframe of sub-industries
        d = {si: subIndustry_returns_dict[si] for si in subIndustry_list}
        subIndustry_returns_df = pd.DataFrame.from_dict(d, orient='index', columns=['Return (%)'])     
        subIndustry_returns_df.index.names = ['Sub-Industry']
        sector_return = sector_returns_df.loc[sector].item()
        
        # Set positions of annotation texts
        if SPY_return > sector_return:
            pos = 'top right'
            pos1 = 'bottom right'
        else:
            pos = 'bottom right'
            pos1 = 'top right'

        # Charts of sub-industry returns
        fig = px.bar(subIndustry_returns_df, x=subIndustry_returns_df.index,
                     y='Return (%)', opacity=0.65)
        fig.add_hline(y=SPY_return, line_color='red', line_width=0.75,
                      annotation_text=f'S&P 500 Return ({SPY_return:,.2f}%)',
                      annotation_position=pos, annotation_bgcolor='#FF7F7F',
                      annotation_bordercolor='red')
        fig.add_hline(y=sector_return, line_color='green', line_width=0.75,
                      annotation_text=f'{sector} Return ({sector_return:,.2f}%)',
                      annotation_position=pos1, annotation_bgcolor='#90ee90', 
                      annotation_bordercolor='green')
        fig.update_layout(title=f'{sector} Sub-Industry Returns', xaxis_title='')
        st.plotly_chart(fig)

    if metric == 'Returns Volatility':
        # Make dataframe of sub-industries
        d = {si: subIndustry_vols_dict[si] for si in subIndustry_list} 
        subIndustry_vols_df = pd.DataFrame.from_dict(d, orient='index', columns=['Volatility (%)'])
        subIndustry_vols_df.index.names = ['Sub-Industry']
        sector_vol = sector_vols_df.loc[sector].item()

        # Set positions of annotation texts
        if SPY_vol > sector_vol:
            pos = 'top right'
            pos1 = 'bottom right'
        else:
            pos = 'bottom right'
            pos1 = 'top right'

        # Charts of sub-industry returns
        fig = px.bar(subIndustry_vols_df, x=subIndustry_vols_df.index,
                     y='Volatility (%)', opacity=0.65)
        fig.add_hline(y=SPY_vol, line_color='red', line_width=0.75,
                      annotation_text=f'S&P 500 Returns Volatility ({SPY_vol:,.2f}%)',
                      annotation_position=pos, annotation_bgcolor='#FF7F7F',
                      annotation_bordercolor='red')
        fig.add_hline(y=sector_vol, line_color='green', line_width=0.75,
                      annotation_text=f'{sector} Returns Volatility ({sector_vol:,.2f}%)',
                      annotation_position=pos1, annotation_bgcolor='#90ee90',
                      annotation_bordercolor='green')
        fig.update_layout(title=f'{sector} Sub-Industry Returns Volatility', xaxis_title='')
        st.plotly_chart(fig)

    if metric == 'Sharpe Ratio':
        # Make dataframe of sub-industries
        d = {si: subIndustry_sharpes_dict[si] for si in subIndustry_list}
        subIndustry_sharpes_df = pd.DataFrame.from_dict(d, orient='index', columns=['Sharpe Ratio'])    
        subIndustry_sharpes_df.index.names = ['Sub-Industry']
        sector_sharpe = sector_sharpes_df.loc[sector].item()

        # Set positions of annotation texts
        if SPY_sharpe > sector_sharpe:
            pos = 'top right'
            pos1 = 'bottom right'
        else:
            pos = 'bottom right'
            pos1 = 'top right'

        # Charts of sub-industry returns
        fig = px.bar(subIndustry_sharpes_df, x=subIndustry_sharpes_df.index,
                     y='Sharpe Ratio', opacity=0.65)
        fig.add_hline(y=SPY_sharpe, line_color='red', line_width=0.75,
                      annotation_text=f'S&P 500 Sharpe Ratio ({SPY_sharpe:.4f})',
                      annotation_position=pos, annotation_bgcolor='#FF7F7F',
                      annotation_bordercolor='red')
        fig.add_hline(y=sector_sharpe, line_color='green', line_width=0.75,
                      annotation_text=f'{sector} Sharpe Ratio ({sector_sharpe:.4f})',
                      annotation_position=pos1, annotation_bgcolor='#90ee90',
                      annotation_bordercolor='green')
        fig.update_layout(title=f'{sector} Sub-Industry Sharpe Ratio', xaxis_title='')
        st.plotly_chart(fig)

    if metric == 'Betas':
        # Make dataframe of sub-industries
        d = {si: subIndustry_betas_dict[si] for si in subIndustry_list}
        subIndustry_betas_df = pd.DataFrame.from_dict(d, orient='index', columns=['Beta'])
        subIndustry_betas_df.index.names = ['Sub-Industry']
        sector_beta = sector_betas_df.loc[sector].item()

        # Set positions of annotation texts
        if sector_beta > 1:
            pos = 'top right'
            pos1 = 'bottom right'
        else:
            pos = 'bottom right'
            pos1 = 'top right'

        fig = px.bar(subIndustry_betas_df, x=subIndustry_betas_df.index, y='Beta', opacity=0.65)
        fig.add_hline(y=1, line_color='red', line_width=0.75,
                      annotation_text=f'S&P 500 Beta (1.00)',
                      annotation_position=pos1, annotation_bgcolor='#FF7F7F',
                      annotation_bordercolor='red')
        fig.add_hline(y=sector_beta, line_color='green', line_width=0.75,
                      annotation_text=f'{sector} Beta ({sector_beta:,.2f})',
                      annotation_position=pos, annotation_bgcolor='#90ee90',
                      annotation_bordercolor='green')
        fig.update_layout(title=f'{sector} Sub-Industry Betas', xaxis_title='')
        st.plotly_chart(fig)

    if metric == 'Financial Ratios':
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

    # Select sub-industry    
    subIndustry = st.selectbox('GICS Sub-Industry', subIndustry_list)
    tickers = SPY_info_df[SPY_info_df['GICS Sub-Industry'] == subIndustry] \
              ['Symbol'].to_list()

    if metric != 'Financial Ratios':
        # Show how many stocks in the sub-industry are missing data
        try:
            missing = missing_mkt_data[sector][subIndustry]
            i = 0 
            for stock in missing:
                i += 1
        except:
            missing = {}

        if len(missing) > 0:
            s = f"{i}/{len(tickers)} stocks in the {subIndustry} sub-industry have \
                  data that begins after {start_date.strftime('%B %d, %Y')}"
            st.info(s)
            with st.expander("Sub-Industry Stocks Missing Data"):
                st.write(missing)

    if metric == 'Returns':
        subIndustry_return = subIndustry_returns_df.loc[subIndustry].item()

        d = {ticker: ticker_cols_dict[ticker] for ticker in tickers}            
        ticker_cols_df = pd.DataFrame.from_dict(d, orient='index', columns=['Return (%)', 'Company'])
        ticker_cols_df.index.names = ['Ticker']
        returns = [SPY_return, sector_return, subIndustry_return]

        # Set positions of annotation texts
        if SPY_return == min(returns):
            pos = 'bottom left'
        else:
            pos = 'top left'

        if sector_return > subIndustry_return:
            pos1 = 'top right'
            pos2 = 'bottom right'
        else:
            pos1 = 'bottom right'
            pos2 = 'top right'

        y = 'Return (%)'

        # Charts of ticker returns
        fig = px.bar(ticker_cols_df, x=ticker_cols_df.index, y=y,
                     opacity=0.65, hover_data={y:True, 'Company':True})
        fig.add_hline(y=SPY_return, line_color='red', line_width=0.75,
                      annotation_text=f'S&P 500 Return ({SPY_return:,.2f}%)',
                      annotation_position=pos, annotation_bgcolor='#FF7F7F',
                      annotation_bordercolor='red')
        fig.add_hline(y=sector_return, line_color='purple', line_width=0.75,
                      annotation_text=f'{sector} Return ({sector_return:,.2f}%)',
                      annotation_position=pos1, annotation_bgcolor='#CBC3E3',
                      annotation_bordercolor='purple')
        fig.add_hline(y=subIndustry_return, line_color='green', line_width=0.75,
                      annotation_text=f'{subIndustry} Return ({subIndustry_return:,.2f}%)',
                      annotation_position=pos2, annotation_bgcolor='#90ee90',
                      annotation_bordercolor='green')
        fig.update_layout(title=f'{subIndustry} Company Returns', xaxis_title='')
        st.plotly_chart(fig)

        # Dataframe of stocks ranked by returns
        st.subheader('Stocks Ranked By Returns')
        st.markdown('For top *n* returns enter a positive integer, \
                    and for bottom *n* returns enter a negative integer')
        n = st.text_input(label='Number of Stocks to Show', value='25')
        n = int(n)
        all_returns_df = pd.DataFrame.from_dict(ticker_cols_dict, orient='index')
        all_returns_df.sort_values('Return (%)', ascending=False, inplace=True)
        all_returns_df.reset_index(inplace=True)
        all_returns_df.rename(columns={'index': 'Ticker'}, inplace=True)
        all_returns_df.index += 1
        
        if n > 0:
            all_returns_df = all_returns_df[:n]
        else:
            all_returns_df = all_returns_df[n:]
            all_returns_df.sort_values('Return (%)', ascending=True, inplace=True)

        st.dataframe(all_returns_df)

    if metric == 'Returns Volatility':
        subIndustry_vol = subIndustry_vols_df.loc[subIndustry].item()

        # Make dataframe of tickers
        d = {ticker: ticker_cols_dict[ticker] for ticker in tickers}                  
        ticker_vols_df = pd.DataFrame.from_dict(d, orient='index', 
                                                columns=['Volatility (%)', 'Company'])
        ticker_vols_df.index.names = ['Ticker']
        vols = [SPY_vol, sector_vol, subIndustry_vol]

        # Set positions of annotation texts
        if SPY_vol == min(vols):
            pos = 'bottom left'
        else:
            pos = 'top left'

        if sector_vol > subIndustry_vol:
            pos1 = 'top right'
            pos2 = 'bottom right'
        else:
            pos1 = 'bottom right'
            pos2 = 'top right'

        y = 'Volatility (%)'

        # Charts of ticker returns
        fig = px.bar(ticker_vols_df, x=ticker_vols_df.index, y=y,
                     opacity=0.65, hover_data={y:True, 'Company':True})
        fig.add_hline(y=SPY_vol, line_color='red', line_width=0.75,
                      annotation_text=f'S&P 500 Returns Volatility ({SPY_vol:,.2f}%)',
                      annotation_position=pos, annotation_bgcolor='#FF7F7F',
                      annotation_bordercolor='red')
        fig.add_hline(y=sector_vol, line_color='purple', line_width=0.75,
                      annotation_text=f'{sector} Returns Volatility ({sector_vol:,.2f}%)',
                      annotation_position=pos1, annotation_bgcolor='#CBC3E3',
                      annotation_bordercolor='purple')
        fig.add_hline(y=subIndustry_vol, line_color='green', line_width=0.75,
                      annotation_text=f'{subIndustry} Returns Volatility ({subIndustry_vol:,.2f}%)',
                      annotation_position=pos2, annotation_bgcolor='#90ee90',
                      annotation_bordercolor='green')
        fig.update_layout(title=f'{subIndustry} Company Returns Volatility', xaxis_title='')
        st.plotly_chart(fig)

        # Dataframe of stocks ranked by returns volatility
        st.subheader('Stocks Ranked By Returns Volatility')
        st.markdown('For top *n* volatile returns enter a positive integer, \
                     and for bottom *n* volatile returns enter a negative integer')
        n = st.text_input(label='Number of Stocks to Show', value='25')
        n = int(n)
        all_vols_df = pd.DataFrame.from_dict(ticker_cols_dict, orient='index')
        all_vols_df.sort_values('Volatility (%)', ascending=False, inplace=True)
        all_vols_df.reset_index(inplace=True)
        all_vols_df.rename(columns={'index': 'Ticker'}, inplace=True)
        all_vols_df.index += 1
        
        if n > 0:
            all_vols_df = all_vols_df[:n]
        else:
            all_vols_df = all_vols_df[n:]
            all_vols_df.sort_values('Volatility (%)', ascending=True, inplace=True)

        st.dataframe(all_vols_df)

    if metric == 'Sharpe Ratio':
        subIndustry_sharpe = subIndustry_sharpes_df.loc[subIndustry].item()
        
        # Make dataframe of tickers
        d = {ticker: ticker_cols_dict[ticker] for ticker in tickers}            
        ticker_sharpes_df = pd.DataFrame.from_dict(d, orient='index', 
                                                   columns=['Sharpe Ratio', 'Company'])
        ticker_sharpes_df.index.names = ['Ticker']
        sharpes = [SPY_sharpe, sector_sharpe, subIndustry_sharpe]
        
        # Set positions of annotation texts
        if SPY_sharpe == min(sharpes):
            pos = 'bottom left'
        else:
            pos = 'top left'
        
        if sector_sharpe > subIndustry_sharpe:
            pos1 = 'top right'
            pos2 = 'bottom right'
        else:
            pos1 = 'bottom right'
            pos2 = 'top right'

        y = 'Sharpe Ratio'

        # Charts of ticker returns
        fig = px.bar(ticker_sharpes_df, x=ticker_sharpes_df.index, y=y,
                     opacity=0.65, hover_data={y:True, 'Company':True})
        fig.add_hline(y=SPY_sharpe, line_color='red', line_width=0.75,
                      annotation_text=f'S&P 500 Sharpe Ratio ({SPY_sharpe:.4f})',
                      annotation_position=pos, annotation_bgcolor='#FF7F7F',
                      annotation_bordercolor='red')
        fig.add_hline(y=sector_sharpe, line_color='purple', line_width=0.75,
                      annotation_text=f'{sector} Sharpe Ratio ({sector_sharpe:.4f})',
                      annotation_position=pos1, annotation_bgcolor='#CBC3E3',
                      annotation_bordercolor='purple')
        fig.add_hline(y=subIndustry_sharpe, line_color='green', line_width=0.75,
                      annotation_text=f'{subIndustry} Sharpe Ratio ({subIndustry_sharpe:.4f})',
                      annotation_position=pos2, annotation_bgcolor='#90ee90',
                      annotation_bordercolor='green')
        fig.update_layout(title=f'{subIndustry} Company Sharpe Ratio', xaxis_title='')
        st.plotly_chart(fig)

        # Dataframe of stocks ranked by Sharpe Ratio
        st.subheader('Stocks Ranked By Sharpe Ratio')
        st.markdown('For top *n* Sharpe Ratio enter a positive integer, \
                     and for bottom *n* Sharpe Ratio enter a negative integer')
        n = st.text_input(label='Number of Stocks to Show', value='25', key='A1')
        n = int(n)
        # all_vols_df['Return/Volatility'] = all_vols_df['Return (%)'] / all_vols_df['Volatility (%)']
        all_sharpes_df = pd.DataFrame.from_dict(ticker_cols_dict, orient='index')
        all_sharpes_df.sort_values('Sharpe Ratio', ascending=False, inplace=True)
        all_sharpes_df.reset_index(inplace=True)
        all_sharpes_df.rename(columns={'index': 'Ticker'}, inplace=True)
        all_sharpes_df.index += 1

        if n > 0:
            all_sharpes_df = all_sharpes_df[:n]
        else:
            all_sharpes_df = all_sharpes_df[n:]
            all_sharpes_df.sort_values('Sharpe Ratio', ascending=True, inplace=True)
        
        st.dataframe(all_sharpes_df)

    if metric == 'Betas':
        subIndustry_beta = subIndustry_betas_df.loc[subIndustry].item()

        # Make dataframe of tickers
        d = {ticker: ticker_betas_dict[ticker] for ticker in tickers}                   
        ticker_betas_df = pd.DataFrame.from_dict(d, orient='index', columns=['Beta', 'Company'])
        ticker_betas_df.index.names = ['Ticker']
        betas = [sector_beta, subIndustry_beta]

        # Set positions of annotation texts
        if min(betas) > 1:
            pos = 'bottom left'
        else:
            pos = 'top left'

        if sector_beta > subIndustry_beta:
            pos1 = 'top right'
            pos2 = 'bottom right'
        else:
            pos1 = 'bottom right'
            pos2 = 'top right'

        y = 'Beta'

        # Chart of ticker betas
        fig = px.bar(ticker_betas_df, x=ticker_betas_df.index, y=y,
                     opacity=0.65, hover_data={y:True, 'Company':True})
        fig.add_hline(y=1, line_color='red', line_width=0.75,
                      annotation_text=f'S&P 500 Beta (1.00)',
                      annotation_position=pos, annotation_bgcolor='#FF7F7F',
                      annotation_bordercolor='red')
        fig.add_hline(y=sector_beta, line_color='green', line_width=0.75,
                      annotation_text=f'{sector} Beta ({sector_beta:,.2f})',
                      annotation_position=pos1, annotation_bgcolor='#90ee90',
                      annotation_bordercolor='green')
        fig.add_hline(y=subIndustry_beta, line_color='purple', line_width=0.75,
                      annotation_text=f'{subIndustry} Beta ({subIndustry_beta:,.2f})',
                      annotation_position=pos2, annotation_bgcolor='#CBC3E3',
                      annotation_bordercolor='purple')
        fig.update_layout(title=f'{subIndustry} Company Betas', xaxis_title='')
        st.plotly_chart(fig)

        # Dataframe of stocks ranked by betas
        st.subheader('Stocks Ranked By Beta')
        st.markdown('For top *n* betas enter a positive integer, \
                    and for bottom *n* betas enter a negative integer')
        n = st.text_input(label='Number of Stocks to Show', value='25')
        n = int(n)

        all_betas_df = pd.DataFrame.from_dict(ticker_betas_dict, orient='index')
        all_betas_df.sort_values('Beta', ascending=False, inplace=True)
        all_betas_df.reset_index(inplace=True)
        all_betas_df.rename(columns={'index': 'Ticker'}, inplace=True)
        all_betas_df.index += 1
        
        if n > 0:
            all_betas_df = all_betas_df[:n]
        else:
            all_betas_df = all_betas_df[n:]
            all_betas_df.sort_values('Beta', ascending=True, inplace=True)

        st.dataframe(all_betas_df)

    if metric == 'Financial Ratios':
        subIndustry_ratio = subIndustry_ratios_df.loc[subIndustry].item()

        # Make dataframe of tickers
        d = {ticker: ticker_ratios_dict[ticker] for ticker in tickers}            
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

        # Dataframe of stocks ranked by ratios
        st.subheader(f'Stocks Ranked By {ratio}')
        st.markdown('For top *n* ratios enter a positive integer, \
                    and for bottom *n* ratios enter a negative integer')
        n = st.text_input(label='Number of Stocks to Show', value='25')
        n = int(n)

        all_ratios_df = pd.DataFrame.from_dict(ticker_ratios_dict, orient='index')
        all_ratios_df.sort_values(ratio, ascending=False, inplace=True)
        all_ratios_df.reset_index(inplace=True)
        all_ratios_df.rename(columns={'index': 'Ticker'}, inplace=True)
        all_ratios_df.index += 1
        
        if n > 0:
            all_ratios_df = all_ratios_df[:n]
        else:
            all_ratios_df = all_ratios_df[n:]
            all_ratios_df.sort_values(ratio, ascending=True, inplace=True)

        st.dataframe(all_ratios_df)
    

if option == 'Technical Analysis':
    # 'Bollinger Squeeze'
    indicators = ['Simple Moving Average Crossovers', 'TTM Squeeze', 'Fibonacci Retracement Levels']
    indicator = st.selectbox('Technical Indicator', indicators)
    
    if indicator == 'TTM Squeeze':           
        st.write('For more details on the TTM Squeeze visit \
                  https://school.stockcharts.com/doku.php?id=technical_indicators:ttm_squeeze')

        if dt.now() == last_date:
            date = 'today'
        else:
            date = last_date.strftime('%B %d, %Y')
    
        st.info(f'{len(coming_out)} stocks have **broken out of the squeeze** \
                  in the 5 trading days prior to {date}.')
        
        if len(coming_out) > 0:
            dates = list(set([x[1] for x in coming_out]))
            dates = sorted(dates, reverse=True)
            dates = [date.strftime('%B %d, %Y') for date in dates]

            with st.expander(f'Stocks that have broken out since {dates[-1]}'):
                st.write([x[0] for x in coming_out])
            
            b_date = st.selectbox('Breakout Date', dates)
            broke_out = []

            for item in coming_out:
                if item[1] == dt.strptime(b_date, '%B %d, %Y'):
                    broke_out.append(item)

            st.info(f'{len(broke_out)} stocks **broke out of the squeeze** on {b_date}.') 

            with st.expander(f'Stocks that broke out on {b_date}'):
                st.write([x[0] for x in broke_out])

            make_TTM_squeeze_charts(broke_out)
            
    if indicator == 'Simple Moving Average Crossovers':              
        st.write('For more details on Simple Moving Average (SMA) Crossovers \
                  visit https://www.investopedia.com/terms/c/crossover.asp')
        
        crossovers = ['5/15 Day SMA', '10/20 Day SMA', '15/25 Day SMA',
                      '20/50 Day SMA', '50/100 Day SMA', '50/200 Day SMA']
        crossover = st.selectbox('Crossovers', crossovers)

        golden, death = find_SMA_crossovers(crossover)
        signal = st.selectbox('Signal', ('Bullish', 'Bearish'))
        
        if signal == 'Bullish':
            st.info(f'{len(golden)} stocks had a *Golden Cross* in the last 5 days.')
            
            if len(golden) > 0:
                with st.expander('Golden Cross Stocks'):
                    st.write(golden)
                
                rng = [x for x in range(0, len(golden), 10)]
                
                if len(rng) > 1:
                    st.write('Graphs of Stocks[*n: n+10*]')
                    n = st.selectbox('Select n', rng)
                else:
                    n = 0
                
                make_crossover_charts(crossover, golden, n)
        
        else:
            st.info(f'{len(death)} stocks had a *Death Cross* in the last 5 days.')
            
            if len(death) > 0:
                with st.expander('Death Cross Stocks'):
                    st.write(death)
                
                rng = [x for x in range(0, len(death), 10)]
                
                if len(rng) > 1:
                    st.write('Graphs of Stocks[*n: n+10*]')
                    n = st.selectbox('Select n', rng)
                else:
                    n = 0

                make_crossover_charts(crossover, death, n)

    if indicator == 'Fibonacci Retracement Levels':
        st.write('For more details on Fibonacci Retracement Levels visit \
                  https://www.investopedia.com/terms/f/fibonacciretracement.asp')

        with st.form(key='my_form'):
            ticker = st.selectbox('Stock Ticker', ticker_list)
            start_date = st.date_input("Start Date", yr_ago, min_value=first_dates[0][1])
            end_date = st.date_input("End Date", last_date)
            submit_button = st.form_submit_button(label='Submit')
    
        plot_fibonacci_levels(ticker, start_date, end_date)


if option == 'News':
    col1, col2 = st.columns(2)
    ticker = col1.selectbox('Stock Ticker', ticker_list)
    date = col2.date_input('Date', dt.now())
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
