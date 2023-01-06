import numpy as np
import pandas as pd
import requests
from datetime import datetime as dt

import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import streamlit as st
import nltk
from newspaper import Article

from config import TWITTER_USERNAMES
from info import SPY_INFO, FINANCIAL_RATIOS
from functions import *


nltk.download([
     "names",
     "stopwords",
     "averaged_perceptron_tagger",
     "vader_lexicon",
     "punkt"
      ])
      

# st.title('S&P 500 Dashboard')

options = ('S&P 500 Information', 'Stock Information',
           'Sector Analysis', 'Technical Analysis',
           'News', 'Social Media')

option = st.sidebar.selectbox("Select Dashboard", options)
st.title(option)

if option == 'S&P 500 Information':
    st.info(SPY_INFO)
    st.subheader('Market Data')
    start_date, end_date = set_form_dates()

    df = get_SPY_data()[start_date : end_date]
    t = len(df) / 252
    cagr = ((df['Close'][-1] / df['Open'][0])**(1 / t) - 1) * 100
    std = df['Return'].std() * np.sqrt(252) * 100
    rf = rf_rates.loc[start_date : end_date, 'Close'].mean() / 100
    sr = (cagr - rf) / std

    pos = df[df['Return'] >= 0]
    neg = df[df['Return'] < 0]

    s1 = f'Positive Daily Returns: {(len(pos) / len(df)) * 100:.2f}%'
    s2 = f'Negative Daily Returns: {(len(neg) / len(df)) * 100:.2f}%'
    s3 = f'CAGR: {cagr:,.2f}%'
    s4 = f'Annualised Volatility: {std:,.2f}%'
    s5 = f'Sharpe Ratio: {sr:,.2f}' # (RF = {rf * 100:,.2f}%)
    s6 = f'{s2} | {s1}'

    # Candlestick chart
    qf = cf.QuantFig(df, title='S&P 500 Daily Values', name='SPY')
    qf.add_volume()
    fig = qf.iplot(asFigure=True, yTitle='Value')
    st.plotly_chart(fig)
    st.info(f'{s3}  \n{s4}  \n{s5}  \n{s6}')


if option == 'Stock Information':
    c1, c2 = st.columns(2)
    search = c1.radio('Search by', ('Ticker', 'Company Name'), horizontal=True)
    
    if search == 'Ticker':
        ticker = c2.selectbox(search, ticker_list)
        name = SPY_info_df[SPY_info_df['Symbol'] == ticker]['Security'].item()
    else:
        names_list = SPY_info_df['Security'].to_list()
        name = c2.selectbox(search, names_list)
        ticker = SPY_info_df[SPY_info_df['Security'] == name]['Symbol'].item()
    
    sector = SPY_info_df[SPY_info_df['Symbol'] == ticker]['GICS Sector'].item()
    subIndustry = SPY_info_df[SPY_info_df['Symbol'] == ticker]['GICS Sub-Industry'].item()
    location = SPY_info_df[SPY_info_df['Symbol'] == ticker]['Headquarters Location'].item()
    founded = SPY_info_df[SPY_info_df['Symbol'] == ticker]['Founded'].item()
    date_added = SPY_info_df[SPY_info_df['Symbol'] == ticker]['Date first added'].item()
    
    try:
        date_added = dt.strptime(date_added, '%Y-%m-%d')
        date_added = date_added.strftime('%B %d, %Y')
    except:
        date_added = 'N/A'

    try:
        tickerData = yf.Ticker(ticker)              
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

    ticker_df = get_ticker_data(ticker)
    first_date = ticker_df.iloc[0].name.date()
    
    start_date, end_date = set_form_dates() 

    ticker_df = ticker_df[start_date : end_date]
    ticker_df.rename(columns={'volume': 'Volume'}, inplace=True)
  
    if start_date > end_date:
        st.error('*Start Date* must be before *End Date*')
    if start_date < first_date:
        st.error(f"Market data before {first_date.strftime('%B %d, %Y')} is unavailable.")
    if end_date > last_date:
        st.error(f"Market data after {last_date.strftime('%B %d, %Y')} is unavailable.")

    # Candlestick charts
    qf = cf.QuantFig(ticker_df, title=f'{name} Daily Prices', name=ticker)
    qf.add_volume()
    fig = qf.iplot(asFigure=True, yTitle='Price')
    st.plotly_chart(fig)

    # Histogram of daily returns
    ticker_df['Return'] = ticker_df['adjclose'].pct_change() * 100
    gt0 = (len(ticker_df[ticker_df['Return'] >= 0]) / (len(ticker_df) - 1)) * 100
    lt0 = (len(ticker_df[ticker_df['Return'] < 0]) / (len(ticker_df) - 1)) * 100
    
    if ticker_df['Return'].mean() < ticker_df['Return'].median():
        pos1 = 'top left'
        pos2 = 'top right'
    else:
        pos1 = 'top right'
        pos2 = 'top left'
    
    s = f'Negative Daily Returns: {lt0:,.2f}% | Positive Daily Returns: {gt0:,.2f}%'
    
    fig = px.histogram(ticker_df,
                       x='Return',
                       title='Daily Returns Distribution',
                       opacity=0.5)
    fig.add_vline(x=ticker_df['Return'].mean(),
                  line_color='red',
                  line_width=0.65, 
                  annotation_text=f"Mean ({ticker_df['Return'].mean():.2f}%)",
                  annotation_position=pos1, 
                  annotation_bgcolor='#FF7F7F',
                  annotation_bordercolor='red')
    fig.add_vline(x=ticker_df['Return'].median(),
                  line_color='green',
                  line_width=0.65, 
                  annotation_text=f"Median ({ticker_df['Return'].median():.2f}%)",
                  annotation_position=pos2, 
                  annotation_bgcolor='green',
                  annotation_bordercolor='green')
    fig.update_layout(xaxis_title=s)
    st.plotly_chart(fig)

    # Comparison to S&P500 daily returns
    SPY_df = get_SPY_data()[start_date : end_date]
    SPY_df['Return'] = SPY_df['Close'].pct_change() * 100
    beta = calculate_beta(ticker, start_date, end_date)

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
                name=ticker,
                mode='lines',
                line_width=1.25,
                line_color='blue',
                showlegend=True)    
            ])
    fig.layout.yaxis.tickformat = ',.0%'
    fig.update_layout(title=f'{name} 20-Day Moving Average of Returns',
                      xaxis={'title': f"Beta = {beta:,.2f}",
                             'showgrid': False}, yaxis_title='Return')
    st.plotly_chart(fig)
    
    # Period returns
    t1 = len(ticker_df) / 252
    t2 = len(SPY_df) / 252
    ticker_return = ((ticker_df.iloc[-1]['adjclose'] / ticker_df.iloc[0]['open'])**(1 / t1) - 1) * 100
    SPY_return = ((SPY_df.iloc[-1]['Close'] / SPY_df.iloc[0]['Open'])**(1 / t2) - 1) * 100    
    
    sectors_df, subIndustries_df, tickers_df, SPY_metrics, rf = calculate_metrics(start_date, end_date)
    
    sector = tickers_df.loc[ticker, 'Sector']
    sector_return = sectors_df.loc[sector, 'Return'] * 100
    sector_returns = tickers_df[tickers_df.Sector == sector] \
                        .sort_values(by='Return', ascending=False).reset_index()
    sector_returns.index += 1
    ticker_rank1 = sector_returns[sector_returns.Ticker == ticker].index.item()
    nsector = len(sector_returns)
    
    subIndustry = tickers_df.loc[ticker, 'Sub-Industry']
    subIndustry_return = subIndustries_df.loc[subIndustry, 'Return'] * 100
    subIndustry_returns = tickers_df[tickers_df['Sub-Industry'] == subIndustry] \
                            .sort_values(by='Return', ascending=False).reset_index()
    subIndustry_returns.index += 1
    ticker_rank2 = subIndustry_returns[subIndustry_returns.Ticker == ticker].index.item()
    nsubIndustry = len(subIndustry_returns)
    

    # Graph of returns
    returns = [ticker_return, sector_return, subIndustry_return, SPY_return]
    
    if SPY_return == max(returns) or SPY_return > subIndustry_return:
        pos = 'top right'
    else:
        pos = 'bottom right'

    returns_dict = {sector: sector_return, subIndustry: subIndustry_return, ticker: ticker_return}
    returns_df = pd.DataFrame.from_dict(returns_dict, orient='index', columns=['Return(%)'])
    # returns_df.index.name = 'Item'
    
    fig = px.bar(returns_df, y='Return(%)', opacity=0.65)
    fig.add_hline(y=SPY_return, 
                  line_color='red', 
                  line_width=1,
                  annotation_text=f'S&P 500 Return ({SPY_return:,.2f}%)', 
                  annotation_bgcolor='#FF7F7F', 
                  annotation_bordercolor='red',
                  annotation_position=pos)
    fig.update_layout(title='Returns Comparison', xaxis_title='')
    st.plotly_chart(fig)

    if ticker_return > SPY_return:
        s = f'{name} outperforms the S&P 500 by {(ticker_return - SPY_return):,.2f}%.'
    elif ticker_return < SPY_return:
        s = f'{name} underperforms the S&P 500 by {(SPY_return - ticker_return):,.2f}%.'
    else:
        s = f'{name} and the S&P 500 have comparable performance.'
    
    if ticker_return > sector_return:
        s1 = f'{name} outperforms the {sector} sector by {(ticker_return - sector_return):,.2f}%.'
    elif ticker_return < sector_return:
        s1 = f'{name} underperforms the {sector} sector by {(sector_return - ticker_return):,.2f}%.'
    else:
        s1 = f'{name} and the {sector} sector have comparable performance.'

    if ticker_return > subIndustry_return:
        s2 = f'{name} outperforms the {subIndustry} sub-industry by \
            {(ticker_return - subIndustry_return):,.2f}%.'
    elif ticker_return < subIndustry_return:
        s2 = f'{name} underperforms the {subIndustry} sub-industry by \
            {(subIndustry_return - ticker_return):,.2f}%.'
    else:
        s2 = f'{name} and the {subIndustry} sub-industry have comparable performance.'    
    
    if nsubIndustry == 1:
        s3 = f'{name} is the only stock in the {subIndustry} sub-industry'
    else:
        s3 = f"- CAGR for the {nsubIndustry} \
             stocks in the {subIndustry} sub-industry is {subIndustry_return:,.2f}% \
             \n- {s2} \
             \n- {name}'s performance is ranked {ticker_rank2}/{nsubIndustry} \
             in the sub-industry"

    st.write(f"Period: {start_date.strftime('%d/%m/%y')} - {end_date.strftime('%d/%m/%y')}")
    st.info(f'- {s}')
    st.info(f"- CAGR for the {nsector}\
            stocks in the {sector} sector is {sector_return:,.2f}% \
            \n- {s1} \
            \n- {name}'s performance is ranked {ticker_rank1}/{nsector} \
            in the sector")
    st.info(f"{s3}")


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
        
        category = c1.selectbox('Categories', categories)

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

        fig = px.bar(subIndustry_ratios_df, 
                     x=subIndustry_ratios_df.index,
                     y=ratio, opacity=0.65)
        fig.add_hline(y=sector_ratio, 
                      line_color='red', 
                      line_width=1,
                      annotation_text=f'{sector} {ratio} ({sector_ratio:.2f})', 
                      annotation_bgcolor='#FF7F7F', 
                      annotation_bordercolor='red')
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
            pos1 = 'top right'
            pos2 = 'bottom right'
        else:
            pos1 = 'bottom right'
            pos2 = 'top right'

        # Chart of ticker ratios
        fig = px.bar(ticker_ratios_df, 
                     x=ticker_ratios_df.index, 
                     y=ratio,
                     opacity=0.65, 
                     hover_data={'Company':True})
        fig.add_hline(y=sector_ratio, 
                      line_color='red', 
                      line_width=1,
                      annotation_text=f'{sector} {ratio} ({sector_ratio:.2f})',
                      annotation_position=pos1, 
                      annotation_bgcolor='#FF7F7F',
                      annotation_bordercolor='red')
        fig.add_hline(y=subIndustry_ratio, 
                      line_color='green', 
                      line_width=1,
                      annotation_text=f'{subIndustry} {ratio} ({subIndustry_ratio:.2f})',
                      annotation_position=pos2,
                      annotation_bgcolor='green',
                      annotation_bordercolor='green')
        fig.update_layout(title=f'{subIndustry} Company {ratio}s', xaxis_title='')
        st.plotly_chart(fig)


if option == 'News':
    c1, c2 = st.columns(2)
    ticker = c1.selectbox('Ticker', ticker_list)
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
    # platform = st.selectbox('Platform', ('StockTwits'))
    
    # if platform == 'StockTwits':
    ticker = st.selectbox('Ticker', ticker_list)
    r = requests.get(f'https://api.stocktwits.com/api/2/streams/ticker/{ticker}.json')
    data = r.json()

    for message in data['messages']:
        st.image(message['user']['avatar_url'])
        st.info(f'''
                {message['user']['username']} 
                \n{message['created_at']} 
                \n{message['body']}
                ''')
