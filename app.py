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

from info import SPY_INFO, FINANCIAL_RATIOS
from functions import *


# nltk.download([
#      "names",
#      "stopwords",
#      "averaged_perceptron_tagger",
#      "vader_lexicon",
#      "punkt"
#       ])
      

options = ('S&P 500 Information', 'Stock Information', 'Stock Analysis',
           'Sector Analysis', 'News', 'Social Media')

option = st.sidebar.selectbox("Select Dashboard", options)
st.title(option)

if option == 'S&P 500 Information':
    st.info(SPY_INFO)
    st.subheader('Market Data')
    start, end = set_form_dates()

    df = get_SPY_data()[start : end]
    t = len(df) / 252
    cagr = ((df['Close'][-1] / df['Open'][0])**(1 / t) - 1)
    std = df['Return'].std() * np.sqrt(252)
    rf = rf_rates.loc[start : end, 'Close'].mean() / 100
    sr = (cagr - rf) / std

    s1 = f'CAGR: {cagr:.2%}'
    s2 = f'Annualised Volatility: {std:.2%}'
    s3 = f'Risk-Free Rate: {rf:.2%}'
    s4 = f'Sharpe Ratio: {sr:,.2f}' 
    
    # Candlestick chart
    qf = cf.QuantFig(df, title='S&P 500 Daily Values', name='SPY')
    qf.add_volume()
    fig = qf.iplot(asFigure=True, yTitle='Value')
    st.plotly_chart(fig)
    
    fig = make_returns_histogram(df)
    st.plotly_chart(fig)

    st.info(f'{s1}  \n{s2}  \n{s3}  \n{s4}')


if option == 'Stock Information':
    c1, c2 = st.columns(2)
    search = c1.radio('Search by', ('Ticker', 'Company Name'), horizontal=True)
    
    if search == 'Ticker':
        ticker = c2.selectbox(search, ticker_list)
        name = SPY_info_df.loc[ticker, 'Security']
    else:
        names_list = SPY_info_df['Security'].to_list()
        name = c2.selectbox(search, names_list)
        ticker = SPY_info_df[SPY_info_df['Security'] == name].index.name
    
    sector = SPY_info_df.loc[ticker, 'Sector']
    subIndustry = SPY_info_df.loc[ticker, 'Sub-Industry']
    location = SPY_info_df.loc[ticker, 'Headquarters Location']
    founded = SPY_info_df.loc[ticker, 'Founded']
    date_added = SPY_info_df.loc[ticker, 'Date first added']
    
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
    
    start, end = set_form_dates() 

    ticker_df = ticker_df[start : end]
    ticker_df.rename(columns={'volume': 'Volume'}, inplace=True)
  
    if start > end:
        st.error('*Start Date* must be before *End Date*')
    if start < first_date:
        st.error(f"Market data before {first_date.strftime('%B %d, %Y')} is unavailable.")
    if end > last_date:
        st.error(f"Market data after {last_date.strftime('%B %d, %Y')} is unavailable.")

    # Candlestick chart
    qf = cf.QuantFig(ticker_df, title=f'{name} Daily Prices', name=ticker)
    qf.add_volume()
    fig = qf.iplot(asFigure=True, yTitle='Price')
    st.plotly_chart(fig)

    fig = make_returns_histogram(ticker_df)
    st.plotly_chart(fig)

    SPY_df = get_SPY_data()[start : end]
    SPY_df['Return'] = np.log1p(SPY_df['Close'].pct_change())
    beta = calculate_beta(ticker, start, end)

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
    fig.layout.yaxis.tickformat = ',.2%'
    fig.update_layout(title=f'20-Day Moving Average of Returns',
                      xaxis=dict(title=f'Beta = {beta:,.2f}', showgrid=False), 
                      yaxis_title='Return')
    st.plotly_chart(fig)
    
    t1 = len(ticker_df) / 252
    t2 = len(SPY_df) / 252
    ticker_return = ((ticker_df['adjclose'][-1] / ticker_df['open'][0])**(1 / t1) - 1)
    SPY_return = ((SPY_df['Close'][-1] / SPY_df['Open'][0])**(1 / t2) - 1)    
    
    sectors_df, subIndustries_df, tickers_df, SPY_metrics, rf = calculate_metrics(start, end)
    
    sector = tickers_df.loc[ticker, 'Sector']
    sector_return = sectors_df.loc[sector, 'Return']
    sector_returns = tickers_df[tickers_df.Sector == sector] \
                        .sort_values(by='Return', ascending=False).reset_index()
    ticker_rank1 = sector_returns[sector_returns.Ticker == ticker].index.item() + 1
    nsector = len(sector_returns)
    
    subIndustry = tickers_df.loc[ticker, 'Sub-Industry']
    subIndustry_return = subIndustries_df.loc[subIndustry, 'Return']
    subIndustry_returns = tickers_df[tickers_df['Sub-Industry'] == subIndustry] \
                            .sort_values(by='Return', ascending=False).reset_index()
    ticker_rank2 = subIndustry_returns[subIndustry_returns.Ticker == ticker].index.item() + 1
    nsubIndustry = len(subIndustry_returns)
    
    returns = [ticker_return, sector_return, subIndustry_return, SPY_return]
    
    if SPY_return == max(returns) or SPY_return > subIndustry_return:
        pos = 'top right'
    else:
        pos = 'bottom right'

    returns_dict = {sector: sector_return, subIndustry: subIndustry_return, ticker: ticker_return}
    returns_df = pd.DataFrame.from_dict(returns_dict, orient='index', columns=['Return'])
    
    fig = px.bar(returns_df, y='Return', opacity=0.65)
    fig.add_hline(y=SPY_return, 
                  line_color='red', 
                  line_width=1,
                  annotation_text=f'S&P 500 Return ({SPY_return:.2%})', 
                  annotation_bgcolor='indianred', 
                  annotation_bordercolor='red',
                  annotation_position=pos)
    fig.update_layout(title='Sector, Sub-Industry & Company Returns', xaxis_title='')
    fig.layout.yaxis.tickformat = ',.0%'

    st.plotly_chart(fig)

    if ticker_return > SPY_return:
        s0 = f'{name} outperforms the S&P 500 by {(ticker_return - SPY_return):.2%}'
    elif ticker_return < SPY_return:
        s0 = f'{name} underperforms the S&P 500 by {(SPY_return - ticker_return):.2%}'
    else:
        s0 = f'{name} and the S&P 500 have comparable performance'
    
    if ticker_return > sector_return:
        s1 = f'{name} outperforms the {sector} sector by \
               {(ticker_return - sector_return):.2%}'
    elif ticker_return < sector_return:
        s1 = f'{name} underperforms the {sector} sector by \
               {(sector_return - ticker_return):.2%}'
    else:
        s1 = f'{name} and the {sector} sector have comparable performance'

    if ticker_return > subIndustry_return:
        s2 = f'{name} outperforms the {subIndustry} sub-industry by \
               {(ticker_return - subIndustry_return):.2%}'
    elif ticker_return < subIndustry_return:
        s2 = f'{name} underperforms the {subIndustry} sub-industry by \
               {(subIndustry_return - ticker_return):.2%}'
    else:
        s2 = f'{name} and the {subIndustry} sub-industry have comparable performance'    
    
    if nsubIndustry == 1:
        s3 = f'{name} is the only stock in the {subIndustry} sub-industry'
    else:
        s3 = f"- CAGR for the {nsubIndustry} stocks in the {subIndustry} \
             sub-industry is {subIndustry_return:.2%} \
             \n- {s2} \
             \n- {name}'s performance is ranked {ticker_rank2}/{nsubIndustry} \
             in the sub-industry"

    st.write(f"Period: {start.strftime('%d/%m/%y')} - {end.strftime('%d/%m/%y')}")
    st.info(f"- {name}'s CAGR is {ticker_return:.2%}")
    st.info(f"- S&P 500 CAGR is {SPY_return:.2%} \
              \n- {s0}")
    st.info(f"- CAGR for the {nsector} stocks in the {sector} sector is \
              {sector_return:.2%} \
              \n- {s1} \
              \n- {name}'s performance is ranked {ticker_rank1}/{nsector} in the sector")
    st.info(f"{s3}")


if option == 'Sector Analysis':
    # Metrics to display graphs of
    metrics = ('Return', 'Volatility', 'Sharpe Ratio', 'Beta', 'Financial Ratio')
    metric = st.selectbox('Metric', metrics)
    
    if metric != 'Financial Ratio':
        #------------------SECTORS--------------------
        start, end = set_form_dates() # Date input

        sectors_df, subIndustries_df, tickers_df, SPY_metrics, rf = calculate_metrics(start, end)

        # Show date range of ticker data and raise error messages
        if start > end:
            st.error('*Start Date* must be before *End Date*')

        missing_mkt_data, rpt1, rpt2, rpt3 = find_stocks_missing_data(start, end)

        # Provide information on data that is available between chosen dates
        if rpt1 != '':
            st.error(rpt1)
        if rpt2 != '':
            st.error(rpt2)
        if len(missing_mkt_data) > 0:
            with st.expander("Stocks Missing Data"):
                st.warning(rpt3)
                st.write(missing_mkt_data)

        fig = plot_sector_metric(sectors_df, SPY_metrics, metric)
        st.plotly_chart(fig)

        #------------------SUB-INDUSTRIES--------------------
        sector = st.selectbox('GICS Sector', sector_list)

        nsector_tickers = len(SPY_info_df[SPY_info_df['Sector'] == sector].index)
        subIndustry_list = subIndustries_df[subIndustries_df.Sector == sector].index.to_list()
        missing = missing_mkt_data.get(sector, {})
        n_missing = len(missing)  

        if n_missing > 0:
            s = f"{n_missing}/{nsector_tickers} stocks in the {sector} sector have \
                  data that begins after {start.strftime('%B %d, %Y')}"
            with st.expander("Stocks Missing Data by Sub-Industry"):
                st.info(s)
                st.write(missing)

        fig = plot_subIndustry_metric(sectors_df, subIndustries_df, SPY_metrics, sector, metric)
        st.plotly_chart(fig)

        #------------------COMPANIES--------------------
        subIndustry = st.selectbox('GICS Sub-Industry', subIndustry_list)
        
        si_tickers = SPY_info_df[SPY_info_df['Sub-Industry'] == subIndustry].index.to_list()
        missing = missing.get(subIndustry, [])
        n_missing = len(missing)  

        if n_missing > 0:
            s = f"{n_missing}/{len(si_tickers)} stocks in the {subIndustry} sub-industry have \
                  data that begins after {start.strftime('%B %d, %Y')}"
            with st.expander("Sub-Industry Stocks Missing Data"):
                st.info(s)
                st.write(missing)

        fig = plot_si_tickers_metric(sectors_df, subIndustries_df, tickers_df, 
                                     SPY_metrics, sector, subIndustry, metric)
        st.plotly_chart(fig)

    if metric == 'Financial Ratio':
        #------------------SECTORS--------------------
        c1, c2 = st.columns(2)

        if ratios_data_report != "The data reported is today's.":
            st.info(ratios_data_report)
            
        categories = ('Investment Valuation Ratios',
                      'Profitability Indicator Ratios',
                      'Liquidity Measurement Ratios',
                      'Debt Ratios',
                      'Operating Performance Ratios',
                      'Cash Flow Indicator Ratios')
        
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
            st.markdown(f'       {formula}\n{definition}')

        sectors_df, subIndustries_df, tickers_df = get_TTM_ratios(ratios, ratio)

        # Charts of sector ratios
        fig = px.bar(sectors_df, x=sectors_df.index, y=ratio, opacity=0.65)
        fig.update_layout(title=f'Sector {ratio}', xaxis_title='')
        st.plotly_chart(fig)

        #------------------SUB-INDUSTRIES--------------------
        sector = st.selectbox('GICS Sector', sector_list)

        sector_ratio = sectors_df.loc[sector].item()
        df = subIndustries_df[subIndustries_df.Sector == sector].sort_values(by=ratio, ascending=False)
        subIndustry_list = df.index.to_list()

        # Chart of sub-industry ratios
        fig = px.bar(df, x=df.index, y=ratio, opacity=0.65)
        fig.add_hline(y=sector_ratio, 
                      line_color='red', 
                      line_width=1,
                      annotation_text=f'{sector} {ratio} ({sector_ratio:,.2f})', 
                      annotation_bgcolor='indianred', 
                      annotation_bordercolor='red')
        fig.update_layout(title=f'{sector} Sub-Industry {ratio}s', xaxis_title='')
        st.plotly_chart(fig)

        #------------------COMPANIES--------------------
        subIndustry = st.selectbox('GICS Sub-Industry', subIndustry_list)

        subIndustry_ratio = df.loc[subIndustry, ratio]

        df = tickers_df[tickers_df['Sub-Industry'] == subIndustry].sort_values(by=ratio, ascending=False)

        # Chart of ticker ratios
        fig = px.bar(df, x=df.index, y=ratio, opacity=0.65, hover_data={'Company': True})
        fig.add_hline(y=sector_ratio, 
                      line_color='red', 
                      line_width=1,
                      annotation_text=f'{sector} {ratio} ({sector_ratio:,.2f})',
                      annotation_position='bottom left', 
                      annotation_bgcolor='indianred',
                      annotation_bordercolor='red')
        fig.add_hline(y=subIndustry_ratio, 
                      line_color='green', 
                      line_width=1,
                      annotation_text=f'{subIndustry} {ratio} ({subIndustry_ratio:,.2f})',
                      annotation_position='bottom left',
                      annotation_bgcolor='lightgreen',
                      annotation_bordercolor='green')
        fig.update_layout(title=f'{subIndustry} Company {ratio}s', xaxis_title='')
        st.plotly_chart(fig)


if option == 'Stock Analysis':
    start, end = set_form_dates() # Date input

    _, _, tickers_df, _, _ = calculate_metrics(start, end)

    metrics = ('Return', 'Volatility', 'Sharpe Ratio', 'Beta', 'Piotroski F-Score')
    metric = st.selectbox('Metric', metrics)

    df = tickers_df.sort_values(by=metric, ascending=False)
    df.reset_index(inplace=True)
    df.index += 1
    cols = df.columns.to_list()
    cols.pop(cols.index(metric))
    cols.insert(4, metric)
    df = df[cols]

    def make_pretty(styler):
        format = {'Return': lambda x: '{:,.2%}'.format(x),
                  'Volatility': lambda x: '{:,.2%}'.format(x)}
        styler.format(precision=2, formatter=format)
        styler.set_properties(**{'background-color': 'lightblue'}, subset=[metric])  
        return styler

    st.write(f'Stocks Ranked by {metric}')
    st.dataframe(df.style.pipe(make_pretty))

        
if option == 'News':
    c1, c2 = st.columns(2)
    tz = timezone('EST')
    ticker = c1.selectbox('Ticker', ticker_list)
    date = c2.date_input('Date', dt.now(tz)).strftime('%Y-%m-%d')
    name = SPY_info_df.loc[ticker, 'Security']
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
                \n_**Published:** {published}_
                _**Full story:** {url}_
                ''')
        

if option == 'Social Media':
    platform = st.selectbox('Platform', ('StockTwits'))
    
    if platform == 'StockTwits':
        ticker = st.selectbox('Ticker', ticker_list)
        try:
            url = f'https://api.stocktwits.com/api/2/streams/ticker/{ticker}.json'
            r = requests.get(url)
            data = r.json()

            for message in data['messages']:
                st.image(message['user']['avatar_url'])
                st.info(f'''
                        {message['user']['username']} \n
                        {message['created_at']} \n
                        {message['body']}
                        ''')
        except:
            st.error(f'{platform} API is unavailable')