import numpy as np
from datetime import datetime as dt
import warnings

import cufflinks as cf
import plotly.express as px
import yfinance as yf
import streamlit as st
from newspaper import Article

from info import SPY_INFO, FINANCIAL_RATIOS
from functions import *
from technical_analysis import *

warnings.simplefilter(action='ignore', category=FutureWarning)


options = ('S&P 500 Information', 'Stock Information', 'Stock Analysis',
           'Sector Analysis', 'Technical Analysis', 'News')

option = st.sidebar.selectbox("Select Dashboard", options)
st.title(option)

if option == 'S&P 500 Information':
    st.info(SPY_INFO)
    st.subheader('Market Data')
    start, end = set_form_dates()

    df = SPY_df[start : end]
    df['Return'] = np.log1p(df['Close'].pct_change())
    t = len(df) / 252
    cagr = ((df['Close'][-1] / df['Open'][0])**(1 / t) - 1)
    std = df['Return'].std() * np.sqrt(252)
    rf = rf_rates.loc[start : end, 'Close'].mean() / 100
    sr = (cagr - rf) / std

    s1 = f'Annualized Return: {cagr:.2%}'
    s2 = f'Annualized Volatility: {std:.2%}'
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
    search = c1.radio('Search by', ('Ticker', 'Company'), horizontal=True)
    
    if search == 'Ticker':
        ticker = c2.selectbox(search, ticker_list)
        cname = SPY_info_df.loc[ticker, 'Security']
    else:
        names_list = SPY_info_df['Security'].to_list()
        cname = c2.selectbox(search, names_list)
        ticker = SPY_info_df[SPY_info_df['Security'] == cname].index.name
    
    sector = SPY_info_df.loc[ticker, 'Sector']
    subIndustry = SPY_info_df.loc[ticker, 'Sub-Industry']
    location = SPY_info_df.loc[ticker, 'Headquarters Location']
    founded = SPY_info_df.loc[ticker, 'Founded']
    date_added = SPY_info_df.loc[ticker, 'Date added']
    
    try:
        date_added = dt.strptime(date_added, '%Y-%m-%d')
        date_added = date_added.strftime('%B %d, %Y')
    except:
        date_added = 'N/A'

    info = get_ticker_info()             
    website = info[ticker]['Website']
    summary = info[ticker]['Business Summary']
    
    st.header(f'**{cname}**')
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
    ticker_df['Return'] = np.log1p(ticker_df['adjclose'].pct_change())
    ticker_df.rename(columns={'volume': 'Volume'}, inplace=True)
  
    if start > end:
        st.error('*Start Date* must be before *End Date*')
    if start < first_date:
        st.error(f"Market data before {first_date.strftime('%B %d, %Y')} is unavailable")
    if end > last_date:
        st.error(f"Market data after {last_date.strftime('%B %d, %Y')} is unavailable")

    # Candlestick chart
    qf = cf.QuantFig(ticker_df, title=f'{cname} Daily Prices', name=ticker)
    qf.add_volume()
    fig = qf.iplot(asFigure=True, yTitle='Price')
    st.plotly_chart(fig)

    fig = make_returns_histogram(ticker_df)
    st.plotly_chart(fig)

    window = st.number_input('Moving Average Window (Days)', value=20)
    fig = plot_sma_returns(ticker, start, end, window)
    st.plotly_chart(fig)

    sectors_df, subIndustries_df, tickers_df, SPY_metrics, rf = calculate_metrics(start, end)
    metrics = ('Return', 'Volatility') # Metrics to display graphs of
    
    st.subheader('Peers Comparison')
    metric = st.selectbox('Metric', metrics)

    rank_df = tickers_df.sort_values(by=metric, ascending=False).reset_index()
    rank_df.index += 1
    rank = rank_df[rank_df['Ticker'] == ticker].index.item()
    metric_val = rank_df.loc[rank, metric]
 
    st.info(f"{cname}'s {metric.lower()} ({metric_val:,.2%}) is ranked \
              {rank}/{len(tickers_df)} in the S&P 500")

    # Graph of all tickers in sector
    fig = plot_sector_tickers_metric(sectors_df, subIndustries_df, tickers_df, 
                                     SPY_metrics, sector, subIndustry, metric, ticker)
    st.plotly_chart(fig)

    # Graph of all tickers in sub-industry
    fig = plot_si_tickers_metric(sectors_df, subIndustries_df, tickers_df, 
                                 SPY_metrics, sector, subIndustry, metric, ticker)
    st.plotly_chart(fig)
  

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

        missing_data, rpt1, rpt2, rpt3 = find_stocks_missing_data(start, end)

        # Provide information on data that is available between chosen dates
        if rpt1 != '':
            st.error(rpt1)
        if rpt2 != '':
            st.error(rpt2)
        if len(missing_data) > 0:
            with st.expander("Stocks Missing Data"):
                st.warning(rpt3)
                st.write(missing_data)

        fig = plot_sector_metric(sectors_df, SPY_metrics, metric)
        st.plotly_chart(fig)

        #------------------SUB-INDUSTRIES--------------------
        sector = st.selectbox('GICS Sector', sector_list)

        nsector_tickers = len(SPY_info_df[SPY_info_df['Sector'] == sector].index)
        subIndustry_list = subIndustries_df[subIndustries_df['Sector'] == sector].index.to_list()
        missing = missing_data.get(sector, {})
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
        ratios = FINANCIAL_RATIOS['ratios'][category]
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
        df = subIndustries_df[subIndustries_df['Sector'] == sector].sort_values(by=ratio, ascending=False)
        subIndustry_list = df.index.to_list()

        # Chart of sub-industry ratios
        fig = px.bar(df, x=df.index, y=ratio, opacity=0.65)
        fig.layout.yaxis.tickformat = ',.2f'
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
        fig.layout.yaxis.tickformat = ',.2f'
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
                      annotation_bgcolor='darkgreen',
                      annotation_bordercolor='green')
        fig.update_layout(title=f'{subIndustry} Company {ratio}s', xaxis_title='')
        st.plotly_chart(fig)


if option == 'Stock Analysis':
    start, end = set_form_dates() # Date input

    _, _, tickers_df, _, _ = calculate_metrics(start, end)
    metrics = ('Return', 'Volatility', 'Sharpe Ratio', 'Beta')
    
    metric = st.selectbox('Metric', metrics)

    df = tickers_df.sort_values(by=metric, ascending=False).reset_index()
    df.index += 1
    cols = df.columns.to_list()
    cols.pop(cols.index(metric))
    cols.insert(4, metric)
    df = df[cols]

    def make_pretty(styler):
        format = {'Return': lambda x: '{:,.2%}'.format(x),
                  'Volatility': lambda x: '{:,.2%}'.format(x)}
        styler.format(precision=2, formatter=format)
        styler.set_properties(**{'background-color': 'darkblue'}, subset=[metric])  
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
        st.write(f'There are no stories about {name}')
    elif len(news) == 1:
        st.write(f'There is {len(news)} story about {name}')
    else:    
        st.write(f'There are {len(news)} stories about {name}')

    for story in news:
        headline = story['headline']
        source = story['source']
        url = story['url']
        # Convert timestamp to datetime and get string of hours & min
        published = dt.fromtimestamp(story['datetime']).strftime('%d %b, %Y %I:%M%p')
        
        # Get summary of the article
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


if option == 'Technical Analysis':
    time_frame = st.selectbox('Time Frame', ('Daily', 'Weekly', 'Monthly'))

    if time_frame == 'Daily':
        v = [50, 100, 200]
    elif time_frame == 'Weekly':
        v = [4, 10, 36]
    else:
        v = [1, 3, 9]

    with st.form(key='ta_form'):
        c1, c2, c3 = st.columns(3)
        short = c1.number_input('Short-Term Trend', value=v[0])
        inter = c2.number_input('Intermediate Trend', value=v[1])
        long = c3.number_input('Primary Trend', value=v[2])
        start_date = last_date - timedelta(365*2)
        start = c1.date_input('Start Date', start_date, min_value=first_date)
        end = c2.date_input('End Date', last_date, max_value=last_date)
        c1.form_submit_button(label='Submit')

    c1, c2, c3 = st.columns(3)
    sector = c1.selectbox('Sectors', ['ALL'] + sector_list)
    search = c2.radio('Search by', ('Ticker', 'Company'))
    
    if sector != 'ALL':
        df = SPY_info_df[SPY_info_df['Sector'] == sector]
        tickers = df.index.to_list()
        names = df['Security'].to_list()
    else:
        tickers = ticker_list
        names = SPY_info_df['Security'].to_list()

    if search == 'Ticker':
        ticker = c3.selectbox(search, ticker_list)
        cname = SPY_info_df.loc[ticker, 'Security']
    else:
        cname = c3.selectbox(search, names)
        ticker = SPY_info_df[SPY_info_df['Security'] == cname].index.name

    df = get_ticker_data(ticker)
    fig = plot_trends(df, start, end, time_frame, short, inter, long)
    st.plotly_chart(fig) 

# if option == 'Social Media':
#     platform = st.selectbox('Platform', 'StockTwits')
    
#     if platform == 'StockTwits':
#         ticker = st.selectbox('Ticker', ticker_list)
#         try:
#             url = f'https://api.stocktwits.com/api/2/streams/ticker/{ticker}.json'
#             r = requests.get(url)
#             data = r.json()

#             for message in data['messages']:
#                 st.image(message['user']['avatar_url'])
#                 st.info(f'''
#                         {message['user']['username']} \n
#                         {message['created_at']} \n
#                         {message['body']}
#                         ''')
#         except:
#             st.error(f'{platform} API is unavailable')