import numpy as np
from datetime import datetime as dt
import warnings

import cufflinks as cf
import plotly.express as px
import streamlit as st
from newspaper import Article

from info import *
from functions import *
from technical_analysis import *

warnings.simplefilter(action='ignore', category=FutureWarning)


options = ('S&P 500 Information', 'Stock Information', 'Stock Comparisons',
           'Sector Analysis', 'Technical Analysis', 'News')

option = st.sidebar.selectbox("Dashboard", options)
st.title(option)

if option == 'S&P 500 Information':
    st.info(SPY_INFO)
    st.subheader('Market Data')
    start, end = set_form_dates()

    df = SPY_df[start: end]
    df['Return'] = np.log1p(df['Close'].pct_change())
    t = len(df) / 252
    cagr = ((df['Close'][-1] / df['Open'][0])**(1 / t) - 1)
    std = df['Return'].std() * np.sqrt(252)
    rf = rf_rates.loc[start: end, 'Close'].mean() / 100
    sr = (cagr - rf) / std

    s1 = f'Annualized Return: {cagr:.2%}'
    s2 = f'Annualized Volatility: {std:.2%}'
    s3 = f'Risk-Free Rate: {rf:.2%}'
    s4 = f'Sharpe Ratio: {sr:,.2f}' 
    
    # Candlestick chart
    qf = cf.QuantFig(df, name='SPY', title='S&P 500')
    qf.add_volume()

    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)
    
    fig = plot_returns_histogram(df)
    st.plotly_chart(fig)

    st.info(f'{s1}  \n{s2}  \n{s3}  \n{s4}')


if option == 'Stock Information':
    c1, c2 = st.columns(2)
    search = c1.radio('Search', ('Ticker', 'Company'), horizontal=True)
    
    if search == 'Ticker':
        ticker = c2.selectbox(search, ticker_list)
        cname = SPY_info_df.loc[ticker, 'Security']
    else:
        names_list = SPY_info_df['Security'].to_list()
        cname = c2.selectbox(search, names_list)
        ticker = SPY_info_df[SPY_info_df['Security'] == cname].index.item()
    
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
             
    website = tickers_info[ticker]['Website']
    summary = tickers_info[ticker]['Business Summary']
    
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

    ticker_df = ticker_df[start: end]
    ticker_df['Return'] = np.log1p(ticker_df['Adj Close'].pct_change())
    ticker_df.rename(columns={'volume': 'Volume'}, inplace=True)
  
    if start > end:
        st.error('*Start Date* must be before *End Date*')
    if start < first_date:
        st.error(f"Market data before {first_date.strftime('%B %d, %Y')} is unavailable")
    if end > last_date:
        st.error(f"Market data after {last_date.strftime('%B %d, %Y')} is unavailable")

    # Candlestick chart
    qf = cf.QuantFig(ticker_df, title=f'{cname} Daily Prices', name=ticker)
    qf.add_volume(colorchange=False)
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)

    fig = plot_returns_histogram(ticker_df)
    st.plotly_chart(fig)

    window = st.number_input('Moving Average (Days)', value=20)
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
    metrics = ('Return', 'Volatility', 'Sharpe Ratio', 'Beta', 'Financial Ratios')
    metric = st.selectbox('Metric', metrics)
    
    if metric != 'Financial Ratios':
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
        subIndustry_list = sorted(subIndustry_list)
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

    if metric == 'Financial Ratios':
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
        fig = plot_sector_financial_ratios(sectors_df, ratio)
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


if option == 'Stock Comparisons':
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
        styler.set_properties(**{'background-color': 'steelblue',
                                 'color': 'white'},
                              subset=[metric])  
        return styler

    st.write(f'Stocks Ranked by {metric}')
    st.dataframe(df.style.pipe(make_pretty))

        
if option == 'News':
    c1, c2 = st.columns(2)
    tz = timezone('EST')
    names = SPY_info_df.loc[ticker_list, 'Security']
    tickers = [f'{ticker} - {name}' for ticker, name in zip(ticker_list, names)]
    ticker = c1.selectbox('Ticker - Security', tickers)
    ticker, name = ticker.split(' - ')
    date = c2.date_input('Date', dt.now(tz)).strftime('%Y-%m-%d')
    news = get_news(ticker, date)
    
    if len(news) == 0:
        st.write(f'There are no stories about {name}')
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

        with st.expander(headline):
            st.info(f'''
                    ### {headline}
                    \n\n##### Summary:
                    \n\n{summary}
                    \n_**Source:** {source}_  
                    \n_**Published:** {published}_
                    _**Full story:** {url}_
                    ''')


if option == 'Technical Analysis':
    c1, c2 = st.columns(2)
    setups = ('Investor', 'Swing Trader', 'Day Trader')
    setup = c1.selectbox('Trader Setup', setups)
    
    if setup == 'Investor':
        periods = ('Weekly', 'Daily', '30 Min')
    elif setup == 'Swing Trader':
        periods = ('Daily', '30 Min', '5 Min')
    else:
        periods = ('30 Min', '5 Min', '1 Min')

    period = c2.radio('Timeframe', periods, horizontal=True)
    period_d = TA_PERIODS[period]
    MAs = period_d['MA']
    days = period_d['days']

    if last_date.weekday() == 0:
        days += 3
    start = last_date - timedelta(days)

    c1, c2 = st.columns(2)
    start = c1.date_input('Start Date', value=start, min_value=first_date, max_value=last_date)
    end = c2.date_input('End Date', value=last_date, min_value=first_date, max_value=last_date)
    
    c1, c2 = st.columns(2)
    data = c1.radio('Stocks', ('Trend-Aligned', 'Trending', 'All'), horizontal=True)

    if data != 'All':
        trend = c2.radio('Trend', ('Up', 'Down'), horizontal=True)
        # st.write('') # for aligning rows in columns
        if data == 'Trending':
            up_tickers, down_tickers = get_trending_stocks(start, end, period, MAs)
            if trend == 'Up':
                tickers = up_tickers
            else:
                tickers = down_tickers
        else:
            up_aligned, down_aligned = get_trend_aligned_stocks(TA_PERIODS, periods, end)
            if trend == 'Up':
                tickers = up_aligned
            else:
                tickers = down_aligned        
    else:
        c2.empty()
        tickers = ticker_list
    
    c1, c2, c3 = st.columns(3)
    sectors = ['-' * 30]
    sectors += SPY_info_df.loc[tickers, 'Sector'].unique().tolist()
    sector = c1.selectbox('Sector', sorted(sectors))

    if sector != sectors[0]:
        df = SPY_info_df[SPY_info_df['Sector'] == sector]
        tickers = list(set(df.index.to_list()) & set(tickers))    
    
    subIndustries = ['-' * 30] 
    subIndustries += SPY_info_df.loc[tickers, 'Sub-Industry'].unique().tolist()
    subIndustry = c2.selectbox('Sub-Industry', sorted(subIndustries))
    
    if subIndustry != subIndustries[0]:
        df = SPY_info_df[SPY_info_df['Sub-Industry'] == subIndustry]
        tickers = list(set(df.index.to_list()) & set(tickers))
        
    if data == 'All':   
        text = f'{len(tickers)} stocks'
    else:
        text = f'{len(tickers)} {trend.lower()}{data.lower()}'
        
    ticker_lbl = 'Ticker - Security' 
    names = SPY_info_df.loc[tickers, 'Security'].to_list()
    tickers = [f'{ticker} - {name}' for ticker, name in zip(tickers, names)]
    
    ticker = c3.selectbox(ticker_lbl, sorted(tickers), help=text)
    short_ma, inter_ma, long_ma, *_ = MAs
    
    if tickers:
        c1, c2, c3 = st.columns(3)
        ticker = ticker.split(' - ')[0]
        show_sr = c1.checkbox('Add Support-Resistance (S/R) Levels', True)
        show_fib = c2.checkbox('Add Fibonacci Retracement Levels (FRLs)')
        adjust_ma = c3.checkbox('Adjust Moving Averages (MAs)')
        log_y = c1.checkbox('Logarithmic Scale', True)
        show_prices = c2.checkbox('Display Candlestick Data')

        if adjust_ma:
            c1, c2, c3 = st.columns(3)
            short_ma = c1.number_input('Short-Term MA', value=MAs[0])
            inter_ma = c2.number_input('Intermediate-Term MA', value=MAs[1])
            long_ma = c3.number_input('Long-Term MA', value=MAs[2])

        fig = plot_trends(ticker, start, end, period, short_ma, inter_ma, long_ma,
                          show_sr, show_prices, show_fib, log_y)
        st.plotly_chart(fig)

        # file = 'watchlist.pickle'

        # if os.path.isfile(file):
        #     with open(file, 'rb') as f:
        #         watchlist = pickle.load(f)
        # else:
        #     watchlist = []

        # if len(watchlist) > 0:
        #     with c1.expander("Watchlist", expanded=False):
        #         st.write(watchlist)
            

        
        # Add option to view stocks in watchlist
        # save = c1.button('Add Stock to Watchlist')
        
        # if save:
        #     with open(file, 'wb') as f:
        #         watchlist.append(ticker)
        #         pickle.dump(watchlist, f)


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