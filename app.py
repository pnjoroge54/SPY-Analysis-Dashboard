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
        for i, rpt in enumerate((rpt1, rpt2, rpt3)):
            if rpt != '':
                if i < 2:
                    st.error(rpt)
                else:
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
    st.subheader('Data Selection')

    with st.expander('Options'):
        c1, c2 = st.columns(2)
        setup = c1.selectbox('Trader Setup', ('Investor', 'Swing Trader', 'Day Trader'))
        data = c2.radio('Stocks', ('All', 'Trend-Aligned', 'Trending'), horizontal=True)
        
        if setup == 'Investor':
            index = 1
            periods = ['W1', 'D1', '30m']
        elif setup == 'Swing Trader':
            index = 2
            periods = ['D1', '30m', '10m']
        else:
            index = 3
            periods = ['30m', '10m', '5m', '1m']

        period = periods[0]
        period_d = TA_PERIODS[period]
        days = period_d['days']
        MAs = period_d['MA']
        plot_MAs = MAs[:3]
        plot_data = {'MAs': plot_MAs,
                     'Adv MAs': [int(ma**(1/2)) for ma in plot_MAs]}
        end = last_date
        start = last_date - timedelta(days)

        if data != 'All':
            c1, c2 = st.columns(2)

            if data == 'Trending':
                if setup == 'Investor':
                    periods.insert(0, 'M1')
                period = c1.radio('Timeframe', periods, horizontal=True, key='period1')
                up_tickers, down_tickers = get_trending_stocks(start, end, period, MAs)
            else:
                end = c1.date_input('End Date', value=last_date, min_value=first_date, 
                                    max_value=last_date, key='end_date1')
                up_tickers, down_tickers = get_trend_aligned_stocks(TA_PERIODS, periods, end)
            
            n = max(up_tickers, down_tickers, key=len)
            index = 1 if len(up_tickers) == 0 else 0

            if n:
                trend = c2.radio('Trend', ('Up', 'Down'), index=index, horizontal=True)
                tickers = up_tickers if trend == 'Up' else down_tickers
                if not tickers:
                    c2.warning(f'No stocks {data} {trend}'.upper())
            else:
                c2.warning(f'No stocks {data}'.upper())
                tickers = None        
                    
        else:
            tickers = ticker_list
        
        c1, c2 = st.columns(2)
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
        tickers = sorted([f'{ticker} - {name}' for ticker, name in zip(tickers, names)])
    
    # Chart Display
    if tickers:
        st.subheader('Chart')
        graph = st.radio('Type', ('Candlesticks', 'Line'), horizontal=True)
        
        periods = ['M1', 'W1', 'D1', '30m', '10m', '5m', '1m']

        with st.expander('Signal Settings'):
            tab1, tab2 = st.tabs(('Signals', 'Subplots'))

        with tab1:
            c1, c2 = st.columns(2)
            show_sr = c1.checkbox('Support / Resistance (SR)', True)
            show_fr = c1.checkbox('Fibonacci Retracements (FR)')
            show_bb = c1.checkbox('Bollinger Bands (BB)')
            show_MAs = c1.checkbox('Moving Averages (MA)')
            show_adv_MAs = c1.checkbox('Advanced MAs')
            placeholder1 = c1.empty()
            show_trend_analysis = c2.checkbox('Trend Analysis') 
            show_trendlines_c = c2.checkbox('Trendlines (Close)')
            show_trendlines_hl = c2.checkbox('Trendlines (High-Low)')
            
            if show_MAs or show_adv_MAs:
                period_d = TA_PERIODS[period]
                MAs = period_d['MA']
                days = period_d['days']
                minor_ma, secondary_ma, primary_ma, *_ = MAs
                plot_MAs = [minor_ma, secondary_ma, primary_ma]
                plot_data = {'MAs': plot_MAs,
                            'Adv MAs': [int(ma**(1/2)) for ma in plot_MAs]}
                
                adjust_MAs = placeholder1.checkbox('Adjust MA Windows')
                c1, c2, c3 = st.columns(3)

                if adjust_MAs:
                    if show_MAs:
                        minor_ma = c1.number_input('Minor MA', value=MAs[0])
                        secondary_ma  = c2.number_input('Secondary MA', value=MAs[1])
                        primary_ma  = c3.number_input('Primary MA', value=MAs[2])
                        plot_MAs = [minor_ma, secondary_ma, primary_ma]
                        plot_data['MAs'] = plot_MAs
                    if show_adv_MAs:
                        advanced_MAs = [int(ma**(1/2)) for ma in plot_MAs]
                        minor_adv_ma = c1.number_input(f'Advance MA{minor_ma}', 
                                                       value=advanced_MAs[0])
                        secondary_adv_ma = c2.number_input(f'Advance MA{secondary_ma}', 
                                                           value=advanced_MAs[1])
                        primary_adv_ma = c3.number_input(f'Advance MA{primary_ma}', 
                                                         value=advanced_MAs[2])
                        plot_adv_MAs = [minor_adv_ma, secondary_adv_ma, primary_adv_ma]
                        plot_data['Adv MAs'] = plot_adv_MAs
        
        with tab2:
            show_vol = st.checkbox('Volume', True)
            show_rsi = st.checkbox('Relative Strength Index (RSI)')
            show_macd = st.checkbox('Moving Average Convergence / Divergence (MACD)')

        c1, c2 = st.columns(2)
        ticker = c1.selectbox(ticker_lbl, tickers, help=text)
        period = c2.radio('Timeframe', periods, index=index, horizontal=True, key='periods3')
        rangeslider = st.checkbox('Date Range Slider')

        period_d = TA_PERIODS[period]
        MAs = period_d['MA']
        days = period_d['days']
        start = None if rangeslider else start - timedelta(days)
        minor_ma, secondary_ma, primary_ma, *_ = MAs
        plot_MAs = [minor_ma, secondary_ma, primary_ma]
        plot_data = {'MAs': plot_MAs,
                     'Adv MAs': [int(ma**(1/2)) for ma in plot_MAs]}
        
        fig = plot_signals(graph, rangeslider, ticker, start, end, period, plot_data,
                           show_vol, show_rsi, show_macd, show_sr, show_fr, show_bb,
                           show_MAs, show_adv_MAs, show_trend_analysis, show_trendlines_c, 
                           show_trendlines_hl)
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
