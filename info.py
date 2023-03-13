SPY_INFO = '''
   The Standard and Poor's 500, or simply the S&P 500, is a free-float weighted stock market index that tracks 500 large companies
   listed on stock exchanges in the United States. It is one of the most commonly followed equity indices. 
   As of December 31st, 2020, more than $4.6 trillion was invested in assets tied to the performance of the index.

   The index value is updated every 15 seconds, or 1,559 times per trading day, with price updates disseminated by Reuters.

   The S&P 500 is maintained by S&P Dow Jones Indices, a joint venture majority-owned by S&P Global, and its components are
   selected by a committee.

   Requirements to be added to the index include:
   
    1. Market capitalization must be greater than or equal to US$12.7 billion.
    2. Annual dollar value traded to float-adjusted market capitalization is greater than 0.75.
    3. Minimimum monthly trading volume of 250,000 shares in each of the six months leading up to the evaluation date.
    4. Must be publicly listed on either the New York Stock Exchange (including NYSE Arca or NYSE American) or NASDAQ (NASDAQ Global Select Market, NASDAQ Select Market or the NASDAQ Capital Market).
    5. The company should be from the U.S.
    6. Securities that are ineligible for inclusion in the index are limited partnerships, master limited partnerships and their investment trust units, OTC Bulletin Board issues, closed-end funds, exchange-traded funds, Exchange-traded notes, royalty trusts, tracking stocks, preferred stock, unit trusts, equity warrants, convertible bonds, investment trusts, American depositary receipts, and American depositary shares.
    7. Since 2017, companies with dual share classes are not added to the index.

   To remain indicative of the largest public companies in the United States, the index is reconstituted quarterly; however, 
   efforts are made to minimize turnover in the index as a result of declines in value of constituent companies.
'''


# Dictionaries of the formulas & definitions of financial ratios.

# For more information on the calculation of financial ratios
# visit https://en.wikipedia.org/wiki/Financial_ratio
       
RATIOS = {
    'Investment Valuation Ratios':
        {'Price to Earnings Ratio': 'priceEarningsRatioTTM', 
         'Price to Book Value Ratio': 'priceToBookRatioTTM', 
         'Price to Sales Ratio': 'priceToSalesRatioTTM', 
         'Price to Earnings to Growth Ratio': 'priceEarningsToGrowthRatioTTM',
         'Price to Free Cash Flows Ratio': 'priceToFreeCashFlowsRatioTTM', 
         'Enterprise Value Multiplier': 'enterpriseValueMultipleTTM', 
         'Dividend Yield': 'dividendYieldTTM'},

    'Profitability Indicator Ratios':
        {'Gross Profit Margin': 'grossProfitMarginTTM',
         'Net Profit Margin': 'netProfitMarginTTM',
         'Operating Profit Margin': 'operatingProfitMarginTTM',
         'Pre-Tax Profit Margin': 'pretaxProfitMarginTTM',
         'Effective Tax Rate': 'effectiveTaxRateTTM',
         'Return On Assets': 'returnOnAssetsTTM',
         'Return On Equity': 'returnOnEquityTTM',
         'Return On Capital Employed': 'returnOnCapitalEmployedTTM'},
    
    'Liquidity Measurement Ratios':
        {'Current Ratio': 'currentRatioTTM',
         'Quick Ratio': 'quickRatioTTM',
         'Cash Ratio': 'cashRatioTTM',
         'Days Of Sales Outstanding': 'daysOfSalesOutstandingTTM',
         'Days Of Inventory Outstanding': 'daysOfInventoryOutstandingTTM',
         'Operating Cycle': 'operatingCycleTTM',
         'Days Of Payables Outstanding': 'daysOfPayablesOutstandingTTM',
         'Cash Conversion Cycle': 'cashConversionCycleTTM'},
    
    'Debt Ratios':
        {'Debt Ratio': 'debtRatioTTM',
         'Debt to Equity Ratio': 'debtEquityRatioTTM',
         'Long-Term Debt to Capitalisation': 'longTermDebtToCapitalizationTTM',
         'Total Debt to Capitalisation': 'totalDebtToCapitalizationTTM',
         'Interest Coverage Ratio': 'interestCoverageTTM',
         'Cash Flow to Debt Ratio': 'cashFlowToDebtRatioTTM',
         'Company Equity Multiplier': 'companyEquityMultiplierTTM'},
    
    'Operating Performance Ratios':
        {'Asset Turnover': 'assetTurnoverTTM',
         'Fixed Asset Turnover': 'fixedAssetTurnoverTTM',
         'Inventory Turnover': 'inventoryTurnoverTTM',
         'Receivables Turnover': 'receivablesTurnoverTTM',
         'Payables Turnover': 'payablesTurnoverTTM'},

    'Cash Flow Indicator Ratios':
        {'Operating Cash Flow per Share': 'operatingCashFlowPerShareTTM',
         'Free Cash Flow per Share': 'freeCashFlowPerShareTTM',
         'Cash per Share': 'cashPerShareTTM',
         'Operating Cash Flow to Sales Ratio': 'operatingCashFlowSalesRatioTTM',
         'Free Cash Flow to Operating Cash Flow Ratio': 'freeCashFlowOperatingCashFlowRatioTTM',
         'Cash Flow Coverage Ratio': 'cashFlowCoverageRatiosTTM',
         'Short-Term Coverage Ratio': 'shortTermCoverageRatiosTTM',
         'Capex Coverage Ratio': 'capitalExpenditureCoverageRatioTTM',
         'Dividend Paid & Capex Coverage Ratio': 'dividendPaidAndCapexCoverageRatioTTM',
         'Dividend Payout Ratio': 'payoutRatioTTM'}
    }
    
FORMULAS = {
    'Investment Valuation Ratios':
        {'Price to Earnings Ratio': r'$P/E = \frac{\text{Stock Price}}{\text{Diluted EPS}}$', 
         'Price to Book Value Ratio': r'$P/B = \frac{\text{Stock Price}}{\text{Book Value per Share}}$', 
         'Price to Sales Ratio': r'$P/S = \frac{\text{Stock Price}}{\text{Gross Sales}}$', 
         'Price to Earnings to Growth Ratio': r'$PEG = \frac{\text{Stock Price/EPS}}{\text{Annual EPS Growth}}$', 
         'Price to Free Cash Flows Ratio': r'$P/FCFF = \frac{\text{Stock Price}}{\text{Free Cash Flow per Share}}$', 
         'Enterprise Value Multiplier': r'$EV/EBITDA = \frac{\text{Enterprise Value}}{EBITDA}$', 
         'Dividend Yield': r'$Dividend Yield = \frac{Dividend}{\text{Stock Price}}$'}, 

    'Profitability Indicator Ratios':
        {'Gross Profit Margin': r'$Gross Profit Margin = \frac{\text{Gross Profit}}{\text{Net Sales}}$',
         'Net Profit Margin': r'$Net Profit Margin = \frac{\text{Net Profit}}{\text{Net Sales}}$',
         'Operating Profit Margin': r'$Operating Profit Margin = \frac{EBIT}{\text{Net Sales}}$',
         'Pre-Tax Profit Margin': r'$Pre-Tax Profit Margin = \frac{\text{Stock Price}}{\text{Net Sales}}$',
         'Effective Tax Rate': r'$Effective Tax Rate = \frac{\text{Stock Price}}{Free Cash Flow per Share}$',
         'Return On Assets': r'$ROA = \frac{\text{Net Income}}{\text{Average Total Assets}}$',
         'Return On Equity': r'$ROE = \frac{\text{Net Income}}{\text{Average Shareholders Equity}}$',
         'Return On Capital Employed': r'$ROCE = \frac{EBIT}{\text{Capital Employed}}$'},
    
    'Liquidity Measurement Ratios':
        {'Current Ratio': r'$Current Ratio = \frac{\text{Current Assets}}{\text{Current Liabilities}}$',
         'Quick Ratio': r'$Quick Ratio = \frac{\text{Current Assets − (Inventories + Prepayments)}}{\text{Current Liabilities}}$',
         'Cash Ratio': r'$Cash Ratio = \frac{\text{Cash + Marketable Securities}}{\text{Current Liabilities}}$',
         'Days Of Sales Outstanding': r'$DSO = \frac{\text{Accounts Receivable}}{\text{Total Annual Sales}} * \text{365 Days}$',
         'Days Of Inventory Outstanding': r'$Inventory conversion period = \frac{\text{Inventory}}{\text{COGS}} * \text{365 Days}$',
         'Operating Cycle': 'operatingCycleTTM',
         'Days Of Payables Outstanding': r'$Payables conversion period = \frac{\text{Accounts Payable}}{\text{Purchases}} * \text{365 Days}$',
         'Cash Conversion Cycle': 'cashConversionCycleTTM'},
    
    'Debt Ratios':
        {'Debt Ratio': 'debtRatioTTM',
         'Debt to Equity Ratio': 'debtEquityRatioTTM',
         'Long-Term Debt to Capitalisation': 'longTermDebtToCapitalizationTTM',
         'Total Debt to Capitalisation': 'totalDebtToCapitalizationTTM',
         'Interest Coverage Ratio': 'interestCoverageTTM',
         'Cash Flow to Debt Ratio': 'cashFlowToDebtRatioTTM',
         'Company Equity Multiplier': 'companyEquityMultiplierTTM'},
    
    'Operating Performance Ratios':
        {'Asset Turnover': 'assetTurnoverTTM',
         'Fixed Asset Turnover': 'fixedAssetTurnoverTTM',
         'Inventory Turnover': 'inventoryTurnoverTTM',
         'Receivables Turnover': 'receivablesTurnoverTTM',
         'Payables Turnover': 'payablesTurnoverTTM'},
    
    'Cash Flow Indicator Ratios':
        {'Operating Cash Flow per Share': 'operatingCashFlowPerShareTTM',
         'Free Cash Flow per Share': 'freeCashFlowPerShareTTM',
         'Cash per Share': 'cashPerShareTTM',
         'Operating Cash Flow to Sales Ratio': 'operatingCashFlowSalesRatioTTM',
         'Free Cash Flow to Operating Cash Flow Ratio': 'freeCashFlowOperatingCashFlowRatioTTM',
         'Cash Flow Coverage Ratio': 'cashFlowCoverageRatiosTTM',
         'Short-Term Coverage Ratio': 'shortTermCoverageRatiosTTM',
         'Capex Coverage Ratio': 'capitalExpenditureCoverageRatioTTM',
         'Dividend Paid & Capex Coverage Ratio': 'dividendPaidAndCapexCoverageRatioTTM',
         'Dividend Payout Ratio': 'payoutRatioTTM'}
}

DEFINITIONS = {
    'Investment Valuation Ratios':
        {'Price to Earnings Ratio':
         '''
         The financial reporting of both companies and investment research services use a basic 
         earnings per share (EPS) figure divided into the current stock price to calculate the P/E 
         multiple (i.e. how many times a stock is trading (its price) per each dollar of EPS).
         ''', 

         'Price to Book Value Ratio':
         '''
         The price-to-book value ratio, expressed as a multiple (i.e. how many times a company's 
         stock is trading per share compared to the company's book value per share), is an 
         indication of how much shareholders are paying for the net assets of a company.
         ''', 

         'Price to Sales Ratio':
         '''
         The P/E ratio and P/S reflects how many times investors are paying for every dollar of a 
         company's sales. Since earnings are subject, to one degree or another, to accounting 
         estimates and management manipulation, many investors consider a company's sales (revenue) 
         figure a more reliable ratio component in calculating a stock's price multiple than the 
         earnings figure.
         ''', 

         'Price to Earnings to Growth Ratio':
         '''
         The PEG ratio is a refinement of the P/E ratio and factors in a stock's estimated earnings 
         growth into its current valuation.The general consensus is that if the PEG ratio indicates 
         a value of 1, this means that the market is correctly valuing (the current P/E ratio) a 
         stock in accordance with the stock's current estimated earnings per share growth. If the 
         PEG ratio is less than 1, this means that EPS growth is potentially able to surpass the 
         market's current valuation. In other words, the stock's price is being undervalued. On 
         the other hand, stocks with high PEG ratios can indicate just the opposite - that the 
         stock is currently overvalued.
         ''',

         'Price to Free Cash Flows Ratio':
         '''
         The price/free cash flow ratio is used by investors to evaluate the investment 
         attractiveness, from a value standpoint, of a company's stock.
         ''', 

         'Enterprise Value Multiplier':
         '''
         Overall, this measurement allows investors to assess a company on the same basis as that 
         of an acquirer. As a rough calculation, enterprise value multiple serves as a proxy for 
         how long it would take for an acquisition to earn enough to pay off its costs in 
         years(assuming no change in EBITDA).
         ''',

         'Dividend Yield':
         '''
         Income investors value a dividend-paying stock, while growth investors have little 
         interest in dividends, preferring to capture large capital gains. Whatever your 
         investing style, it is a matter of historical record that dividend-paying stocks 
         have performed better than non-paying-dividend stocks over the long term.
         '''
        },

    'Profitability Indicator Ratios':
        {'Gross Profit Margin': 'grossProfitMarginTTM',
         
         'Net Profit Margin': 'netProfitMarginTTM',
         
         'Operating Profit Margin': 
         '''
         Operating income is the difference between operating revenues and operating expenses, 
         but it is also sometimes used as a synonym for EBIT and operating profit. 
         This is true if the firm has no non-operating income.
         ''',

         'Pre-Tax Profit Margin': 'pretaxProfitMarginTTM',
         'Effective Tax Rate': 'effectiveTaxRateTTM',
         'Return On Assets': 'returnOnAssetsTTM',
         'Return On Equity': 'returnOnEquityTTM',
         'Return On Capital Employed': 'returnOnCapitalEmployedTTM'},
    
    'Liquidity Measurement Ratios':
        {'Current Ratio': 'currentRatioTTM',
         'Quick Ratio': 'quickRatioTTM',
         'Cash Ratio': 'cashRatioTTM',
         'Days Of Sales Outstanding': 'daysOfSalesOutstandingTTM',
         'Days Of Inventory Outstanding': 'daysOfInventoryOutstandingTTM',
         'Operating Cycle': 'operatingCycleTTM',
         'Days Of Payables Outstanding': 'daysOfPayablesOutstandingTTM',
         'Cash Conversion Cycle': 'cashConversionCycleTTM'},
    
    'Debt Ratios':
        {'Debt Ratio': 'debtRatioTTM',
         'Debt to Equity Ratio': 'debtEquityRatioTTM',
         'Long-Term Debt to Capitalisation': 'longTermDebtToCapitalizationTTM',
         'Total Debt to Capitalisation': 'totalDebtToCapitalizationTTM',
         'Interest Coverage Ratio': 'interestCoverageTTM',
         'Cash Flow to Debt Ratio': 'cashFlowToDebtRatioTTM',
         'Company Equity Multiplier': 'companyEquityMultiplierTTM'},
    
    'Operating Performance Ratios':
        {'Asset Turnover': 'assetTurnoverTTM',
         'Fixed Asset Turnover': 'fixedAssetTurnoverTTM',
         'Inventory Turnover': 'inventoryTurnoverTTM',
         'Receivables Turnover': 'receivablesTurnoverTTM',
         'Payables Turnover': 'payablesTurnoverTTM'},

    'Cash Flow Indicator Ratios':
        {'Operating Cash Flow per Share': 'operatingCashFlowPerShareTTM',
         'Free Cash Flow per Share': 'freeCashFlowPerShareTTM',
         'Cash per Share': 'cashPerShareTTM',
         'Operating Cash Flow to Sales Ratio': 'operatingCashFlowSalesRatioTTM',
         'Free Cash Flow to Operating Cash Flow Ratio': 'freeCashFlowOperatingCashFlowRatioTTM',
         'Cash Flow Coverage Ratio': 'cashFlowCoverageRatiosTTM',
         'Short-Term Coverage Ratio': 'shortTermCoverageRatiosTTM',
         'Capex Coverage Ratio': 'capitalExpenditureCoverageRatioTTM',
         'Dividend Paid & Capex Coverage Ratio': 'dividendPaidAndCapexCoverageRatioTTM',
         'Dividend Payout Ratio': 'payoutRatioTTM'}
}

FINANCIAL_RATIOS = {'formulas': FORMULAS, 'definitions': DEFINITIONS, 'ratios': RATIOS}

# p.106 of Brian Shannon - Technical Analysis Using Multiple Timeframes (2008)
TA_PERIODS = {
    '1wk': {'MA': [10, 20, 40], 'days': 365 * 2},
    '1d': {'MA': [10, 20, 50, 200], 'days': 180},
    '30m': {'MA': [7, 17, 33, 65], 'days': 20},
    '10m': {'MA': [20, 50, 100, 195], 'days': 7},
    '5m': {'MA': [40, 100, 200], 'days': 3},
    '2m': {'MA': [20, 50, 100], 'days': 1},
    '1m': {'MA': [50, 100, 200], 'days': 0}
    }
