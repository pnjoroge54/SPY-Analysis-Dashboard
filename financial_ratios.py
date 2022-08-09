'''
For more information on the calculation of financial ratios
visit https://financialmodelingprep.com/financial-ratios/
'''        

FORMULAS = {
    'Investment Valuation Ratios':
        {'Price to Earnings Ratio': r'$P/E = \frac{StockPrice/Share}{EPS}$', 
         'Price to Book Value Ratio': r'$\frac{StockPrice/Share}{Equity/Share}$', 
         'Price to Sales Ratio': 'priceToSalesRatioTTM', 
         'Price to Earnings to Growth Ratio': 'priceEarningsToGrowthRatioTTM',
         'Price to Free Cash Flows Ratio': r'$\frac{StockPrice/Share}{FreeCashFlow/Share}$', 
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

MEANINGS = {
    'Investment Valuation Ratios':
        {'Price to Earnings Ratio':
         '''
         The financial reporting of both companies and investment research services use a basic 
         earnings per share (EPS) figure divided into the current stock price to calculate the P/E 
         multiple (i.e. how many times a stock is trading (its price) per each dollar of EPS).
         ''', 

         'Price to Book Value Ratio':
         '''
         The price-to-book value ratio, expressed as a multiple (i.e. how many times a company's \n
         stock is trading per share compared to the company's book value per share), is an \n
         indication of how much shareholders are paying for the net assets of a company.
         ''', 

         'Price to Sales Ratio':
         '''
         The P/E ratio and P/S reflects how many times investors are paying for every dollar of a \n
         company's sales. Since earnings are subject, to one degree or another, to accounting \n
         estimates and management manipulation, many investors consider a company's sales (revenue) \n
         figure a more reliable ratio component in calculating a stock's price multiple than the \n
         earnings figure.
         ''', 

         'Price to Earnings to Growth Ratio':
         '''
         The PEG ratio is a refinement of the P/E ratio and factors in a stock's estimated earnings \n
         growth into its current valuation.The general consensus is that if the PEG ratio indicates \n
         a value of 1, this means that the market is correctly valuing (the current P/E ratio) a \n
         stock in accordance with the stock's current estimated earnings per share growth. If the \n
         PEG ratio is less than 1, this means that EPS growth is potentially able to surpass the \n
         market's current valuation. In other words, the stock's price is being undervalued. On \n
         the other hand, stocks with high PEG ratios can indicate just the opposite - that the \n
         stock is currently overvalued.
         ''',

         'Price to Free Cash Flows Ratio':
         '''
         The price/free cash flow ratio is used by investors to evaluate the investment \n
         attractiveness, from a value standpoint, of a company's stock.
         ''', 

         'Enterprise Value Multiplier':
         '''
         Overall, this measurement allows investors to assess a company on the same basis as that \n
         of an acquirer. As a rough calculation, enterprise value multiple serves as a proxy for \n
         how long it would take for an acquisition to earn enough to pay off its costs in \n
         years(assuming no change in EBITDA).
         ''',

         'Dividend Yield':
         '''
         Income investors value a dividend-paying stock, while growth investors have little \n
         interest in dividends, preferring to capture large capital gains. Whatever your \n
         investing style, it is a matter of historical record that dividend-paying stocks \n
         have performed better than non-paying-dividend stocks over the long term.
         '''
        },

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
