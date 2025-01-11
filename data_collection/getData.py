import yfinance as yf
import pandas as pd
import yahoo_fin.stock_info as si


#close prices
ticker_symbols = ["CM.TO","RY.TO","BMO.TO","TD.TO","BNS.TO","EQB.TO","NA.TO","POW.TO"]

close_data = pd.DataFrame({})
dividend_data = pd.DataFrame({})
pe_data = pd.DataFrame({})
# Fetch historical market data

for symbol in ticker_symbols :
    ticker = yf.Ticker(symbol)
    historical_data = ticker.history(period="2y")
    financials = ticker.financials
    close = historical_data["Close"]
    close_data[symbol] = close
    dividend_data[symbol] = ticker.get_dividends()
    pe_ratios = historical_data['Close'] / financials.loc['Net Income'][-1] 
    pe_ratios = pe_ratios.to_frame(name='P/E Ratio')
    
print(pe_ratios)



