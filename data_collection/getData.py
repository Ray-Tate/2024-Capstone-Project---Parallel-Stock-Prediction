import yfinance as yf
import pandas as pd


#close prices
ticker_symbols = ["CM.TO","RY.TO","BMO.TO","TD.TO","BNS.TO","EQB.TO","NA.TO"]

close_data = pd.DataFrame({})

# Fetch historical market data
for symbol in ticker_symbols :
    ticker = yf.Ticker(symbol)
    historical_data = ticker.history(period="2y")  # data for the last year
    close = historical_data["Close"]
    close_data[symbol] = close
print(historical_data)
print(close_data)

