import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyvalet import ValetInterpreter
import stats_can

#close prices
ticker_symbols = ["CM.TO","RY.TO","BMO.TO","TD.TO","BNS.TO","EQB.TO","NA.TO","POW.TO"]
START_DATE = "2014-01-01"
END_DATE = "2019-01-30"

full_date_range = pd.date_range(start=START_DATE, end=END_DATE)
close_data = pd.DataFrame({})
dividend_data = pd.DataFrame({})

if True:
    symbol = "RY.TO"
    ticker = yf.Ticker(symbol)
    historical_data = ticker.history(start=START_DATE, end=END_DATE)
    dividend_data = historical_data['Dividends']
    dividend_data =pd.DataFrame({"Date": pd.DatetimeIndex(dividend_data.axes[0].date), "value": dividend_data.values}) 
    print(dividend_data)

if False:
    boc = ValetInterpreter()
    interest_rates = boc.get_series_observations('V39079', start_date=START_DATE, end_date=END_DATE, response_format='csv')
    interest_rates['date'] = pd.to_datetime(interest_rates['date'])
    interest_rates = interest_rates[interest_rates['date'].isin(dividend_data.axes[0].date)]
    interest_rates = [[float(rate)] for rate in interest_rates['V39079'].values]
    print(interest_rates)
    
if True:   
    ticker = yf.Ticker("CAD=X")
    historical_data = ticker.history(start=START_DATE, end=END_DATE)
    USD_CAD_exhcange_rate = historical_data['Close']
    USD_CAD_exhcange_rate =pd.DataFrame({"Date": pd.DatetimeIndex(USD_CAD_exhcange_rate.axes[0].date), "value": USD_CAD_exhcange_rate.values})
    USD_CAD_exhcange_rate = USD_CAD_exhcange_rate[USD_CAD_exhcange_rate['Date'].isin(dividend_data['Date'])]
    date_list = np.union1d(USD_CAD_exhcange_rate['Date'].values, dividend_data['Date'].values)
    date_list_df = pd.DataFrame({"Date": pd.to_datetime(date_list)})
    USD_CAD_exhcange_rate = pd.merge(date_list_df,USD_CAD_exhcange_rate, on="Date",how="left")
    USD_CAD_exhcange_rate['value'] = USD_CAD_exhcange_rate['value'].ffill()
    print(USD_CAD_exhcange_rate)
    

if False:
    sc = stats_can.StatsCan()
    gdp_data = sc.table_to_df('36-10-0104-01')
    gdp_data['REF_DATE'] = gdp_data['REF_DATE'].astype(str)
    gdp_growth = gdp_data[gdp_data['REF_DATE'].str.startswith('202')][['REF_DATE', 'VALUE']] 
    gdp_growth['REF_DATE'] = pd.to_datetime(gdp_growth['REF_DATE'])
    print(gdp_growth)