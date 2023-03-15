


import yfinance as yf
import pandas as pd

symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2022-12-31'

# Use yfinance API to fetch historical stock price data
ticker = yf.Ticker(symbol)
data = ticker.history(start=start_date, end=end_date)
# Store it in apple_stock_data.csv
data.to_csv('apple_stock_data.csv')