import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Get data from Yahoo Finance
df = yf.download('^HSI', start="2017-01-01", end="2021-12-31")
df.to_csv('^HSI.csv')

# Read data from CSV file
df = pd.read_csv('^HSI.csv', parse_dates=True, index_col=0)

# Calculate the 100-day moving average
df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()

# Drop rows with missing values
df.dropna(inplace=True)

# Resample the data to get OHLC and volume data for each 10-day period
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Plot the adjusted close and 100-day moving average on the first axis
ax1.plot(df.index, df['Adj Close'], label='Adj Close')
ax1.plot(df.index, df['100ma'], label='100-day MA')
ax1.legend()

# Plot the volume data on the second axis
ax2.bar(df.index, df['Volume'], label='Volume')
ax2.legend()

# Set the x-axis to display dates
ax1.xaxis_date()
ax2.xaxis_date()

# Show the plot
plt.show()


# current data

# Get the Hang Seng Index
hsi = yf.Ticker("^HSI")

# Get the index information
index_info = hsi.info

# Print all the information from the info dictionary
for key, value in index_info.items():
    print(f"{key}: {value}")

# Print the PB ratio, PE ratio, dividend yield, Earnings per share, Return on assets, and Return on equity
print(f"PB ratio: {index_info['priceToBook']}")
print(f"PE ratio: {index_info['trailingPE']}")
print(f"Dividend yield: {index_info['dividendYield']}")
print(f"EPS: {index_info['regularMarketEPS']}")
print(f"ROA: {index_info['returnOnAssets']}")
print(f"ROE: {index_info['returnOnEquity']}")

# financial data

# Get the Hang Seng Index
hsi = yf.Ticker("^HSI")

# Get the financials data
financial_info = hsi.financials

# Print all the information from the financial dictionary
for key, value in financial_info.items():
    print(f"{key}: {value}")

# get specific stock data from hsi
# get "0700.HK" stock data from hsi
# Read the stock data from the CSV file
stock_data = pd.read_csv('^HSI.csv', index_col='Date', parse_dates=True)

# Get the stock data for the stock with ticker symbol "0700.HK"
stock_data = stock_data[stock_data['Ticker'] == "0700.HK"]

# Print the stock data
print(stock_data)


# get 0700.HK data separately
stock = yf.Ticker("0700.HK")
# get historical market data
hist = stock.history(period="max")
# show actions (dividends, splits)
actions = stock.actions
# show dividends
dividends = stock.dividends
# show splits
splits = stock.splits
# show financials
financials = stock.financials
# show major holders
major_holders = stock.major_holders
# show institutional holders
institutional_holders = stock.institutional_holders
# show balance sheet
balance_sheet = stock.balance_sheet
# show cashflow
cashflow = stock.cashflow
# show earnings
earnings = stock.earnings
# show sustainability
sustainability = stock.sustainability
# show analysts recommendations
recommendations = stock.recommendations
# show next event (earnings, etc)
calendar = stock.calendar
# show ISIN code - *experimental*
isin = stock.isin
# show options expirations
options = stock.options

# print all the information
print(f"hist: {hist}")
print(f"actions: {actions}")
print(f"dividends: {dividends}")
print(f"splits: {splits}")
print(f"financials: {financials}")
print(f"major_holders: {major_holders}")
print(f"institutional_holders: {institutional_holders}")
print(f"balance_sheet: {balance_sheet}")
print(f"cashflow: {cashflow}")
print(f"earnings: {earnings}")
print(f"sustainability: {sustainability}")
print(f"recommendations: {recommendations}")
print(f"calendar: {calendar}")
print(f"isin: {isin}")
print(f"options: {options}")