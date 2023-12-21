import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

# Fetch historical data
data = pd.read_csv("apple_stock_data.csv")

# Simple Moving Average (SMA) strategy
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Signal for long position (SMA_50 crosses above SMA_200)
data['Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1.0, 0.0)

# Shift the signal by one day to simulate real trading scenario
data['Signal'] = data['Signal'].shift()

# Backtest the strategy
data['Returns'] = data['Close'].pct_change()
data['Strategy_Returns'] = data['Returns'] * data['Signal']

# Cumulative returns
data['Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()

# Plot the results
data[['Cumulative_Returns']].plot(figsize=(10, 5))
plt.show()