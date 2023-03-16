
"""
There are several time series models that you can use to predict stock prices, 
including ARIMA (Autoregressive Integrated Moving Average), SARIMA (Seasonal ARIMA), and Prophet. 
Each model has its strengths and weaknesses, and you should choose the one that best fits your data.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyrsistent import m
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Step 1: Data Collection
df = pd.read_csv('apple_stock_data.csv', index_col=0, parse_dates=True)

"""
# Step 2: Data Preprocessing
# Check for missing values and outliers
if df.isna().values.any():
    df.plot()
    plt.show()
# Check for missing values
if df.isnull().values.any():
    df = df.dropna()

# Step 3: Data Visualization
plt.figure(figsize=(10,5))
plt.plot(df['Close'])
plt.title('Stock Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.show()
"""

# Step 4: Check for Stationarity
result = adfuller(df['Close'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# If the data is not stationary, take first differences
diff = df['Close'].diff().dropna()
diff.index = pd.to_datetime(diff.index, utc=True)

# Check for stationarity again
result = adfuller(diff)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Step 5: Split the data into training and testing sets
# using last 90 days as test data
split_index = len(diff) - 90
train_data, test_data = diff[:split_index], diff[split_index:]

"""
# Step 6: Determine the order of differencing
plot_acf(train_data)
plot_pacf(train_data, method='ywm')
plt.show()
"""

# The parameters of the ARIMA model are defined as follows:
# p: The number of lag observations included in the model, also called the lag order.
# d: The number of times that the raw observations are differenced, also called the degree of differencing.
# q: The size of the moving average window, also called the order of moving average.
# Based on the ACF and PACF plots, choose the order of the ARIMA model
# Step 7: Determine the order of the ARIMA model
model = ARIMA(train_data, order=(3, 1, 3))
# Perform parameter estimation
model_fit = model.fit()
# print(model_fit.summary())

# Step 8: Evaluate the ARIMA model
# Evaluate the model using MSE and RMSE
predictions_arima = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, typ='levels')

# Step 9: Evaluate the model using MSE and RMSE
error = mean_squared_error(test_data, predictions_arima)
print(f'RMSE: {np.sqrt(error)}')

# Step 10: Visualize the predictions and the actual values
plt.figure(figsize=(10,5))
plt.plot(test_data.index, test_data, color='blue', label='Actual')
plt.plot(test_data.index, predictions_arima, color='red', label='Predicted')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()


# Step 11: Forecast future stock prices from the last date in the dataset
future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')

# Forecast the future values using the ARIMA model
future_values_arima = model_fit.forecast(steps=len(future_dates), typ='levels')

# Visualize the predictions
plt.figure(figsize=(10,5))
plt.plot(future_dates, future_values_arima, color='red', label='Forecasted')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Invert the differencing to get the forecasted stock prices of the next year along with the historical data
forecasted = df['Close'].iloc[-1] + future_values_arima
plt.figure(figsize=(10,5))
plt.plot(df['Close'], color='blue', label='Historical')
plt.plot(future_dates, forecasted, color='red', label='Forecasted')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()



# https://www.ichdata.com/use-python-to-do-time-series.html
# https://medium.com/@raj.saha3382/forecasting-of-stock-market-using-arima-in-python-cd4fe76fc58a
