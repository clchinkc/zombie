
"""
There are several time series models that you can use to predict stock prices, 
including ARIMA (Autoregressive Integrated Moving Average), SARIMA (Seasonal ARIMA), and Prophet. 
Each model has its strengths and weaknesses, and you should choose the one that best fits your data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Step 1: Data Collection
df = pd.read_csv('apple_stock_data.csv', index_col=0, parse_dates=True)

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

# Step 4: Check for Stationarity
result = adfuller(df['Close'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# If the data is not stationary, take first differences
diff = df['Close'].diff().dropna()

# Check for stationarity again
result = adfuller(diff)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Step 5: Split the data into training and testing sets
# using last 90 days as test data
split_index = len(diff) - 90
train_data, test_data = diff[:split_index], diff[split_index:]


# Step 6: Determine the order of differencing
plot_acf(train_data)
plot_pacf(train_data)
plt.show()

# Because the autocorrelation plot shows there's a high correlation between the current and past values up to around 10 lags
# the partial autocorrelation plot shows a significant lag at 3
# Based on the ACF and PACF plots, choose an ARIMA(10, 1, 3) model
# Step 7: Determine the order of the ARIMA model
model = ARIMA(train_data, order=(10, 1, 3))
# Perform parameter estimation
model_fit = model.fit()

# Step 8: Evaluate the ARIMA model
# Evaluate the model using MSE and RMSE
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, typ='levels')
error = mean_squared_error(test_data, predictions)
print(f'RMSE: {np.sqrt(error)}')
# Evaluate the model using AIC and BIC scores
print(model_fit.summary())

# Visualize the predictions and the actual values
plt.figure(figsize=(10,5))
plt.plot(test_data.index, test_data, color='blue', label='Actual')
plt.plot(predictions.index, predictions, color='red', label='Predicted')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Predicted values should be set with the same index as the actual values

# Step 9: Forecast future stock prices
future_dates = pd.date_range(start='2022-03-08', end='2023-03-08')
# Forecast the future values using the trained model
future_values = model_fit.forecast(steps=len(future_dates), typ='levels')
plt.plot(df['Close'].diff(), label='Actual')
plt.plot(future_dates, future_values, color='red', label='Forecasted')
plt.show()

# ValueError: x and y must have same first dimension, but have shapes (366,) and (30,)
# The error is because the future dates have a different shape than the actual values


# Step 10: Invert the differencing to get the forecasted stock prices
