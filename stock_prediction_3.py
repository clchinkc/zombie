

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Data Collection
df = pd.read_csv('apple_stock_data.csv', index_col='Date', parse_dates=True)

# Step 2: Data Preprocessing
# Check for missing values
if df.isnull().values.any():
    df = df.dropna()

# Step 3: Model Selection
# Plot autocorrelation and partial autocorrelation plots to determine ARIMA model parameters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df['Close'])
plot_pacf(df['Close'])

# From the autocorrelation plot, it looks like there's a high correlation between the current and past values up to around 10 lags. 
# From the partial autocorrelation plot, it looks like there's a significant correlation with the first lag.

# So, we can select an ARIMA(10, 1, 1) model
model = ARIMA(df['Close'], order=(10, 1, 1))

# Step 4: Parameter Estimation
model_fit = model.fit()

# Step 5: Model Validation
# Evaluate the model using AIC and BIC scores
print(model_fit.summary())

# Step 6: Forecasting
# Forecast the future values using the trained model
forecasted_values = model_fit.forecast(steps=30)

# Step 7: Performance Evaluation
# Evaluate the model's performance using MSE and RMSE
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(df['Close'][-30:], forecasted_values)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)

# Plot the forecasted values against the actual values
plt.figure(figsize=(10, 6))
plt.plot(df['Close'][-30:], label='Actual')
plt.plot(forecasted_values, label='Forecasted')
plt.legend()
plt.show()
"""

"""
import matplotlib.pyplot as plt
import numpy as np

# Importing libraries
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Reading the dataset
data = pd.read_csv('apple_stock_data.csv', index_col='Date', parse_dates=True)

# Plotting the dataset
plt.figure(figsize=(10,5))
plt.plot(data['Close'])
plt.title('Stock Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.show()

# Splitting the dataset into training and testing data
split_index = len(data['Close']) - 90 # using last 90 days as test data
train_data = data['Close'][:split_index]
test_data = data['Close'][split_index:]

# Creating the ARIMA model
arima_model = ARIMA(train_data, order=(2,1,2))

# Fitting the ARIMA model
arima_result = arima_model.fit()

# Predicting the stock prices
predictions = arima_result.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, typ='levels')

# Plotting the predictions and the actual data
plt.figure(figsize=(10,5))
plt.plot(test_data, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()
"""


import matplotlib.pyplot as plt
import numpy as np

# Import necessary libraries
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Load the stock price data
df = pd.read_csv('apple_stock_data.csv', index_col=0, parse_dates=True)

# Check for missing values and outliers
df.isna().sum()
df.plot(figsize=(12,6))
plt.show()

# Check for stationarity using ADF test
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['Close'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# If the data is not stationary, take first differences
diff = df['Close'].diff().dropna()

# Check for stationarity again
result = adfuller(diff)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Split the data into training and testing sets
train_size = int(len(diff) * 0.8)
train, test = diff[:train_size], diff[train_size:]

# Determine the order of differencing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train)
plot_pacf(train)
plt.show()

# Based on the ACF and PACF plots, choose d = 1

# Determine the order of the ARIMA model
model = ARIMA(train, order=(1,1,0))
model_fit = model.fit()

# Evaluate the ARIMA model
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
error = mean_squared_error(test, predictions)
print(f'RMSE: {np.sqrt(error)}')

# Visualize the predicted vs actual values
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

# Forecast future stock prices
future_dates = pd.date_range(start='2022-03-08', end='2023-03-08')
future_values = model_fit.forecast(len(future_dates), typ='levels')
plt.plot(df['Close'].diff())
plt.plot(future_dates, future_values, color='red')
plt.show()
