
"""
Both STL decomposition and EMD (Empirical Mode Decomposition) decomposition are methods used for time series analysis in statistics.

STL decomposition (Seasonal and Trend decomposition using Loess) is a popular method used to decompose a time series into three components: trend, seasonal, and residual. The trend component represents the long-term trend of the time series, the seasonal component represents the cyclical variation that occurs at fixed intervals, and the residual component represents the random variation in the time series that cannot be explained by the trend and seasonal components. This method is particularly useful for data with strong seasonality and trends that are not linear.

On the other hand, EMD decomposition decomposes a time series into intrinsic mode functions (IMFs) that represent the different scales of variation present in the data. The method is based on the assumption that any signal can be decomposed into a finite number of IMFs, which are oscillatory functions with different characteristic scales. The final decomposition includes an IMF that represents the trend of the data and a set of IMFs that capture the cyclical variation at different scales. EMD is particularly useful for non-stationary and non-linear data.
"""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyEMD import EMD
from statsmodels.tsa.seasonal import seasonal_decompose

# Load stock price data
df = pd.read_csv('apple_stock_data.csv', index_col=0, parse_dates=True)

# Extract closing price data
signal = df['Close'].values

# Perform STL decomposition
stl_decomp = seasonal_decompose(signal, model='additive', period=252)

# Get trend, seasonal, and residual components
stl_trend = stl_decomp.trend
stl_trend = pd.Series(stl_trend).interpolate().values
stl_seasonal = stl_decomp.seasonal
stl_seasonal = pd.Series(stl_seasonal).interpolate().values
stl_residual = stl_decomp.resid
stl_residual = pd.Series(stl_residual).interpolate().values

# Initialize EMD object
emd = EMD()

# Decompose residual component into IMFs
IMFs = emd(signal)

# Select relevant IMFs for prediction
emd_trend = IMFs[0]
emd_trend = pd.Series(emd_trend).interpolate().values
emd_cyclical = IMFs[1:]
emd_cyclical = [pd.Series(imf).interpolate().values for imf in emd_cyclical]

# Define function to predict future prices
def predict_stock_price(stl_trend, stl_seasonal, emd_trend, emd_cyclical, days):
    # Extrapolate stl trend component
    stl_trend_ext = np.linspace(stl_trend[-1], stl_trend[-1]+(days/252)*(stl_trend[-1] - stl_trend[-252]), days)
    
    # Extrapolate stl seasonal component
    stl_seasonal_ext = np.tile(stl_seasonal[-252:], (int(np.ceil(days/252)), 1)).flatten()[:days]
    
    # Extrapolate emd trend component
    emd_trend_ext = np.linspace(emd_trend[-1], emd_trend[-1]+(days/252)*(emd_trend[-1] - emd_trend[-252]), days)
    
    # Extrapolate emd cyclical components
    emd_cyclical_ext = np.zeros((days, len(emd_cyclical)))
    for i in range(len(emd_cyclical)):
        emd_cyclical_ext[:, i] = np.linspace(emd_cyclical[i][-1], emd_cyclical[i][-1]+(days/252)*(emd_cyclical[i][-1] - emd_cyclical[i][-252]), days)
    
    # Combine extrapolated components
    signal_ext = (stl_trend_ext + emd_trend_ext + stl_seasonal_ext + np.sum(emd_cyclical_ext, axis=1)) / 2
    
    return signal_ext, stl_trend_ext, stl_seasonal_ext, emd_trend_ext, emd_cyclical_ext


# Number of days to predict
n_days = 30

# Predict the stock price for the next n days
predicted_prices, stl_predicted_trend, stl_predicted_seasonal, emd_predicted_trend, emd_predicted_cyclical = predict_stock_price(stl_trend, stl_seasonal, emd_trend, emd_cyclical, n_days)

# Create a new date range for the predicted prices
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_days, freq='D')

# Create a DataFrame for the predicted prices
predicted_df = pd.DataFrame({'Date': future_dates, 'Close': predicted_prices})
predicted_df = predicted_df.set_index('Date')

# Combine actual and predicted data
combined_df = pd.concat([df, predicted_df])

# Plot actual and predicted prices, and their components
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# Plot historical prices
axs[0].plot(df.index, df['Close'], label='Historical Prices', linestyle='-')

# Plot predicted prices
axs[0].plot(predicted_df.index, predicted_df['Close'], label='Predicted Prices', linestyle='--')

# Format the plot
axs[0].legend(loc='best')
axs[0].set_title('Actual and Predicted Stock Prices')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Price')

# Plot trend, seasonal, and cyclical components for predicted prices
axs[1].plot(df.index, stl_trend, label='Historical Trend', linestyle='-')
axs[1].plot(future_dates, stl_predicted_trend, label='Predicted Trend', linestyle='--')
axs[1].plot(df.index, stl_seasonal, label='Historical Seasonal', linestyle='-')
axs[1].plot(future_dates, stl_predicted_seasonal, label='Predicted Seasonal', linestyle='--')
axs[1].plot(df.index, stl_residual, label='Historical Residual', linestyle='-')
axs[1].plot(future_dates, np.zeros(len(future_dates)), label='Predicted Residual', linestyle='--')

# Format the plot
axs[1].legend(loc='best')
axs[1].set_title('Trend and Seasonality Components of Predicted Prices')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Price')

# Plot cyclical components for predicted prices
axs[2].plot(df.index, emd_trend, label='Historical Trend', linestyle='-')
axs[2].plot(future_dates, emd_predicted_trend, label='Predicted Trend', linestyle='--')
for i in range(1, len(emd_cyclical)):
    axs[2].plot(df.index, emd_cyclical[i], label='Historical Cyclical {}'.format(i+1), linestyle='-')
    axs[2].plot(future_dates, emd_predicted_cyclical[:, i], label='Predicted Cyclical {}'.format(i+1), linestyle='--')

# Format the plot
axs[2].legend(loc='best')
axs[2].set_title('Cyclical Components of Predicted Prices')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Price')

plt.tight_layout()
plt.show()





# In this example, we first perform STL decomposition on the original stock price data to get the trend, seasonal, and residual components. Then, we decompose the residual component using EMD to get the cyclical components. We use the extrapolated trend, seasonal, and cyclical components to predict the future stock prices, and combine them to get the predicted prices. Finally, we plot the actual and predicted prices, as well as their components, to visualize the prediction results.

# may use the predicted next value instead of the difference between the last two values

