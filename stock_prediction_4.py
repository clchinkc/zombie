
import arch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit

# Load historical stock price data
data = pd.read_csv('apple_stock_data.csv', index_col=0, parse_dates=True)
data = data['Close']

log_returns = np.log(data/data.shift(1)).fillna(np.mean(np.log(data/data.shift(1))))

# Estimate the parameters of the GARCH model
garch_model = arch.arch_model(log_returns, vol='GARCH', p=1, o=0, q=1, rescale=False)
res = garch_model.fit(disp='off')

# Set simulation parameters
T = 365 # time horizon
N = 365 # number of time steps
dt = T/N # time step size
t = np.linspace(0, T, N+1)
W = np.random.normal(scale=np.sqrt(dt), size=N+1)
S0 = data.iloc[-1] # initial stock price

# Define a JIT-compiled function for simulating the exponential Brownian motion
# @jit(nopython=True)
def EBM(S0, mu, sigma, W, N, dt):
    S = np.zeros(N+1)
    S[0] = S0
    for i in range(1, N+1):
        drift = mu[i-1]*S[i-1]*dt
        diffusion = sigma[i-1]*S[i-1]*W[i]
        S[i] = S[i-1] + drift + diffusion
    return S

# Generate forecasts for the GARCH model
forecasts = res.forecast(horizon=N+1, reindex=False)
mu = np.array(forecasts.mean.values.ravel())
sigma = np.array((forecasts.residual_variance.values ** 0.5).ravel())

# Modify the drift and diffusion components to incorporate time-varying volatility and drift
drift = np.array(mu)
diffusion = np.array(sigma)*np.sqrt(dt)

# Simulate the exponential Brownian motion for each forecasted mean and volatility
S = EBM(S0, drift, diffusion, W, N, dt)

# Plot the historical data
plt.plot(data.index, data.values, label='Historical Data')

# Plot the forecasted prices
forecast_dates = pd.date_range(start=data.index[-1], periods=N+1, freq='D')
plt.plot(forecast_dates, S.T, label='Forecasted Prices')

# Add labels and legend to the plot
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.title('Apple Stock Price Forecast')
plt.legend()

# Show the plot
plt.show()


# https://arch.readthedocs.io/en/latest/univariate/introduction.html
