
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
S0 = data.iloc[-1] # initial stock price

# Define exponential Brownian motion parameters
# Generate forecasts for the GARCH model
forecasts = res.forecast(horizon=N+1, reindex=False)
mu = np.array(forecasts.mean.values.ravel())
sigma = np.array((forecasts.residual_variance.values ** 0.5).ravel())
# Modify the drift and diffusion components to incorporate time-varying volatility and drift
drift = np.array(mu)
diffusion = np.array(sigma)*np.sqrt(dt)
W = np.random.normal(scale=np.sqrt(dt), size=N+1)

# Define jump diffusion parameters
lambd = 0.05 # jump intensity
jump_mean = 0.002 # jump size
jump_sd = 0.02 # jump size volatility

# Define a JIT-compiled function for simulating the stock price
@jit(nopython=True)
def EBM_jump(S0, mu, sigma, W, N, dt, lambd, jump_mean, jump_sd):
    S = np.zeros(N+1)
    S[0] = S0
    for i in range(1, N+1):
        
        # Add drift and diffusion components of the exponential Brownian motion
        drift = mu[i-1]*S[i-1]*dt
        diffusion = sigma[i-1]*S[i-1]*W[i]
        
        # Add jump component of the jump diffusion process
        jump_count = np.random.poisson(lambd*dt) # Number of jumps in the current time step
        jump_sizes = np.random.normal(jump_mean, jump_sd, jump_count) # Jump sizes
        jump_total = np.sum(jump_sizes) # Total jump effect
        
        S[i] = S[i-1] + drift + diffusion + S[i-1]*jump_total
    return S

# Simulate the exponential Brownian motion for each forecasted mean and volatility
S = EBM_jump(S0, drift, diffusion, W, N, dt, lambd, jump_mean, jump_sd)

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
