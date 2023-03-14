
"""
As for a project to predict stock prices, one possibility would be to use a stochastic process such as the Wiener process or exponential Brownian motion to model the behavior of stock prices over time. This would involve collecting historical data on the stock prices and using statistical methods to estimate the parameters of the model. The model could then be used to make predictions about future stock prices. Show me the Wiener process or exponential Brownian motion.

Sure, I can explain the Wiener process and exponential Brownian motion, which are both commonly used to model stock prices in finance.

The Wiener process, also known as Brownian motion, is a stochastic process that models the random movement of particles in a fluid. In finance, it is often used to model the price movements of stocks, where the stock price is assumed to follow a random walk. The Wiener process is characterized by the following properties:

The increments of the process are normally distributed, with mean zero and variance proportional to the time interval.

The process has independent increments, meaning that the future movements of the process are not affected by its past movements.

The process has continuous paths, meaning that it is a smooth function of time.

The Wiener process can be written mathematically as:

dS(t) = μ dt + σ dW(t)

where S(t) is the stock price at time t, μ is the expected return, σ is the volatility, dt is a small time interval, and dW(t) is a random variable that represents the increment of a Wiener process at time t.

The exponential Brownian motion is a modification of the Wiener process that takes into account the fact that stock prices are often subject to trends and mean reversion. It is characterized by the following properties:

The process has drift, meaning that it has a long-term mean that it tends to revert to over time.

The increments of the process are normally distributed, with mean equal to the drift and variance proportional to the time interval.

The process has independent increments and continuous paths, like the Wiener process.

The exponential Brownian motion can be written mathematically as:

dS(t) = μS(t) dt + σS(t) dW(t)

where S(t) is the stock price at time t, μ is the expected return, σ is the volatility, dt is a small time interval, and dW(t) is a random variable that represents the increment of a Wiener process at time t.

Both the Wiener process and exponential Brownian motion are stochastic processes that can be used to model the behavior of stock prices over time. By estimating the parameters of the model from historical data, we can make predictions about future stock prices. However, it's worth noting that stock prices are notoriously difficult to predict accurately, and any model is likely to have a degree of uncertainty associated with it.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit

# Load historical stock price data
data = pd.read_csv('apple_stock_data.csv', index_col=0, parse_dates=True)
data = data['Close']

# Calculate log returns
log_returns = np.log(data/data.shift(1))

# Estimate the parameters of the model
mu = np.mean(log_returns) # expected return
sigma = np.std(log_returns) # volatility
S0 = data.iloc[-1] # initial stock price

# Set simulation parameters
T = 30 # time horizon
N = 252 # number of time steps
dt = T/N # time step size

# Generate the Wiener process
t = np.linspace(0, T, N+1)
W = np.zeros(N+1)
W[0:] = np.random.normal(scale=np.sqrt(dt), size=N+1)

# Define a JIT-compiled function for simulating the exponential Brownian motion
@jit(nopython=True)
def EBM(S0, mu, sigma, W, N, dt):
    S = np.zeros(N+1)
    S[0] = S0
    for i in range(1, N+1):
        drift = mu*S[i-1]*dt
        diffusion = sigma*S[i-1]*W[i]
        S[i] = S[i-1] + drift + diffusion
    return S

# Simulate the exponential Brownian motion
S = EBM(S0, mu, sigma, W, N, dt)

# Plot the historical data
plt.plot(data.index, data.values, label='Historical Data')

# Plot the simulated exponential Brownian motion
plt.plot(data.index[-1] + pd.to_timedelta(t, unit='D'), S, label='Simulated EBM')

# Add labels and legend to the plot
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.title('Apple Stock Price Simulation')
plt.legend()

# Show the plot
plt.show()


