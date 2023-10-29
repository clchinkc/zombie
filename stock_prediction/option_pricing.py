
import numpy as np


def binomial_tree(S, K, r, t, T, sigma, n, option_type):
    """
    Calculate the price of a European call or put option using the binomial tree model.
    
    S: float, the current stock price
    K: float, the strike price of the option
    r: float, the risk-free interest rate
    t: float, the current time
    T: float, the expiration time of the option
    sigma: float, the volatility of the stock price
    n: int, the number of time periods in the binomial tree
    option_type: str, "call" or "put"
    
    Returns the price of the option.
    """
    
    dt = (T - t) / n  # size of each time step
    u = np.exp(sigma * np.sqrt(dt))  # up factor
    d = 1 / u  # down factor
    p = (np.exp(r * dt) - d) / (u - d)  # probability of an up move
    q = 1 - p  # probability of a down move
    
    # Calculate stock prices at each node of the tree
    stock_prices = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            stock_prices[j, i] = S * (u ** (i - j)) * (d ** j)
    
    # Calculate option values at each node of the tree
    option_values = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        if option_type == "call":
            option_values[i, n] = max(0, stock_prices[i, n] - K)
        else:
            option_values[i, n] = max(0, K - stock_prices[i, n])
    
    # Work backwards through the tree to calculate option values at earlier time periods
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            if option_type == "call":
                option_values[j, i] = np.exp(-r * dt) * (p * option_values[j, i + 1] + q * option_values[j + 1, i + 1])
            else:
                option_values[j, i] = np.exp(-r * dt) * (p * option_values[j, i + 1] + q * option_values[j + 1, i + 1])
    
    return option_values[0, 0]


S = 100  # current stock price
K = 110  # strike price
r = 0.05  # risk-free interest rate
t = 0  # current time
T = 10  # expiration time
sigma = 0.2  # volatility
n = 100  # number of time periods
option_type = "call"  # option type

price = binomial_tree(S, K, r, t, T, sigma, n, option_type)
print(price)



import math

import numpy as np
from scipy.stats import norm


def black_scholes_merton(S, K, r, t, T, sigma, option_type):
    """
    Calculate the price of a European call or put option using the Black-Scholes-Merton model.
    
    S: float, the current stock price
    K: float, the strike price of the option
    r: float, the risk-free interest rate
    t: float, the current time
    T: float, the expiration time of the option
    sigma: float, the volatility of the stock price
    option_type: str, "call" or "put"
    
    Returns the price of the option.
    """
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * math.sqrt(T - t))
    d2 = d1 - sigma * math.sqrt(T - t)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * (T - t)) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * (T - t)) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

S = 100  # current stock price
K = 110  # strike price
r = 0.05  # risk-free interest rate
t = 0  # current time
T = 10  # expiration time
sigma = 0.2  # volatility
# n = 100  # number of time periods
option_type = "call"  # option type

price = black_scholes_merton(S, K, r, t, T, sigma, option_type)
print(price)


