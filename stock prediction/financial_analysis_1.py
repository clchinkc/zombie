
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prophet
import yfinance as yf
from pmdarima.arima import auto_arima
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('apple_stock_data.csv')
dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None) for date in data['Date'].values]
data['Date'] = pd.to_datetime(dates)


# Plot daily closing prices, daily returns, and volatility

# plot daily closing prices
plt.plot(data['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.title('AAPL Daily Closing Prices')
plt.show()


# calculate daily returns
daily_returns = data['Close'].pct_change()

# plot histogram of daily returns
plt.hist(daily_returns, bins=50)
plt.xlabel('Daily Returns')
plt.ylabel('Frequency')
plt.title('AAPL Daily Returns')
plt.show()


# calculate volatility using rolling standard deviation
volatility = daily_returns.rolling(window=30).std()

# plot volatility
plt.plot(volatility)
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('AAPL Volatility')
plt.show()



# Financial risk estimation with Value-at-Risk (VaR)

def value_at_risk(data, confidence_level=0.95):
    log_returns = np.log(data / data.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()

    var = norm.ppf(1 - confidence_level, mu, sigma)
    return var

stock_var = value_at_risk(closing_prices)
print(f"Value at Risk (VaR) : {stock_var:.4f}")

# Financial risk estimation with Conditional Value-at-Risk (CVaR)

def conditional_value_at_risk(data, confidence_level=0.95):
    log_returns = np.log(data / data.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()

    var = norm.ppf(1 - confidence_level, mu, sigma)
    cvar = log_returns[log_returns <= var].mean()
    return cvar

stock_cvar = conditional_value_at_risk(closing_prices)
print(f"Conditional Value at Risk (CVaR) : {stock_cvar:.4f}")


# Financial risk estimation with Sharpe ratio

def sharpe_ratio_plot(data, risk_free_rate=0.02):
    log_returns = np.log(data / data.shift(1)).fillna(np.mean(np.log(data / data.shift(1)))).values.reshape(-1, 1)
    
    scaler = StandardScaler()
    log_returns = scaler.fit_transform(log_returns)
    
    returns = log_returns.mean()
    volatility = log_returns.std()
    sharpe_ratios = (returns - risk_free_rate) / volatility
    
    return returns * 252, volatility * np.sqrt(252), sharpe_ratios * np.sqrt(252)

stock_returns, stock_volatility, stock_sharpe_ratios = sharpe_ratio_plot(closing_prices)

plt.figure(figsize=(12, 6))
plt.scatter(stock_volatility, stock_returns, c=stock_sharpe_ratios, cmap='RdYlGn')
plt.colorbar()
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()


# Portfolio optimization using Markowitz model
from scipy.optimize import minimize


def portfolio_optimization(symbols):
    data = yf.download(symbols)['Adj Close']
    
    log_returns = np.log(data/data.shift(1)).dropna()
    
    num_assets = len(symbols)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights)))

    def negative_sharpe_ratio(weights):
        return -(np.sum(log_returns.mean() * weights) * 252 - 0.02) / portfolio_volatility(weights)

    def check_sum(weights):
        return np.sum(weights) - 1

    constraints = ({'type': 'eq', 'fun': check_sum})
    bounds = [(0, 1) for i in range(num_assets)]

    opt_results = minimize(negative_sharpe_ratio, weights, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = opt_results.x
    
    return opt_weights

symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
opt_weights = portfolio_optimization(symbols)
print(opt_weights)


# Portfolio optimization using Efficient Frontier
from scipy.optimize import minimize


def portfolio_optimization(symbols):
    data = yf.download(symbols)['Adj Close']
    
    log_returns = np.log(data/data.shift(1)).dropna()
    
    num_assets = len(symbols)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights)))

    def negative_sharpe_ratio(weights):
        return -(np.sum(log_returns.mean() * weights) * 252 - 0.02) / portfolio_volatility(weights)

    def check_sum(weights):
        return np.sum(weights) - 1

    constraints = ({'type': 'eq', 'fun': check_sum})
    bounds = [(0, 1) for i in range(num_assets)]
    initial_guess = num_assets * [1 / num_assets]

    optimized_weights = minimize(negative_sharpe_ratio, initial_guess, bounds=bounds, constraints=constraints)

    return optimized_weights.x

symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
optimized_weights = portfolio_optimization(symbols)
print("Optimized portfolio weights:", optimized_weights)






# https://www.youtube.com/watch?v=pryXhOgDY9A
