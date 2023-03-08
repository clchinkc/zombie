
"""
Monte Carlo methods can be applied to stock price prediction by simulating future stock prices using random sampling from the historical price data. The basic idea is to generate a large number of possible future price trajectories and use the average or expected value of these trajectories as the predicted stock price.

Here is a possible approach using Monte Carlo simulation for stock price prediction:

Collect historical price data for the stock using yfinance API.
Choose a time horizon for the prediction (e.g., next 30 days).
Calculate the daily returns for the historical data.
Calculate the mean and standard deviation of the daily returns.
Generate a large number of possible future price trajectories by simulating daily returns using a normal distribution with the mean and standard deviation calculated in step 4.
For each simulated trajectory, calculate the final price at the end of the time horizon.
Calculate the average or expected value of the final prices from all the simulated trajectories.
Use the average or expected value as the predicted stock price for the end of the time horizon.
Plot the predicted stock price alongside with the history stock price using seaborn.

Write code in python.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

# Step 1: Collect historical price data for the stock
# stock = yf.Ticker("AAPL")
# data = stock.history(period="max")
# data.to_csv("apple_stock_data.csv")

# load data from csv file
data = pd.read_csv("apple_stock_data.csv", index_col="Date", parse_dates=True)

# Step 2: Choose a time horizon for the prediction
days = 30

# Step 3: Calculate the daily returns for the historical data
returns = data['Close'].pct_change().dropna()

# Step 4: Calculate the mean and standard deviation of the daily returns
mu = np.mean(returns)
sigma = np.std(returns)

# Step 5: Generate a large number of possible future price trajectories
simulations = 1000
last_price = data['Close'][-1]
price_matrix = np.zeros((days, simulations))
for d in range(days):
    if d == 0:
        price_matrix[d] = last_price
    else:
        shock = np.random.normal(mu, sigma, simulations)
        price_matrix[d] = price_matrix[d - 1] * (1 + shock)

# Step 6: For each simulated trajectory, calculate the final price at the end of the time horizon
simulations_df = pd.DataFrame(price_matrix)
last_prices = simulations_df.iloc[-1, :]

# Step 7: Calculate the average or expected value of the final prices from all the simulated trajectories
mean_price = last_prices.mean()

# Step 8: Plot the predicted stock price alongside with the history stock price using seaborn
sns.set()
plt.figure(figsize=(10, 6))
plt.hist(last_prices, bins=30)
plt.axvline(mean_price, color='r', linestyle='dashed', linewidth=2)
plt.title("Monte Carlo Simulation: AAPL")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Step 9: Calculate the mean price from all the simulated trajectories
mean_price = np.mean(price_matrix, axis=1)

# Step 10: Plot all the simulated price trajectories using a line plot
sns.set()
plt.figure(figsize=(10, 6))
plt.plot(price_matrix, color='gray', alpha=0.2)
plt.plot(mean_price, color='blue', linewidth=3)
plt.title("Monte Carlo Simulation: AAPL")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()