
"""
Markov Chain Monte Carlo methods can be applied to stock price prediction by generate future price trajectories using Metropolis-Hastings algorithm depending on the historical price data. The basic idea is to generate a large number of possible future price trajectories and use the average or expected value of these trajectories as the predicted stock price.

Here is a possible approach using Markov Chain Monte Carlo (MCMC) simulation for stock price prediction:
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

# Step 5: Define the Metropolis-Hastings algorithm for the MCMC simulation
def metropolis_hastings(likelihood_func, prior_sampler, proposal_sampler, param_init, iterations):
    # Initialize the chain
    chain = np.zeros(iterations)
    # Set the initial parameter value
    param = param_init
    # Calculate the likelihood of the initial parameter value
    likelihood = likelihood_func(param)
    # Add the initial parameter value to the chain
    chain[0] = param
    # Run the chain
    for i in range(1, iterations):
        # Sample a new parameter from the proposal distribution
        proposal = proposal_sampler(param)
        # Calculate the likelihood of the proposed parameter value
        proposal_likelihood = likelihood_func(proposal)
        # Calculate the acceptance ratio
        acceptance_ratio = min(1, proposal_likelihood / likelihood)
        # Generate a uniform random number between 0 and 1
        u = np.random.uniform(0, 1)
        # Accept or reject the proposal based on the acceptance ratio and the random number
        if u < acceptance_ratio:
            param = proposal
            likelihood = proposal_likelihood
        # Add the parameter value to the chain
        chain[i] = param
    return chain 

# Define the likelihood function for the MCMC simulation
def likelihood(param):
    # Simulate the price trajectory using a random walk process
    price = data['Close'][-1]
    for i in range(days):
        shock = np.random.normal(mu, sigma)
        proposal = price * (1 + shock)
        acceptance_ratio = min(1, np.exp(-0.5 * ((np.log(proposal) - np.log(price)) / sigma)**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)))
        u = np.random.uniform(0, 1)
        if u < acceptance_ratio:
            price = proposal
    # Calculate the likelihood of the final price
    log_likelihood = -0.5 * ((np.log(price) - np.log(last_price)) / sigma)**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    return np.exp(log_likelihood)

# Define the prior sampler for the MCMC simulation
def prior_sampler():
    return np.random.normal(last_price, sigma)

# Define the proposal sampler for the MCMC simulation
def proposal_sampler(param):
    return np.random.normal(param, sigma)

# Step 6: Generate a large number of possible future price trajectories using the MCMC simulation
simulations = 1000
last_price = data['Close'][-1]
price_matrix = np.zeros((days, simulations))
for d in range(days):
    if d == 0:
        price_matrix[d] = last_price
    else:
        # Run the MCMC simulation
        chain = metropolis_hastings(likelihood, prior_sampler, proposal_sampler, last_price, simulations)
        price_matrix[d] = chain
        last_price = chain[-1]
        
# Step 7: For each simulated trajectory, calculate the final price at the end of the time horizon and the median or expected value of the final prices from all the simulated trajectories
simulations_df = pd.DataFrame(price_matrix)
last_prices = simulations_df.iloc[-1, :]
mean_price = last_prices.mean()

# Step 8: Plot the predicted stock price alongside with the history stock price using seaborn
sns.set()
plt.figure(figsize=(10, 6))
plt.hist(last_prices, bins=30)
plt.axvline(mean_price, color='r', linestyle='dashed', linewidth=2)
plt.title("MCMC Simulation: AAPL")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Step 9: Calculate the median price from all the simulated trajectories
median_price = np.median(price_matrix, axis=1)
# 25th percentile
lower_price = np.percentile(price_matrix, 25, axis=1)
# 75th percentile
upper_price = np.percentile(price_matrix, 75, axis=1)

# Step 10: Plot all the simulated price trajectories using a line plot
plt.figure(figsize=(10, 6))
plt.plot(simulations_df, color='gray', alpha=0.25)
plt.plot(median_price, color='blue', linewidth=2)
plt.plot(lower_price, color='red', linewidth=1)
plt.plot(upper_price, color='green', linewidth=1)
plt.title("MCMC Simulation: AAPL")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()
