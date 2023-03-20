


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from numba import jit
from scipy.stats import norm


# Collect historical price data for the stock
def collect_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

def load_stock_data(file_name):
    data = pd.read_csv(file_name, index_col="Date", parse_dates=True)
    return data

def calculate_daily_returns(prices):
    returns = prices.pct_change().dropna()
    mu = np.mean(returns) # expected return
    sigma = np.std(returns) # volatility
    theta = mu - 0.5 * sigma ** 2 # mean reversion level
    return mu, sigma, theta

# Define the Metropolis-Hastings algorithm for the MCMC simulation
def metropolis_hastings(likelihood_func, proposal_sampler, last_price, num_iterations):
    # Initialize the chain
    chain = np.zeros(num_iterations)
    # Set the initial parameter value
    previous_price = last_price
    # Calculate the likelihood of the initial parameter value
    likelihood = likelihood_func(previous_price, last_price, mu, sigma, theta)
    # Add the initial parameter value to the chain
    chain[0] = previous_price
    # Run the chain
    for i in range(1, num_iterations):
        # Sample a new parameter from the proposal distribution
        proposal = proposal_sampler(previous_price, mu, sigma)
        # Calculate the likelihood of the proposed parameter value
        proposal_likelihood = likelihood_func(proposal, previous_price, mu, sigma, theta)
        # Calculate the acceptance ratio
        acceptance_ratio = min(1, proposal_likelihood / likelihood)
        # Generate a uniform random number between 0 and 1
        u = np.random.uniform(0, 1)
        # Accept or reject the proposal based on the acceptance ratio and the random number
        if u < acceptance_ratio:
            previous_price = proposal
            likelihood = proposal_likelihood
        # Add the parameter value to the chain
        chain[i] = previous_price
    return chain 

# Define the likelihood function for the MCMC simulation

# geometric Brownian motion model modelled with a Wiener process
# def likelihood(final_price, previous_price, mu, sigma, theta):
#     # Calculate the parameters of the geometric Brownian motion model
#     alpha = theta / sigma
#     beta = mu - 0.5 * sigma ** 2 / theta
#     # Calculate the expected log price and log volatility at the final time point
#     time = 1
#     log_mean = np.log(previous_price) + (beta - 0.5 * alpha ** 2) * time
#     log_var = (alpha ** 2) * time
#     # Calculate the log-likelihood of the final price
#     log_likelihood = norm.logpdf(np.log(final_price), loc=log_mean, scale=np.sqrt(log_var + sigma ** 2))
#     return np.exp(log_likelihood)

@jit(nopython=True)
def pdf(x, loc, scale):
    return np.exp(-0.5 * ((x - loc) / scale) ** 2) / np.sqrt(2 * np.pi) / scale

# mean-reverting model with Ornstein-Uhlenbeck process
# @jit(nopython=True)
# def likelihood(final_price, previous_price, mu, sigma, theta):
#     alpha = theta / sigma # mean reversion speed
#     beta = mu - 0.5 * sigma**2 / theta # mean reversion level
#     delta_t = 1 # time interval
#     # Calculate the price trajectory for the proposed parameter value
#     price_trajectory = beta + (previous_price - beta) * np.exp(-alpha * delta_t) + sigma * np.sqrt((1 - np.exp(-2 * alpha * delta_t)) / (2 * alpha))
#     # Calculate the likelihood of the final price
#     likelihood = pdf(final_price, loc=price_trajectory, scale=sigma) + 1e-10
#     return likelihood

# mean-reverting model with exponential Brownian motion process
@jit(nopython=True)
def likelihood(final_price, previous_price, mu, sigma, theta, randomness = np.random.normal()):
    delta_t = 1 # time interval
    drift = mu * previous_price * delta_t
    diffusion = sigma * previous_price * np.sqrt(delta_t) * randomness
    price_trajectory = previous_price + drift + diffusion
    # Calculate the likelihood of the final price
    likelihood = pdf(final_price, loc=price_trajectory, scale=sigma) + 1e-10
    return likelihood

# Define the proposal sampler for the MCMC simulation

# random normal proposal sampler
@jit(nopython=True)
def proposal_sampler(previous_price, mu, sigma):
    return np.random.normal(previous_price, sigma)

# Define the function to generate a large number of possible future price trajectories using the MCMC simulation
def price_prediction(last_price, days, num_simulations):
    price_matrix = np.zeros((days, num_simulations))
    for d in range(days):
        if d == 0:
            price_matrix[d] = last_price
        else:
            # Run the MCMC simulation
            chain = metropolis_hastings(likelihood, proposal_sampler, last_price, num_simulations)
            price_matrix[d] = chain
            last_price = chain[-1]
    return price_matrix

# For each simulated trajectory, calculate the final price at the end of the time horizon and the median or expected value of the final prices from all the simulated trajectories
def calculate_final_price(price_matrix):
    final_prices = price_matrix[-1]
    mean_price = final_prices.mean()
    return final_prices, mean_price

# Calculate the median price from all the simulated trajectories
def calculate_median_price(price_matrix):
    median_price = np.median(price_matrix, axis=1)
    lower_price = np.percentile(price_matrix, 25, axis=1)
    upper_price = np.percentile(price_matrix, 75, axis=1)
    return median_price, lower_price, upper_price



# Collect historical price data for the stock
# data = collect_stock_data("AAPL", "max")
# data.to_csv("apple_stock_data.csv")

# Load data from csv file
data = load_stock_data("apple_stock_data.csv")

# Calculate the mean and standard deviation of the daily returns for the historical data
mu, sigma, theta = calculate_daily_returns(data['Close'])

# Generate a large number of possible future price trajectories using the MCMC simulation
days = 365
num_simulations = 1000
last_price = data['Close'][-1]

# Run the MCMC simulation to generate a large number of possible future price trajectories
price_matrix = price_prediction(last_price, days, num_simulations)

# Calculate the final price at the end of the time horizon and the median or expected value of the final prices from all the simulated trajectories
final_prices, mean_price = calculate_final_price(price_matrix)

# Plot the predicted stock price alongside with the history stock price using seaborn
sns.set()
plt.figure(figsize=(10, 6))
plt.hist(final_prices, bins=30)
plt.axvline(mean_price, color='r', linestyle='dashed', linewidth=2)
plt.title("MCMC Simulation: AAPL")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Calculate the median price from all the simulated trajectories
median_price, lower_price, upper_price = calculate_median_price(price_matrix)

# Plot the price trajectories using a line plot
plt.plot(price_matrix, color='gray', alpha=0.25)
plt.plot(median_price, color='blue', linewidth=2)
plt.plot(lower_price, color='red', linewidth=1)
plt.plot(upper_price, color='green', linewidth=1)
plt.title("MCMC Simulation: AAPL")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()

# Plot historical data and all the simulated price trajectories using a line plot
plt.plot(data.index, data['Close'].values, label='Historical Data')
plt.plot(pd.date_range(start=data.index[-1], periods=days, freq='D'), price_matrix, color='gray', alpha=0.25)
plt.plot(pd.date_range(start=data.index[-1], periods=days, freq='D'), median_price, color='blue', linewidth=2)
plt.plot(pd.date_range(start=data.index[-1], periods=days, freq='D'), lower_price, color='red', linewidth=1)
plt.plot(pd.date_range(start=data.index[-1], periods=days, freq='D'), upper_price, color='green', linewidth=1)
# fill the area between the 25th and 75th percentiles
plt.title("MCMC Simulation: AAPL")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()



"""
The implementation assumes that the daily returns are normally distributed, which may not always be the case in practice. Different distributions or models may be more appropriate depending on the data and context.
The implementation does not take into account any external factors or events that may affect the stock price, such as news or market trends. Incorporating such information may improve the accuracy of the predictions.
The implementation uses the same proposal distribution for all the iterations of the MCMC simulation, which may not be optimal. Tuning the proposal distribution or using adaptive MCMC methods may improve the efficiency and convergence of the simulation.
The implementation uses a single value for the standard deviation of the daily returns, which may not capture the full range of variability in the data. Using a more robust estimator or considering different levels of volatility may improve the accuracy of the predictions.
"""