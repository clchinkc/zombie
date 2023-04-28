
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Load the stock price data
data = pd.read_csv('apple_stock_data.csv', parse_dates=['Date'], index_col='Date')


def log_likelihood(theta, data):
    mu, sigma = theta
    log_likelihood = np.sum(norm.logpdf(data, loc=mu, scale=sigma))
    return log_likelihood

def log_prior(theta):
    mu, sigma = theta
    if 0 < sigma < 1000:
        return norm.logpdf(mu, loc=0, scale=100) + norm.logpdf(sigma, loc=0, scale=10)
    return -np.inf

def log_posterior(theta, data):
    log_prior_ = log_prior(theta)
    if np.isinf(log_prior_):
        return log_prior_
    log_likelihood_ = log_likelihood(theta, data)
    return log_prior_ + log_likelihood_

# Set the initial values of the parameters
theta = np.array([0, 1])

# Set the number of iterations and the burn-in period
n_iter = 10000
burn_in = 1000

# Set the standard deviation of the proposal distribution
proposal_sigma = 0.1

# Create an empty array to store the samples
samples = np.zeros((n_iter, 2))

# Run the Metropolis-Hastings algorithm
for i in range(n_iter):
    # Generate a proposal sample from the normal distribution
    theta_proposal = np.random.normal(theta, proposal_sigma)
    
    # Calculate the log-ratio of the posterior probabilities
    log_alpha = log_posterior(theta_proposal, data) - log_posterior(theta, data)
    
    # Accept or reject the proposal sample based on the log-ratio and a random number
    if np.log(np.random.rand()) < log_alpha:
        theta = theta_proposal
    
    # Store the current sample
    samples[i, :] = theta

# Discard the burn-in period and thin the samples
samples = samples[burn_in::10, :]

# Calculate the expected value of the stock price tomorrow
mu_pred = np.mean(samples[:, 0])
sigma_pred = np.mean(samples[:, 1])
stock_price_pred = np.random.normal(mu_pred, sigma_pred)

# Calculate the 95% credible interval of the stock price tomorrow
credible_interval = np.percentile(np.random.normal(samples[:, 0], samples[:, 1]), [2.5, 97.5])

print(f"Expected stock price tomorrow: {stock_price_pred:.2f}")
print(f"95% credible interval: {credible_interval[0]:.2f} - {credible_interval[1]:.2f}")
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Replace this with your actual stock price data
data = pd.read_csv('apple_stock_data.csv', parse_dates=['Date'], index_col='Date')
close_prices = data['Close'].values
X = close_prices[:-14]   # use all data except last 14 days for training
Y = close_prices[14:]    # use last 14 days as target for prediction


# Define the prior distribution for the weights (intercept and slope) of the linear model
prior_mean = np.array([0, 0])
prior_cov = np.array([[1, 0], [0, 1]])

# Define the likelihood function for the data
def likelihood(X, Y, params, sigma):
    a, b = params
    Y_pred = a * X + b
    squared_errors = (Y - Y_pred) ** 2
    return np.exp(-0.5 * np.sum(squared_errors) / sigma ** 2)

def posterior(X, Y, prior_mean, prior_cov, sigma):
    N = len(X)
    X_ = np.column_stack((X, np.ones(N)))
    post_cov = np.linalg.inv(np.linalg.inv(prior_cov) + (1 / sigma ** 2) * X_.T @ X_)
    post_mean = post_cov @ (np.linalg.inv(prior_cov) @ prior_mean + (1 / sigma ** 2) * X_.T @ Y)
    return post_mean, post_cov

# Set the noise level (you would need to estimate this from your data)
sigma = 1.0

# Compute the posterior distribution for the model parameters
post_mean, post_cov = posterior(X, Y, prior_mean, prior_cov, sigma)

# Generate predictions for future data
X_future = np.linspace(X[-1], X[-1]+13, 14)  # predict 14 future days
Y_future_mean = post_mean[0] * X_future + post_mean[1]
Y_future_std = np.sqrt(post_cov[0, 0] * X_future ** 2 + post_cov[1, 1] + 2 * post_cov[0, 1] * X_future)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(X, Y, "o", label="Observed data")
plt.plot(X_future, Y_future_mean, "ro", label="Predicted mean")
plt.errorbar(X_future, Y_future_mean, yerr=1.96 * Y_future_std, capsize=5, fmt='none', ecolor='r', label="95% confidence interval")
plt.xlabel("Previous Day's Close Price")
plt.ylabel("Current Day's Close Price")
plt.legend()
plt.show()

