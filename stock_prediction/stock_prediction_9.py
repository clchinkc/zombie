import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

data = pd.read_csv('apple_stock_data.csv', parse_dates=['Date'], index_col='Date')
close_prices = data['Close'].values

# 1. Extended Model:
X = close_prices[:-14]
Y = close_prices[14:]

# 2. Prior Distributions:
prior_mean = np.array([0, 0, 0, 1])  # a, b, mu, sigma
prior_cov = np.diag([1, 1, 100, 10])  

# 3. Likelihood Function:
def likelihood(X, Y, params):
    a, b, mu, sigma = params
    Y_pred = a * X + b
    residuals = Y - Y_pred
    return np.sum(norm.logpdf(residuals, loc=mu, scale=sigma))

# 4. Posterior using Metropolis-Hastings:
def log_posterior(theta, X, Y):
    a, b, mu, sigma = theta
    if 0 < sigma < 1000:
        prior_prob = np.sum(norm.logpdf(theta, loc=prior_mean, scale=np.sqrt(np.diag(prior_cov))))
        likelihood_val = likelihood(X, Y, theta)
        return prior_prob + likelihood_val
    return -np.inf

# Metropolis-Hastings parameters
theta = prior_mean
n_iter = 10000
burn_in = 1000
proposal_sigma = np.diag([0.1, 0.1, 0.1, 0.01])
samples = np.zeros((n_iter, 4))

for i in range(n_iter):
    theta_proposal = np.random.multivariate_normal(theta, proposal_sigma)
    log_alpha = log_posterior(theta_proposal, X, Y) - log_posterior(theta, X, Y)
    if np.log(np.random.rand()) < log_alpha:
        theta = theta_proposal
    samples[i, :] = theta

samples = samples[burn_in::10, :]
a_samples = samples[:, 0]
b_samples = samples[:, 1]
mu_samples = samples[:, 2]
sigma_samples = samples[:, 3]

# 5. Recursive Predictions for the next 14 days:
future_predictions = np.zeros((len(a_samples), 14))

last_observed_value = close_prices[-14]  # Start with the last observed value

for j in range(14):
    daily_predictions = np.zeros(len(a_samples))
    for i, (a, b, mu, sigma) in enumerate(zip(a_samples, b_samples, mu_samples, sigma_samples)):
        pred = (a * last_observed_value + b) + np.random.normal(mu, sigma)
        daily_predictions[i] = pred
    
    mean_daily_prediction = np.mean(daily_predictions)
    future_predictions[:, j] = daily_predictions
    last_observed_value = mean_daily_prediction  # Use the mean prediction as the input for the next day

# Aggregate predictions
mean_predictions = np.mean(future_predictions, axis=0)
credible_intervals = np.percentile(future_predictions, [2.5, 97.5], axis=0)

# 6. Visualization and Analysis:
plt.figure(figsize=(10, 5))
plt.plot(data.index[-28:-14], close_prices[-28:-14], 'o-', label='Observed data (last 14 days)')
plt.plot(data.index[-14:], close_prices[-14:], 's-', color="green", label="Actual Prices for Forecasted Period")
plt.plot(data.index[-14:], mean_predictions, 'ro-', label='Predicted Prices')
plt.fill_between(data.index[-14:], credible_intervals[0, :], credible_intervals[1, :], color='skyblue', alpha=0.5, label='95% Credible Interval')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Recursive Stock Price Predictions Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
