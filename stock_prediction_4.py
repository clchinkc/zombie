
"""
To design an MCMC simulation for stock price prediction, we can follow the steps below:

Step 1: Define the Model
Choose a stochastic process to model the stock price. The model should take into account the past performance of the stock and any relevant external factors that may influence the stock price. One commonly used process is the geometric Brownian motion (GBM) model, which assumes that the logarithm of the stock price follows a normal distribution with a drift term and a volatility term. The GBM model can be expressed as follows:

dS = μSdt + σSdz

where S is the stock price, μ is the drift rate, σ is the volatility, dt is a small time interval, and dz is a standard normal random variable.

S(t+1) = S(t) * exp((μ - 0.5 * σ^2) * Δt + σ * ε * sqrt(Δt))

where S(t) is the stock price at time t, μ is the expected return, σ is the volatility, Δt is the time interval, and ε is a random variable drawn from a normal distribution.

dln(S_t) = (μ - 0.5σ^2) dt + σ dW_t

where S_t is the stock price at time t, μ is the expected return, σ is the volatility, W_t is the Brownian motion, and dln(S_t) is the differential of the logarithm of S_t.

Step 2: Define the Prior Distribution
The next step is to define the prior distribution for the parameters in the model. Choose an initial value for the stock price, S_0, and set the time interval, Δt, between each time step in the simulation. These priors can be based on historical data or expert knowledge. For example, we can use a normal distribution for μ and a uniform distribution for σ.

Step 3: Define the Likelihood Function
The likelihood function represents the probability of the observed data given the model parameters. The likelihood function should be consistent with the assumptions made in the model. In the case of stock prices, the likelihood function can be defined as the product of the probabilities of each observed stock price given the model parameters. We can assume that the errors are normally distributed and use the following likelihood function:

L(μ, σ | S) = Π (1 / sqrt(2 * π * σ^2)) * exp(-(S(t+1) - S(t) * exp((μ - 0.5 * σ^2) * Δt))^2 / (2 * σ^2 * Δt))

where S is the vector of observed stock prices.

L(θ|data) = ∏i=1n (1/(√(2π)σSi))exp(-(ln(Si/Si-1) - μΔt)^2/(2σ^2Δt))

where θ represents the parameters μ and σ, data represents the historical stock prices, Si represents the stock price at time i, and Δt represents the time interval between observations.

Step 4: Define the Metropolis-Hastings Algorithm
The Metropolis-Hastings algorithm is a common MCMC algorithm used for sampling from the posterior distribution. The algorithm involves proposing a new set of parameters based on the prior distribution and calculating the acceptance probability based on the likelihood function and the previous set of parameters. Then, generate a Markov chain of stock price values. At each step of the algorithm, propose a new value for the stock price by adding a random increment to the current value and simulating the Brownian motion term in the GBM model, such that:

S' = S + ε

where ε is a normally distributed random number with mean 0 and variance Δt. The acceptance probability for the proposed value is given by:

α = min(1, exp(ln(S'_t/S_t)))

where S_t and S'_t are the current and proposed stock prices, respectively. If the proposed value is accepted, set the current value to the proposed value. Otherwise, keep the current value. The algorithm should be run for a sufficient number of iterations to ensure convergence.

Step 5: Repeat the Monte Carlo process
Repeat step 4 for a large number of iterations to generate a long Markov chain of stock price values.

Step 6: Estimate the distribution of the stock price
Use the Markov chain to estimate the distribution of the stock price at a given time in the future. We can compute the mean and standard deviation of the stock price distribution from the Markov chain, and use these values to estimate the probability of the stock price being within a certain range at a future time.

Step 7: Evaluate the results
Repeat steps 4-6 for different values of the initial stock price, μ, and σ to explore the effect of these parameters on the stock price distribution and determine the most likely values for the parameters μ and σ. We can look at the posterior distribution of the parameters to see how much uncertainty there is in the predictions. We can also compare the predicted values to actual stock prices to see how well the model performed in practice.

Write complete code in python.
"""
"""
import numpy as np

# define parameters
S0 = 100.0 # initial stock price
mu = 0.05 # expected return
sigma = 0.2 # volatility
T = 1.0 # time horizon
N = 252 # number of time steps
dt = T/N # time interval

# simulate stock prices using GBM model
t = np.linspace(0, T, N+1)
S = np.zeros(N+1)
S[0] = S0
for i in range(1, N+1):
    epsilon = np.random.normal(0, 1)
    S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * epsilon * np.sqrt(dt))

# plot the stock prices
import matplotlib.pyplot as plt

plt.plot(t, S)
plt.xlabel('Time (years)')
plt.ylabel('Stock price ($)')
plt.title('Geometric Brownian Motion Model')
plt.show()
"""


"""
import numpy as np


# Define the Model
def GBM_model(S, mu, sigma, dt, epsilon):
    return S * np.exp((mu - 0.5 * sigma**2) * dt + sigma * epsilon * np.sqrt(dt))



# Define the Prior Distribution
S_0 = 100.0 # initial stock price
mu_prior = (0.05, 0.1) # prior range for mu
sigma_prior = (0.1, 0.3) # prior range for sigma

# Define the Likelihood Function
def likelihood(mu, sigma, S):
    n = len(S)
    log_return = np.diff(np.log(S))
    ll = -0.5 * n * np.log(2 * np.pi) - n * np.log(sigma) - 0.5 * np.sum(log_return - mu * dt)**2 / sigma**2
    return ll

# Define the Metropolis-Hastings Algorithm
def metropolis_hastings(S, mu, sigma, iterations):
    samples = np.zeros((iterations, 2))
    samples[0, :] = mu, sigma
    for i in range(1, iterations):
        mu_p = np.random.normal(mu, 0.1)
        sigma_p = np.random.uniform(sigma - 0.1, sigma + 0.1)
        mu_likelihood = likelihood(mu_p, sigma, S)
        mu_prior_prob = np.log(np.random.normal(mu_prior[0], mu_prior[1]))
        sigma_likelihood = likelihood(mu, sigma_p, S)
        sigma_prior_prob = np.log(1 / (sigma_prior[1] - sigma_prior[0]))
        current_likelihood = likelihood(mu, sigma, S)
        current_prior_prob = np.log(np.random.normal(mu_prior[0], mu_prior[1])) + np.log(1 / (sigma_prior[1] - sigma_prior[0]))
        acceptance_prob = np.exp(mu_likelihood - current_likelihood + mu_prior_prob - current_prior_prob +
                                sigma_likelihood - current_likelihood + sigma_prior_prob - current_prior_prob)
        acceptance_prob = min(1, acceptance_prob)
        if np.random.uniform() < acceptance_prob:
            mu, sigma = mu_p, sigma_p
        samples[i, :] = mu, sigma
    return samples

# Generate a Markov Chain of Stock Price Values
def simulate_stock_price(S_0, mu, sigma, dt, T, iterations):
    S = np.zeros(iterations)
    S[0] = S_0
    for i in range(1, iterations):
        epsilon = np.random.normal()
        S[i] = GBM_model(S[i-1], mu, sigma, dt, epsilon)
    return S

# Repeat the Monte Carlo process
iterations = 1000 # number of iterations for MCMC
T = 1.0 # # total time horizon
dt = 1.0 / iterations # time interval
mu = 0.05 # expected return
sigma = 0.2 # volatility
S = simulate_stock_price(S_0, mu, sigma, dt, T, iterations)
samples = metropolis_hastings(S, mu_prior, sigma_prior, iterations)

# Estimate the distribution of the stock price
S_future = simulate_stock_price(S[-1], np.mean(samples[:, 0]), np.mean(samples[:, 1]), dt, T, iterations)
mean = np.mean(S_future)
std = np.std(S_future)

import matplotlib.pyplot as plt

# Evaluate the results
# Compare predicted values to actual stock prices and analyze the posterior distribution of the parameters
import pandas as pd

# load the actual stock prices
df = pd.read_csv('apple_stock_data.csv', index_col='Date', parse_dates=True)

# plot the actual stock prices
plt.plot(df.index, df['Close'], label='Actual Stock Price')

# plot the predicted stock prices
plt.plot(df.index, S_future, label='Predicted Stock Price')

# plot the mean and standard deviation of the predicted stock prices
plt.plot(df.index, mean + std, label='Mean + Std')
plt.plot(df.index, mean - std, label='Mean - Std')

plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.title('Apple Stock Price Prediction')
plt.legend()
plt.show()
"""


import matplotlib.pyplot as plt
import numpy as np


# Define the Model
def GBM(S0, mu, sigma, dt, T):
    N = round(T/dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N)
    W = np.cumsum(W)*np.sqrt(dt) 
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = S0*np.exp(X) 
    return S

# Define the Prior Distribution
S0 = 100 # initial stock price
mu_prior = 0.05 # prior mean for mu
sigma_prior = 0.2 # prior range for sigma
dt = 1/252 # time step (1 trading day)
T = 1 # time horizon (1 year)
N = round(T/dt)
t = np.linspace(0, T, N)

# Define the Likelihood Function
def likelihood(S, mu, sigma, dt):
    N = len(S)-1
    log_S = np.log(S)
    log_ret = np.diff(log_S)
    llh = -0.5*N*np.log(2*np.pi) - N*np.log(sigma) - 0.5*np.sum(log_ret-mu*dt)**2/sigma**2
    return llh

# Define the Metropolis-Hastings Algorithm
def MCMC(S, mu_prior, sigma_prior, dt, T, n_iter):
    # initialize arrays
    mu_array = np.zeros(n_iter)
    sigma_array = np.zeros(n_iter)
    llh_array = np.zeros(n_iter)
    S_array = np.zeros((n_iter, len(S)))
    S_array[0,:] = S

    # initialize starting values
    mu = mu_prior
    sigma = np.random.uniform(sigma_prior[0], sigma_prior[1])
    llh = likelihood(S, mu, sigma, dt)

    # run MCMC algorithm
    for i in range(1, n_iter):
        # generate proposal
        mu_p = np.random.normal(mu, 0.01)
        sigma_p = np.random.normal(sigma, 0.01)
        if sigma_p < 0:
            sigma_p = -sigma_p
        # compute likelihood for proposal
        llh_p = likelihood(S, mu_p, sigma_p, dt)
        # compute acceptance probability
        alpha = np.exp(llh_p - llh)
        if alpha > 1:
            mu = mu_p
            sigma = sigma_p
            llh = llh_p
        else:
            u = np.random.uniform(0, 1)
            if u < alpha:
                mu = mu_p
                sigma = sigma_p
                llh = llh_p
        # store results
        mu_array[i] = mu
        sigma_array[i] = sigma
        llh_array[i] = llh
        S_array[i,:] = GBM(S[-1], mu, sigma, dt, T)

    return mu_array, sigma_array, llh_array, S_array

# Run the MCMC algorithm
S = GBM(S0, mu_prior, sigma_prior, dt, T)
n_iter = 10000
mu_array, sigma_array, llh_array, S_array = MCMC(S, mu_prior, [0.001, 0.5], dt, T, n_iter)

# Estimate the distribution of the stock price
S_mean = np.mean(S_array, axis=0)
S_std = np.std(S_array, axis=0)

# Evaluate the results
# Compare predicted values to actual stock prices and analyze the posterior distribution of the parameters
import pandas as pd

# load the actual stock prices
df = pd.read_csv('apple_stock_data.csv', index_col='Date', parse_dates=True)

# plot the actual stock prices
plt.plot(df.index, df['Close'], label='Actual Stock Price')

# plot the mean and standard deviation of the predicted stock prices
plt.plot(df.index, S_mean[1:], label='Predicted Stock Price')
plt.plot(df.index, S_mean[1:] + S_std[1:], label='Mean + Std')
plt.plot(df.index, S_mean[1:] - S_std[1:], label='Mean - Std')

plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.title('Apple Stock Price Prediction')
plt.legend()
plt.show()

