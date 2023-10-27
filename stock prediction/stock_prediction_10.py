
"""
Stochastic control theory can be used in stock price prediction by providing a framework for making decisions that optimize the behavior of the stock price over time. The stock price is a random variable that evolves over time, and stochastic control theory provides a way to design control strategies that take into account the uncertainty and randomness of the stock price.
One way to use stochastic control theory in stock price prediction is to model the stock price as a stochastic process and to design a control policy that maximizes the expected return or minimizes the risk of the stock portfolio. This can be done by using tools such as dynamic programming or stochastic differential equations to model the behavior of the stock price over time and to optimize the control policy.
"""
"""
This code is using dynamic programming and Bellman's equation to solve a continuous-time stochastic control problem known as the Merton problem. The model assumes that the stock price follows a geometric Brownian motion (GBM) and uses Monte Carlo simulation to estimate the transition probabilities and rewards for the discretized state and action spaces. The optimal policy is obtained by iteratively solving the Bellman's equation until convergence.
"""

import matplotlib.pyplot as plt
import numpy as np

# Parameters
initial_stock_price = 100
expected_return = 0.05
stock_volatility = 0.2
time_step = 1
risk_free_rate = 0.03
n_simulation = 1000
n_actions = 101

# Discretize action space: Fractions representing percentage invested in stocks
action_space = np.linspace(0, 1, n_actions)
stock_price_space = np.linspace(0, 2 * initial_stock_price, 2 * n_actions)

# Initialize rewards
rewards = np.zeros((len(stock_price_space), len(action_space)))
transition_probs = np.zeros((len(stock_price_space), len(action_space), len(stock_price_space)))

np.random.seed(42)
simulated_returns = np.random.normal(expected_return * time_step, stock_volatility * np.sqrt(time_step), n_simulation)

for current_price_idx, current_price in enumerate(stock_price_space):
    for action_idx, fraction_invested in enumerate(action_space):
        combined_rewards = []
        
        for simulated_return in simulated_returns:
            stock_return = fraction_invested * simulated_return
            cash_return = (1 - fraction_invested) * risk_free_rate * time_step
            total_return = stock_return + cash_return
            combined_rewards.append(current_price * total_return)
            
            next_price = current_price * (1 + total_return)
            idx_next_price = np.argmin(np.abs(stock_price_space - next_price))
            transition_probs[current_price_idx, action_idx, idx_next_price] += 1 / n_simulation

        rewards[current_price_idx, action_idx] = np.mean(combined_rewards)

# Value iteration
value_function = np.zeros(len(stock_price_space))
max_iterations = 100
convergence_tolerance = 1e-6
discount_factor = np.exp(-risk_free_rate * time_step)

for _ in range(max_iterations):
    new_value_function = np.zeros(len(stock_price_space))
    for i in range(len(stock_price_space)):
        action_values = rewards[i] + discount_factor * np.dot(transition_probs[i], value_function)
        new_value_function[i] = np.max(action_values)
    if np.max(np.abs(value_function - new_value_function)) < convergence_tolerance:
        break
    value_function = new_value_function

# Simulate stock price path and find the optimal investment fraction for each stock price
T = 100
stock_price_path = [initial_stock_price]
optimal_fraction_path = []

for t in range(T - 1):
    dt_return = np.random.normal(expected_return * time_step, stock_volatility * np.sqrt(time_step))
    next_stock_price = stock_price_path[-1] * (1 + dt_return)
    stock_price_path.append(next_stock_price)

    # Derive the optimal action for the current stock price
    idx_closest_price = np.argmin(np.abs(stock_price_space - stock_price_path[-1]))
    optimal_fraction = action_space[np.argmax(rewards[idx_closest_price] + discount_factor * np.dot(transition_probs[idx_closest_price], value_function))]
    optimal_fraction_path.append(optimal_fraction)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Stock Price', color=color)
ax1.plot(stock_price_path, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Fraction Invested in Stock', color=color)
ax2.plot(optimal_fraction_path, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Stock Price and Optimal Fraction Invested in Stock Over Time')
plt.show()




"""
A more appropriate approach to this problem would be to use more sophisticated models and techniques like continuous-time portfolio optimization, which involves solving a Hamilton-Jacobi-Bellman (HJB) partial differential equation.
"""
