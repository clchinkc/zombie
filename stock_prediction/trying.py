import matplotlib.pyplot as plt
import numpy as np

# Parameters
initial_stock_price = 100.
expected_return = 0.05
stock_volatility = 0.2
time_step = 1/252
risk_free_rate = 0.03
n_simulation = 1000
n_actions = 101
risk_aversion = 0.01  # risk aversion parameter
discount_factor = np.exp(-risk_free_rate * time_step)

def utility(wealth, rho=risk_aversion):
    if rho == 1:
        return np.log(wealth)
    else:
        return wealth**(1-rho) / (1-rho)

def kalman_predict(state_estimates, covariances, A, Q):
    next_state_estimates = A @ state_estimates
    next_covariances = A @ covariances @ A.T + Q
    return next_state_estimates, next_covariances

def kalman_update(state_estimates, covariances, observations, C, R):
    observations = np.atleast_2d(observations).T  # Ensure observations is a 2D column vector

    # Compute Kalman Gain
    C_covariance_product = C @ covariances
    kalman_gains = covariances @ C.T @ np.linalg.inv(C_covariance_product @ C.T + R)

    # Update State Estimate and Covariance
    updated_state_estimates = state_estimates + kalman_gains @ (observations - C @ state_estimates)
    updated_covariances = covariances - kalman_gains @ C @ covariances

    return updated_state_estimates, updated_covariances

def simulate_stock_price(T, initial_state, initial_covariance, A, C, Q, R, expected_return, stock_volatility, time_step):
    state_estimates = np.full((1, T), initial_state)
    covariances = np.full((1, 1, T), initial_covariance)
    true_prices = np.full(T, initial_stock_price)
    observations = np.full(T, initial_stock_price)
    
    for t in range(1, T):
        true_price = true_prices[t-1] * np.exp(np.random.normal(expected_return * time_step, stock_volatility * np.sqrt(time_step)))
        observation = true_price + np.random.normal(0, stock_volatility * np.sqrt(time_step))
        state_estimates[:, t:t+1], covariances[:, :, t:t+1] = kalman_predict(state_estimates[:, t-1:t], covariances[:, :, t-1:t], A, Q)
        state_estimates[:, t:t+1], covariances[:, :, t:t+1] = kalman_update(state_estimates[:, t:t+1], covariances[:, :, t:t+1], np.log(observation), C, R)

        true_prices[t] = true_price
        observations[t] = observation

    estimates = np.exp(state_estimates.flatten())

    return true_prices, observations, estimates

def get_stock_price_space(prices, buffer_percent=0.05):
    min_price_with_buffer = min(prices) * (1 - buffer_percent)
    max_price_with_buffer = max(prices) * (1 + buffer_percent)
    return np.linspace(min_price_with_buffer, max_price_with_buffer, 2 * n_actions)

def compute_final_wealth(current_price, fraction_invested, simulated_return, risk_free_rate, time_step):
    stock_return = fraction_invested * simulated_return
    cash_return = (1 - fraction_invested) * risk_free_rate * time_step
    total_return = stock_return + cash_return
    final_wealth = current_price * (1 + total_return)
    return final_wealth

def update_transition_probs(transition_probs, current_price_idx, action_idx, next_price, stock_price_space, n_simulation):
    idx_next_price = np.argmin(np.abs(stock_price_space - next_price))
    transition_probs[current_price_idx, action_idx, idx_next_price] += 1 / n_simulation

def compute_action_utility(current_price, fraction_invested, simulated_return, risk_free_rate, time_step, utility_function):
    final_wealth = compute_final_wealth(current_price, fraction_invested, simulated_return, risk_free_rate, time_step)
    return current_price * (final_wealth - current_price) + utility_function(final_wealth)

def compute_utilities_and_transitions(stock_price_space, n_simulation, action_space, expected_return, stock_volatility, risk_free_rate, time_step, utility):
    utilities = np.zeros((len(stock_price_space), len(action_space)))
    transition_probs = np.zeros((len(stock_price_space), len(action_space), len(stock_price_space)))
    simulated_returns = np.random.normal(expected_return * time_step, stock_volatility * np.sqrt(time_step), n_simulation)

    for current_price_idx, current_price in enumerate(stock_price_space):
        for action_idx, fraction_invested in enumerate(action_space):
            # Compute final_wealth for all simulated returns
            final_wealths = compute_final_wealth(current_price, fraction_invested, simulated_returns, risk_free_rate, time_step)
            
            # Compute utility values for all simulated returns
            utility_values = current_price * (final_wealths - current_price) + utility(final_wealths)
            
            # Compute mean of utility values
            utilities[current_price_idx, action_idx] = np.mean(utility_values)

            # Update transition probabilities
            for idx, next_price in enumerate(final_wealths):
                idx_next_price = np.argmin(np.abs(stock_price_space - next_price))
                transition_probs[current_price_idx, action_idx, idx_next_price] += 1 / n_simulation

    return utilities, transition_probs

def compute_state_value(i, value_function, utilities, transition_probs, discount_factor):
    action_values = utilities[i] + discount_factor * np.dot(transition_probs[i], value_function)
    return np.max(action_values)

def has_converged(old_value_function, new_value_function, tolerance):
    return np.max(np.abs(old_value_function - new_value_function)) < tolerance

def value_iteration(utilities, transition_probs, discount_factor, stock_price_space, max_iterations=100, convergence_tolerance=1e-6):
    value_function = np.zeros(len(stock_price_space))
    
    for _ in range(max_iterations):
        new_value_function = np.zeros(len(stock_price_space))
        for i in range(len(stock_price_space)):
            new_value_function[i] = compute_state_value(i, value_function, utilities, transition_probs, discount_factor)
            
        if has_converged(value_function, new_value_function, convergence_tolerance):
            break
            
        value_function = new_value_function

    return value_function

def calculate_next_stock_price(current_price, expected_return, stock_volatility, time_step):
    dt_return = np.random.normal(expected_return * time_step, stock_volatility * np.sqrt(time_step))
    next_stock_price = current_price * (1 + dt_return)
    return next_stock_price

def find_optimal_fraction(current_stock_price, stock_price_space, action_space, utilities, transition_probs, value_function, discount_factor):
    idx_closest_price = np.argmin(np.abs(stock_price_space - current_stock_price))
    optimal_fraction = action_space[np.argmax(utilities[idx_closest_price] + discount_factor * np.dot(transition_probs[idx_closest_price], value_function))]
    return optimal_fraction

def simulate_optimal_fraction_path(T, initial_stock_price, stock_price_space, action_space, utilities, transition_probs, value_function, expected_return, stock_volatility, time_step, discount_factor):
    stock_price_path = [initial_stock_price]
    optimal_fraction_path = []

    for t in range(T - 1):
        next_stock_price = calculate_next_stock_price(stock_price_path[-1], expected_return, stock_volatility, time_step)
        stock_price_path.append(next_stock_price)

        optimal_fraction = find_optimal_fraction(next_stock_price, stock_price_space, action_space, utilities, transition_probs, value_function, discount_factor)
        optimal_fraction_path.append(optimal_fraction)

    return stock_price_path, optimal_fraction_path

def plot_results(stock_price_path, optimal_fraction_path):
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
    
if __name__ == "__main__":
    # Parameters for Kalman Filter
    A = np.array([[1]])  # Identity for random walk
    C = np.array([[1]])  # Observation matrix
    Q = stock_volatility**2 * time_step  # Process noise
    R = stock_volatility**2 * time_step  # Observation noise
    initial_state = np.array([np.log(initial_stock_price)])
    initial_covariance = np.array([[stock_volatility**2 * time_step]])

    # Simulate stock price path
    T = 252
    true_stock_prices, observations, kalman_estimates = simulate_stock_price(T, initial_state, initial_covariance, A, C, Q, R, expected_return, stock_volatility, time_step)

    # Get stock price space
    stock_price_space = get_stock_price_space(true_stock_prices)

    # Get action space
    action_space = np.linspace(0, 1, n_actions)

    # Compute utilities and transition probabilities
    utilities, transition_probs = compute_utilities_and_transitions(stock_price_space, n_simulation, action_space, expected_return, stock_volatility, risk_free_rate, time_step, utility)

    # Value iteration
    value_function = value_iteration(utilities, transition_probs, discount_factor, stock_price_space)

    # Simulate stock price path and find the optimal investment fraction for each stock price
    _, optimal_fraction_path = simulate_optimal_fraction_path(T, initial_stock_price, stock_price_space, action_space, utilities, transition_probs, value_function, expected_return, stock_volatility, time_step, discount_factor)

    # Plotting
    plot_results(kalman_estimates, optimal_fraction_path)