import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eig, svd


def load_stock_data(filename):
    data = pd.read_csv(filename, index_col='Date', parse_dates=True)
    prices = data['Close'].values
    return prices

def compute_dmd_matrix(X):
    U, Sigma, Vh = svd(X, full_matrices=False)
    S_inv = np.diag(1 / Sigma)
    A_tilde = U.T @ X @ Vh.T @ S_inv
    return A_tilde

def find_dominant_modes(eigenvalues, eigenvectors, num_modes):
    dominant_indices = np.argsort(np.abs(eigenvalues))[-num_modes:]
    dominant_eigenvalues = eigenvalues[dominant_indices]
    dominant_eigenvectors = eigenvectors[:, dominant_indices]
    return dominant_eigenvalues, dominant_eigenvectors

def predict_future_prices(A_tilde, X, dominant_eigenvalues, dominant_eigenvectors, forecast_steps):
    n_steps = X.shape[1]
    state = X[:, -1]

    # Compute future states using matrix power in a loop
    future_states = np.empty((forecast_steps, A_tilde.shape[0]))
    for i in range(forecast_steps):
        future_state = np.linalg.matrix_power(A_tilde, i + 1) @ state
        future_states[i] = future_state

    future_prices = future_states[:, 1]

    return future_prices

def plot_results(prices, future_prices):
    n_steps = len(prices)
    forecast_steps = len(future_prices)
    time = np.arange(n_steps + forecast_steps)

    plt.plot(time[:n_steps], prices, label='Actual Prices')
    plt.plot(time[n_steps:], future_prices, label='Predicted Prices', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Apple Stock Prices')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load data
    prices = load_stock_data('apple_stock_data.csv')

    # Create data matrix
    X = np.vstack((prices[:-1], prices[1:]))

    # Perform SVD
    A_tilde = compute_dmd_matrix(X)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(A_tilde)

    # Find dominant modes
    num_modes = 5
    dominant_eigenvalues, dominant_eigenvectors = find_dominant_modes(eigenvalues, eigenvectors, num_modes)

    # Forecast future prices
    forecast_steps = 30
    future_prices = predict_future_prices(A_tilde, X, dominant_eigenvalues, dominant_eigenvectors, forecast_steps)

    # Plot actual and predicted prices
    plot_results(prices, future_prices)