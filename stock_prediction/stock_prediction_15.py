import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from hurst import compute_Hc
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values

def detrend_normalize(data, window_size=7):
    trend = np.polyfit(np.arange(len(data)), data, 1)[0] * np.arange(len(data))
    detrended = data - trend
    normalized = detrended / np.linalg.norm(detrended)
    return normalized, trend

def extrapolate_trend(trend, next_n):
    """ Extrapolate the trend using linear regression for the next 'next_n' points. """
    X = np.arange(len(trend)).reshape(-1, 1)
    y = trend.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    future_x = np.arange(len(trend), len(trend) + next_n).reshape(-1, 1)
    future_trend = reg.predict(future_x).reshape(-1)
    return future_trend

def time_delay_embedding(data, dimension, delay):
    N = len(data)
    X = np.zeros((dimension, N - (dimension-1)*delay))
    for i in range(dimension):
        X[i] = data[i*delay : i*delay + X.shape[1]]
    return X


def calculate_rolling_hurst(stock_data, window_size=252):
    """ Calculate the rolling Hurst exponent """
    hurst_values = []
    for i in range(len(stock_data) - window_size):
        H = compute_Hc(stock_data[i:i+window_size], kind='price', simplified=True)[0]
        hurst_values.append(H)
    return hurst_values


def construct_data_matrices(feature_matrix):
    """Construct the data matrices X and X_prime for DMD."""
    X = feature_matrix[:, :-1]
    X_prime = feature_matrix[:, 1:]
    return X, X_prime

def compute_dmd_matrix(X):
    """ Compute the DMD matrix using X and its shifted version X'. """
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    # Singular Value Decomposition
    U, Sigma, Vh = np.linalg.svd(X1, full_matrices=False)

    # Invert Sigma
    Sigma_inv = np.diag(1.0 / Sigma)

    # Compute A_tilde
    A_tilde = np.dot(U.T, np.dot(X2, np.dot(Vh.T, Sigma_inv)))

    return A_tilde

def compute_dmd_modes_and_freqs(X, A_tilde):
    """ Compute DMD modes and frequencies using eigen decomposition. """
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
    
    # Project onto original data
    DMD_modes = np.dot(X[:, 1:], np.dot(np.linalg.pinv(eigenvectors), np.diag(np.exp(np.log(eigenvalues)))))

    # Compute DMD frequencies
    DMD_freqs = np.log(eigenvalues).imag
    
    return DMD_modes, DMD_freqs

def predict_using_dmd_modes(X, DMD_modes, DMD_freqs, forecast_steps=50):
    """Predict future states using DMD modes and frequencies."""
    b = np.linalg.lstsq(DMD_modes, X[:, 0], rcond=None)[0]
    time_dynamics = np.zeros((DMD_modes.shape[1], forecast_steps))
    
    for t in range(forecast_steps):
        time_dynamics[:, t] = np.exp(DMD_freqs * t) * b
    
    return np.abs(DMD_modes @ time_dynamics)[-1]


def lorenz_system(t, xyz, sigma, beta, rho):
    """ Lorenz System dynamics """
    x, y, z = xyz
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def fit_to_lorenz(embedded_data, forecast_steps):
    data_len = embedded_data.shape[1]
    initial_condition = embedded_data[:, -1]  # Use the last known point as the initial condition

    def cost_function(params):
        sol = solve_ivp(lorenz_system, [0, data_len + forecast_steps], initial_condition, args=params, t_eval=np.arange(data_len + forecast_steps))
        return (sol.y[:, -forecast_steps:] - embedded_data[:, -forecast_steps:]).flatten()

    res = least_squares(cost_function, x0=[10, 28, 8/3])

    # Once we have the best fit parameters, we'll forecast the next values
    sol = solve_ivp(lorenz_system, [0, forecast_steps], initial_condition, args=res.x, t_eval=np.arange(forecast_steps))
    return sol.y[0]  # Return x-values from Lorenz system


def predict_using_clusters(embedded_data, forecast_steps):
    kmeans = KMeans(n_clusters=3, n_init="auto").fit(embedded_data.T)
    cluster_centers = kmeans.cluster_centers_
    
    predictions = [embedded_data[:, -1]]
    for _ in range(forecast_steps - 1):  
        distances = np.linalg.norm(cluster_centers - predictions[-1].reshape(-1, 1), axis=1)
        predictions.append(cluster_centers[np.argmin(distances)])
    return np.array(predictions).T[0]  # Return the x-values from the cluster centers


def denormalize_trend(data_normalized_detrended, min_val, max_val, trend):
    data_detrended = data_normalized_detrended * (max_val - min_val) + min_val
    original_data = data_detrended + trend
    return original_data

def consolidate_predictions(lorenz_pred, cluster_pred, hurst_value):
    lorenz_weight = hurst_value
    return lorenz_pred * lorenz_weight + cluster_pred * (1 - lorenz_weight)


def visualize_combined_data(data, dmd_predictions, lorenz_predictions, cluster_predictions, combined_predictions, trend, hurst_values, window_size):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    time = np.arange(len(data) - len(hurst_values))
    
    # Assuming Hurst values can range between 0 and 1
    for i, H in enumerate(hurst_values):
        if i + window_size < len(data):
            color = plt.cm.RdYlGn(H)  # Get color based on Hurst value
            ax.axvspan(i, i + window_size, facecolor=color, alpha=0.1)
    
    ax.plot(data, label='Actual Stock Prices', color='black')
    ax.plot(np.arange(len(data), len(data) + len(dmd_predictions)), dmd_predictions, label='DMD Predictions', color='red')
    ax.plot(np.arange(len(data), len(data) + len(lorenz_predictions)), lorenz_predictions, label='Lorenz Predictions', color='blue')
    ax.plot(np.arange(len(data), len(data) + len(cluster_predictions)), cluster_predictions, label='Cluster Predictions', color='green')
    ax.plot(np.arange(len(data), len(data) + len(combined_predictions)), combined_predictions, label='Combined Predictions', color='purple')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Price Predictions using DMD, Lorenz System, and Clustering')
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    # Data acquisition and preprocessing
    stock_data = fetch_data('AAPL', '2020-01-01', '2022-12-31')
    min_val = np.min(stock_data)
    max_val = np.max(stock_data)
    normalized_data, trend = detrend_normalize(stock_data, window_size=7)
    embedded_data = time_delay_embedding(normalized_data, dimension=3, delay=1)
    
    window_size = 252
    forecast_steps = 30

    # Advanced Time-Series Analysis
    hurst_values = calculate_rolling_hurst(stock_data, window_size=window_size)

    # Extrapolate the trend
    future_trend = extrapolate_trend(trend, next_n=forecast_steps)

    # DMD Analysis
    X, X_prime = construct_data_matrices(embedded_data.T)
    dmd_matrix = compute_dmd_matrix(X)
    DMD_modes, DMD_freqs = compute_dmd_modes_and_freqs(X, dmd_matrix)
    dmd_predictions = predict_using_dmd_modes(X, DMD_modes, DMD_freqs, forecast_steps)
    dmd_predictions = denormalize_trend(dmd_predictions, min_val, max_val, future_trend)

    
    # Lorenz System Prediction
    lorenz_predictions = fit_to_lorenz(embedded_data, forecast_steps)
    lorenz_predictions = denormalize_trend(lorenz_predictions, min_val, max_val, future_trend)

    # Clustering-based Prediction
    cluster_predictions = predict_using_clusters(embedded_data, forecast_steps)
    cluster_predictions = denormalize_trend(cluster_predictions, min_val, max_val, future_trend)

    # Consolidate Predictions
    combined_predictions = consolidate_predictions(lorenz_predictions, cluster_predictions, hurst_values[-1])

    # Visualization
    visualize_combined_data(stock_data, dmd_predictions, lorenz_predictions, cluster_predictions, combined_predictions, trend, hurst_values, window_size)


"""
Takens' Theorem, introduced by Floris Takens in 1981, is a fundamental result in the theory of dynamical systems which provides a foundation for reconstructing a dynamical system from a series of observations. The theorem is especially significant in the analysis of time series data and is a cornerstone for the method of time-delay embeddings.

In essence, Takens' Theorem states that under certain conditions, the dynamics of a system can be fully reconstructed from a single scalar observation function. That is, if you have a sequence of observations taken at successive times, you can recreate a space that is topologically equivalent to the original phase space of the dynamical system. This is known as a reconstructed phase space.

Here's a simplified explanation of how it works:

1. **Observation Function**: You start with a function that records observations from the system over time. For instance, this could be a single variable from a system that evolves over time, such as a stock price.

2. **Time-Delay Embedding**: You construct vectors where each vector component is an observation separated by a fixed time delay. For example, if your observations are \( x(t) \), then a reconstructed state might look like \( (x(t), x(t + \tau), x(t + 2\tau), ..., x(t + (m-1)\tau)) \), where \( \tau \) is the time delay and \( m \) is the embedding dimension.

3. **Embedding Dimension**: Takens' Theorem tells us that as long as the embedding dimension \( m \) is large enough (more precisely, at least twice the dimension of the original attractor of the dynamical system), the reconstructed dynamics will be equivalent to the true underlying dynamics of the system, up to a diffeomorphism (a kind of smooth, invertible map).

This embedding technique allows for the examination of the dynamical properties of the system, such as cycles, periodicity, attractors, and chaotic behavior, using a time series of scalar measurements. The theorem is especially useful when the system in question has many unobservable or unmeasured variables.

In practical applications, such as in analyzing financial time series data, Takens' theorem suggests that it may be possible to capture the essential dynamics of the stock market using just the historical price data, even though the market is influenced by a large number of unobserved factors. However, identifying the correct time delay \( \tau \) and embedding dimension \( m \) is non-trivial and typically requires careful analysis and sometimes trial and error.
"""